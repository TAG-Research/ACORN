/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/HNSW.h>

#include <string>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/IDSelector.h>

// added
#include <sys/time.h>
#include <stdio.h>
#include <iostream>

/*******************************************************
 * Added for debugging
 *******************************************************/


// const int debugFlag = 2;

// void debugTime() {
// 	if (debugFlag) {
//         struct timeval tval;
//         gettimeofday(&tval, NULL);
//         struct tm *tm_info = localtime(&tval.tv_sec);
//         char timeBuff[25] = "";
//         strftime(timeBuff, 25, "%H:%M:%S", tm_info);
//         char timeBuffWithMilli[50] = "";
//         sprintf(timeBuffWithMilli, "%s.%06ld ", timeBuff, tval.tv_usec);
//         std::string timestamp(timeBuffWithMilli);
// 		std::cout << timestamp << std::flush;
//     }
// }

// //needs atleast 2 args always
// //  alt debugFlag = 1 // fprintf(stderr, fmt, __VA_ARGS__); 
// #define debug(fmt, ...) \
//     do { \
//         if (debugFlag == 1) { \
//             fprintf(stdout, "--" fmt, __VA_ARGS__);\
//         } \
//         if (debugFlag == 2) { \
//             debugTime(); \
//             fprintf(stdout, "%s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, __VA_ARGS__); \
//         } \
//     } while (0)



// double elapsed() {
//     struct timeval tv;
//     gettimeofday(&tv, NULL);
//     return tv.tv_sec + tv.tv_usec * 1e-6;
// }

namespace faiss {




/**************************************************************
 * HNSW structure implementation
 **************************************************************/

int HNSW::nb_neighbors(int layer_no) const {
    return cum_nneighbor_per_level[layer_no + 1] -
            cum_nneighbor_per_level[layer_no];
}

void HNSW::set_nb_neighbors(int level_no, int n) {
    FAISS_THROW_IF_NOT(levels.size() == 0);
    int cur_n = nb_neighbors(level_no);
    for (int i = level_no + 1; i < cum_nneighbor_per_level.size(); i++) {
        cum_nneighbor_per_level[i] += n - cur_n;
    }
}

int HNSW::cum_nb_neighbors(int layer_no) const {
    return cum_nneighbor_per_level[layer_no];
}

void HNSW::neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end)
        const {
    size_t o = offsets[no];
    // debug("offset o: %ld\n", o);
    *begin = o + cum_nb_neighbors(layer_no);
    // debug("begin: %ln\n", begin);
    *end = o + cum_nb_neighbors(layer_no + 1);
    // debug("end: %ln\n", end);
}

// HNSW::HNSW(int M) : rng(12345) {
//     set_default_probas(M, 1.0 / log(M));
//     max_level = -1;
//     entry_point = -1;
//     efSearch = 16;
//     efConstruction = 40 * 10;
//     upper_beam = 1;
//     gamma = 1;
//     offsets.push_back(0);
// }

HNSW::HNSW(int M, int gamma) : rng(12345) {
    set_default_probas(M, 1.0 / log(M), gamma);
    max_level = -1;
    entry_point = -1;
    efSearch = 16;
    efConstruction = M * gamma; //added gamma
    upper_beam = 1;
    this->gamma = gamma;
    this->M = M;
    // gamma = gamma;
    offsets.push_back(0);
    for (int i = 0; i < assign_probas.size(); i++) nb_per_level.push_back(0);
}

int HNSW::random_level() {
    double f = rng.rand_float();
    // could be a bit faster with bissection
    for (int level = 0; level < assign_probas.size(); level++) {
        if (f < assign_probas[level]) {
            return level;
        }
        f -= assign_probas[level];
    }
    // happens with exponentially low probability
    return assign_probas.size() - 1;
}

void HNSW::set_default_probas(int M, float levelMult, int gamma) {
    int nn = 0;
    cum_nneighbor_per_level.push_back(0);
    // printf("---set_default_probas: gamma: %d\n", this->gamma);
    // printf("---set_default_probas: gamma: %d\n", gamma);
    for (int level = 0;; level++) {
        float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
        if (proba < 1e-9)
            break;
        assign_probas.push_back(proba);
        nn += level == 0 ? M * 2 * gamma: M * gamma; // added gamma
        cum_nneighbor_per_level.push_back(nn);
    }
}

void HNSW::clear_neighbor_tables(int level) {
    for (int i = 0; i < levels.size(); i++) {
        size_t begin, end;
        neighbor_range(i, level, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            neighbors[j] = -1;
        }
    }
}

void HNSW::reset() {
    max_level = -1;
    entry_point = -1;
    offsets.clear();
    offsets.push_back(0);
    levels.clear();
    neighbors.clear();
}

void HNSW::print_neighbor_stats(int level) const {
    FAISS_THROW_IF_NOT(level < cum_nneighbor_per_level.size());
    printf("* stats on level %d, max %d neighbors per vertex:\n",
           level,
           nb_neighbors(level));

    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
#pragma omp parallel for reduction(+: tot_neigh) reduction(+: tot_common) \
  reduction(+: tot_reciprocal) reduction(+: n_node)
    for (int i = 0; i < levels.size(); i++) {
        if (levels[i] > level) {
            n_node++;
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            std::unordered_set<int> neighset;
            for (size_t j = begin; j < end; j++) {
                if (neighbors[j] < 0)
                    break;
                neighset.insert(neighbors[j]);
            }
            int n_neigh = neighset.size();
            int n_common = 0;
            int n_reciprocal = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t i2 = neighbors[j];
                if (i2 < 0)
                    break;
                FAISS_ASSERT(i2 != i);
                size_t begin2, end2;
                neighbor_range(i2, level, &begin2, &end2);
                for (size_t j2 = begin2; j2 < end2; j2++) {
                    storage_idx_t i3 = neighbors[j2];
                    if (i3 < 0)
                        break;
                    if (i3 == i) {
                        n_reciprocal++;
                        continue;
                    }
                    if (neighset.count(i3)) {
                        neighset.erase(i3);
                        n_common++;
                    }
                }
            }
            tot_neigh += n_neigh;
            tot_common += n_common;
            tot_reciprocal += n_reciprocal;
        }
    }
    float normalizer = n_node;
    printf("   1. nb of nodes: %zd\n", n_node);
    printf("   2. neighbors per node: %.2f (%zd)\n",
           tot_neigh / normalizer,
           tot_neigh);
    printf("   3. nb of reciprocal neighbors: %.2f\n",
           tot_reciprocal / normalizer);
    printf("   4. nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer,
           tot_common);
}


void HNSW::hybrid_print_neighbor_stats(int level) const {
    FAISS_THROW_IF_NOT(level < cum_nneighbor_per_level.size());
    printf("* stats on level %d, max %d neighbors per vertex:\n",
           level,
           nb_neighbors(level));

    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
#pragma omp parallel for reduction(+: tot_neigh) reduction(+: tot_common) \
  reduction(+: tot_reciprocal) reduction(+: n_node)
    for (int i = 0; i < levels.size(); i++) {
        if (levels[i] > level) {
            n_node++;
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            std::unordered_set<int> neighset;
            for (size_t j = begin; j < end; j++) {
                if (hybrid_neighbors[j].first < 0) // mod
                    break;
                neighset.insert(hybrid_neighbors[j].first); // mod
            }
            int n_neigh = neighset.size();
            int n_common = 0;
            int n_reciprocal = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t i2 = hybrid_neighbors[j].first; // mod
                if (i2 < 0)
                    break;
                FAISS_ASSERT(i2 != i);
                size_t begin2, end2;
                neighbor_range(i2, level, &begin2, &end2);
                for (size_t j2 = begin2; j2 < end2; j2++) {
                    storage_idx_t i3 = hybrid_neighbors[j2].first;  //mod
                    if (i3 < 0)
                        break;
                    if (i3 == i) {
                        n_reciprocal++;
                        continue;
                    }
                    if (neighset.count(i3)) {
                        neighset.erase(i3);
                        n_common++;
                    }
                }
            }
            tot_neigh += n_neigh;
            tot_common += n_common;
            tot_reciprocal += n_reciprocal;
        }
    }
    float normalizer = n_node;
    printf("   1. nb of nodes: %zd\n", n_node);
    printf("   2. neighbors per node: %.2f (%zd)\n",
           tot_neigh / normalizer,
           tot_neigh);
    printf("   3. nb of reciprocal neighbors: %.2f\n",
           tot_reciprocal / normalizer);
    printf("   4. nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer,
           tot_common);
}

// same as print_neighbor_stats with additional edge lists printed, no parallelism
void HNSW::print_edges(int level) const {
    FAISS_THROW_IF_NOT(level < cum_nneighbor_per_level.size());
    printf("* stats on level %d, max %d neighbors per vertex:\n",
           level,
           nb_neighbors(level));

    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
    printf("\t edges lists:\n");
//#pragma omp parallel for reduction(+: tot_neigh) reduction(+: tot_common) \
  reduction(+: tot_reciprocal) reduction(+: n_node)
    for (int i = 0; i < levels.size(); i++) {
        if (levels[i] > level) {
            n_node++;
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            std::unordered_set<int> neighset;
            printf("\t\t %d: [", i);
            for (size_t j = begin; j < end; j++) {
                if (neighbors[j] < 0)
                    break;
                printf("%d, ", neighbors[j]);
                neighset.insert(neighbors[j]);
            }
            printf("]\n");
            int n_neigh = neighset.size();
            int n_common = 0;
            int n_reciprocal = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t i2 = neighbors[j];
                if (i2 < 0)
                    break;
                FAISS_ASSERT(i2 != i);
                size_t begin2, end2;
                neighbor_range(i2, level, &begin2, &end2);
                for (size_t j2 = begin2; j2 < end2; j2++) {
                    storage_idx_t i3 = neighbors[j2];
                    if (i3 < 0)
                        break;
                    if (i3 == i) {
                        n_reciprocal++;
                        continue;
                    }
                    if (neighset.count(i3)) {
                        neighset.erase(i3);
                        n_common++;
                    }
                }
            }
            tot_neigh += n_neigh;
            tot_common += n_common;
            tot_reciprocal += n_reciprocal;
        }
    }
    float normalizer = n_node;

    printf("   0. level: %d\n", level);
    printf("   1. max neighbors per node: %d\n", nb_neighbors(level));
    printf("   2. nb of nodes: %zd\n", n_node);
    printf("   3. neighbors per node: %.2f (%zd)\n",
           tot_neigh / normalizer,
           tot_neigh);
    printf("   4. nb of reciprocal neighbors: %.2f\n",
           tot_reciprocal / normalizer);
    printf("   5. nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer,
           tot_common);
}


// same as print_neighbor_stats with additional edge lists printed, no parallelism
void HNSW::hybrid_print_edges(int level) const {
    FAISS_THROW_IF_NOT(level < cum_nneighbor_per_level.size());
    printf("* stats on level %d, max %d neighbors per vertex:\n",
           level,
           nb_neighbors(level));

    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
    printf("\t edges lists:\n");
//#pragma omp parallel for reduction(+: tot_neigh) reduction(+: tot_common) \
  reduction(+: tot_reciprocal) reduction(+: n_node)
    for (int i = 0; i < levels.size(); i++) {
        if (levels[i] > level) {
            n_node++;
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            std::unordered_set<int> neighset;
            printf("\t\t %d: [", i);
            for (size_t j = begin; j < end; j++) {
                if (hybrid_neighbors[j].first < 0) // mod
                    break;
                printf("%d, (%d)", hybrid_neighbors[j].first, hybrid_neighbors[j].second); // mod
                neighset.insert(hybrid_neighbors[j].first); // mod
            }
            printf("]\n");
            int n_neigh = neighset.size();
            int n_common = 0;
            int n_reciprocal = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t i2 = hybrid_neighbors[j].first; //mod
                if (i2 < 0)
                    break;
                FAISS_ASSERT(i2 != i);
                size_t begin2, end2;
                neighbor_range(i2, level, &begin2, &end2);
                for (size_t j2 = begin2; j2 < end2; j2++) {
                    storage_idx_t i3 = hybrid_neighbors[j2].first; // mod
                    if (i3 < 0)
                        break;
                    if (i3 == i) {
                        n_reciprocal++;
                        continue;
                    }
                    if (neighset.count(i3)) {
                        neighset.erase(i3);
                        n_common++;
                    }
                }
            }
            tot_neigh += n_neigh;
            tot_common += n_common;
            tot_reciprocal += n_reciprocal;
        }
    }
    float normalizer = n_node;

    printf("   0. level: %d\n", level);
    printf("   1. max neighbors per node: %d\n", nb_neighbors(level));
    printf("   2. nb of nodes: %zd\n", n_node);
    printf("   3. neighbors per node: %.2f (%zd)\n",
           tot_neigh / normalizer,
           tot_neigh);
    printf("   4. nb of reciprocal neighbors: %.2f\n",
           tot_reciprocal / normalizer);
    printf("   5. nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer,
           tot_common);
}




// stats for all levels 
void HNSW::print_neighbor_stats(bool edge_list, bool is_hybrid) const {
    printf("========= METADATA =======\n");

    printf("\t* cumulative max num neighbors per level\n");
    for (int i = 0; i < cum_nneighbor_per_level.size() - 1; i++) {
        printf("\t\tindx %d: %d\n", i, cum_nneighbor_per_level[i]);
    }
    printf("\t* level probabilities\n");
    for (int level = 0; level < assign_probas.size(); level++) {
        printf("\t\tlevel %d: %f\n", level, assign_probas[level]);
    }
    printf("\t* efConstruction: %d\n", efConstruction);
    printf("\t* efSearch: %d\n", efSearch);
    printf("\t* max_level: %d\n", max_level);
    printf("\t* entry_point: %d\n", entry_point);
    printf("\t* gamma: %d\n", gamma);


    // per level stats
    if (edge_list){
        for (int level = 0; level <= max_level; level++) {
            printf("========= LEVEL %d =======\n", level);
            if (is_hybrid) {
                hybrid_print_edges(level);
            } else {
                print_edges(level); // this is more detailed version with edge lists
            }
            
        } 
    } else {
        for (int level = 0; level <= max_level; level++) {
            printf("========= LEVEL %d =======\n", level);
            if (is_hybrid) {
                hybrid_print_neighbor_stats(level);
            } else {
                print_neighbor_stats(level);
            }
            
        }
    }
}



void HNSW::fill_with_random_links(size_t n) {
    int max_level = prepare_level_tab(n);
    RandomGenerator rng2(456);

    for (int level = max_level - 1; level >= 0; --level) {
        std::vector<int> elts;
        for (int i = 0; i < n; i++) {
            if (levels[i] > level) {
                elts.push_back(i);
            }
        }
        printf("linking %zd elements in level %d\n", elts.size(), level);

        if (elts.size() == 1)
            continue;

        for (int ii = 0; ii < elts.size(); ii++) {
            int i = elts[ii];
            size_t begin, end;
            neighbor_range(i, 0, &begin, &end);
            for (size_t j = begin; j < end; j++) {
                int other = 0;
                do {
                    other = elts[rng2.rand_int(elts.size())];
                } while (other == i);

                neighbors[j] = other;
            }
        }
    }
}

// n is the number of vectors that will be added (this gets called in IndexHSNW.hnsw_add_vertices)
int HNSW::prepare_level_tab(size_t n, bool preset_levels) {
    size_t n0 = offsets.size() - 1;

    if (preset_levels) {
        FAISS_ASSERT(n0 + n == levels.size());
    } else {
        FAISS_ASSERT(n0 == levels.size());
        for (int i = 0; i < n; i++) {
            int pt_level = random_level();
            levels.push_back(pt_level + 1);
        }
    }

    int max_level = 0;
    for (int i = 0; i < n; i++) {
        int pt_level = levels[i + n0] - 1;
        if (pt_level > max_level)
            max_level = pt_level;
        offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
        neighbors.resize(offsets.back(), -1);
    }

    return max_level;
}

/** Enumerate vertices from farthest to nearest from query, keep a
 * neighbor only if there is no previous neighbor that is closer to
 * that vertex than the query.
 */
void HNSW::shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistFarther>& input,
        std::vector<NodeDistFarther>& output,
        int max_size, int gamma) {
    while (input.size() > 0) {
        NodeDistFarther v1 = input.top();
        input.pop();
        float dist_v1_q = v1.d;

        bool good = true;
        for (NodeDistFarther v2 : output) {
            float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);

            // TODO check
            if (dist_v1_v2 < dist_v1_q /*&& gamma == 1*/) {
                good = false;
                break;
            }
        }

        if (good) {
            output.push_back(v1);
            if (output.size() >= max_size) {
                return;
            }
        }
    }
}

namespace {

using storage_idx_t = HNSW::storage_idx_t;
using NodeDistCloser = HNSW::NodeDistCloser;
using NodeDistFarther = HNSW::NodeDistFarther;

/**************************************************************
 * Addition subroutines
 **************************************************************/

/// remove neighbors from the list to make it smaller than max_size
void shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& resultSet1,
        int max_size, int gamma) {
    // debug("shrink_neighbor_list from size %ld, to max size %d\n", resultSet1.size(), max_size);
    if (resultSet1.size() < max_size) {
        return;
    }
    std::priority_queue<NodeDistFarther> resultSet;
    std::vector<NodeDistFarther> returnlist;

    while (resultSet1.size() > 0) {
        resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
        resultSet1.pop();
    }

    HNSW::shrink_neighbor_list(qdis, resultSet, returnlist, max_size, gamma);

    for (NodeDistFarther curen2 : returnlist) {
        resultSet1.emplace(curen2.d, curen2.id);
    }
}

/// add a link between two elements, possibly shrinking the list
/// of links to make room for it.
void add_link(
        HNSW& hnsw,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest,
        int level) {
    size_t begin, end;
    hnsw.neighbor_range(src, level, &begin, &end);
    if (hnsw.neighbors[end - 1] == -1) {
        // there is enough room, find a slot to add it
        size_t i = end;
        while (i > begin) {
            if (hnsw.neighbors[i - 1] != -1)
                break;
            i--;
        }
        hnsw.neighbors[i] = dest;
        return;
    }

    // otherwise we let them fight out which to keep

    // copy to resultSet...
    std::priority_queue<NodeDistCloser> resultSet;
    resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
    for (size_t i = begin; i < end; i++) { // HERE WAS THE BUG
        storage_idx_t neigh = hnsw.neighbors[i];
        resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
    }

    // TODO revisit in a better way
    shrink_neighbor_list(qdis, resultSet, end - begin, hnsw.gamma);
    
    

    // ...and back
    size_t i = begin;
    while (resultSet.size()) {
        hnsw.neighbors[i++] = resultSet.top().id;
        resultSet.pop();
    }
    // they may have shrunk more than just by 1 element
    while (i < end) {
        hnsw.neighbors[i++] = -1;
    }
}

// HYBRID version
/// add a link between two elements, possibly shrinking the list
/// of links to make room for it.
void hybrid_add_link(
        HNSW& hnsw,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest,
        int level) {
    size_t begin, end;
    hnsw.neighbor_range(src, level, &begin, &end);
    if (hnsw.hybrid_neighbors[end - 1].first == -1) { // access to std::pair
        // there is enough room, find a slot to add it
        size_t i = end;
        while (i > begin) {
            if (hnsw.hybrid_neighbors[i - 1].first != -1)
                break;
            i--;
        }
        hnsw.hybrid_neighbors[i] = std::make_pair(dest, hnsw.metadata[dest]); // mod
        return;
    }

    // otherwise we let them fight out which to keep

    // copy to resultSet...
    std::priority_queue<NodeDistCloser> resultSet;
    resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
    for (size_t i = begin; i < end; i++) { // HERE WAS THE BUG
        auto [neigh, metadata] = hnsw.hybrid_neighbors[i]; // mod
        resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
    }

    // TODO revisit in a better way
    shrink_neighbor_list(qdis, resultSet, end - begin, hnsw.gamma);
    
    

    // ...and back
    size_t i = begin;
    while (resultSet.size()) {
        hnsw.hybrid_neighbors[i++] = std::make_pair(resultSet.top().id, hnsw.metadata[resultSet.top().id]); // mod
        resultSet.pop();
    }
    // they may have shrunk more than just by 1 element
    while (i < end) {
        hnsw.hybrid_neighbors[i++].first = -1; // TODO fix
    }
}



/// search neighbors on a single level, starting from an entry point
// this only gets called in construction
void search_neighbors_to_add(
        HNSW& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& results,
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt,
        std::vector<storage_idx_t> ep_per_level = {}) {
    // debug("search_neighbors to add, entrypoint: %d\n", entry_point);
    // top is nearest candidate
    std::priority_queue<NodeDistFarther> candidates;

    NodeDistFarther ev(d_entry_point, entry_point);
    candidates.push(ev);
    results.emplace(d_entry_point, entry_point);
    vt.set(entry_point);

    // number of neighbors we want for hybrid construction
    int M = hnsw.nb_neighbors(level);
    // debug("desired resuts size: %d, at level: %d\n", M, level);
    

    // for backtracking in hybrid version
    int backtrack_level = level;

    while (!candidates.empty()) {
        // get nearest
        const NodeDistFarther& currEv = candidates.top();

        // MOD greedy break - added break conditional on size of results
        if ((currEv.d > results.top().d && hnsw.gamma == 1) || results.size() >= M) {
            // debug("greedy stop in construction, results size: %ld, desired size: %d, gamma = %d\n", results.size(), M, hnsw.gamma);
            break;
        }
        int currNode = currEv.id;
        candidates.pop();

        // loop over neighbors
        size_t begin, end;
        hnsw.neighbor_range(currNode, level, &begin, &end);

        // // limit the number of neighbors visited during construction
        // if (end - begin > 64) {
        //     end = begin + 64;
        //     // debug("truncating neighbor range -- M: %d, begin: %ld, new end: %ld", hnsw.M, begin, end);
        // }

        for (size_t i = begin; i < end; i++) {
            storage_idx_t nodeId = hnsw.neighbors[i];
            if (nodeId < 0)
                break;
            if (vt.get(nodeId))
                continue;
            vt.set(nodeId);

            float dis = qdis(nodeId);
            NodeDistFarther evE1(dis, nodeId);

            // debug("while checking neighbors, efc: %d, results size: %ld\n", hnsw.efConstruction, results.size());
            if (results.size() < hnsw.efConstruction || results.top().d > dis) {
                results.emplace(dis, nodeId);
                candidates.emplace(dis, nodeId);
                if (results.size() > hnsw.efConstruction) {
                    results.pop();
                }
            }
        }
        // TODO - if candidates is empty, add the next non-visited neighbor from neighbors of ep or query at level l-1
        // only do this step if we have a hybrid index (gamma > 1)
        // TODO should only backtrack if there are unvisited nodes at this level
        // debug("during BFS, gamma: %d, candidates size: %ld, results size: %ld, vt.num_visited: %d, nb on level: %d, backtrack_level: %d, level: %d\n", hnsw.gamma, candidates.size(), results.size(), vt.num_visited(), hnsw.nb_per_level[level], backtrack_level, level);
        // if (hnsw.gamma > 1 && candidates.empty() && results.size() < M && vt.num_visited() < hnsw.nb_per_level[level] && backtrack_level <= level) {
        // if (hnsw.gamma > 1 && candidates.empty() && results.size() < M && vt.num_visited() < hnsw.nb_per_level[level]) {
        //     debug("connected component exhausted too early, results size: %ld, desired size: %d, gamma = %d\n", results.size(), M, hnsw.gamma);
        //     // TODO should add a while loop here
        //     while(candidates.empty() && backtrack_level < hnsw.max_level) {
        //         backtrack_level = backtrack_level + 1;
        //         debug("backtracking to level: %d\n", backtrack_level);
        //         size_t begin, end;
        //         debug("\tep is: %d\n", ep_per_level[backtrack_level]);
        //         hnsw.neighbor_range(ep_per_level[backtrack_level], backtrack_level, &begin, &end); // TODO can also do this for query instead of ep
        //         debug("\tchecking %ld neighbors of node %d at level %d\n", (end - begin), entry_point, backtrack_level);
        //         for (size_t i = begin; i < end; i++) {
        //             // debug("\t\titer %d\n", i);
        //             storage_idx_t nodeId = hnsw.neighbors[i];
        //             debug("\t\tchecking node %d\n", nodeId);
        //             if (nodeId < 0) {
        //                 debug("\t\t\t%s\n", "-1 reached, backtrack ends");
        //                 break;
        //             }  
        //             if (vt.get(nodeId)) {
        //                 debug("\t\t\t%d already visited\n", nodeId);
        //                 continue;
        //             }   
        //             debug("\t\t\tbacktracking adds node: %d\n", nodeId);
        //             float dis = qdis(nodeId);
        //             candidates.emplace(dis, nodeId);
        //         }
        //     }
            
        // } 
    }
    // debug("search_neighbors to add finds %ld nn's\n", results.size());
    vt.advance();
}

// HYBRID version
/// search neighbors on a single level, starting from an entry point
// this only gets called in construction
void hybrid_search_neighbors_to_add(
        HNSW& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& results,
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt,
        std::vector<storage_idx_t> ep_per_level = {}) {
    // debug("HYBRID search_neighbors to add, entrypoint: %d\n", entry_point);
    // top is nearest candidate
    std::priority_queue<NodeDistFarther> candidates;

    NodeDistFarther ev(d_entry_point, entry_point);
    candidates.push(ev);
    results.emplace(d_entry_point, entry_point);
    vt.set(entry_point);

    // number of neighbors we want for hybrid construction
    int M = hnsw.nb_neighbors(level);
    // debug("desired resuts size: %d, at level: %d\n", M, level);
    

    // for backtracking in hybrid version
    int backtrack_level = level;

    while (!candidates.empty()) {
        // get nearest
        const NodeDistFarther& currEv = candidates.top();

        // MOD greedy break - added break conditional on size of results
        if ((currEv.d > results.top().d && hnsw.gamma == 1) || results.size() >= M) {
            // debug("greedy stop in construction, results size: %ld, desired size: %d, gamma = %d\n", results.size(), M, hnsw.gamma);
            break;
        }
        int currNode = currEv.id;
        candidates.pop();

        // loop over neighbors
        size_t begin, end;
        hnsw.neighbor_range(currNode, level, &begin, &end);
        for (size_t i = begin; i < end; i++) {
            // storage_idx_t nodeId = hnsw.neighbors[i];
            auto [nodeId, metadata] = hnsw.hybrid_neighbors[i]; // storage_idx_t, int
            if (nodeId < 0)
                break;
            if (vt.get(nodeId))
                continue;
            vt.set(nodeId);

            float dis = qdis(nodeId);
            NodeDistFarther evE1(dis, nodeId);

            // debug("while checking neighbors, efc: %d, results size: %ld\n", hnsw.efConstruction, results.size());
            if (results.size() < hnsw.efConstruction || results.top().d > dis) {
                results.emplace(dis, nodeId);
                candidates.emplace(dis, nodeId);
                if (results.size() > hnsw.efConstruction) {
                    results.pop();
                }
            }
        }
        // TODO - if candidates is empty, add the next non-visited neighbor from neighbors of ep or query at level l-1
        // only do this step if we have a hybrid index (gamma > 1)
        // TODO should only backtrack if there are unvisited nodes at this level
        // debug("during BFS, gamma: %d, candidates size: %ld, results size: %ld, vt.num_visited: %d, nb on level: %d, backtrack_level: %d, level: %d\n", hnsw.gamma, candidates.size(), results.size(), vt.num_visited(), hnsw.nb_per_level[level], backtrack_level, level);
    }
    // debug("search_neighbors to add finds %ld nn's\n", results.size());
    vt.advance();
}

/**************************************************************
 * Searching subroutines
 **************************************************************/

/// greedily update a nearest vector at a given level
/// node construction uses this also
int greedy_update_nearest(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {
    // debug("%s\n", "reached");
    int ndis = 0;
    for (;;) {
        storage_idx_t prev_nearest = nearest;

        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);
        //  // limit the number of neighbors visited during construction
        // if (end - begin > 64) {
        //     end = begin + 64;
        //     // debug("truncating neighbor range -- M: %d, begin: %ld, new end: %ld", hnsw.M, begin, end);
        // }
        for (size_t i = begin; i < end; i++) {
            storage_idx_t v = hnsw.neighbors[i];
            if (v < 0)
                break;
            float dis = qdis(v);
            ndis++;
            if (dis < d_nearest) {
                nearest = v;
                d_nearest = dis;
            }
        }
        if (nearest == prev_nearest) {
            return ndis;
        }
    }
    return ndis;

    
}

/// for hybrid constructionh TODO
void hybrid_constr_greedy_update_nearest(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {
    // debug("%s\n", "reached");
    for (;;) {
        // debug("%s", "here 1\n");
        storage_idx_t prev_nearest = nearest;

        // debug("%s", "here 2\n");
        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);
        // debug("%s", "here 3\n");
        for (size_t i = begin; i < end; i++) {
            // storage_idx_t v = hnsw.neighbors[i];
            auto [v, metadata] = hnsw.hybrid_neighbors[i]; // storate_idx_t, int
            if (v < 0)
                break;
            float dis = qdis(v);
            if (dis < d_nearest) {
                nearest = v;
                d_nearest = dis;
            }
        }
        if (nearest == prev_nearest) {
            return;
        }
    }
}

/// for hybrid search TODO
int hybrid_greedy_update_nearest(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int filter,
        Operation op,
        std::string regex,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {
    // debug("%s\n", "reached");
    int ndis = 0;
    for (;;) {
        storage_idx_t prev_nearest = nearest;
        int num_found = 0;

        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);
        for (size_t i = begin; i < end; i++) {
            storage_idx_t v = hnsw.neighbors[i];
            if (v < 0)
                break;
            // TODO check - skips vertex if it has the wrong attribute
            // debug("looking at vertex %d with metadata %d\n", v, hnsw.metadata[v]);
            // if (hnsw.metadata[v] != filter) {
            //     // debug("%s\n", "skipping vertex");
            //     continue;
            // }
            // check filter
            if (((op == EQUAL) && (hnsw.metadata[v] == filter)) || (op == OR && (hnsw.metadata[v] & filter != 0)) || (op == REGEX && CHECK_REGEX(hnsw.metadata_strings[v], regex))) {
                num_found = num_found + 1;
            }

            // check filter
            if (((op == EQUAL) && (hnsw.metadata[v] == filter)) || (op == OR && (hnsw.metadata[v] & filter != 0)) || (op == REGEX && CHECK_REGEX(hnsw.metadata_strings[v], regex))) {
                float dis = qdis(v);
                ndis++;
                if (dis < d_nearest || ((op == EQUAL) && (hnsw.metadata[nearest] != filter)) || (op == OR && (hnsw.metadata[nearest] & filter == 0)) || (op == REGEX && !CHECK_REGEX(hnsw.metadata_strings[nearest], regex))) {
                    nearest = v;
                    d_nearest = dis;
                }
                if (num_found >= hnsw.M) {
                    break;
                }
            }

            // expand list by one hop
            size_t begin2, end2;
            hnsw.neighbor_range(v, level, &begin2, &end2);
            for (size_t j = begin2; j < end2; j++) {
                storage_idx_t v2 = hnsw.neighbors[j];
                if (v2 < 0)
                    break;
             
                // check filter
                if (((op == EQUAL) && (hnsw.metadata[v2] == filter)) || (op == OR && (hnsw.metadata[v2] & filter != 0)) || (op == REGEX && CHECK_REGEX(hnsw.metadata_strings[v2], regex))) {
                    num_found = num_found + 1;
                    float dis2 = qdis(v2);
                    ndis += 1;

                    if (dis2 < d_nearest || ((op == EQUAL) && (hnsw.metadata[nearest] != filter)) || (op == OR && (hnsw.metadata[nearest] & filter == 0)) || (op == REGEX && !CHECK_REGEX(hnsw.metadata_strings[nearest], regex))) {
                        nearest = v2;
                        d_nearest = dis2;
                    }
                    if (num_found >= hnsw.M) {
                        // keep_expanding = false;
                        break;
                    }
                }
            }

            
        }
        if (nearest == prev_nearest) {
            return ndis;
        }
    }
    return ndis;
}

} // namespace

/// Finds neighbors and builds links with them, starting from an entry
/// point. The own neighbor list is assumed to be locked.
void HNSW::add_links_starting_from(
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        int level,
        omp_lock_t* locks,
        VisitedTable& vt,
        std::vector<storage_idx_t> ep_per_level) {
    // debug("add_links_starting_from at level: %d, nearest: %d\n", level, nearest);
    std::priority_queue<NodeDistCloser> link_targets;

    search_neighbors_to_add(
            *this, ptdis, link_targets, nearest, d_nearest, level, vt, ep_per_level);

    // but we can afford only this many neighbors
    int M = nb_neighbors(level);

    // debug("add_links_starting_from will shrink results list to size: %d\n", M);
    
    // TODO check this just ignore this step if we have hybrid search, becuase we need big neighbor lists
    ::faiss::shrink_neighbor_list(ptdis, link_targets, M, gamma);
    
    
    // debug("add_links_starting_from gets edge link size: %ld\n", link_targets.size());

    std::vector<storage_idx_t> neighbors;
    neighbors.reserve(link_targets.size());
    while (!link_targets.empty()) {
        storage_idx_t other_id = link_targets.top().id;
        add_link(*this, ptdis, pt_id, other_id, level); // for hybrid search we also will store the metadata in the neighbor list
        neighbors.push_back(other_id);
        link_targets.pop();
    }

    omp_unset_lock(&locks[pt_id]);
    for (storage_idx_t other_id : neighbors) {
        omp_set_lock(&locks[other_id]);
        add_link(*this, ptdis, other_id, pt_id, level);
        omp_unset_lock(&locks[other_id]);
    }
    omp_set_lock(&locks[pt_id]);
}

// HYBRID version, calls hybrid_ subroutiens
/// Finds neighbors and builds links with them, starting from an entry
/// point. The own neighbor list is assumed to be locked.
void HNSW::hybrid_add_links_starting_from(
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        int level,
        omp_lock_t* locks,
        VisitedTable& vt,
        std::vector<storage_idx_t> ep_per_level) {
    // debug("add_links_starting_from at level: %d, nearest: %d\n", level, nearest);
    std::priority_queue<NodeDistCloser> link_targets;

    hybrid_search_neighbors_to_add(
            *this, ptdis, link_targets, nearest, d_nearest, level, vt, ep_per_level);

    // but we can afford only this many neighbors
    int M = nb_neighbors(level);

    // debug("add_links_starting_from will shrink results list to size: %d\n", M);
    
    // TODO check this just ignore this step if we have hybrid search, becuase we need big neighbor lists
    ::faiss::shrink_neighbor_list(ptdis, link_targets, M, gamma);
    
    
    // debug("add_links_starting_from gets edge link size: %ld\n", link_targets.size());

    std::vector<storage_idx_t> neighbors;
    neighbors.reserve(link_targets.size());
    while (!link_targets.empty()) {
        storage_idx_t other_id = link_targets.top().id;
        hybrid_add_link(*this, ptdis, pt_id, other_id, level); // for hybrid search we also will store the metadata in the neighbor list
        neighbors.push_back(other_id);
        link_targets.pop();
    }

    omp_unset_lock(&locks[pt_id]);
    for (storage_idx_t other_id : neighbors) {
        omp_set_lock(&locks[other_id]);
        hybrid_add_link(*this, ptdis, other_id, pt_id, level);
        omp_unset_lock(&locks[other_id]);
    }
    omp_set_lock(&locks[pt_id]);
}

/**************************************************************
 * Building, parallel
 **************************************************************/

void HNSW::add_with_locks(
        DistanceComputer& ptdis,
        int pt_level,
        int pt_id,
        std::vector<omp_lock_t>& locks,
        VisitedTable& vt) {
    //  greedy search on upper levels
    // debug("add_with_locks called for node %d, level %d\n", pt_id, pt_level);
    // printf("add_with_locks called for node %d, level %d\n", pt_id, pt_level);
    // std::cout << "-Starting!" << std::endl;
    storage_idx_t nearest;
#pragma omp critical
    {
        nearest = entry_point;
        // std::cout << "--add_with_locks here0.1" << std::endl;

        if (nearest == -1) {
            // std::cout << "--add_with_locks here0.2" << std::endl;
            max_level = pt_level;
            entry_point = pt_id;
            for (int i=0; i <= max_level; i++){
                nb_per_level[i] = nb_per_level[i] + 1;
            }
            
        }
        // std::cout << "--add_with_locks here0.3" << std::endl;
    }

    if (nearest < 0) {
        // printf("finished add_with_locks called for node %d, level %d\n", pt_id, pt_level);
        // std::cout << "FINISHED add_with_locks finished" << std::endl;
        return;
    }
    // std::cout << "--add_with_locks here1" << std::endl;

    omp_set_lock(&locks[pt_id]);

    int level = max_level; // level at which we start adding neighbors
    float d_nearest = ptdis(nearest);

    // needed for backtracking in hybrid search
    std::vector<storage_idx_t> ep_per_level(max_level); // idx of nearest node per level
    ep_per_level[level] = nearest;

    // std::cout << "--add_with_locks here2" << std::endl;


    for (; level > pt_level; level--) {
        // debug("--greedy update nearest at level: %d, ep: %d\n", level, nearest);
        greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
        ep_per_level[level] = nearest;
        // std::cout << "--add_with_locks here3" << std::endl;

    }

    for (int i = 0; i <= max_level; i++) {
        // debug("--at level: %d, ep: %d\n", i, ep_per_level[i]);
    }

    // std::cout << "--add_with_locks here4" << std::endl;


    for (; level >= 0; level--) {
        // debug("--add_links_starting_from at level: %d\n", level);
        add_links_starting_from(
                ptdis, pt_id, nearest, d_nearest, level, locks.data(), vt, ep_per_level);
        nb_per_level[level] = nb_per_level[level] + 1;
    }
    // std::cout << "--add_with_locks here5" << std::endl;


    omp_unset_lock(&locks[pt_id]);

    if (pt_level > max_level) {
        max_level = pt_level;
        entry_point = pt_id;
    }
    // printf("finished add_with_locks called for node %d, level %d\n", pt_id, pt_level);
    // std::cout << "FINISHED add_with_locks finished" << std::endl;
}


// hybrid version which calls hybrid_* subroutines
void HNSW::hybrid_add_with_locks(
        DistanceComputer& ptdis,
        int pt_level,
        int pt_id,
        std::vector<omp_lock_t>& locks,
        VisitedTable& vt) {
    //  greedy search on upper levels
    printf("HYBRID add_with_locks called for node %d, level %d\n", pt_id, pt_level);

    storage_idx_t nearest;
#pragma omp critical
    {
        nearest = entry_point;

        if (nearest == -1) {
            max_level = pt_level;
            entry_point = pt_id;
            for (int i=0; i <= max_level; i++){
                nb_per_level[i] = nb_per_level[i] + 1;
            }
            
        }
    }

    if (nearest < 0) {
        return;
    }

    omp_set_lock(&locks[pt_id]);

    int level = max_level; // level at which we start adding neighbors
    float d_nearest = ptdis(nearest);

    // needed for backtracking in hybrid search
    std::vector<storage_idx_t> ep_per_level(max_level); // idx of nearest node per level
    ep_per_level[level] = nearest;

    for (; level > pt_level; level--) {
        // debug("--greedy update nearest at level: %d, ep: %d\n", level, nearest);
        hybrid_constr_greedy_update_nearest(*this, ptdis, level, nearest, d_nearest); // hybrid method
        ep_per_level[level] = nearest;
    }

    for (int i = 0; i <= max_level; i++) {
        // debug("--at level: %d, ep: %d\n", i, ep_per_level[i]);
    }

    for (; level >= 0; level--) {
        // debug("--add_links_starting_from at level: %d\n", level);
        add_links_starting_from(
                ptdis, pt_id, nearest, d_nearest, level, locks.data(), vt, ep_per_level);
        nb_per_level[level] = nb_per_level[level] + 1;
    }

    omp_unset_lock(&locks[pt_id]);

    if (pt_level > max_level) {
        max_level = pt_level;
        entry_point = pt_id;
    }
}

/**************************************************************
 * Searching
 **************************************************************/

namespace {

using MinimaxHeap = HNSW::MinimaxHeap;
using Node = HNSW::Node;
/** Do a BFS on the candidates list */
// this is called in search and search_from_level_0
int search_from_candidates(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in = 0,
        const SearchParametersHNSW* params = nullptr) {
    // debug("%s\n", "reached");
    int nres = nres_in;
    int ndis = 0;

    // can be overridden by search params
    bool do_dis_check = params ? params->check_relative_distance
                               : hnsw.check_relative_distance;
    int efSearch = params ? params->efSearch : hnsw.efSearch;
    const IDSelector* sel = params ? params->sel : nullptr;

    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (nres < k) {
                faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
                faiss::maxheap_replace_top(nres, D, I, d, v1);
            }
        }
        vt.set(v1);
    }

    int nstep = 0;

    while (candidates.size() > 0) { // candidates is heap of size max(efs, k)
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0

            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                break;
            }
        }

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0)
                break;
            if (vt.get(v1)) {
                continue;
            }
            vt.set(v1);
            ndis++;
            float d = qdis(v1);
            if (!sel || sel->is_member(v1)) {
                if (nres < k) {
                    faiss::maxheap_push(++nres, D, I, d, v1);
                } else if (d < D[0]) {
                    faiss::maxheap_replace_top(nres, D, I, d, v1);
                }
            }
            candidates.push(v1, d);
        }

        nstep++;
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.n3 += ndis;
    }

    return nres;
}

// for hybrid search TODO
// modified to impl alt postfiltering with neighbor expansion
int hybrid_search_from_candidates(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int filter,
        Operation op,
        std::string regex,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in = 0,
        const SearchParametersHNSW* params = nullptr) {
    // debug("%s\n", "reached");
    int nres = nres_in;
    int ndis = 0;

    // can be overridden by search params
    bool do_dis_check = params ? params->check_relative_distance
                               : hnsw.check_relative_distance;
    int efSearch = params ? params->efSearch : hnsw.efSearch;
    const IDSelector* sel = params ? params->sel : nullptr;

    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (nres < k) {
                faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
                faiss::maxheap_replace_top(nres, D, I, d, v1);
            }
        }
        vt.set(v1);
    }

    int nstep = 0;

    while (candidates.size() > 0) { // candidates is heap of size max(efs, k)
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        // TODO - this may cause stopping before all neighbors are found
        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0
            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                // debug("%s\n", "n_dis_below >= efSearch BREAK cond reached");
                break;
            }
        }

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);
        int num_found = 0;
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0)
                break;
            // check filter
            if (((op == EQUAL) && (hnsw.metadata[v1] == filter)) || (op == OR && ((hnsw.metadata[v1] & filter) != 0)) || (op == REGEX && CHECK_REGEX(hnsw.metadata_strings[v1], regex))) {
                num_found = num_found + 1;
            }
            if (vt.get(v1)) {
                continue;
            }
            // TODO - add check for if attr of v1 passes filter
            // possible problem if none of them pass, you would then want
            // the closest of all the ones that did pass, for now just return err
            // skips vertex if it has the wrong attribute
            // debug("looking at vertex %d with metadata %d\n", v1, hnsw.metadata[v1]);
            // if (hnsw.metadata[v1] != filter) {
            //     // debug("%s\n", "skipping vertex");
            //     continue;
            // }
            // if (hnsw.metadata[v1] == filter) {
            if (((op == EQUAL) && (hnsw.metadata[v1] == filter)) || (op == OR && ((hnsw.metadata[v1] & filter) != 0)) || (op == REGEX && CHECK_REGEX(hnsw.metadata_strings[v1], regex))) {
                // printf("found vertex with metadata %d\n", hnsw.metadata[v1]);
                vt.set(v1);
                ndis++;
                float d = qdis(v1);
                if (!sel || sel->is_member(v1)) {
                    if (nres < k) {
                        faiss::maxheap_push(++nres, D, I, d, v1);
                    } else if (d < D[0]) {
                        faiss::maxheap_replace_top(nres, D, I, d, v1);
                    }
                }
                candidates.push(v1, d);
                if (num_found >= 2* hnsw.M) {
                    // debug("%s\n", "num_found >= hnsw.M BREAK cond reached");
                    break;
                }
            }

            // expand neighbor list
            size_t begin2, end2;
            hnsw.neighbor_range(v1, level, &begin2, &end2);
            for (size_t j2 = begin2; j2 < end2; j2+=1) {
                int v2 = hnsw.neighbors[j2];
                // printf("expanding to neighbor %d with metadata %d\n", v2, hnsw.metadata[v2]);
                // std::cout << "expanding to neighbor " << v2 << " with metadata " << hnsw.metadata[v2] << "\n" << std::endl;
                if (v2 < 0)
                    break;
                // if (hnsw.metadata[v2] != filter) {
                if (((op == EQUAL) && (hnsw.metadata[v2] == filter)) || (op == OR && ((hnsw.metadata[v2] & filter) != 0)) || (op == REGEX && CHECK_REGEX(hnsw.metadata_strings[v2], regex))) {
                    num_found = num_found + 1;
                } else {
                    continue;
                }

                if (vt.get(v2)) {
                    continue;
                }
                vt.set(v2);
                ndis++;
                float d2 = qdis(v2);
                if (!sel || sel->is_member(v2)) {
                    if (nres < k) {
                        faiss::maxheap_push(++nres, D, I, d2, v2);
                    } else if (d2 < D[0]) {
                        faiss::maxheap_replace_top(nres, D, I, d2, v2);
                    }
                }
                candidates.push(v2, d2);
                if (num_found >= 2* hnsw.M) {
                    // debug("%s\n", "num_found >= hnsw.M BREAK cond reached");
                    break;
                }
            }
            
        }
        
        nstep++; // TODO - might want to only increment this if we find a neighbor that passes attr
        // debug("nstep incr'd to %d\n", nstep);
        if (!do_dis_check && nstep > efSearch) {
            // debug("BREAK cond reached - nstep=%d > efsSearch\n", nstep);
            break;
        }
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.n3 += ndis;
    }

    return nres;
}

std::priority_queue<HNSW::Node> search_from_candidate_unbounded(
        const HNSW& hnsw,
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        HNSWStats& stats) {
    //  debug("%s\n", "reached");
    int ndis = 0;
    std::priority_queue<Node> top_candidates;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;

    top_candidates.push(node);
    candidates.push(node);

    vt->set(node.second);

    while (!candidates.empty()) {
        float d0;
        storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (d0 > top_candidates.top().first) {
            break;
        }

        candidates.pop();

        size_t begin, end;
        hnsw.neighbor_range(v0, 0, &begin, &end);

        for (size_t j = begin; j < end; ++j) {
            int v1 = hnsw.neighbors[j];

            if (v1 < 0) {
                break;
            }
            if (vt->get(v1)) {
                continue;
            }

            vt->set(v1);

            float d1 = qdis(v1);
            ++ndis;

            if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
                candidates.emplace(d1, v1);
                top_candidates.emplace(d1, v1);

                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
            }
        }
    }

    ++stats.n1;
    if (candidates.size() == 0) {
        ++stats.n2;
    }
    stats.n3 += ndis;

    return top_candidates;
}

} // anonymous namespace

HNSWStats HNSW::search(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        VisitedTable& vt,
        const SearchParametersHNSW* params) const {
    // debug("%s\n", "reached");
    HNSWStats stats;
    if (entry_point == -1) {
        return stats;
    }
    if (upper_beam == 1) { // common branch
        // debug("%s\n", "reached upper beam == 1");

        //  greedy search on upper levels
        storage_idx_t nearest = entry_point;
        float d_nearest = qdis(nearest);

        int ndis_upper = 0;
        for (int level = max_level; level >= 1; level--) {
            ndis_upper += greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
            
        }
        stats.n3 += ndis_upper;
        // stats.n_upper += ndis_upper;

        int ef = std::max(efSearch, k);
        if (search_bounded_queue) { // this is the most common branch
            // debug("%s\n", "reached search bounded queue");

            MinimaxHeap candidates(ef);

            candidates.push(nearest, d_nearest);

            search_from_candidates(
                    *this, qdis, k, I, D, candidates, vt, stats, 0, 0, params);
        } else {
            // debug("%s\n", "reached search_bounded_queue == False");

            std::priority_queue<Node> top_candidates =
                    search_from_candidate_unbounded(
                            *this,
                            Node(d_nearest, nearest),
                            qdis,
                            ef,
                            &vt,
                            stats);

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }

            int nres = 0;
            while (!top_candidates.empty()) {
                float d;
                storage_idx_t label;
                std::tie(d, label) = top_candidates.top();
                faiss::maxheap_push(++nres, D, I, d, label);
                top_candidates.pop();
            }
        }

        vt.advance();

    } else {
        // debug("%s\n", "reached upper beam != 1");

        int candidates_size = upper_beam;
        MinimaxHeap candidates(candidates_size);

        std::vector<idx_t> I_to_next(candidates_size);
        std::vector<float> D_to_next(candidates_size);

        int nres = 1;
        I_to_next[0] = entry_point;
        D_to_next[0] = qdis(entry_point);

        for (int level = max_level; level >= 0; level--) {
            // copy I, D -> candidates

            candidates.clear();

            for (int i = 0; i < nres; i++) {
                candidates.push(I_to_next[i], D_to_next[i]);
            }

            if (level == 0) {
                nres = search_from_candidates(
                        *this, qdis, k, I, D, candidates, vt, stats, 0);
            } else {
                nres = search_from_candidates(
                        *this,
                        qdis,
                        candidates_size,
                        I_to_next.data(),
                        D_to_next.data(),
                        candidates,
                        vt,
                        stats,
                        level);
            }
            vt.advance();
        }
    }

    return stats;
}

// hybrid search TODO
HNSWStats HNSW::hybrid_search(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        VisitedTable& vt,
        int filter,
        Operation op,
        std::string regex,
        const SearchParametersHNSW* params) const {
    // printf("reach hybrid_search\n");
    // printf("hybrid search upper_beam: %d\n", upper_beam);
    // debug("%s\n", "reached");
    HNSWStats stats;
    if (entry_point == -1) {
        // printf("entry point == -1\n");
        return stats;
    }
    if (upper_beam == 1) { // common branch
        // debug("%s\n", "reached upper beam == 1");
        // printf("reach upper beam == 1\n");

        //  greedy search on upper levels
        storage_idx_t nearest = entry_point;
        float d_nearest = qdis(nearest);

        int ndis_upper = 0;
        for (int level = max_level; level >= 1; level--) {
            ndis_upper += hybrid_greedy_update_nearest(*this, qdis, filter, op, regex, level, nearest, d_nearest);
        }
        stats.n3 += ndis_upper;

        int ef = std::max(efSearch, k);
        if (search_bounded_queue) { // this is the most common branch
            // debug("%s\n", "reached search bounded queue");

            MinimaxHeap candidates(ef);

            candidates.push(nearest, d_nearest);
            // printf("calling hybrid_search_from_candidates\n");
            // std::cout << "calling hybrid_search_form_Candidiat" << std::endl;
            hybrid_search_from_candidates(
                    *this, qdis, filter, op, regex, k, I, D, candidates, vt, stats, 0, 0, params);
        } else {
            // TODO
            printf("UNIMPLEMENTED BRANCH for hybid search\n");
            // debug("%s\n", "reached search_bounded_queue == False");

            std::priority_queue<Node> top_candidates =
                    search_from_candidate_unbounded(
                            *this,
                            Node(d_nearest, nearest),
                            qdis,
                            ef,
                            &vt,
                            stats);

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }

            int nres = 0;
            while (!top_candidates.empty()) {
                float d;
                storage_idx_t label;
                std::tie(d, label) = top_candidates.top();
                faiss::maxheap_push(++nres, D, I, d, label);
                top_candidates.pop();
            }
        }

        vt.advance();

    } else {
        // debug("%s\n", "reached upper beam != 1");

        int candidates_size = upper_beam;
        MinimaxHeap candidates(candidates_size);

        std::vector<idx_t> I_to_next(candidates_size);
        std::vector<float> D_to_next(candidates_size);

        int nres = 1;
        I_to_next[0] = entry_point;
        D_to_next[0] = qdis(entry_point);

        for (int level = max_level; level >= 0; level--) {
            // copy I, D -> candidates

            candidates.clear();

            for (int i = 0; i < nres; i++) {
                candidates.push(I_to_next[i], D_to_next[i]);
            }

            if (level == 0) {
                nres = hybrid_search_from_candidates(
                        *this, qdis, filter, op, regex, k, I, D, candidates, vt, stats, 0);
            } else {
                nres = hybrid_search_from_candidates(
                        *this,
                        qdis,
                        filter,
                        op,
                        regex,
                        candidates_size,
                        I_to_next.data(),
                        D_to_next.data(),
                        candidates,
                        vt,
                        stats,
                        level);
            }
            vt.advance();
        }
    }

    return stats;
}

void HNSW::search_level_0(
        DistanceComputer& qdis,
        int k,
        idx_t* idxi,
        float* simi,
        idx_t nprobe,
        const storage_idx_t* nearest_i,
        const float* nearest_d,
        int search_type,
        HNSWStats& search_stats,
        VisitedTable& vt) const {
    // debug("%s\n", "reached");

    const HNSW& hnsw = *this;

    if (search_type == 1) {
        int nres = 0;

        for (int j = 0; j < nprobe; j++) {
            storage_idx_t cj = nearest_i[j];

            if (cj < 0)
                break;

            if (vt.get(cj))
                continue;

            int candidates_size = std::max(hnsw.efSearch, int(k));
            MinimaxHeap candidates(candidates_size);

            candidates.push(cj, nearest_d[j]);

            nres = search_from_candidates(
                    hnsw,
                    qdis,
                    k,
                    idxi,
                    simi,
                    candidates,
                    vt,
                    search_stats,
                    0,
                    nres);
        }
    } else if (search_type == 2) {
        int candidates_size = std::max(hnsw.efSearch, int(k));
        candidates_size = std::max(candidates_size, int(nprobe));

        MinimaxHeap candidates(candidates_size);
        for (int j = 0; j < nprobe; j++) {
            storage_idx_t cj = nearest_i[j];

            if (cj < 0)
                break;
            candidates.push(cj, nearest_d[j]);
        }

        search_from_candidates(
                hnsw, qdis, k, idxi, simi, candidates, vt, search_stats, 0);
    }
}

/**************************************************************
 * MinimaxHeap
 **************************************************************/

void HNSW::MinimaxHeap::push(storage_idx_t i, float v) {
    if (k == n) {
        if (v >= dis[0])
            return;
        faiss::heap_pop<HC>(k--, dis.data(), ids.data());
        --nvalid;
    }
    faiss::heap_push<HC>(++k, dis.data(), ids.data(), v, i);
    ++nvalid;
}

float HNSW::MinimaxHeap::max() const {
    return dis[0];
}

int HNSW::MinimaxHeap::size() const {
    return nvalid;
}

void HNSW::MinimaxHeap::clear() {
    nvalid = k = 0;
}

int HNSW::MinimaxHeap::pop_min(float* vmin_out) {
    assert(k > 0);
    // returns min. This is an O(n) operation
    int i = k - 1;
    while (i >= 0) {
        if (ids[i] != -1)
            break;
        i--;
    }
    if (i == -1)
        return -1;
    int imin = i;
    float vmin = dis[i];
    i--;
    while (i >= 0) {
        if (ids[i] != -1 && dis[i] < vmin) {
            vmin = dis[i];
            imin = i;
        }
        i--;
    }
    if (vmin_out)
        *vmin_out = vmin;
    int ret = ids[imin];
    ids[imin] = -1;
    --nvalid;

    return ret;
}

int HNSW::MinimaxHeap::count_below(float thresh) {
    int n_below = 0;
    for (int i = 0; i < k; i++) {
        if (dis[i] < thresh) {
            n_below++;
        }
    }

    return n_below;
}

} // namespace faiss
