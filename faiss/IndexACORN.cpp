/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexACORN.h>

#include <omp.h>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <queue>
#include <unordered_set>

#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <faiss/Index2Layer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/sorting.h>

// # added
#include <sys/time.h>
#include <stdio.h>
#include <iostream>





extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

namespace faiss {

using MinimaxHeap = ACORN::MinimaxHeap;
using storage_idx_t = ACORN::storage_idx_t;
using NodeDistFarther = ACORN::NodeDistFarther;

ACORNStats acorn_stats;

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/

namespace {

/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCE search easier */

struct NegativeDistanceComputer : DistanceComputer {
    /// owned by this
    DistanceComputer* basedis;

    explicit NegativeDistanceComputer(DistanceComputer* basedis)
            : basedis(basedis) {}

    void set_query(const float* x) override {
        basedis->set_query(x);
    }

    /// compute distance of vector i to current query
    float operator()(idx_t i) override {
        return -(*basedis)(i);
    }

    /// compute distance between two stored vectors
    float symmetric_dis(idx_t i, idx_t j) override {
        return -basedis->symmetric_dis(i, j);
    }

    virtual ~NegativeDistanceComputer() {
        delete basedis;
    }
};

DistanceComputer* storage_distance_computer(const Index* storage) {
    if (storage->metric_type == METRIC_INNER_PRODUCT) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

// TODO
void acorn_add_vertices(
        IndexACORN& index_acorn,
        size_t n0,
        size_t n,
        const float* x,
        bool verbose,
        bool preset_levels = false) {
    size_t d = index_acorn.d;
    ACORN& acorn = index_acorn.acorn;
    size_t ntotal = n0 + n;
    double t0 = getmillisecs();
    if (verbose) {
        printf("acorn_add_vertices: adding %zd elements on top of %zd "
               "(preset_levels=%d)\n",
               n,
               n0,
               int(preset_levels));
    }

    if (n == 0) {
        return;
    }

    int max_level = acorn.prepare_level_tab(n, preset_levels);

    if (verbose) {
        printf("  max_level = %d\n", max_level);
    }

    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

    // add vectors from highest to lowest level
    std::vector<int> hist;
    std::vector<int> order(n);

    { // make buckets with vectors of the same level

        // build histogram
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = acorn.levels[pt_id] - 1;
            while (pt_level >= hist.size())
                hist.push_back(0);
            hist[pt_level]++;
        }

        // accumulate
        std::vector<int> offsets(hist.size() + 1, 0);
        for (int i = 0; i < hist.size() - 1; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        // bucket sort
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = acorn.levels[pt_id] - 1;
            order[offsets[pt_level]++] = pt_id;
        }
    }

    idx_t check_period = InterruptCallback::get_period_hint(
            max_level * index_acorn.d * acorn.efConstruction);

    { // perform add
        RandomGenerator rng2(789);

        int i1 = n;

        for (int pt_level = hist.size() - 1; pt_level >= 0; pt_level--) {
            int i0 = i1 - hist[pt_level];

            if (verbose) {
                printf("Adding %d elements at level %d\n", i1 - i0, pt_level);
            }

            // random permutation to get rid of dataset order bias
            for (int j = i0; j < i1; j++)
                std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

            bool interrupt = false;

#pragma omp parallel if (i1 > i0 + 100)
            {
                VisitedTable vt(ntotal);

                DistanceComputer* dis =
                        storage_distance_computer(index_acorn.storage);
                ScopeDeleter1<DistanceComputer> del(dis);
                int prev_display =
                        verbose && omp_get_thread_num() == 0 ? 0 : -1;
                size_t counter = 0;

                // here we should do schedule(dynamic) but this segfaults for
                // some versions of LLVM. The performance impact should not be
                // too large when (i1 - i0) / num_threads >> 1
#pragma omp for schedule(static)
                for (int i = i0; i < i1; i++) {
                    storage_idx_t pt_id = order[i];
                    dis->set_query(x + (pt_id - n0) * d);

                    // cannot break
                    if (interrupt) {
                        continue;
                    }

                    acorn.add_with_locks(*dis, pt_level, pt_id, locks, vt);

                    if (prev_display >= 0 && i - i0 > prev_display + 10000) {
                        prev_display = i - i0;
                        printf("  %d / %d\r", i - i0, i1 - i0);
                        fflush(stdout);
                    }
                    if (counter % check_period == 0) {
                        if (InterruptCallback::is_interrupted()) {
                            interrupt = true;
                        }
                    }
                    counter++;
                }
            }
            if (interrupt) {
                FAISS_THROW_MSG("computation interrupted");
            }
            i1 = i0;
        }
        FAISS_ASSERT(i1 == 0);
    }
    if (verbose) {
        printf("Done in %.3f ms\n", getmillisecs() - t0);
    }

    for (int i = 0; i < ntotal; i++) {
        omp_destroy_lock(&locks[i]);
    }
}


} // namespace

/**************************************************************
 * IndexACORN implementation
 **************************************************************/

IndexACORN::IndexACORN(int d, int M, int gamma, std::vector<int>& metadata, int M_beta, MetricType metric)
        : Index(d, metric),
          acorn(M, gamma, metadata, M_beta),
          own_fields(false),
          storage(nullptr)
          /* reconstruct_from_neighbors(nullptr)*/ {}

IndexACORN::IndexACORN(Index* storage, int M, int gamma, std::vector<int>& metadata, int M_beta)
        : Index(storage->d, storage->metric_type),
          acorn(M, gamma, metadata, M_beta), // TOOD acorn needs to keep metadata now
          own_fields(false),
          storage(storage)
          /* reconstruct_from_neighbors(nullptr) */ {}

IndexACORN::~IndexACORN() {
    if (own_fields) {
        delete storage;
    }
}

void IndexACORN::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexACORNFlat (or variants) instead of IndexACORN directly");
    // acorn structure does not require training
    storage->train(n, x);
    is_trained = true;
}

// overloaded search for hybrid search 
void IndexACORN::search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            char* filter_id_map,
            const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexACORNFlat (or variants) instead of IndexACORN directly");
    const SearchParametersACORN* params = nullptr;

    int efSearch = acorn.efSearch;
    if (params_in) {
        params = dynamic_cast<const SearchParametersACORN*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "params type invalid");
        efSearch = params->efSearch;
    }
    size_t n1 = 0, n2 = 0, n3 = 0, ndis = 0, nreorder = 0;
    double candidates_loop = 0, neighbors_loop = 0, tuple_unwrap = 0, skips = 0, visits = 0; //added for profiling

    idx_t check_period =
            InterruptCallback::get_period_hint(acorn.max_level * d * efSearch);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel
        {
            VisitedTable vt(ntotal);

            DistanceComputer* dis = storage_distance_computer(storage);
            ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for reduction(+ : n1, n2, n3, ndis, nreorder, candidates_loop)
            for (idx_t i = i0; i < i1; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                char* filters = filter_id_map + i * ntotal;
                dis->set_query(x + i * d);

                maxheap_heapify(k, simi, idxi);
                ACORNStats stats = acorn.hybrid_search(*dis, k, idxi, simi, vt, filters, params); //TODO edit to hybrid search

                // ACORNStats stats = acorn.hybrid_search(*dis, k, idxi, simi, vt, filters[i], op, regex, params); //TODO edit to hybrid search
                n1 += stats.n1;
                n2 += stats.n2;
                n3 += stats.n3;
                ndis += stats.ndis;
                nreorder += stats.nreorder;
                // printf("index -- stats updates: %f\n", stats.candidates_loop);
                // printf("index -- stats updates: %f\n", stats.neighbors_loop);
                //added for profiling
                candidates_loop += stats.candidates_loop; 
                neighbors_loop += stats.neighbors_loop;
                tuple_unwrap += stats.tuple_unwrap;
                skips += stats.skips;
                visits += stats.visits;
                maxheap_reorder(k, simi, idxi);
                
            }
        }
        InterruptCallback::check();
    }

    if (metric_type == METRIC_INNER_PRODUCT) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }

    acorn_stats.combine({n1, n2, n3, ndis, nreorder, candidates_loop, neighbors_loop, tuple_unwrap, skips, visits}); //added for profiling
}

// TODO figure out what do with this
void IndexACORN::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexACORNFlat (or variants) instead of IndexACORN directly");
    const SearchParametersACORN* params = nullptr;

    int efSearch = acorn.efSearch;
    if (params_in) {
        params = dynamic_cast<const SearchParametersACORN*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "params type invalid");
        efSearch = params->efSearch;
    }
    size_t n1 = 0, n2 = 0, n3 = 0, ndis = 0, nreorder = 0;
    
    idx_t check_period =
            InterruptCallback::get_period_hint(acorn.max_level * d * efSearch);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel
        {
            VisitedTable vt(ntotal);

            DistanceComputer* dis = storage_distance_computer(storage);
            ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for reduction(+ : n1, n2, n3, ndis, nreorder)
            for (idx_t i = i0; i < i1; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                dis->set_query(x + i * d);

                maxheap_heapify(k, simi, idxi);
                ACORNStats stats = acorn.search(*dis, k, idxi, simi, vt, params);
                n1 += stats.n1;
                n2 += stats.n2;
                n3 += stats.n3;
                ndis += stats.ndis;
                nreorder += stats.nreorder;
                maxheap_reorder(k, simi, idxi);

                
            }
        }
        InterruptCallback::check();
    }

    if (metric_type == METRIC_INNER_PRODUCT) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }

    acorn_stats.combine({n1, n2, n3, ndis, nreorder});
}

// add n vectors of dimension d to the index, x is the matrix of vectors TODO
void IndexACORN::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexACORNFlat (or variants) instead of IndexACORN directly");
    FAISS_THROW_IF_NOT(is_trained);
    int n0 = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;

    acorn_add_vertices(*this, n0, n, x, verbose, acorn.levels.size() == ntotal);
}

void IndexACORN::reset() {
    acorn.reset();
    storage->reset();
    ntotal = 0;
}

void IndexACORN::reconstruct(idx_t key, float* recons) const {
    storage->reconstruct(key, recons);
}





// added for debugging TODO
void IndexACORN::printStats(bool print_edge_list, bool print_filtered_edge_lists, int filter,  Operation op) {
    acorn.print_neighbor_stats(print_edge_list, print_filtered_edge_lists, filter, op);
    printf("METADATA VEC for number nodes per level\n");
    for (int i = 0; i < acorn.nb_per_level.size(); i++) {
        printf("\tlevel %d: %d nodes\n", i, acorn.nb_per_level[i]);
    }
}



/**************************************************************
 * IndexACORNFlat implementation
 **************************************************************/




IndexACORNFlat::IndexACORNFlat(int d, int M, int gamma, std::vector<int>& metadata, int M_beta, MetricType metric)
        : IndexACORN(new IndexFlat(d, metric), M, gamma, metadata, M_beta) {
    own_fields = true;
    is_trained = true;
}





} // namespace faiss
