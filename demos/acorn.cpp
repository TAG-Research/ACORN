#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>
#include <sys/time.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexACORN.h>
#include <faiss/index_io.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iosfwd>
#include <faiss/impl/platform_macros.h>
#include <cassert>
#include <thread>
#include <set>
#include <numeric>
#include <nlohmann/json.hpp>
#include "utils.cpp"

// Additional includes
#include <getopt.h>

// Custom classes (SingleLabel and MultiLabel)

// Configuration structure
struct Config {
    size_t N;                       // Number of vectors
    size_t nq;                      // Number of queries
    std::string dataset_path;       // Path to the dataset
    int efc = 40;                   // HNSW construction parameter
    int efs = 16;                   // HNSW search parameter
    int k = 10;                     // Number of nearest neighbors
    int d = 384;                 // Dimension of vectors (to be overwritten)
    int M = 16;                     // HNSW parameter M
    int M_beta = 16;                // Compression parameter
    int gamma = 12;                  // ACORN parameter
    std::string assignment_type = "rand";
    int alpha = 0;
    std::string dataset_name = "default_dataset"; // Dataset name
    int n_centroids;                // Number of centroids (gamma)
};

// Function prototypes
void parseArguments(int argc, char* argv[], Config& config);
void loadMetadata(const Config& config, std::vector<int>& metadata, std::vector<std::string>& metadata_strings);
void loadQueryData(const Config& config, float*& xq, std::vector<int>& aq, std::vector<std::string>& aq_strings);
void loadGroundTruth(const Config& config, const std::vector<faiss::idx_t>& nns2);
void addVectorsToIndexes(const Config& config, faiss::IndexHNSWFlat& base_index_HNSWFlat, faiss::IndexACORNFlat& hybrid_index_ACORNFlat);
void writeIndexesToFiles(const Config& config, faiss::IndexHNSWFlat& base_index_HNSWFlat, faiss::IndexACORNFlat& hybrid_index_ACORNFlat);
void printIndexStats( faiss::IndexHNSWFlat& base_index_HNSWFlat,  faiss::IndexACORNFlat& hybrid_index_ACORNFlat);
void searchIndexes(const Config& config, faiss::IndexHNSWFlat& base_index_HNSWFlat, faiss::IndexACORNFlat& hybrid_index_ACORNFlat, float* xq, std::vector<std::string>& aq_strings, const std::vector<std::string>& metadata_strings);
void prepareFilterIdsMap(const Config& config, const std::vector<std::string>& aq_strings, const std::vector<std::string>& metadata_strings, std::vector<char>& filter_ids_map);
// Main function
int main(int argc, char* argv[]) {
    unsigned int nthreads = std::thread::hardware_concurrency();
    double t0 = elapsed();
    std::cout << "====================\nSTART: Running TEST_ACORN with " << nthreads << " cores\n" << std::endl;

    Config config;
    parseArguments(argc, argv, config);

    std::vector<int> metadata;
    std::vector<std::string> metadata_strings;
    loadMetadata(config, metadata, metadata_strings);

    float* xq;
    std::vector<int> aq;
    std::vector<std::string> aq_strings;
    loadQueryData(config, xq, aq, aq_strings);



        // base HNSW index
    faiss::IndexHNSWFlat base_index_HNSWFlat(config.d, config.M, 1); // gamma = 1
    base_index_HNSWFlat.hnsw.efConstruction = config.efc; // default is 40  in HNSW.capp
    base_index_HNSWFlat.hnsw.efSearch = config.efs; // default is 16 in HNSW.capp
    
    // ACORN-gamma
    faiss::IndexACORNFlat hybrid_index_ACORNFlat(config.d, config.M, config.gamma, metadata, config.M_beta);
    hybrid_index_ACORNFlat.acorn.efSearch = config.efs; // default is 16 HybridHNSW.capp
    debug("ACORN index created%s\n", "");


    // ACORN-1
    faiss::IndexACORNFlat hybrid_index_gamma1(config.d, config.M, 1, metadata, config.M*2);
    hybrid_index_gamma1.acorn.efSearch = config.efs; // default is 16 HybridHNSW.capp


    addVectorsToIndexes(config, base_index_HNSWFlat, hybrid_index_ACORNFlat);

    writeIndexesToFiles(config, base_index_HNSWFlat, hybrid_index_ACORNFlat);

    printIndexStats(base_index_HNSWFlat, hybrid_index_ACORNFlat);

    searchIndexes(config, base_index_HNSWFlat, hybrid_index_ACORNFlat, xq, aq_strings, metadata_strings);


  

    printf("[%.3f s] -----DONE-----\n", elapsed() - t0);

    delete[] xq;
    return 0;
}

// Function implementations
void parseArguments(int argc, char* argv[], Config& config) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <number of vectors N> <number of queries nq> <dataset path>\n", argv[0]);
        exit(1);
    }
    config.N = strtoul(argv[1], NULL, 10);
    config.nq = strtoul(argv[2], NULL, 10);
    config.dataset_path = argv[3];

    // Optional: parse more arguments for other configurations
    // For example:
    // config.gamma = atoi(argv[4]);
    // config.M = atoi(argv[5]);
    // config.M_beta = atoi(argv[6]);

    config.n_centroids = config.gamma;
    std::cout << "[Config] N: " << config.N << ", nq: " << config.nq << ", dataset_path: " << config.dataset_path << std::endl;
}

void loadMetadata(const Config& config, std::vector<int>& metadata, std::vector<std::string>& metadata_strings) {
    double t0 = elapsed();
    // Placeholder for loading metadata
    // Implement actual data loading based on your dataset format
    // For now, we simulate loading metadata

    // fake the orginal metadata since it will have segment fault
    metadata = load_ab("sift1M", config.gamma, config.assignment_type, config.N);
    metadata_strings = load_metadata_strings(config.dataset_path + "/base_labels.txt", config.N);

    //assert(config.N == metadata.size());
    printf("[%.3f s] Loaded metadata, %ld attributes found\n", elapsed() - t0, metadata.size());
}

void loadQueryData(const Config& config, float*& xq, std::vector<int>& aq, std::vector<std::string>& aq_strings) {
    double t0 = elapsed();
    printf("[%.3f s] Loading query vectors and attributes\n", elapsed() - t0);

    size_t d2;
    size_t nq_loaded;
    // Placeholder for loading query vectors
    // Implement actual data loading based on your dataset format

    // Example: Load from files (to be replaced with actual file paths)

    xq = fbin_read((config.dataset_path + "/query.fbin").c_str(), &d2, &nq_loaded);
    if (config.d != d2) {
        printf("Error: query vectors have different dimensionality (%ld) than base vectors (%d)\n", d2, config.d);
    }

    printf("[%.3f s] Loaded query vectors from %s\n", elapsed() - t0, (config.dataset_path + "/query.fbin").c_str());
    aq = load_aq(config.dataset_name, config.n_centroids, config.alpha, config.N);
    aq_strings = load_metadata_strings(config.dataset_path + "/query_labels.txt", nq_loaded);

    // Adjust nq if necessary
    if (config.nq > nq_loaded) {
        printf("Warning: nq is greater than the number of loaded queries, setting nq to %ld\n", nq_loaded);
    }
    //aq.resize(config.nq);
    //aq_strings.resize(config.nq);

    printf("[%.3f s] Loaded %ld queries\n", elapsed() - t0, config.nq);
}

void loadGroundTruth(const Config& config, const std::vector<faiss::idx_t>& nns2) {
    size_t nq = config.nq;
    size_t k = config.k;
    
    const std::string& gt_filename = config.dataset_path + "/gt.bin.txt";
    // Open the ground truth file
    std::ifstream reader(gt_filename);
    if (!reader.is_open()) {
        std::cerr << "Error opening ground truth file " << gt_filename << std::endl;
        return  ;
    }

    size_t npts, ndims;
    // Read header (number of points and dimensions)
    reader >> npts >> ndims;
    reader.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Skip to the next line

    if (npts != nq) {
        std::cerr << "Mismatch between number of queries in ground truth file and nns2" << std::endl;
     }

    size_t gt_k = std::min(k, ndims);

    // Read the ground truth IDs
    std::vector<std::unordered_set<int32_t>> gt_sets(nq);
    std::string line;
    for (size_t i = 0; i < nq; ++i) {
        std::getline(reader, line);
        std::istringstream iss(line);
        int32_t id;
        size_t count = 0;
        // Read up to gt_k IDs per query
        while (iss >> id && count < gt_k) {
            gt_sets[i].insert(id);
            count++;
        }
    }

    // Skip the distances
    reader.close();

    // Compute recall
    size_t total_correct = 0;
    for (size_t i = 0; i < nq; ++i) {
        size_t correct = 0;
        const auto& gt_set = gt_sets[i];
        for (size_t j = 0; j < k; ++j) {
            faiss::idx_t ann_id = nns2[j + i * k];
            if (gt_set.find(ann_id) != gt_set.end()) {
                correct++;
            }
        }
        total_correct += correct;
    }
    double recall = static_cast<double>(total_correct) / (nq * k);
    
    printf("Recall@%ld: %.3f\n", k, recall);
}
    
void addVectorsToIndexes(const Config& config, faiss::IndexHNSWFlat& base_index_HNSWFlat, faiss::IndexACORNFlat& hybrid_index_ACORNFlat) {
    double t0 = elapsed();
    printf("[%.3f s] Loading database vectors\n", elapsed() - t0);

    size_t nb, d2;
    // Placeholder for loading base vectors
    float* xb = fbin_read((config.dataset_path + "/base.fbin").c_str(), &d2, &nb);

    if (config.d != d2) {
        printf("Error: base vectors have different dimensionality (%ld) than query vectors (%d)\n", d2, config.d);
    }

    printf("[%.3f s] Loaded base vectors from %s\n", elapsed() - t0, (config.dataset_path + "/base.fbin").c_str());

    printf("[%.3f s] Adding vectors to indexes\n", elapsed() - t0);
    //base_index_HNSWFlat.add(config.N, xb);
    hybrid_index_ACORNFlat.add(config.N, xb);

    //delete[] xb;

    printf("[%.3f s] Added %ld vectors to indexes\n", elapsed() - t0, config.N);
}

void writeIndexesToFiles(const Config& config, faiss::IndexHNSWFlat& base_index_HNSWFlat, faiss::IndexACORNFlat& hybrid_index_ACORNFlat) {
    double t0 = elapsed();
    std::cout << "====================Write Index====================\n" << std::endl;

    std::stringstream filepath_stream;
    filepath_stream << config.dataset_path << "/hybrid_index_M=" << config.M << "_efc=" << config.efc << "_Mb=" << config.M_beta << "_gamma=" << config.gamma << ".index";
    std::string hybrid_index_path = filepath_stream.str();

    write_index(&hybrid_index_ACORNFlat, hybrid_index_path.c_str());
    printf("[%.3f s] Wrote hybrid index to file: %s\n", elapsed() - t0, hybrid_index_path.c_str());

    filepath_stream.str(std::string()); // Clear the stream
    filepath_stream << config.dataset_path << "/base_index_M=" << config.M << "_efc=" << config.efc << ".index";
    std::string base_index_path = filepath_stream.str();

    write_index(&base_index_HNSWFlat, base_index_path.c_str());
    printf("[%.3f s] Wrote base index to file: %s\n", elapsed() - t0, base_index_path.c_str());
}

void printIndexStats( faiss::IndexHNSWFlat& base_index_HNSWFlat,  faiss::IndexACORNFlat& hybrid_index_ACORNFlat) {
    printf("====================================\n");
    printf("============ BASE INDEX =============\n");
    printf("====================================\n");
    base_index_HNSWFlat.printStats(false);
    printf("====================================\n");
    printf("============ ACORN INDEX =============\n");
    printf("====================================\n");
    hybrid_index_ACORNFlat.printStats(false);
}

void searchIndexes(const Config& config, faiss::IndexHNSWFlat& base_index_HNSWFlat, faiss::IndexACORNFlat& hybrid_index_ACORNFlat, float* xq, std::vector<std::string>& aq_strings, const std::vector<std::string>& metadata_strings) {
    double t0 = elapsed();
    printf("==============================================\n");
    printf("====================Search Results====================\n");
    printf("==============================================\n");

    // Searching the base index
    printf("====================HNSW INDEX====================\n");
    printf("[%.3f s] Searching the %d nearest neighbors of %ld vectors in the index, efsearch %d\n",
           elapsed() - t0, config.k, config.nq, base_index_HNSWFlat.hnsw.efSearch);

    std::vector<faiss::idx_t> nns(config.k * config.nq);
    std::vector<float> dis(config.k * config.nq);

    double t1 = elapsed();
    base_index_HNSWFlat.search(config.nq, xq, config.k, dis.data(), nns.data());
    double t2 = elapsed();

    printf("[%.3f s] Query results (vector ids, then distances):\n", elapsed() - t0);
    int nq_print = std::min(5, (int)config.nq);
    for (int i = 0; i < nq_print; i++) {
        printf("query %2d nn's: ", i);
        for (int j = 0; j < config.k; j++) {
            printf("%7ld (%s) ", nns[j + i * config.k], metadata_strings[nns[j + i * config.k]].c_str());
        }
        printf("\n     dis: \t");
        for (int j = 0; j < config.k; j++) {
            printf("%7g ", dis[j + i * config.k]);
        }
        printf("\n");
    }
    printf("[%.3f s] *** Query time: %f\n", elapsed() - t0, t2 - t1);

    // Searching the hybrid index
    printf("==================== ACORN INDEX ====================\n");
    printf("[%.3f s] Searching the %d nearest neighbors of %ld vectors in the index, efsearch %d\n",
           elapsed() - t0, config.k, config.nq, hybrid_index_ACORNFlat.acorn.efSearch);

    std::vector<faiss::idx_t> nns2(config.k * config.nq);
    std::vector<float> dis2(config.k * config.nq);

    printf("[%.3f s] *** Start filter_ids_map %.3f\n", elapsed() - t0);
    // Prepare filter_ids_map
    double t1_f = elapsed();

    std::vector<char> filter_ids_map(config.nq * config.N);
    for (size_t xq_idx = 0; xq_idx < config.nq; xq_idx++) {
        MultiLabel label_query = MultiLabel::fromQuery(aq_strings[xq_idx]);
        for (size_t xb_idx = 0; xb_idx < config.N; xb_idx++) {
            MultiLabel label_base = MultiLabel::fromBase(metadata_strings[xb_idx]);
            filter_ids_map[xq_idx * config.N + xb_idx] =  label_query.isSubsetOf(label_base);
        }
    }
        double t2_f = elapsed();

    printf("[%.3f s] *** Done filter_ids_map %.3f\n", elapsed() - t0, t2_f - t1_f);

    printf("[%.3f s] *** Start search %f\n", elapsed() - t0);
    double t1_x = elapsed();
    hybrid_index_ACORNFlat.search(config.nq, xq, config.k, dis2.data(), nns2.data(), filter_ids_map.data());
    double t2_x = elapsed();

    printf("[%.3f s] *** Search done. Query time: %f\n", elapsed() - t0, t2_x - t1_x);
   
   
    // print config.nq, xq, config.k, dis2.data(), nns2.data(), filter_ids_map.data()
    std::cout << "config nq: " << config.nq <<" dis size:"<< dis2.size() << "nns size:"<< nns2.size() <<"id map:"<< filter_ids_map.size() << std::endl;

    std::cout << "aq_strings size: " << aq_strings.size() << std::endl;
    std::cout<< "metadata_strings size: " << metadata_strings.size() << std::endl;


    printf("[%.3f s] Query results (vector ids, then distances):\n", elapsed() - t0);
    for (int i = 0; i < nq_print; i++) {
        printf("query %2d nn's (%s): ", i, aq_strings[i].c_str());
        for (int j = 0; j < config.k; j++) {
            int id = nns2[j + i * config.k];
            if (id < 0 || id >= metadata_strings.size() ){
                printf("%d (Not Found) ", id);
            }else{
            printf("%7ld (%s) ", id, metadata_strings[id].c_str());
            }
        }
        printf("\n     dis: \t");
        for (int j = 0; j < config.k; j++) {
            printf("%7g ", dis2[j + i * config.k]);
        }
        printf("\n");
    }
    

    loadGroundTruth(config, nns2);
}

void prepareFilterIdsMap(
        const Config& config,
        const std::vector<std::string>& aq_strings,
        const std::vector<std::string>& metadata_strings,
        std::vector<char>& filter_ids_map) {
            
        }
