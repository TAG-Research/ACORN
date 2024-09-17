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


// added these
#include <faiss/Index.h>
#include <stdlib.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <pthread.h>
#include <iostream>
#include <sstream>      // for ostringstream
#include <fstream>  
#include <iosfwd>
#include <faiss/impl/platform_macros.h>
#include <assert.h>     /* assert */
#include <thread>
#include <set>
#include <math.h>  
#include <numeric> // for std::accumulate
#include <cmath>   // for std::mean and std::stdev
#include <nlohmann/json.hpp>
#include "utils.cpp"




// create indices for debugging, write indices to file, and get recall stats for all queries
int main(int argc, char *argv[]) {
    unsigned int nthreads = std::thread::hardware_concurrency();
    std::cout << "====================\nSTART: running TEST_ACORN for hnsw, sift data --" << nthreads << "cores\n" << std::endl;
    // printf("====================\nSTART: running MAKE_INDICES for hnsw --...\n");
    double t0 = elapsed();
    
    int efc = 40; // default is 40
    int efs = 16; //  default is 16
    int k = 10; // search parameter
    size_t d = 128; // dimension of the vectors to index - will be overwritten by the dimension of the dataset
    int M; // HSNW param M TODO change M back
    int M_beta; // param for compression
    // float attr_sel = 0.001;
    // int gamma = (int) 1 / attr_sel;
    int gamma;
    int n_centroids;
    // int filter = 0;
    std::string dataset; // must be sift1B or sift1M or tripclick
    int test_partitions = 0;
    int step = 10; //2
    
    std::string assignment_type = "rand";
    int alpha = 0;

    srand(0); // seed for random number generator
    int num_trials = 60;


    size_t N = 0; // N will be how many we truncate nb from sift1M to

    int opt;
    {// parse arguments

        if (argc < 6 || argc > 8) {
            fprintf(stderr, "Syntax: %s <number vecs> <gamma> [<assignment_type>] [<alpha>] <dataset> <M> <M_beta>\n", argv[0]);
            exit(1);
        }

        N = strtoul(argv[1], NULL, 10);
        printf("N: %ld\n", N);
     

        gamma = atoi(argv[2]);
        printf("gamma: %d\n", gamma);


        
        
        dataset = argv[3];
        printf("dataset: %s\n", dataset.c_str());
        if (dataset != "sift1M" && dataset != "sift1M_test" && dataset != "sift1B" && dataset != "tripclick" && dataset != "paper" && dataset != "paper_rand2m") {
            printf("got dataset: %s\n", dataset.c_str());
            fprintf(stderr, "Invalid <dataset>; must be a value in [sift1M, sift1B]\n");
            exit(1);
        }

        M = atoi(argv[4]);
        printf("M: %d\n", M);

        M_beta = atoi(argv[5]);
        printf("M_beta: %d\n", M_beta);

    }
    



    // load metadata
    n_centroids = gamma;

    std::vector<int> metadata = load_ab(dataset, gamma, 
    , N);
    metadata.resize(N);
    assert(N == metadata.size());
    printf("[%.3f s] Loaded metadata, %ld attr's found\n", 
        elapsed() - t0, metadata.size());

   

    size_t nq;
    float* xq;
    std::vector<int> aq;
    { // load query vectors and attributes
        printf("[%.3f s] Loading query vectors and attributes\n", elapsed() - t0);

        size_t d2;
        // xq = fvecs_read("sift1M/sift_query.fvecs", &d2, &nq);
        bool is_base = 0;
        // load_data(dataset, is_base, &d2, &nq, xq);
        std::string filename = get_file_name(dataset, is_base);
        xq = fvecs_read(filename.c_str(), &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as expected 128");
        if (d != d2) {
            d = d2;
        }
        
        std::cout << "query vecs data loaded, with dim: " << d2 << ", nb=" << nq << std::endl;
        printf("[%.3f s] Loaded query vectors from %s\n", elapsed() - t0, filename.c_str());
        aq = load_aq(dataset, n_centroids, alpha, N);
        printf("[%.3f s] Loaded %ld %s queries\n", elapsed() - t0, nq, dataset.c_str());
 
    }
    // nq = 1;
    int gt_size = 100;
    if (dataset=="sift1M_test" || dataset=="paper") {
        gt_size = 10;
    } 
    std::vector<faiss::idx_t> gt(gt_size * nq);
    { // load ground truth
        gt = load_gt(dataset, gamma, alpha, assignment_type, N);
        printf("[%.3f s] Loaded ground truth, gt_size: %d\n", elapsed() - t0, gt_size);
    }

    // create normal (base) and hybrid index
    printf("[%.3f s] Index Params -- d: %ld, M: %d, N: %ld, gamma: %d\n",
               elapsed() - t0, d, M, N, gamma);
    // base HNSW index
    faiss::IndexHNSWFlat base_index(d, M, 1); // gamma = 1
    base_index.hnsw.efConstruction = efc; // default is 40  in HNSW.capp
    base_index.hnsw.efSearch = efs; // default is 16 in HNSW.capp
    
    // ACORN-gamma
    faiss::IndexACORNFlat hybrid_index(d, M, gamma, metadata, M_beta);
    hybrid_index.acorn.efSearch = efs; // default is 16 HybridHNSW.capp
    debug("ACORN index created%s\n", "");


    // ACORN-1
    faiss::IndexACORNFlat hybrid_index_gamma1(d, M, 1, metadata, M*2);
    hybrid_index_gamma1.acorn.efSearch = efs; // default is 16 HybridHNSW.capp




    { // populating the database
        std::cout << "====================Vectors====================\n" << std::endl;
        // printf("====================Vectors====================\n");
       
        printf("[%.3f s] Loading database\n", elapsed() - t0);

        size_t nb, d2;
        bool is_base = 1;
        std::string filename = get_file_name(dataset, is_base);
        float* xb = fvecs_read(filename.c_str(), &d2, &nb);
        assert(d == d2 || !"dataset does not dim 128 as expected");
        printf("[%.3f s] Loaded base vectors from file: %s\n", elapsed() - t0, filename.c_str());

       

        std::cout << "data loaded, with dim: " << d2 << ", nb=" << nb << std::endl;

        printf("[%.3f s] Indexing database, size %ld*%ld from max %ld\n",
               elapsed() - t0, N, d2, nb);

        // index->add(nb, xb);

        printf("[%.3f s] Adding the vectors to the index\n", elapsed() - t0);
        
        base_index.add(N, xb);
        printf("[%.3f s] Vectors added to base index \n", elapsed() - t0);
        std::cout << "Base index vectors added: " << nb << std::endl;

        hybrid_index.add(N, xb);
        printf("[%.3f s] Vectors added to hybrid index \n", elapsed() - t0);
        std::cout << "Hybrid index vectors added" << nb << std::endl;
        // printf("SKIPPED creating ACORN-gamma\n");


        hybrid_index_gamma1.add(N, xb);
        printf("[%.3f s] Vectors added to hybrid index with gamma=1 \n", elapsed() - t0);
        std::cout << "Hybrid index with gamma=1 vectors added" << nb << std::endl;

    

        delete[] xb;       
    }
   

   // write hybrid index and partition indices to files
    {
        std::cout << "====================Write Index====================\n" << std::endl;
        // write hybrid index
        // std::string filename = "hybrid_index" + dataset + ".index";
        std::stringstream filepath_stream;
        if (dataset == "sift1M" || dataset == "sift1B") {
            filepath_stream << "./tmp/hybrid_"  << (int) (N / 1000 / 1000) << "m_nc=" << n_centroids << "_assignment=" << assignment_type << "_alpha=" << alpha << ".json";

        } else {
            filepath_stream << "./tmp/" << dataset << "/hybrid" << "_M=" << M << "_efc" << efc << "_Mb=" << M_beta << "_gamma=" << gamma << ".json";
        }
        std::string filepath = filepath_stream.str();
        write_index(&hybrid_index, filepath.c_str());
        printf("[%.3f s] Wrote hybrid index to file: %s\n", elapsed() - t0, filepath.c_str());
        
        // write hybrid_gamma1 index
        std::stringstream filepath_stream2;
        if (dataset == "sift1M" || dataset == "sift1B") {
            filepath_stream2 << "./tmp/hybrid_gamma1_"  << (int) (N / 1000 / 1000) << "m_nc=" << n_centroids << "_assignment=" << assignment_type << "_alpha=" << alpha << ".json";

        } else {
            filepath_stream2 << "./tmp/" << dataset << "/hybrid" << "_M=" << M << "_efc" << efc << "_Mb=" << M_beta << "_gamma=" << 1 << ".json";
        }
        std::string filepath2 = filepath_stream2.str();
        write_index(&hybrid_index_gamma1, filepath2.c_str());
        printf("[%.3f s] Wrote hybrid_gamma1 index to file: %s\n", elapsed() - t0, filepath2.c_str());

        
        { // write base index
            std::stringstream filepath_stream;
            if (dataset == "sift1M" || dataset == "sift1B") {
                filepath_stream << "./tmp/base_"  << (int) (N / 1000 / 1000) << "m_nc=" << n_centroids << "_assignment=" << assignment_type << "_alpha=" << alpha << ".json";

            } else {
                filepath_stream << "./tmp/" << dataset << "/base" << "_M=" << M << "_efc=" << efc << ".json";
            }
            std::string filepath = filepath_stream.str();
            write_index(&base_index, filepath.c_str());
            printf("[%.3f s] Wrote base index to file: %s\n", elapsed() - t0, filepath.c_str());
        }
      

        
        


        
    }

    { // print out stats
        printf("====================================\n");
        printf("============ BASE INDEX =============\n");
        printf("====================================\n");
        base_index.printStats(false);
        printf("====================================\n");
        printf("============ ACORN INDEX =============\n");
        printf("====================================\n");
        hybrid_index.printStats(false);
       
    }

    
    printf("==============================================\n");
    printf("====================Search Results====================\n");
    printf("==============================================\n");
    // double t1 = elapsed();
    printf("==============================================\n");
    printf("====================Search====================\n");
    printf("==============================================\n");
    double t1 = elapsed();
    
    { // searching the base database
        printf("====================HNSW INDEX====================\n");
        printf("[%.3f s] Searching the %d nearest neighbors "
               "of %ld vectors in the index, efsearch %d\n",
               elapsed() - t0,
               k,
               nq,
               base_index.hnsw.efSearch);

        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<float> dis(k * nq);

        std::cout << "here1" << std::endl;
        std::cout << "nn and dis size: " << nns.size() << " " << dis.size() << std::endl;

 

        double t1 = elapsed();
        base_index.search(nq, xq, k, dis.data(), nns.data());
        double t2 = elapsed();

        printf("[%.3f s] Query results (vector ids, then distances):\n",
               elapsed() - t0);

        // take max of 5 and nq
        int nq_print = std::min(5, (int) nq);
        for (int i = 0; i < nq_print; i++) {
            printf("query %2d nn's: ", i);
            for (int j = 0; j < k; j++) {
                // printf("%7ld (%d) ", nns[j + i * k], metadata.size());
                printf("%7ld (%d) ", nns[j + i * k], metadata[nns[j + i * k]]);
            }
            printf("\n     dis: \t");
            for (int j = 0; j < k; j++) {
                printf("%7g ", dis[j + i * k]);
            }
            printf("\n");
            // exit(0);
        }

        printf("[%.3f s] *** Query time: %f\n",
               elapsed() - t0, t2 - t1);
        
        // print number of distance computations
        // printf("[%.3f s] *** Number of distance computations: %ld\n",
            //    elapsed() - t0, base_index.ntotal * nq);
        std::cout << "finished base index examples" << std::endl;

    }

    {// look at stats
        // const faiss::HybridHNSWStats& stats = index.hnsw_stats;
        const faiss::HNSWStats& stats = faiss::hnsw_stats;

        std::cout << "============= BASE HNSW QUERY PROFILING STATS =============" << std::endl;
        printf("[%.3f s] Timing results for search of k=%d nearest neighbors of nq=%ld vectors in the index\n",
               elapsed() - t0,
               k,
               nq);
        std::cout << "n1: " << stats.n1 << std::endl;
        std::cout << "n2: " << stats.n2 << std::endl;
        std::cout << "n3 (number distance comps at level 0): " << stats.n3 << std::endl;
        std::cout << "ndis: " << stats.ndis << std::endl;
        std::cout << "nreorder: " << stats.nreorder << std::endl;
        printf("average distance computations per query: %f\n", (float)stats.n3 / stats.n1);
    
    }

    { // searching the hybrid database
        printf("==================== ACORN INDEX ====================\n");
        printf("[%.3f s] Searching the %d nearest neighbors "
               "of %ld vectors in the index, efsearch %d\n",
               elapsed() - t0,
               k,
               nq,
               hybrid_index.acorn.efSearch);

        std::vector<faiss::idx_t> nns2(k * nq);
        std::vector<float> dis2(k * nq);

        // create filter_ids_map, ie a bitmap of the ids that are in the filter
        std::vector<char> filter_ids_map(nq * N);
        for (int xq = 0; xq < nq; xq++) {
            for (int xb = 0; xb < N; xb++) {
                filter_ids_map[xq * N + xb] = (bool) (metadata[xb] == aq[xq]);
            }
        }

        double t1_x = elapsed();
        hybrid_index.search(nq, xq, k, dis2.data(), nns2.data(), filter_ids_map.data()); // TODO change first argument back to nq
        double t2_x = elapsed();


        printf("[%.3f s] Query results (vector ids, then distances):\n",
               elapsed() - t0);

        int nq_print = std::min(5, (int) nq);
        for (int i = 0; i < nq_print; i++) {
            printf("query %2d nn's (%d): ", i, aq[i]);
            for (int j = 0; j < k; j++) {
                printf("%7ld (%d) ", nns2[j + i * k], metadata[nns2[j + i * k]]);
            }
            printf("\n     dis: \t");
            for (int j = 0; j < k; j++) {
                printf("%7g ", dis2[j + i * k]);
            }
            printf("\n");
        }


        printf("[%.3f s] *** Query time: %f\n",
               elapsed() - t0, t2_x - t1_x);



         std::cout << "finished hybrid index examples" << std::endl;
    }



    

    // check here

    {// look at stats
        // const faiss::HybridHNSWStats& stats = index.hnsw_stats;
        const faiss::ACORNStats& stats = faiss::acorn_stats;

        std::cout << "============= ACORN QUERY PROFILING STATS =============" << std::endl;
        printf("[%.3f s] Timing results for search of k=%d nearest neighbors of nq=%ld vectors in the index\n",
               elapsed() - t0,
               k,
               nq);
        std::cout << "n1: " << stats.n1 << std::endl;
        std::cout << "n2: " << stats.n2 << std::endl;
        std::cout << "n3 (number distance comps at level 0): " << stats.n3 << std::endl;
        std::cout << "ndis: " << stats.ndis << std::endl;
        std::cout << "nreorder: " << stats.nreorder << std::endl;
        printf("average distance computations per query: %f\n", (float)stats.n3 / stats.n1);
    
    }
  
    

    printf("[%.3f s] -----DONE-----\n", elapsed() - t0);
}