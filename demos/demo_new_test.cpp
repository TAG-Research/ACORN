#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <sys/time.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

// added these
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



/*******************************************************
 * Added for debugging
 *******************************************************/
const int debugFlag = 1;

void debugTime() {
	if (debugFlag) {
        struct timeval tval;
        gettimeofday(&tval, NULL);
        struct tm *tm_info = localtime(&tval.tv_sec);
        char timeBuff[25] = "";
        strftime(timeBuff, 25, "%H:%M:%S", tm_info);
        char timeBuffWithMilli[50] = "";
        sprintf(timeBuffWithMilli, "%s.%06ld ", timeBuff, tval.tv_usec);
        std::string timestamp(timeBuffWithMilli);
		std::cout << timestamp << std::flush;
    }
}

//needs atleast 2 args always
//  alt debugFlag = 1 // fprintf(stderr, fmt, __VA_ARGS__); 
#define debug(fmt, ...) \
    do { \
        if (debugFlag == 1) { \
            fprintf(stdout, "--" fmt, __VA_ARGS__);\
        } \
        if (debugFlag == 2) { \
            debugTime(); \
            fprintf(stdout, "%s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, __VA_ARGS__); \
        } \
    } while (0)



double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/*******************************************************
 * Run tests
 *******************************************************/

//  args are nb, M, gamma
int main(int argc, char *argv[]) {
    printf("====================\nSTART: running tests for hnsw...\n");
    double t0 = elapsed();
    int opt;
    int d = 128; // dimension of the vectors to index
    size_t nb;
    int M;
    int gamma;
    // int d = 128; // dimension of the vectors to index
    // int M = 32 * 1000; // HSNW param M
    // size_t nb = 1000; // size of the database we plan to index


    {// parse arguments

        if (argc != 4) {
            fprintf(stderr, "Syntax: %s <number vecs> <M> <gamma>\n", argv[0]);
            exit(1);
        }

        nb = strtoul(argv[1], NULL, 10);
        debug("nb: %ld\n", nb);

        M = atoi(argv[2]);
        debug("M: %d\n", M);

        gamma = atoi(argv[3]);
        debug("gamma: %d\n", gamma);
    }
    
    printf("[%.3f s] Index Params -- d: %d, M: %d, nb: %ld, gamma: %d\n",
               elapsed() - t0, d, M, nb, gamma);
    faiss::IndexHNSWFlat index(d, M, gamma);
    debug("HNSW index created%s\n", "");
    
    std::mt19937 rng; // random generator to be used for creating vectors

    size_t nq; // num queries
    std::vector<float> queries;

    { // populating the database
        printf("[%.3f s] Building a dataset of %ld vectors to index\n",
               elapsed() - t0,
               nb);

        std::vector<float> database(nb * d);
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < nb * d; i++) {
            database[i] = distrib(rng);
        }

        printf("[%.3f s] Adding the vectors to the index\n", elapsed() - t0);

        index.add(nb, database.data());

        printf("[%.3f s] Vectors added\n", elapsed() - t0);

        // TODO: print out stats here
        // printf("[%.3f s] imbalance factor: %g\n",
        //        elapsed() - t0,
        //        index.invlists->imbalance_factor());

        // remember a few elements from the database as queries
        int i0 = 4;
        int i1 = 8;

        nq = i1 - i0;
        queries.resize(nq * d);
        for (int i = i0; i < i1; i++) {
            for (int j = 0; j < d; j++) {
                queries[(i - i0) * d + j] = database[i * d + j];
            }
        }
    }

    { // print out stats
        index.printStats();
    }

    { // get index size
        
        //  file name
        std::ostringstream ss;
        ss << "./tmp/index_hnsw_N=" << nb << ".faissindex";
        std::string s_tmp = ss.str();
        const char* outfilename = s_tmp.c_str();
        // const char* outfilename = "/tmp/index_hnsw.faissindex";
        printf("[%.3f s] storing the hnsw index to %s\n",
               elapsed() - t0,
               outfilename);

        // write index to disk
        write_index(&index, outfilename);

        //  measure file size
        std::ifstream in_file(outfilename, std::ios::binary);
        in_file.seekg(0, std::ios::end);
        int file_size = in_file.tellg();
        std::cout<<"====Size of the file is"<<" "<< file_size<<" "<<"bytes" << std::endl;
        
    }

    printf("-----DONE-----\n");
}