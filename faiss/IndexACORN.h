/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/ACORN.h>
#include <faiss/utils/utils.h>

// added
#include <sys/time.h>
#include <stdio.h>
#include <iostream>

namespace faiss {

struct IndexACORN;




/** The ACORN index is a normal random-access index with a ACORN
 * link structure built on top */

struct IndexACORN : Index {
    typedef ACORN::storage_idx_t storage_idx_t;

    // the link strcuture
    ACORN acorn; // TODO change to hybrid

    // the sequential storage
    bool own_fields;
    Index* storage;

//     ReconstructFromNeighbors* reconstruct_from_neighbors;

    explicit IndexACORN(int d, int M, int gamma, std::vector<int>& metadata, int M_beta, MetricType metric = METRIC_L2); // defaults d = 0, M=32, gamma=1
    explicit IndexACORN(Index* storage, int M, int gamma, std::vector<int>& metadata, int M_beta);
//     explicit IndexACORN(); // TODO check this is right

    ~IndexACORN() override;

    // add n vectors of dimension d to the index, x is the matrix of vectors
    void add(idx_t n, const float* x) override;

    /// Trains the storage if needed
    void train(idx_t n, const float* x) override;

    /// entry point for search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    // search for metadata
    // this doesn't override normal search since definition has a filter param - search is overloaded
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            char* filter_id_map,
            const SearchParameters* params = nullptr) const;

    void reconstruct(idx_t key, float* recons) const override;

    void reset() override;

    

    // added for debugging
    void printStats(bool print_edge_list=false, bool print_filtered_edge_lists=false, int filter=-1, Operation op=EQUAL);

    private:
        const int debugFlag = 0;

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
};

/** Flat index topped with with a ACORN structure to access elements
 *  more efficiently.
 */

struct IndexACORNFlat : IndexACORN {
    IndexACORNFlat();
    IndexACORNFlat(int d, int M, int gamma, std::vector<int>& metadata, int M_beta, MetricType metric = METRIC_L2);

};






} // namespace faiss
