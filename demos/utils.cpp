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

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <stack>
#include <regex>
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
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
// #include <format>
// for convenience
using json = nlohmann::json;
/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
        -> wget -r ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
        -> cd ftp.irisa.fr/local/texmex/corpus
        -> tar -xf sift.tar.gz
        
 * and unzip it to the sudirectory sift1M.
 **/

// MACRO
#define TESTING_DATA_DIR "./testing_data"


#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <zlib.h>

// using namespace std;






 /*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

bool fileExists(const std::string& filePath) {
    std::ifstream file(filePath);
    return file.good();
}


float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    int size = fread(&d, 1, sizeof(int), f);
    printf("Dimension: %d\n", d);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}



#define PARTSIZE 10000000
#define ALIGNMENT 512
template <class T> T *aligned_malloc(const size_t n, const size_t alignment)
{
#ifdef _WINDOWS
    return (T *)_aligned_malloc(sizeof(T) * n, alignment);
#else
    return static_cast<T *>(aligned_alloc(alignment, sizeof(T) * n));
#endif
}

template <typename T> inline int get_num_parts(const char *filename)
{
    std::ifstream reader;
    reader.exceptions(std::ios::failbit | std::ios::badbit);
    reader.open(filename, std::ios::binary);
    std::cout << "Reading bin file " << filename << " ...\n";
    int npts_i32, ndims_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&ndims_i32, sizeof(int));
    std::cout << "#pts = " << npts_i32 << ", #dims = " << ndims_i32 << std::endl;
    reader.close();
    uint32_t num_parts =
        (npts_i32 % PARTSIZE) == 0 ? npts_i32 / PARTSIZE : (uint32_t)std::floor(npts_i32 / PARTSIZE) + 1;
    std::cout << "Number of parts: " << num_parts << std::endl;
    return num_parts;
}

template <typename T>
inline void load_bin_as_float(const char *filename, float *&data, size_t &npts, size_t &ndims, int part_num)
{
     try {
    std::ifstream reader;
    reader.exceptions(std::ios::failbit | std::ios::badbit);
    reader.open(filename, std::ios::binary);
    
    std::cout << "Reading bin file " << filename << " ...\n";
    int npts_i32, ndims_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&ndims_i32, sizeof(int));
    uint64_t start_id = part_num * PARTSIZE;
    uint64_t end_id = (std::min)(start_id + PARTSIZE, (uint64_t)npts_i32);
    npts = end_id - start_id;
    ndims = (uint64_t)ndims_i32;
    std::cout << "#pts in part = " << npts << ", #dims = " << ndims << ", size = " << npts * ndims * sizeof(T) << "B"
              << std::endl;

    reader.seekg(start_id * ndims * sizeof(T) + 2 * sizeof(uint32_t), std::ios::beg);
    T *data_T = new T[npts * ndims];
    reader.read((char *)data_T, sizeof(T) * npts * ndims);
    std::cout << "Finished reading part of the bin file." << std::endl;
    reader.close();
    data = aligned_malloc<float>(npts * ndims, ALIGNMENT);
#pragma omp parallel for schedule(dynamic, 32768)
    for (int64_t i = 0; i < (int64_t)npts; i++)
    {
        for (int64_t j = 0; j < (int64_t)ndims; j++)
        {
            float cur_val_float = (float)data_T[i * ndims + j];
            std::memcpy((char *)(data + i * ndims + j), (char *)&cur_val_float, sizeof(float));
        }
    }
    delete[] data_T;
    std::cout << "Finished converting part data to float." << std::endl;
    } catch (const std::ios_base::failure& e) {
        std::cerr << "I/O error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
}

float* fbin_read(const char* fname, size_t* d_out, size_t* n_out) {
    float *base_data = nullptr;
    int num_parts = get_num_parts<float>(fname);
 
    size_t npoints, dim;
    for (int p = 0; p < num_parts; p++)
    {
        size_t start_id = p * PARTSIZE;
        load_bin_as_float<float>(fname, base_data, npoints, dim, p);
        size_t end_id = start_id + npoints;

        *d_out = dim;
        *n_out = npoints;
    }

    return base_data;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}



// get file name to load data vectors from
std::string get_file_name(std::string dataset, bool is_base) {
    if (dataset == "sift1M" || dataset == "sift1M_test") {
        return std::string("./Datasets/sift1M/sift_") + (is_base ? "base" : "query") + ".fvecs";
    } else if (dataset == "sift1B") {
        return std::string("./Datasets/sift1B/bigann_") + (is_base ? "base_10m" : "query") + ".fvecs";
    } else if (dataset == "tripclick") {
        return std::string("./Datasets/tripclick/") + (is_base ? "base_vecs_tripclick" : "query_vecs_tripclick_min100") + ".fvecs";
    } else if (dataset == "paper" || dataset == "paper_rand2m") {
        return std::string("./Datasets/paper/") + (is_base ? "paper_base" : "paper_query") + ".fvecs";
    } else {
        std::cerr << "Invalid datset in get_file_name" << std::endl;
        return "";
    }
}

// return name is in arg file_path
void get_index_name(int N, int n_centroids, std::string assignment_type, float alpha, int M_beta, std::string& file_path) {
    std::stringstream filepath_stream;
    filepath_stream << "./tmp/hybrid_"  << (int) (N / 1000 / 1000) << "m_nc=" << n_centroids << "_assignment=" << assignment_type << "_alpha=" << alpha << "Mb=" << M_beta << ".json";
    // copy filepath_stream to file_path
    file_path = filepath_stream.str();
}





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
 * performance testing helpers
 *******************************************************/
std::pair<float, float> get_mean_and_std(std::vector<float>& times) {
    // compute mean
    float total = 0;
    // for (int num: times) {
    for (int i=0; i < times.size(); i++) {
       // printf("%f, ", times[i]); // for debugging
        total = total + times[i];
    }
    float mean = (total / times.size());

    // compute stdev from variance, using computed mean
    float result = 0;
    for (int i=0; i < times.size(); i++) {
        result = result + (times[i] - mean)*(times[i] - mean);
    }
    float variance = result / (times.size() - 1);
    // for debugging
    // printf("variance: %f\n", variance);

    float std = std::sqrt(variance);

    // return 
    return std::make_pair(mean, std);
}




// ground truth labels @gt, results to evaluate @I with @nq queries, returns @gt_size-Recall@k where gt had max gt_size NN's per query
float compute_recall(std::vector<faiss::idx_t>& gt, int gt_size, std::vector<faiss::idx_t>& I, int nq, int k, int gamma=1) {
    // printf("compute_recall params: gt.size(): %ld, gt_size: %d, I.size(): %ld, nq: %d, k: %d, gamma: %d\n", gt.size(), gt_size, I.size(), nq, k, gamma);
    
    int n_1 = 0, n_10 = 0, n_100 = 0;
    for (int i = 0; i < nq; i++) { // loop over all queries
        // int gt_nn = gt[i * k];
        std::vector<faiss::idx_t>::const_iterator first = gt.begin() + i*gt_size;
        std::vector<faiss::idx_t>::const_iterator last = gt.begin() + i*gt_size + (k / gamma);
        std::vector<faiss::idx_t> gt_nns_tmp(first, last);
        // if (gt_nns_tmp.size() > 10) {
        //     printf("gt_nns size: %ld\n", gt_nns_tmp.size());
        // }
        
        // gt_nns_tmp.resize(k); // truncate if gt_size > k
        std::set<faiss::idx_t> gt_nns(gt_nns_tmp.begin(), gt_nns_tmp.end());
        // if (gt_nns.size() > 10) {
        //     printf("gt_nns size: %ld\n", gt_nns.size());
        // }
        
        
        for (int j = 0; j < k; j++) { // iterate over returned nn results
            if (gt_nns.count(I[i * k + j])!=0) {
            // if (I[i * k + j] == gt_nn) {
                if (j < 1 * gamma)
                    n_1++;
                if (j < 10 * gamma)
                    n_10++;
                if (j < 100 * gamma)
                    n_100++;
            }
        }
    }
    // BASE ACCURACY
    // printf("* Base HNSW accuracy relative to exact search:\n");
    // printf("\tR@1 = %.4f\n", n_1 / float(nq) );
    // printf("\tR@10 = %.4f\n", n_10 / float(nq));
    // printf("\tR@100 = %.4f\n", n_100 / float(nq)); // not sure why this is always same as R@10
    // printf("\t---Results for %ld queries, k=%d, N=%ld, gt_size=%d\n", nq, k, N, gt_size);
    return (n_10 / float(nq));

}


template <typename T>
void log_values(std::string annotation, std::vector<T>& values) {
    std::cout << annotation;
    for (int i = 0; i < values.size(); i++) {
        std::cout << values[i];
        if (i < values.size() - 1) {
            std::cout << ", ";
        }
    } 
    std::cout << std::endl;
}




/////////////////////////////////////////////////////////////////////////////////////////////////////////
// 
// FOR CORRELATION TESTING
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
std::vector<T> load_json_to_vector(std::string filepath) {
   // Open the JSON file
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open JSON file" << std::endl;
        // return 1;
    }

    // Parse the JSON data
    json data;
    try {
        file >> data;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse JSON data from " << filepath << ": " << e.what() << std::endl;
        // return 1;
    }

    // Convert data to a vector
    std::vector<T> v =  data.get<std::vector<T>>();

    // print size
    std::cout << "metadata or vector loaded from json, size: " << v.size() << std::endl;
    return v;
}


std::vector<int> load_aq(std::string dataset, int n_centroids, int alpha, int N) {
    if (dataset == "sift1M" || dataset == "sift1B") {
        assert((alpha == -2 || alpha == 0 || alpha == 2) || !"alpha must be value in [-2, 0, 2]");

        // Compose File Name
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_DIR << "/query_filters_sift" << (int) (N / 1000 / 1000)  << "m_nc=" << n_centroids << "_alpha=" << alpha << ".json";
        std::string filepath = filepath_stream.str();

        std::vector<int> v = load_json_to_vector<int>(filepath);
        printf("loaded query attributes from: %s\n", filepath.c_str());

        
        // // print out data for debugging
        // for (int i : v) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        return v;
    } else if (dataset == "tripclick") {
        std::stringstream filepath_stream;   
        filepath_stream << TESTING_DATA_DIR << "/query_filters_tripclick_sample_subset_min100.json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<int> v = load_json_to_vector<int>(filepath);
        printf("loaded query attributes from: %s\n", filepath.c_str());

        
        // // print out data for debugging
        // for (int i : v) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        return v;
    } else if (dataset == "sift1M_test") {
        // return a vector of all int 5 with lenght N
        std::vector<int> v(N, 5);
        printf("made query filters with value %d, length %ld\n", v[0], v.size());
        return v;
    
    } else if (dataset == "paper") {
        std::vector<int> v(N, 5);
        printf("made query filters with value %d, length %ld\n", v[0], v.size());
        return v;
    } else if (dataset == "paper_rand2m") {
        // Compose File Name
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_DIR << "/query_filters_paper_rand2m_nc=12_alpha=0.json";
        std::string filepath = filepath_stream.str();

        std::vector<int> v = load_json_to_vector<int>(filepath);
        printf("loaded query attributes from: %s\n", filepath.c_str());
        return v;
    
    } else {
        std::cerr << "Invalid dataset in load_aq" << std::endl;
        return std::vector<int>();
    }
    

}
 
    // Helper function to replace all occurrences of a substring with another string
    void replaceAll(std::string& str, const std::string& from, const std::string& to) {
        if (from.empty())
            return;
        size_t start_pos = 0;
        while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
            str.replace(start_pos, from.length(), to);
            start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
        }
    }

std::vector<std::string> load_metadata_strings(std::string file_name, int N) {
 
    std::vector<std::string> lines;
    std::ifstream file(file_name);
    std::string line;
 
    while (std::getline(file, line)) {
        lines.push_back(line);
        if (lines.size() == 81546) {
            std::cout << "loaded metadata for 81545: " << lines[81545] << std::endl;
        }
    }
    file.close();

    printf("loaded metadata from: %s\n", file_name.c_str());
    printf("Number of lines loaded: %zu\n", lines.size());
    printf("Value of N: %d\n", N);
    return lines;
}

// assignment_type can be "rand", "soft", "soft_squared", "hard"
std::vector<int> load_ab(std::string dataset, int n_centroids, std::string assignment_type, int N) {
    // Compose File Name
    if (dataset == "sift1M" || dataset == "sift1B") {
        std::stringstream filepath_stream;   
        filepath_stream << TESTING_DATA_DIR << "/base_attrs_sift" <<   (int) (N / 1000 / 1000)   << "m_nc=" << n_centroids << "_assignment=" << assignment_type << ".json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<int> v = load_json_to_vector<int>(filepath);
        printf("loaded base attributes from: %s\n", filepath.c_str());

        
        // // print out data for debugging
        // for (int i : v) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        return v;
    } else if (dataset == "sift1M_test") {
        std::stringstream filepath_stream;   
        filepath_stream << TESTING_DATA_DIR << "/sift_attr" << ".json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<int> v = load_json_to_vector<int>(filepath);
        printf("loaded base attributes from: %s\n", filepath.c_str());
        return v;

    } else if (dataset == "paper") {
        std::stringstream filepath_stream;   
        filepath_stream << TESTING_DATA_DIR << "/paper_attr.json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<int> v = load_json_to_vector<int>(filepath);
        printf("loaded base attributes from: %s\n", filepath.c_str());

        return v;

    } else if (dataset == "paper_rand2m") {
        std::stringstream filepath_stream;   
        filepath_stream << TESTING_DATA_DIR << "/base_attrs_paper_rand2m_nc=12_assignment=rand.json";
        std::string filepath = filepath_stream.str();

        std::vector<int> v = load_json_to_vector<int>(filepath);
        printf("loaded base attributes from: %s\n", filepath.c_str());

        return v;
    } else if (dataset == "tripclick") {
        std::stringstream filepath_stream;   
        filepath_stream << TESTING_DATA_DIR << "/base_attrs_tripclick.json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<int> v = load_json_to_vector<int>(filepath);
        printf("loaded base attributes from: %s\n", filepath.c_str());

        
        // // print out data for debugging
        // for (int i : v) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        return v;
    } else {
        std::cerr << "Invalid dataset in load_ab" << std::endl;
        return std::vector<int>();
    }
    

}

// assignment_type can be "rand", "soft", "soft_squared", "hard"
// alpha can be -2, 0, 2
std::vector<faiss::idx_t> load_gt(std::string dataset, int n_centroids, int alpha, std::string assignment_type, int N) {
    if (dataset == "sift1M" || dataset == "sift1B") {
        // Compose File Name
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_DIR << "/gt_sift" << (int) (N / 1000 / 1000) << "m_nc=" << n_centroids << "_assignment=" << assignment_type << "_alpha=" << alpha << ".json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<int> v_tmp = load_json_to_vector<int>(filepath);
        std::vector<faiss::idx_t> v(v_tmp.begin(), v_tmp.end());
        printf("loaded gt from: %s\n", filepath.c_str());
        printf("gt size: %ld\n", v.size());
        // // print out data for debugging
        // for (faiss::idx_t i : v) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        return v;
    } else if (dataset == "sift1M_test") {
        // Compose File Name
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_DIR << "/sift_gt_5.json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<int> v_tmp = load_json_to_vector<int>(filepath);
        std::vector<faiss::idx_t> v(v_tmp.begin(), v_tmp.end());
        printf("loaded gt from: %s\n", filepath.c_str());

        // // print out data for debugging
        // for (faiss::idx_t i : v) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        return v;

    } else if (dataset == "paper") {
        // Compose File Name
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_DIR << "/paper_gt_5.json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<int> v_tmp = load_json_to_vector<int>(filepath);
        std::vector<faiss::idx_t> v(v_tmp.begin(), v_tmp.end());
        printf("loaded gt from: %s\n", filepath.c_str());

        // // print out data for debugging
        // for (faiss::idx_t i : v) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        return v;
    
    } else if (dataset == "paper_rand2m") {
        // Compose File Name
        std::stringstream filepath_stream;
        filepath_stream << TESTING_DATA_DIR << "/gt_paper_rand2m_nc=12_assignment=rand_alpha=0.json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<int> v_tmp = load_json_to_vector<int>(filepath);
        std::vector<faiss::idx_t> v(v_tmp.begin(), v_tmp.end());
        printf("loaded gt from: %s\n", filepath.c_str());

        return v;
    } else if (dataset == "tripclick") {
        std::stringstream filepath_stream;   
        filepath_stream << TESTING_DATA_DIR << "/gt_tripclick_sample_subset_min100.json";
        std::string filepath = filepath_stream.str();
        // printf("%s\n", filepath.c_str());

        std::vector<int> v_tmp = load_json_to_vector<int>(filepath);
        std::vector<faiss::idx_t> v(v_tmp.begin(), v_tmp.end());
        printf("loaded gt from: %s\n", filepath.c_str());

        // // print out data for debugging
        // for (faiss::idx_t i : v) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        return v;
    } else {
        std::cerr << "Invalid dataset in load_gt" << std::endl;
        return std::vector<faiss::idx_t>();
    }
    

}





class MultiLabel {
public:
    tsl::robin_set<std::string> base_clause;  // For base labels 
    std::vector<std::vector<std::string>> query_clause;  // For query labels,first level is AND, second level is OR
    // Constructors
    MultiLabel() {}
    static MultiLabel fromBase(const std::string& base_label) {
        MultiLabel ml;
        
         std::istringstream new_iss(base_label);
         std::string token;
        while (getline(new_iss, token, ','))
        {
            token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
            token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
            ml.base_clause.insert(token);
        }
        return ml;
    }


    static MultiLabel fromQuery(std::string& query_label) {
        MultiLabel ml;
        std::istringstream new_iss(query_label);
        std::string token;
        std::vector<std::vector<std::string>> lbls(0);
        while (getline(new_iss, token, '&'))
        {
            std::vector<std::string> or_clause(0);
            std::istringstream inner_iss(token);
            while (getline(inner_iss, token, '|'))
            {
                token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
                token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
                or_clause.push_back(token);
            }

            ml.query_clause.push_back(or_clause);
        }
          return ml;
    }

    void printQuery() {
        for (uint32_t k = 0; k < this->query_clause.size(); k++)
        {
            for (uint32_t l = 0; l < this->query_clause[k].size(); l++)
            {
                std::cout << this->query_clause[k][l];
                if (l < this->query_clause[k].size() - 1) {
                    std::cout << "|";
                }
            }
            if(k < this->query_clause.size() - 1) {
            std::cout << "&";
            }
        }
        std::cout << std::endl;
    }

    // Method to check if the query label is a subset of the base label
    bool isSubsetOf(const MultiLabel& base_label) const {
            bool pass = true;
            for (uint32_t k = 0; k < this->query_clause.size(); k++)
            {
                // check OR clause inside AND clause
                bool or_pass = false;
                for (uint32_t l = 0; l < this->query_clause[k].size(); l++)
                {
                    if (base_label.base_clause.find(this->query_clause[k][l]) != base_label.base_clause.end())
                    {
                        or_pass = true;
                        break;
                    }
                }

                // if any OR clause is not satisfied, then AND clause is not satisfied otherwise AND clause is satisfied
                if (or_pass == false) {
                    pass = false;
                    break;
                }
                
            }
        return pass;
    }
};











