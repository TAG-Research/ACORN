# ACORN
ACORN is an index for state-of-the-art search over vector embeddings and structured data (SIGMOD '24)

You can read more about our work in the paper:
[**ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data**](https://dl.acm.org/doi/10.1145/3654923)

This implementation of the ACORN index is built on [**The FAISS Library**](https://github.com/facebookresearch/faiss) in C++.

If you run into any issues, please open an [issue](https://github.com/stanford-futuredata/ACORN/issues) and we'll respond promptly!

## Installation
```
git clone https://github.com/stanford-futuredata/ACORN.git
cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build
make -C build -j faiss
```


## Example Usage
1) Initialize the index
```
d=128;
M=32; 
gamma=12;
M_beta=32;

// ACORN-gamma
faiss::IndexACORNFlat acorn_gamma(d, M, gamma, M_beta);

// ACORN-1
faiss::IndexACORNFlat acorn_1(d, M, 1, M*2);
```
2) Construct the index
```
size_t nb, d2;
std::string filename = // your fvec file
float* xb = fvecs_read(filename.c_str(), &d2, &nb);
assert(d == d2 || !"dataset dimension is not as expected");
acorn_gamma.add(nb, xb);
```

3) Search the index
```
// ... load nq queries, xb
// ... load attribute filters as array aq

std::vector<faiss::idx_t> nns2(k * nq);
std::vector<float> dis2(k * nq);

// create filter_ids_map to specify the passing entities for each predicate
std::vector<char> filter_ids_map(nq * N);
for (int xq = 0; xq < nq; xq++) {
    for (int xb = 0; xb < N; xb++) {
        filter_ids_map[xq * N + xb] = (bool) (metadata[xb] == aq[xq]);
    }
}

// perform efficient hybrid search
acorn_gamma.search(nq, xq, k, dis2.data(), nns2.data(), filter_ids_map.data());
```
