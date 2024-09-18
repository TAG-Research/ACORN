# FAISS HNSW and ACORN Implementation

## Overview

This document provides an overview of the HNSW (Hierarchical Navigable Small World) and ACORN (Approximate COntent-based Retrieval Network) implementations in the FAISS library. These algorithms are used for efficient similarity search in high-dimensional spaces.

## Code Structure

### `faiss/impl/HNSW.h`

The `HNSW` struct in [`faiss/impl/HNSW.h`](faiss/impl/HNSW.h) contains the core implementation of the HNSW algorithm. Key methods include:

- `add_with_locks`: Adds a point to all levels up to `pt_level` and builds the link structure.
- `hybrid_add_with_locks`: Similar to `add_with_locks` but for hybrid indices.
- `search`: Searches for the `k` nearest neighbors of a query point.
- `search_level_0`: Searches only in level 0 from a given vertex.
- `hybrid_search`: Searches for the `k` nearest neighbors with additional filtering options.
- `reset`, `clear_neighbor_tables`, `print_neighbor_stats`, `print_edges`: Utility functions for managing and debugging the HNSW structure.

### `faiss/IndexHNSW.h`

The [`faiss/IndexHNSW.h`](faiss/IndexHNSW.h) file defines several index structures that use HNSW for efficient access:

- `IndexHNSW`: Base class for HNSW indices.
- `IndexHNSWFlat`: Flat index with HNSW structure.
- `IndexHNSWPQ`: PQ (Product Quantization) index with HNSW structure.
- `IndexHNSWSQ`: SQ (Scalar Quantization) index with HNSW structure.
- `IndexHNSW2Level`: Two-level index with HNSW structure.
- `IndexHNSWHybridOld`: Hybrid index inheriting from `IndexHNSWFlat`.



## ACORN Implementation

### Overview

ACORN (Approximate COntent-based Retrieval Network) is an indexing method implemented in FAISS for efficient similarity search. It is designed to handle large-scale datasets with high-dimensional vectors. ACORN uses a combination of hashing and graph-based search to quickly approximate nearest neighbors.

### Code Structure

#### `faiss/impl/ACORN.h`

The `ACORN` struct in [`faiss/impl/ACORN.h`](faiss/impl/ACORN.h) contains the core implementation of the ACORN algorithm. Key methods include:

- `add`: Adds a new vector to the index.
- `search`: Searches for the `k` nearest neighbors of a query vector.
- `remove`: Removes a vector from the index.
- `update`: Updates an existing vector in the index.
- `rebuild`: Rebuilds the index to optimize search performance.

#### `faiss/IndexACORN.h`

The [`faiss/IndexACORN.h`](faiss/IndexACORN.h) file defines the index structure that uses ACORN for efficient access:

- `IndexACORN`: Base class for ACORN indices.

#### `benchs/acorn/README.md`

The [`benchs/acorn/README.md`](benchs/acorn/README.md) file explains the benchmarking process for the ACORN indexing method. It provides instructions for running benchmarks and analyzing the performance of the ACORN index.

### How ACORN Works

1. **Initialization**: The ACORN index is initialized with parameters such as the number of hash tables and the dimensionality of the vectors.
2. **Adding Vectors**: Vectors are added to the index using the `add` method. Each vector is hashed into multiple hash tables.
3. **Searching**: To find the `k` nearest neighbors of a query vector, the `search` method is used. The query vector is hashed, and candidate vectors are retrieved from the hash tables. A graph-based search is then performed to refine the results.
4. **Updating and Removing Vectors**: Vectors can be updated or removed from the index using the `update` and `remove` methods, respectively.
5. **Rebuilding the Index**: The `rebuild` method can be used to optimize the index for better search performance.

