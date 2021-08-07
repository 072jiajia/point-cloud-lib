#ifndef UTILS_H
#define UTILS_H

// #include <stdio.h>
// #include <stdlib.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// #include <google/dense_hash_map>

// #include <cstdint>
// #include <array>
#include <vector>
// #include <queue>

using Int = long;


#define update_min_by_value_and_index(dists, dists_i, idx1, idx2) \
  long *index_pointer1 = &dists_i[idx1]; \
  long *index_pointer2 = &dists_i[idx2]; \
  float *pointer1 = &dists[idx1]; \
  float *pointer2 = &dists[idx2]; \
  if (*pointer2 < *pointer1) { \
    *pointer1 = *pointer2; \
    *index_pointer1 = *index_pointer2; \
  } \
  else if (*pointer2 == *pointer1) { \
    if (*index_pointer1 > *index_pointer2) { \
      *pointer1 = *pointer2; \
      *index_pointer1 = *index_pointer2; \
    } \
  }


#define update_min_by_value(dists, dists_i, idx1, idx2) \
  float *pointer1 = &dists[idx1]; \
  float *pointer2 = &dists[idx2]; \
  if (*pointer2 < *pointer1) { \
    *pointer1 = *pointer2; \
    dists_i[idx1] = dists_i[idx2]; \
  }

#define update_max_by_value(dists, dists_i, idx1, idx2) \
  float *pointer1 = &dists[idx1]; \
  float *pointer2 = &dists[idx2]; \
  if (*pointer2 > *pointer1) { \
    *pointer1 = *pointer2; \
    dists_i[idx1] = dists_i[idx2]; \
  }


#define compare_parallel(update_function, bs, id) \
  if (block_size >= bs) { \
    if (tid < id) { \
      const long tid2 = tid + id; \
      update_function(dists, dists_i, tid, tid2); \
    } \
    __syncthreads(); \
  }


#define compute_global_extremum(parallel_function, update_function) \
  parallel_function(update_function, 1024, 512) \
  parallel_function(update_function, 512, 256) \
  parallel_function(update_function, 256, 128) \
  parallel_function(update_function, 128, 64) \
  parallel_function(update_function, 64, 32) \
  parallel_function(update_function, 32, 16) \
  parallel_function(update_function, 16, 8) \
  parallel_function(update_function, 8, 4) \
  parallel_function(update_function, 4, 2) \
  parallel_function(update_function, 2, 1)


#define MAX_THREADS 1024

inline int opt_n_threads(int work_size) {
  const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
  return max(min(1 << pow_2, MAX_THREADS), 1);
}


#endif