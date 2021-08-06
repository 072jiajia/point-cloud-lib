#include <stdio.h>
#include <stdlib.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


#define __update_min(dists, dists_i, idx1, idx2) \
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

#define min_parallel(bs, id) \
  if (block_size >= bs) { \
    if (tid < id) { \
      const long tid2 = tid + id; \
      __update_min(dists, dists_i, tid, tid2); \
    } \
    __syncthreads(); \
  }


#define MAX_THREADS 1024

inline int opt_n_threads(int work_size) {
  const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
  return max(min(1 << pow_2, MAX_THREADS), 1);
}


