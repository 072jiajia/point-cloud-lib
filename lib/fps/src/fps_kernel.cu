#include <stdio.h>
#include <stdlib.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>



#define max_parallel(bs, id) \
  if (block_size >= bs) { \
    if (tid < id) { \
      __update(dists, dists_i, tid, tid + id); \
    } \
    __syncthreads(); \
  }

__device__ void __update(float *__restrict__ dists, long *__restrict__ dists_i, const long idx1, const long idx2) {
  const float v1 = dists[idx1], v2 = dists[idx2];
  const long i1 = dists_i[idx1], i2 = dists_i[idx2];
  if (v2 > v1) {
    dists[idx1] = v2;
    dists_i[idx1] = i2;
  }
  else {
    dists[idx1] = v1;
    dists_i[idx1] = i1;
  }
}


#define TOTAL_THREADS 1024

__global__ void furthest_point_sampling_kernel(int n, int m, const int block_size,
                                               const float *__restrict__ dataset,
                                               float *__restrict__ temp,
                                               long *__restrict__ idxs) {
  if (m <= 0) return;
  __shared__ float dists[TOTAL_THREADS];
  __shared__ long dists_i[TOTAL_THREADS];

  int tid = threadIdx.x;

  long old = 0;
  idxs[0] = 0;

  // __syncthreads();
  for (long j = 1; j < m; j++) {
    long besti = 0;
    float best = -1;

    long index1 = old * 3;
    float x1 = dataset[index1];
    float y1 = dataset[index1 + 1];
    float z1 = dataset[index1 + 2];
    for (long k = tid; k < n; k += block_size) {
      long index2 = k * 3;

      float x_diff = dataset[index2] - x1;
      float y_diff = dataset[index2 + 1] - y1;
      float z_diff = dataset[index2 + 2] - z1;
      float d = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff;

      float d2 = min(d, temp[k]);
      temp[k] = d2;

      if (d2 > best) {
        besti = k;
        best = d2;
      }
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    max_parallel(1024, 512)

    max_parallel(512, 256)
    max_parallel(256, 128)
    max_parallel(128, 64)
    max_parallel(64, 32)
    max_parallel(32, 16)
    max_parallel(16, 8)
    max_parallel(8, 4)
    max_parallel(4, 2)
    max_parallel(2, 1)

    old = dists_i[0];
    if (tid == 0) idxs[j] = old;
  }
}


inline int opt_n_threads(int work_size) {
  const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
  return max(min(1 << pow_2, TOTAL_THREADS), 1);
}

void furthest_point_sampling_kernel_wrapper(int n, int m, const float *dataset, float *temp, long *idxs) {
  int n_threads = opt_n_threads(n);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  furthest_point_sampling_kernel<<<1, n_threads, 0, stream>>>(n, m, n_threads, dataset, temp, idxs);

}
