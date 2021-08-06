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
  if (v2 < v1) {
    dists[idx1] = v2;
    dists_i[idx1] = dists_i[idx2];
  }
}


#define MAX_THREADS 1024

__global__ void furthest_point_sampling_kernel(int n, int m, int K,
                                               const float *__restrict__ points,
                                               const float *__restrict__ samples,
                                               long *__restrict__ nearest_index) {
  __shared__ float dists[MAX_THREADS];
  __shared__ long dists_i[MAX_THREADS];

  int tid = threadIdx.x;
  long block_size = gridDim.x;

  for (long i = blockIdx.x; i < m; i += gridDim.x) {
    long index1 = i * 3;
    float x1 = points[index1];
    float y1 = points[index1 + 1];
    float z1 = points[index1 + 2];

    long last_besti = 0;
    float last_best = -1;
    for (long kth = 0; kth < K; kth ++) {
      long besti = 0;
      float best = 1e20;
      for (long j = tid; j < n; j += block_size) {
        long index2 = j * 3;

        float x_diff = samples[index2] - x1;
        float y_diff = samples[index2 + 1] - y1;
        float z_diff = samples[index2 + 2] - z1;
        float d = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff;

        if (d <= last_best){
          if (j <= last_besti){
            continue;
          }
        }

        if (d < best) {
          besti = j;
          best = d;
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

      last_besti = dists_i[0];
      last_best = dists[0];
      if (tid == 0) nearest_index[i * K + kth] = last_besti;
    }
  }
}


inline int opt_n_threads(int work_size) {
  const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
  return max(min(1 << pow_2, MAX_THREADS), 1);
}

void furthest_point_sampling_kernel_wrapper(int n, int m, int K, const float *points, const float *samples, long *nearest_index) {
  const int n_threads = opt_n_threads(m);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  furthest_point_sampling_kernel<<<min(n, 16384), n_threads, 0, stream>>>(n, m, K, points, samples, nearest_index);

}
