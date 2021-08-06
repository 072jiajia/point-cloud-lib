#include "utils.h"


#define compute_global_min() \
  min_parallel(1024, 512) \
  min_parallel(512, 256) \
  min_parallel(256, 128) \
  min_parallel(128, 64) \
  min_parallel(64, 32) \
  min_parallel(32, 16) \
  min_parallel(16, 8) \
  min_parallel(8, 4) \
  min_parallel(4, 2) \
  min_parallel(2, 1)


__global__ void knn_kernel(int n, int m, int K,
                           const float *__restrict__ points,
                           const float *__restrict__ samples,
                           long *__restrict__ nearest_index) {
  __shared__ float dists[MAX_THREADS];
  __shared__ long dists_i[MAX_THREADS];

  int tid = threadIdx.x;
  long block_size = blockDim.x;

  for (long i = blockIdx.x; i < n; i += gridDim.x) {
    long index1 = i * 3;
    float x1 = points[index1];
    float y1 = points[index1 + 1];
    float z1 = points[index1 + 2];

    long last_besti = 0;
    float last_best = -1.;
    for (long kth = 0; kth < K; kth++) {
      long besti = 0;
      float best = 1e20;
      for (long j = tid; j < m; j += block_size) {
        long index2 = j * 3;

        float x_diff = samples[index2] - x1;
        float y_diff = samples[index2 + 1] - y1;
        float z_diff = samples[index2 + 2] - z1;
        float d = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff;

        if (d <= last_best){
          if (d < last_best){
            continue;
          }
          else if (j <= last_besti){
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

      compute_global_min();

      last_best = dists[0];
      last_besti = dists_i[0];
      if (tid == 0) nearest_index[i * K + kth] = last_besti;
    }
  }
}



__global__ void knn_k1_kernel(int n, int m,
                              const float *__restrict__ points,
                              const float *__restrict__ samples,
                              long *__restrict__ nearest_index) {
  __shared__ float dists[MAX_THREADS];
  __shared__ long dists_i[MAX_THREADS];

  int tid = threadIdx.x;
  const long block_size = blockDim.x;

  for (long i = blockIdx.x; i < n; i += gridDim.x) {
    long besti = 0;
    float best = 1e20;

    long index1 = i * 3;
    float x1 = points[index1];
    float y1 = points[index1 + 1];
    float z1 = points[index1 + 2];
    for (long j = tid; j < m; j += block_size) {
      long index2 = j * 3;

      float x_diff = samples[index2] - x1;
      float y_diff = samples[index2 + 1] - y1;
      float z_diff = samples[index2 + 2] - z1;
      float d = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff;

      if (d < best) {
        besti = j;
        best = d;
      }
    }
    dists[tid] = best;
    dists_i[tid] = besti;

    __syncthreads();

    compute_global_min();

    if (tid == 0) nearest_index[i] = dists_i[0];
  }
}



at::Tensor knn(at::Tensor points, at::Tensor samples, int K) {

  int n = points.size(0);
  int m = samples.size(0);

  at::Tensor nearest_index = torch::zeros({n, K}, at::device(points.device()).dtype(at::ScalarType::Long));

  const int n_threads = opt_n_threads(m);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (K <= 1) {
    knn_k1_kernel<<<min(n, 16384), n_threads, 0, stream>>>
      (n, m, points.data_ptr<float>(), samples.data_ptr<float>(), nearest_index.data_ptr<long>());
  }
  else {
    knn_kernel<<<min(n, 16384), n_threads, 0, stream>>>
      (n, m, K, points.data_ptr<float>(), samples.data_ptr<float>(), nearest_index.data_ptr<long>());
  }

  return nearest_index;
}
