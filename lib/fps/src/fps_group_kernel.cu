#include "__utils__.h"

__global__ void fps_group_kernel(int n, int m,
                                 const float *__restrict__ dataset,
                                 float *__restrict__ nearest,
                                 long *__restrict__ nearest_index,
                                 long *__restrict__ idxs) {
  __shared__ float dists[MAX_THREADS];
  __shared__ long dists_i[MAX_THREADS];

  int tid = threadIdx.x;
  const long block_size = blockDim.x;

  long old = 0;

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

      float *min_d_pointer = &nearest[k];

      if (d < *min_d_pointer) {
        *min_d_pointer = d;
        nearest_index[k] = j - 1;
      }

      if (*min_d_pointer > best) {
        besti = k;
        best = *min_d_pointer;
      }
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    compute_global_extremum(compare_parallel, update_max_by_value);

    old = dists_i[0];
    if (tid == 0) idxs[j] = old;
  }

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

    if (d < nearest[k]) {
      nearest_index[k] = m - 1;
    }
  }
}



at::Tensor fps_group(at::Tensor points, const int nsamples) {

  int n = points.size(0);
  at::Tensor output = torch::zeros(nsamples, at::device(points.device()).dtype(at::ScalarType::Long));

  at::Tensor nearest = torch::full(n, 1e10, at::device(points.device()).dtype(at::ScalarType::Float));
  at::Tensor nearest_index = torch::zeros(n, at::device(points.device()).dtype(at::ScalarType::Long));


  int n_threads = opt_n_threads(n);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  fps_group_kernel<<<1, n_threads, 0, stream>>>(n, nsamples, points.data_ptr<float>(), nearest.data_ptr<float>(), nearest_index.data_ptr<long>(), output.data_ptr<long>());

  return nearest_index;
}
