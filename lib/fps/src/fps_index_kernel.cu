#include "__utils__.h"



__global__ void fps_index_kernel(int n, int m,
                                 const float *__restrict__ dataset,
                                 float *__restrict__ temp,
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

      float *min_d_pointer = &temp[k];
      *min_d_pointer = min(d, *min_d_pointer);

      if (*min_d_pointer > best) {
        best = *min_d_pointer;
        besti = k;
      }
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    compute_global_extremum(compare_parallel, update_max_by_value);

    old = dists_i[0];
    if (tid == 0) idxs[j] = old;
  }
}


at::Tensor fps_index(at::Tensor points, const int nsamples) {

  int n = points.size(0);

  at::Tensor output = torch::zeros(nsamples, at::device(points.device()).dtype(at::ScalarType::Long));
  at::Tensor tmp = torch::full(n, 1e10, at::device(points.device()).dtype(at::ScalarType::Float));

  int n_threads = opt_n_threads(n);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  fps_index_kernel<<<1, n_threads, 0, stream>>>(n, nsamples, points.data_ptr<float>(), tmp.data_ptr<float>(), output.data_ptr<long>());

  return output;
}
