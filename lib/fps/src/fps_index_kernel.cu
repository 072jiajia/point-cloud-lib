#include "__utils__.h"



__global__ void fps_index_kernel(int n, int m,
                                 float *__restrict__ dataset,
                                 float *__restrict__ temp,
                                 long *__restrict__ idxs) {
  __shared__ float dists[MAX_THREADS];
  __shared__ int dists_i[MAX_THREADS];

  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  int old = 0;

  for (int j = 1; j < m; j++) {
    int besti = 0;
    float best = -1;

    float *point1_base = dataset + old * 3;
    float x1 = *point1_base;
    float y1 = point1_base[1];
    float z1 = point1_base[2];
    for (int k = tid; k < n; k += block_size) {

      float *point2_base = dataset + k * 3;
      float x_diff = *point2_base - x1;
      float y_diff = point2_base[1] - y1;
      float z_diff = point2_base[2] - z1;

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
