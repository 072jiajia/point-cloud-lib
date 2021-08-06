#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>


void furthest_point_sampling_kernel_wrapper(int n, int m, const float *dataset, float *temp, long *idxs);


at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples) {

  int n = points.size(0);
  at::Tensor output = torch::zeros(nsamples, at::device(points.device()).dtype(at::ScalarType::Long));

  at::Tensor tmp = torch::full(n, 1e10, at::device(points.device()).dtype(at::ScalarType::Float));

  furthest_point_sampling_kernel_wrapper(n, nsamples, points.data_ptr<float>(), tmp.data_ptr<float>(), output.data_ptr<long>());

  return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("furthest_point_sampling", &furthest_point_sampling);
}
