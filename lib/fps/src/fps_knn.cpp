#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>


void furthest_point_sampling_kernel_wrapper(int n, int m, const float *dataset, float *nearest, long *nearest_index, long *idxs);


at::Tensor fps_knn(at::Tensor points, const int nsamples) {

  int n = points.size(0);
  at::Tensor output = torch::zeros(nsamples, at::device(points.device()).dtype(at::ScalarType::Long));

  at::Tensor nearest = torch::full(n, 1e10, at::device(points.device()).dtype(at::ScalarType::Float));
  at::Tensor nearest_index = torch::zeros(n, at::device(points.device()).dtype(at::ScalarType::Long));

  furthest_point_sampling_kernel_wrapper(n, nsamples, points.data_ptr<float>(), nearest.data_ptr<float>(), nearest_index.data_ptr<long>(), output.data_ptr<long>());

  return nearest_index;
  return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fps_knn", &fps_knn);
}
