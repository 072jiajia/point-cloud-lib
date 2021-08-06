#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>


void furthest_point_sampling_kernel_wrapper(int n, int m, const float *points, const float *samples, long *nearest_index);


at::Tensor knn(at::Tensor points, at::Tensor samples) {

  int n = points.size(0);
  int m = samples.size(0);

  at::Tensor nearest_index = torch::zeros(n, at::device(points.device()).dtype(at::ScalarType::Long));

  furthest_point_sampling_kernel_wrapper(n, m, points.data_ptr<float>(), samples.data_ptr<float>(), nearest_index.data_ptr<long>());

  return nearest_index;
  // return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("knn", &knn);
}
