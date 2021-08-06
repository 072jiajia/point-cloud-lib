#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>



at::Tensor knn(at::Tensor points, at::Tensor samples, int K);
at::Tensor fps_index(at::Tensor points, const int nsamples);
at::Tensor fps_group(at::Tensor points, const int nsamples);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("knn", &knn);
  m.def("fps_index", &fps_index);
  m.def("fps_group", &fps_group);
}
