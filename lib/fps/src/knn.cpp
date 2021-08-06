#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>



at::Tensor knn(at::Tensor points, at::Tensor samples, int K);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("knn", &knn);
}
