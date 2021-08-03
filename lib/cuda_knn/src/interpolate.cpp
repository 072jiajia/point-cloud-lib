#include "interpolate.h"
#include "utils.h"

void three_nn_kernel_wrapper(int n, int m, const float *unknown, const float *known, int *idx);

at::Tensor three_nn(at::Tensor unknowns, at::Tensor knows, int K) {
  CHECK_CONTIGUOUS(unknowns);
  CHECK_CONTIGUOUS(knows);
  CHECK_IS_FLOAT(unknowns);
  CHECK_IS_FLOAT(knows);


  at::Tensor idx = torch::zeros({unknowns.size(0), K}, at::device(unknowns.device()).dtype(at::ScalarType::Int));

  three_nn_kernel_wrapper(unknowns.size(0), knows.size(0),
                          unknowns.data_ptr<float>(), knows.data_ptr<float>(),
                          idx.data_ptr<int>());

  return idx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("three_nn", &three_nn);
}
