#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "datatype.h"
#include "voxelize/voxelize.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("voxelize_fp", &voxelize_fp<float>, "voxelize_fp");
    m.def("voxelize_bp", &voxelize_bp<float>, "voxelize_bp");
}
