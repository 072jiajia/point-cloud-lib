#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "pointgroup_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("voxelize_fp", &voxelize_fp_feat, "voxelize_fp");
    m.def("voxelize_bp", &voxelize_bp_feat, "voxelize_bp");
}