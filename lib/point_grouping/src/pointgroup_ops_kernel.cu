#include <ATen/ATen.h>
#include "datatype.h"
#include "voxelize/voxelize.cu"

template void voxelize_fp_cuda<float>(float *feats, float *output_feats, Int *rules, Int N, Int C);
template void voxelize_bp_cuda<float>(float *d_output_feats, float *d_feats, Int *rules, Int N, Int C);
