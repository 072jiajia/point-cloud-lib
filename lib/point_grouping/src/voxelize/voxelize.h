#ifndef VOXELIZE_H
#define VOXELIZE_H
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#include "../datatype.h"

/* ================================== voxelize ================================== */
template <typename T>
void voxelize_fp(/* cuda float N*C */ at::Tensor feats, // N * 3 -> M * 3 (N >= M)
              /* cuda float M*C */ at::Tensor output_feats,
              /* cuda Int M*(maxActive+1) */ at::Tensor output_map, Int N, Int C);

template <typename T>
void voxelize_fp_cuda(T *feats, T *output_feats, Int *rules, Int N, Int C);


//
template <typename T>
void voxelize_bp(/* cuda float M*C */ at::Tensor d_output_feats,
                /* cuda float N*C */ at::Tensor d_feats,
                /* cuda Int M*(maxActive+1) */ at::Tensor output_map,
                Int N, Int C);

template <typename T>
void voxelize_bp_cuda(T *d_output_feats, T *d_feats, Int *rules, Int N, Int C);



#endif //VOXELIZE_H
