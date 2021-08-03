#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "datatype/datatype.cpp"
#include "voxelize/voxelize.cpp"


void voxelize_fp_feat(/* cuda float N*C */ at::Tensor feats, // N * 3 -> M * 3 (N >= M)
              /* cuda float M*C */ at::Tensor output_feats,
              /* cuda Int M*(maxActive+1) */ at::Tensor output_map, Int N, Int C){
    voxelize_fp<float>(feats, output_feats, output_map, N, C);
}


void voxelize_bp_feat(/* cuda float M*C */ at::Tensor d_output_feats,
            /* cuda float N*C */ at::Tensor d_feats,
            /* cuda Int M*(maxActive+1) */ at::Tensor output_map,
            Int N, Int C){
    voxelize_bp<float>(d_output_feats, d_feats, output_map, N, C);
}
