#include "voxelize.h"


/* ================================== voxelize ================================== */
template <typename T>
void voxelize_fp(/* cuda float N*C */ at::Tensor feats, // N * 3 -> M * 3 (N >= M)
              /* cuda float M*C */ at::Tensor output_feats,
              /* cuda Int M*(maxActive+1) */ at::Tensor output_map, Int N, Int C){

    auto iF = feats.data<T>();
    auto oF = output_feats.data<T>();

    Int *rules = output_map.data<Int>();

    voxelize_fp_cuda<T>(iF, oF, rules, N, C);
}

template <typename T>
void voxelize_bp(/* cuda float M*C */ at::Tensor d_output_feats,
            /* cuda float N*C */ at::Tensor d_feats,
            /* cuda Int M*(maxActive+1) */ at::Tensor output_map,
            Int N, Int C){
    auto d_oF = d_output_feats.data<T>();
    auto d_iF = d_feats.data<T>();

    Int *rules = output_map.data<Int>();

    voxelize_bp_cuda<T>(d_oF, d_iF, rules, N, C);
}
