#ifndef POINTGROUP_H
#define POINTGROUP_H
#include "datatype/datatype.h"


void voxelize_fp_feat(/* cuda float N*C */ at::Tensor feats, // N * 3 -> M * 3 (N >= M)
              /* cuda float M*C */ at::Tensor output_feats,
              /* cuda Int M*(maxActive+1) */ at::Tensor output_map,
              Int N, Int C);

void voxelize_bp_feat(/* cuda float M*C */ at::Tensor d_output_feats,
            /* cuda float N*C */ at::Tensor d_feats,
            /* cuda Int M*(maxActive+1) */ at::Tensor output_map,
            Int N, Int C);

#endif // POINTGROUP_H