#include "__utils__.h"


__global__ void grouping_fp_kernel(float *feats, float *output_feats, Int *rules, Int N, Int C){
    for(Int data_index = blockIdx.x; data_index < N; data_index += gridDim.x){
        Int group_index = *(rules + data_index);

        float *inp = feats + data_index * C;
        float *out = output_feats + group_index * C;
        for(Int plane = threadIdx.x; plane < C; plane += blockDim.x){
            atomicAdd(&out[plane], inp[plane]);
        }
    }
}



__global__ void grouping_bp_kernel(float *d_output_feats, float *d_feats, Int *rules, Int N, Int C){
    for(Int data_index = blockIdx.x; data_index < N; data_index += gridDim.x){
        Int group_index = *(rules + data_index);

        float *inp = d_feats + data_index * C;
        float *out = d_output_feats + group_index * C;
        for(Int plane = threadIdx.x; plane < C; plane += blockDim.x){
            atomicAdd(&inp[plane], out[plane]);
        }
    }
}



void grouping_fp(/* cuda float N*C */ at::Tensor feats, // N * 3 -> M * 3 (N >= M)
              /* cuda float M*C */ at::Tensor output_feats,
              /* cuda Int M*(maxActive+1) */ at::Tensor output_map, Int N, Int C){

    auto iF = feats.data_ptr<float>();
    auto oF = output_feats.data_ptr<float>();

    Int *rules = output_map.data_ptr<Int>();

    grouping_fp_kernel<<<std::min(N, (Int)32768), std::min(C, (Int)MAX_THREADS)>>>(iF, oF, rules, N, C);
}


void grouping_bp(/* cuda float M*C */ at::Tensor d_output_feats,
            /* cuda float N*C */ at::Tensor d_feats,
            /* cuda Int M*(maxActive+1) */ at::Tensor output_map,
            Int N, Int C){
    auto d_oF = d_output_feats.data_ptr<float>();
    auto d_iF = d_feats.data_ptr<float>();

    Int *rules = output_map.data_ptr<Int>();

    grouping_bp_kernel<<<std::min(N, (Int)65536), std::min(C, (Int)MAX_THREADS)>>>(d_oF, d_iF, rules, N, C);
}
