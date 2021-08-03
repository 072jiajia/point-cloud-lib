#include "voxelize.h"

template <typename T>
__global__ void voxelize_fp_cuda_(T *feats, T *output_feats, Int *rules, Int N, Int C){
    for(Int data_index = blockIdx.x; data_index < N; data_index += gridDim.x){
        Int group_index = *(rules + data_index);

        T *inp = feats + data_index * C;
        T *out = output_feats + group_index * C;
        for(Int plane = threadIdx.x; plane < C; plane += blockDim.x){
            atomicAdd(&out[plane], inp[plane]);
        }
    }
}

template <typename T>
void voxelize_fp_cuda(T *feats, T *output_feats, Int *rules, Int N, Int C){
    voxelize_fp_cuda_<T><<<std::min(N, (Int)32768), std::min(C, (Int)32)>>>(feats, output_feats, rules, N, C);
}


template <typename T>
__global__ void voxelize_bp_cuda_(T *d_output_feats, T *d_feats, Int *rules, Int N, Int C){
    for(Int data_index = blockIdx.x; data_index < N; data_index += gridDim.x){
        Int group_index = *(rules + data_index);

        T *inp = d_feats + data_index * C;
        T *out = d_output_feats + group_index * C;
        for(Int plane = threadIdx.x; plane < C; plane += blockDim.x){
            atomicAdd(&inp[plane], out[plane]);
        }
    }
}

template <typename T>
void voxelize_bp_cuda(T *d_output_feats, T *d_feats, Int *rules, Int N, Int C){
    voxelize_bp_cuda_<T><<<std::min(N, (Int)32768), std::min(C, (Int)32)>>>(d_output_feats, d_feats, rules, N, C);
}

