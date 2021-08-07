#include "__utils__.h"


__global__ void avg_grouping_fp_kernel(float *inp_feats, float *out_feats, Int *rules, Int N, Int C){
    for(Int data_index = blockIdx.x; data_index < N; data_index += gridDim.x){
        Int group_index = *(rules + data_index);

        float *inp = inp_feats + data_index * C;
        float *out = out_feats + group_index * C;
        for(Int plane = threadIdx.x; plane < C; plane += blockDim.x){
            atomicAdd(&out[plane], inp[plane]);
        }
    }
}



__global__ void avg_grouping_bp_kernel(float *inp_grad, float *out_grad, Int *rules, Int N, Int C){
    for(Int data_index = blockIdx.x; data_index < N; data_index += gridDim.x){
        Int group_index = *(rules + data_index);

        float *inp = inp_grad + group_index * C;
        float *out = out_grad + data_index * C;
        for(Int plane = threadIdx.x; plane < C; plane += blockDim.x){
            atomicAdd(&out[plane], inp[plane]);
        }
    }
}



at::Tensor avg_grouping_fp(at::Tensor inp_feats,
                           at::Tensor rules,
                           Int N, Int C, Int M){

    at::Tensor out_feats = torch::zeros({M, C}, at::device(inp_feats.device()).dtype(at::ScalarType::Float));

    avg_grouping_fp_kernel<<<std::min(N, (Int)32768), std::min(C, (Int)MAX_THREADS)>>>
        (inp_feats.data_ptr<float>(), out_feats.data_ptr<float>(), rules.data_ptr<Int>(), N, C);

    return out_feats;
}


at::Tensor avg_grouping_bp(at::Tensor inp_grad,
                           at::Tensor rules,
                           Int N, Int C){

    at::Tensor out_grad = torch::zeros({N, C}, at::device(inp_grad.device()).dtype(at::ScalarType::Float));

    avg_grouping_bp_kernel<<<std::min(N, (Int)65536), std::min(C, (Int)MAX_THREADS)>>>
        (inp_grad.data_ptr<float>(), out_grad.data_ptr<float>(), rules.data_ptr<Int>(), N, C);

    return out_grad;
}
