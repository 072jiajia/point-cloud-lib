#include "__utils__.h"

__device__ void fatomicMax(float *addr, float value)
{
    if (*addr >= value) return;
    float old = *addr;
    int *int_addr = (int*)addr;
    int int_value = __float_as_int(value);
    while (old < value) {
        old = __int_as_float(atomicCAS(int_addr, __float_as_int(old), int_value));
    }

}


// inline __device__ void AtomicArgMax(float *max_pointer, int *max_i_pointer, float cur_value, int cur_index)
// {
//     if ((*max_pointer) >= cur_value) return;

//     int cur_max_i = *max_i_pointer;
//     while (fatomicMax(max_pointer, cur_value)) {
//         cur_max_i = atomicCAS(max_i_pointer, cur_max_i, cur_index);
//         if (cur_max_i == cur_index) {
//             break;
//         }
//     }
// }


__global__ void max_grouping_fp_kernel(float *feats, Int *rules, float *max_value, int *max_index, int N, int C){
    for(int data_index = threadIdx.x; data_index < N; data_index += blockDim.x){
        int group_offset = (*(rules + data_index)) * C;

        float *max_value_p = max_value + group_offset;
        float *value_p = feats + data_index * C;
        for(int plane = blockIdx.x; plane < C; plane += gridDim.x){
            fatomicMax(max_value_p + plane, *(value_p + plane));
        }
    }
    __syncthreads();
    for(int data_index = threadIdx.x; data_index < N; data_index += blockDim.x){
        int group_offset = (*(rules + data_index)) * C;

        float *value_p = feats + data_index * C;
        float *max_value_p = max_value + group_offset;
        int *max_index_p = max_index + group_offset;
        for(int plane = blockIdx.x; plane < C; plane += gridDim.x){
            if (value_p[plane] == max_value_p[plane]) {
                max_index_p[plane] = data_index;
            }
        }
    }
}


// __global__ void max_grouping_fp_kernel(float *feats, Int *rules, float *max_value, int *max_index, int N, int C){
//     for(int data_index = blockIdx.x; data_index < N; data_index += gridDim.x){
//         int group_offset = (*(rules + data_index)) * C;

//         float *inp = feats + data_index * C;
//         float *max_value_p = max_value + group_offset;
//         int *max_index_p = max_index + group_offset;
//         for(int plane = threadIdx.x; plane < C; plane += blockDim.x){
//             AtomicArgMax(max_value_p + plane, max_index_p + plane, *(inp + plane), data_index);
//         }
//     }
// }


__global__ void max_grouping_bp_kernel(float *inp_grad, int *max_index, float *out_grad, int M, int C) {
    for(int data_index = blockIdx.x; data_index < M; data_index += gridDim.x) {
        int offset_2d = data_index * C;
        for(int plane = threadIdx.x; plane < C; plane += blockDim.x){
            int offset_1d = offset_2d + plane;
            int grad_index = max_index[offset_1d];
            float grad_value = inp_grad[offset_1d];
            out_grad[grad_index * C + plane] = grad_value;
        }
    }
}


std::vector<at::Tensor> max_grouping_fp(at::Tensor feats, at::Tensor group_index, int M) {

    int N = feats.size(0);
    int C = feats.size(1);

    at::Tensor max_index = torch::full({M, C},    -1, at::device(feats.device()).dtype(at::ScalarType::Int));
    at::Tensor max_value = torch::full({M, C}, -1e20, at::device(feats.device()).dtype(at::ScalarType::Float));

    max_grouping_fp_kernel<<<std::min(C, 65536), std::min(N, MAX_THREADS)>>>
        (feats.data_ptr<float>(), group_index.data_ptr<Int>(),
         max_value.data_ptr<float>(), max_index.data_ptr<int>(),
         N, C);
    
    return {max_index, max_value};
}


at::Tensor max_grouping_bp(at::Tensor inp_grad, at::Tensor max_index, int N) {
    int M = inp_grad.size(0);
    int C = inp_grad.size(1);
    at::Tensor out_grad = torch::zeros({N, C}, at::device(inp_grad.device()).dtype(at::ScalarType::Float));

    max_grouping_bp_kernel<<<std::min(M, 65536), std::min(C, MAX_THREADS)>>>
        (inp_grad.data_ptr<float>(), max_index.data_ptr<int>(), out_grad.data_ptr<float>(), M, C);

    return out_grad;
}


