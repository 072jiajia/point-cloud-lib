#include "utils.h"

__device__ bool fatomicMax(float *addr, float value)
{
    if (*addr >= value) return (*addr) == value;
    float old = *addr;
    int *int_addr = (int*)addr;
    int int_value = __float_as_int(value);
    while (old < value) {
        old = __int_as_float(atomicCAS(int_addr, __float_as_int(old), int_value));
    }

    return (old == value);
}


inline __device__ void AtomicArgMax(float *max_pointer, int *max_i_pointer, float cur_value, int cur_index)
{
    if ((*max_pointer) >= cur_value) return;

    int cur_max_i = *max_i_pointer;
    while (fatomicMax(max_pointer, cur_value)) {
        cur_max_i = atomicCAS(max_i_pointer, cur_max_i, cur_index);
        if (cur_max_i == cur_index) {
            break;
        }
    }
}


__global__ void max_grouping_fp_kernel(float *feats, Int *rules, float *max_value, int *max_index, int N, int C){
    for(int data_index = blockIdx.x; data_index < N; data_index += gridDim.x){
        int group_offset = (*(rules + data_index)) * C;

        float *inp = feats + data_index * C;
        float *max_value_p = max_value + group_offset;
        int *max_index_p = max_index + group_offset;
        for(int plane = threadIdx.x; plane < C; plane += blockDim.x){
            AtomicArgMax(max_value_p + plane, max_index_p + plane, *(inp + plane), data_index);
        }
    }
}



std::vector<at::Tensor> max_grouping_fp(at::Tensor feats, at::Tensor group_index, int M){

    int N = feats.size(0);
    int C = feats.size(1);


    at::Tensor max_index = torch::full({M, C},    -1, at::device(feats.device()).dtype(at::ScalarType::Int));
    at::Tensor max_value = torch::full({M, C}, -1e20, at::device(feats.device()).dtype(at::ScalarType::Float));

    max_grouping_fp_kernel<<<std::min(N, 65536), std::min(C, MAX_THREADS)>>>
        (feats.data_ptr<float>(), group_index.data_ptr<Int>(),
         max_value.data_ptr<float>(), max_index.data_ptr<int>(),
         N, C);
    
    return {max_index, max_value};
}


