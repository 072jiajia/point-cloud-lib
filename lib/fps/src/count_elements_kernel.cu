#include "__utils__.h"


__global__ void count_elements_kernel(int N, Int *group_index, int *counts){
    for(int data_index = blockIdx.x; data_index < N; data_index += gridDim.x){
        int group_offset = *(group_index + data_index);

        atomicAdd(counts + group_offset, 1);
    }
}


at::Tensor count_elements(at::Tensor group_index, int M) {
    int N = group_index.size(0);
    at::Tensor counts = torch::zeros({M, 1}, at::device(group_index.device()).dtype(at::ScalarType::Int));

    count_elements_kernel<<<std::min(N, 65536), 1>>>(N, group_index.data_ptr<Int>(), counts.data_ptr<int>());

    return counts;
}




