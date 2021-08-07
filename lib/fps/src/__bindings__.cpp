#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>


at::Tensor knn(at::Tensor points, at::Tensor samples, int K);
at::Tensor fps_index(at::Tensor points, const int nsamples);
at::Tensor fps_group(at::Tensor points, const int nsamples);

at::Tensor avg_grouping_fp(at::Tensor inp_feats, at::Tensor rules, long N, long C, long M);
at::Tensor avg_grouping_bp(at::Tensor inp_grad, at::Tensor rules, long N, long C);

std::vector<at::Tensor> max_grouping_fp(at::Tensor feats, at::Tensor group_index, int M);
at::Tensor max_grouping_bp(at::Tensor inp_grad, at::Tensor max_index, int N);

at::Tensor count_elements(at::Tensor group_index, int M);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn", &knn);
    m.def("fps_index", &fps_index);
    m.def("fps_group", &fps_group);
    m.def("avg_grouping_fp", &avg_grouping_fp);
    m.def("avg_grouping_bp", &avg_grouping_bp);
    m.def("max_grouping_fp", &max_grouping_fp);
    m.def("max_grouping_bp", &max_grouping_bp);
    m.def("count_elements", &count_elements);
}
