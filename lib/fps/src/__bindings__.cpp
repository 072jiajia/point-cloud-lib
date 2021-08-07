#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>


at::Tensor knn(at::Tensor points, at::Tensor samples, int K);
at::Tensor fps_index(at::Tensor points, const int nsamples);
at::Tensor fps_group(at::Tensor points, const int nsamples);

void grouping_fp(at::Tensor feats, at::Tensor output_feats, at::Tensor output_map, long N, long C);
void grouping_bp(at::Tensor d_output_feats, at::Tensor d_feats, at::Tensor output_map, long N, long C);

std::vector<at::Tensor> max_grouping_fp(at::Tensor feats, at::Tensor group_index, int M);

at::Tensor count_elements(at::Tensor group_index, int M);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn", &knn);
    m.def("fps_index", &fps_index);
    m.def("fps_group", &fps_group);
    m.def("grouping_fp", &grouping_fp);
    m.def("grouping_bp", &grouping_bp);
    m.def("max_grouping_fp", &max_grouping_fp);
    m.def("count_elements", &count_elements);
}
