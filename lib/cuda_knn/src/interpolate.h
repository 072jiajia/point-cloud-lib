#pragma once

#include <torch/extension.h>
#include <vector>

at::Tensor three_nn(at::Tensor unknowns, at::Tensor knows, int K);
