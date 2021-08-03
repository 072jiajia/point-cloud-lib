#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CHECK_CONTIGUOUS(x)                                          \
  do {                                                               \
    AT_ASSERT(x.is_contiguous(), #x " must be a contiguous tensor"); \
  } while (0)


#define CHECK_IS_FLOAT(x)                               \
  do {                                                  \
    AT_ASSERT(x.scalar_type() == at::ScalarType::Float, \
              #x " must be a float tensor");            \
  } while (0)
