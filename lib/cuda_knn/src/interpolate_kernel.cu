#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: unknown(b, n, 3) known(b, m, 3)
// output: dist2(b, n, 3), idx(b, n, 3)
__global__ void three_nn_kernel(int n, int m,
                                const float *__restrict__ unknown,
                                const float *__restrict__ known,
                                int *__restrict__ idx) {


  // int stride = blockDim.x;
  for (int j = blockIdx.x; j < n; j += gridDim.x) {
    float ux = unknown[j * 3 + 0];
    float uy = unknown[j * 3 + 1];
    float uz = unknown[j * 3 + 2];

    double best1 = 1e40, best2 = 1e40, best3 = 1e40;
    int besti1 = 0, besti2 = 0, besti3 = 0;
    for (int k = 0; k < m; ++k) {
      float x = known[k * 3 + 0];
      float y = known[k * 3 + 1];
      float z = known[k * 3 + 2];
      float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
      if (d < best1) {
        best3 = best2;
        besti3 = besti2;
        best2 = best1;
        besti2 = besti1;
        best1 = d;
        besti1 = k;
      } else if (d < best2) {
        best3 = best2;
        besti3 = besti2;
        best2 = d;
        besti2 = k;
      } else if (d < best3) {
        best3 = d;
        besti3 = k;
      }
    }

    idx[j * 3 + 0] = besti1;
    idx[j * 3 + 1] = besti2;
    idx[j * 3 + 2] = besti3;
  }
}

void three_nn_kernel_wrapper(int n, int m, const float *unknown,
                             const float *known, int *idx) {
  // cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  three_nn_kernel<<<std::min(n, (int)32768), 3>>>(n, m, unknown, known, idx);
}
