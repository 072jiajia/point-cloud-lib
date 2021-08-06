import torch
import fps
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


times = 2
a = torch.rand(4096 * 128, 3).cuda()

# b = fps.furthest_point_sampling(a, 10)

# print((a[b] * 100).int())

import time


st = time.time()
for i in range(times):
    b = fps.furthest_point_sampling(a, 10000)


b = b.cpu()
ed = time.time()

print((ed - st) / times)

print(b[:10])

