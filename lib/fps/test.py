import torch
import fps
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


times = 2
a = torch.rand(4096 * 128, 3).cuda()

# b = fps.furthest_point_sampling(a, 10)

# print((a[b] * 100).int())

import time


st = time.time()

for i in range(times):
    b = fps.fps_knn(a, 1024 * 2)
    b = b.cpu()

# a = a.cpu().numpy()

# plt.subplot(1, 3, 1)
# plt.scatter(a[:, 0], a[:, 1], c=b)

# plt.subplot(1, 3, 2)
# plt.scatter(a[:, 0], a[:, 2], c=b)

# plt.subplot(1, 3, 3)
# plt.scatter(a[:, 1], a[:, 2], c=b)

# plt.savefig('plot.png')



ed = time.time()

print((ed - st) / times)

print(b[:10])

