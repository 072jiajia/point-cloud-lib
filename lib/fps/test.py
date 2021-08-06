import torch
import fps
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


N = 1024 * 128
M = 1024


# N = 1025
# N = 5


# N = 8
# M = 1024
# M = 3

times = 5
a = torch.rand(N, 3).cuda()
b = torch.rand(M, 3).cuda()
# a = torch.tensor([[0,0,0],
#                   [0,0,1],
#                   [0,1,0],
#                   [0,1,1],
#                   [1,0,0],
#                   [1,0,1],
#                   [1,1,0],
#                   [1,1,1],
#                   ]).cuda().float()

# b = torch.rand(M, 3).cuda()
# b = a

import time

st = time.time()


c = fps.knn(a, b, 8)
d = fps.knn(a, b, 1)

print((c[:, 0] != d[:, 0]).sum())
# print(c.shape)
# print(d.shape)
# for i in range(times):
#     c = fps.knn(a, b, 8)
#     c = c.cpu()

# print(c)

# for d in c:
#     if torch.unique(d).shape[0] < 3:
#         print(d)



ed = time.time()

print((ed - st) / times)

# print(b[:10])


# for i in range(100):
#     print(i)
#     time.sleep(1)
