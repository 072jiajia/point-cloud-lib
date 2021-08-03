import lib.point_grouping.point_grouping as point_grouping
import torch
import torch.nn as nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

a = torch.rand(5, 3, requires_grad=True).cuda()
b = torch.tensor([0, 3, 3, 3, 1]).cuda()
print('a =')
print(a)

c = point_grouping.voxelization(a, b)
print('c =')
print(c)

loss = c.sum()
a.retain_grad()
loss.backward()
print('a.grad =')
print(a.grad)






# assert 0
# # performance test
# N = 1000 * 1000 * 100
# a = torch.rand(N, 3).cuda()
# b = torch.randint(0, 1000000, (N, )).cuda()

# print("start")
# import time
# print("start", time.time())
# for i in range(100):
#     c = point_group.voxelization(a,b, True)
#     # print(point_group.voxelization(a,b).shape)
# print("end", time.time())
