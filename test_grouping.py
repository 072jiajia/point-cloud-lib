import lib.point_grouping.point_grouping as point_grouping
import torch
import torch.nn as nn
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def test_simple():
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


def test_speed(N, M, times, average):
    a = torch.rand(N, 300).cuda()
    b = torch.randint(0, M, (N, )).cuda()

    print('start', end='\r')
    st = time.time()
    for i in range(times):
        c = point_grouping.voxelization(a, b, average)
    ed = time.time()
    print(N, M, times, average)
    print("avg", (ed - st) / times)


test_speed(1000 * 1000, 1000, times=1000, average=False)
# test_speed(1000 * 1000 * 100, 1000 * 1000, times=10, average=True)
# test_speed(1000 * 1000 * 100, 1000 * 1000, times=10, average=False)
