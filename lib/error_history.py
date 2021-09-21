import numpy as np
import torch
import time
import glob
import pclib
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def deadlock_test():
    N = 1000000
    x = torch.ones(N, 6).cuda()
    x[N//2:] = 0
    y = torch.zeros(N).cuda().long()

    print('start')

    for i in range(100000):
        output = pclib.max_grouping_fp(x, y, 1)
        print(i, output)


deadlock_test()

def contiguous():
    N = 10
    x = torch.ones(N, 6).cuda()
    y = torch.zeros(N).cuda().long()

    x[:, 3:] = 0

    fps_group_index = pclib.max_grouping(x[:, :3].contiguous(), 3)
    print(fps_group_index)
    fps_group_index = pclib.max_grouping(x[:, :3], 3)
    print(fps_group_index)




