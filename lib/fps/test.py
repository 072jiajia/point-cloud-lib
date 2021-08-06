import torch
import fps
import os
import matplotlib.pyplot as plt
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def test_knn():

    N = 1024 * 128
    M = 1024

    times = 5
    a = torch.rand(N, 3).cuda()
    b = torch.rand(M, 3).cuda()

    st = time.time()


    c = fps.knn(a, b, 8)
    d = fps.knn(a, b, 1)

    print((c[:, 0] != d[:, 0]).sum())


    ed = time.time()

    print((ed - st) / times)

def test_fps():
    a = torch.tensor([[0,0,0],
                      [0,0,1],
                      [0,1,0],
                      [0,1,1],
                      [1,0,0],
                      [1,0,1],
                      [1,1,0],
                      [1,1,1],
                      ]).cuda().float()

    print(fps.fps_index(a,8))

def test_fps_group():
    a = torch.tensor([[0,0,0],
                      [0,0,1],
                      [0,1,0],
                      [0,1,1],
                      [1,0,0],
                      [1,0,1],
                      [1,1,0],
                      [1,1,1],
                      ]).cuda().float()

    print(fps.fps_group(a,1))
    print(fps.fps_group(a,2))
    print(fps.fps_group(a,3))

    print("================")
    index = fps.fps_index(a,8)
    group = fps.fps_group(a,8)

    print(index)
    print(group)
    print("=>", group[index])

test_fps_group()
