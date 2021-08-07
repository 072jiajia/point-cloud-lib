import torch
import pclib
import os
import matplotlib.pyplot as plt
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def wait(TIME=100):
    for i in range(TIME):
        print(f"{i}s / {TIME}s")
        time.sleep(1)


def test_knn():

    N = 1024 * 128
    M = 1024

    times = 5
    a = torch.rand(N, 3).cuda()
    b = torch.rand(M, 3).cuda()

    st = time.time()


    c = pclib.knn(a, b, 8)
    d = pclib.knn(a, b, 1)

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

    print(pclib.fps_index(a,8))

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

    print(pclib.fps_group(a,1))
    print(pclib.fps_group(a,2))
    print(pclib.fps_group(a,3))

    print("================")
    index = pclib.fps_index(a,8)
    group = pclib.fps_group(a,8)

    print(index)
    print(group)
    print("=>", group[index])

def test_grouping():
    import grouping
    a = torch.rand(5, 3, requires_grad=True).cuda()
    b = torch.tensor([0, 3, 3, 3, 1]).cuda()

    print("a =")
    print(a)

    c = grouping.grouping(a, b)
    print('c =')
    print(c)

    loss = c.sum()
    a.retain_grad()
    loss.backward()
    print('a.grad =')
    print(a.grad)


def test_max_grouping():
    a = torch.randn(5, 3).cuda()
    a = (10 * a).int().float()
    b = torch.tensor([0, 3, 3, 3, 1]).cuda()

    print("a =")
    print(a)

    c = pclib.max_grouping_fp(a, b, b.max()+1)
    print('c =')
    print(c[0])
    print(c[1])


def test_max_grouping_v2():
    N = 10000000
    times = 2
    a = torch.randn(N, 3).cuda()
    b = torch.zeros(N).cuda().long()
    st = time.time()
    for _ in range(times):

        # print("a =")
        # print(a)

        c = pclib.max_grouping_fp(a, b, b.max()+1)
        # print('c =')
        # print(c[0])
        # print(c[1])
        c[0] = c[0].cpu()
        c[1] = c[1].cpu()
    ed = time.time()
    print((ed -st) / times)


def test_grouping_v2():
    N = 1000000
    times = 2
    a = torch.randn(N, 3).cuda()
    b = torch.zeros(N).cuda().long()
    st = time.time()
    for _ in range(times):

        # print("a =")
        # print(a)

        M = b.max() + 1
        output_feats = torch.zeros(M, 3).cuda()

        pclib.grouping_fp(a, output_feats, b, N, 3)

        # pclib.grouping_fp(a, b, b.max()+1)
        # print('c =')
        # print(c[0])
        # print(c[1])
        output_feats = output_feats.cpu()
    ed = time.time()
    print((ed -st) / times)



def test_max_grouping_v3():
    N = 100000000
    times = 2
    a = torch.randn(N, 3).cuda()
    b = (torch.rand(N).cuda() * 1000).long()
    # wait()
    st = time.time()
    for _ in range(times):

        # print("a =")
        # print(a)

        c = pclib.max_grouping_fp(a, b, b.max()+1)
        # print('c =')
        # print(c[0])
        # print(c[1])
        # c[0] = c[0].cpu()
        # c[1] = c[1].cpu()
    ed = time.time()
    print((ed -st) / times)
    wait()


test_max_grouping_v3()
# test_grouping_v2()
