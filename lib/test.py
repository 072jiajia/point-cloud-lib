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



def test_max_grouping_v3(N=100000000, times=100):
    a = torch.randn(N, 3).cuda()
    b = (torch.rand(N).cuda() * 1).long()
    # wait()
    st = time.time()
    for _ in range(times):
        c1, c2 = pclib.max_grouping_fp(a, b, b.max()+1)

        c1 = c1.cpu()
        c2 = c2.cpu()
    ed = time.time()
    print((ed -st) / times)
    wait()



def test_count():
    N = 1000000
    times = 200
    a = (torch.rand(N) * 1000).cuda().long()
    st = time.time()
    for _ in range(times):
        M = a.max() + 1
        M = M.cpu()

    M += 1
    ed = time.time()
    print((ed -st) / times)


def test_avg_grouping():
    import grouping
    a = torch.rand(5, 3, requires_grad=True).cuda()
    b = torch.tensor([0, 3, 3, 3, 1]).cuda()

    print("a =")
    print(a)

    c = grouping.avg_grouping(a, b)
    print('c =')
    print(c)

    loss = c.sum()
    a.retain_grad()
    loss.backward()
    print('a.grad =')
    print(a.grad)


def test_max_grouping():
    import grouping
    a = torch.rand(5, 3, requires_grad=True).cuda()
    b = torch.tensor([0, 3, 3, 3, 7]).cuda()

    print("a =")
    print(a)

    c = grouping.max_grouping(a, b)
    print('c =')
    print(c)

    loss = c.sum()
    a.retain_grad()
    loss.backward()
    print('a.grad =')
    print(a.grad)


def test_fps_group_v2(N=100000, M=512, times=10):
    a = torch.randn(N, 3).cuda()

    import tqdm
    st = time.time()
    for _ in tqdm.tqdm(range(times)):
        sampled = pclib.fps_group(a,M)
        sampled = sampled.cpu()
        # print(sampled.shape)

    ed = time.time()
    print((ed -st) / times)


def test_fps_index_v2(N=1000000, M=128, times=100):
    a = torch.randn(N, 3).cuda()

    import tqdm
    st = time.time()
    # for _ in tqdm.tqdm(range(times)):
    for _ in range(times):
        sampled = pclib.fps_index(a,M)
        # sampled = pclib.fps_index_v2(a,M)
        sampled = sampled.cpu()
        # print(sampled)

    ed = time.time()
    print((ed -st) / times)

def test_fps_index_same(N=1000000, M=512, times=10):
    a = torch.randn(N, 3).cuda()

    import tqdm
    st = time.time()
    for _ in tqdm.tqdm(range(times)):
        sampled_v1 = pclib.fps_index(a,M)
        sampled_v2 = pclib.fps_index_v2(a,M)

        print((sampled_v1 != sampled_v2).sum())        
        # print(sampled.shape)

    ed = time.time()
    print((ed -st) / times)


test_max_grouping()
# test_fps_index_v2()
# test_fps_index_same()

# test_fps_group_v2()
# test_max_grouping()
# test_max_grouping_v3(N=10000000)
