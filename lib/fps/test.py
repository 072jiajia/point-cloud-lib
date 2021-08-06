import torch
import fps
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


N = 1024 * 128

times = 2
a = torch.rand(N, 3).cuda()
b = torch.rand(N, 3).cuda()
# b = a

import time

st = time.time()

for i in range(times):
    c = fps.knn(a, b)
    c = c.cpu()

for d in c[:100]:
    print(d)



ed = time.time()

print((ed - st) / times)

# print(b[:10])


for i in range(100):
    print(i)
    time.sleep(1)
