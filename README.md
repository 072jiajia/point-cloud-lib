# Point-Cloud-Lib

This is a repo aims at modulize / optimize / extent the library for point cloud analysis.

## Installation
```
conda create -n pclib python=3.7
source activate pclib
conda install -c bioconda google-sparsehash
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
CUDA_HOME=/usr/local/cuda-10.2
cd lib
python setup.py install
```

## How to Use

### To Import PCLIB
```
import torch
import pclib
```

### KNN
```
N = NUM_POINTS
M = NUM_QUERIES
K = NUM_KNN

points = torch.rand(N, 3).cuda()
queries = torch.rand(M, 3).cuda()

knn_indices = pclib.knn(points, queries, K)
# knn_indices.shape: (M, K)
```

### Farthest Point Sampling
```
N = NUM_RAW_POINTS
M = NUM_SAMPLED_POINTS

raw_points = torch.rand(N, 3).cuda()

sampled_indices = pclib.fps_index(raw_points, M)
# sampled_indices.shape = (M, )

sampled_points = raw_points[sampled_indices]
# sampled_indices.shape = (M, 3)
```

### Average / Max Pooling
```
import grouping

points = torch.ones(5, 3, requires_grad=True).cuda()
groups = torch.tensor([0, 3, 3, 3, 1]).cuda()
# point 0 belongs to group 0
# point 1 belongs to group 3
# point 2 belongs to group 3
# point 3 belongs to group 3
# point 4 belongs to group 1

result = grouping.avg_grouping(points, groups) # avg_grouping / max_grouping
print('result =')
print(result)
# result = 
# [[  1.,  1.,  1. ]
#  [  1.,  1.,  1. ]
#  [ nan, nan, nan ]
#  [  1.,  1.,  1. ]]
# Since group 2 doesn't contains any points

loss = result.sum()
result.retain_grad()
loss.backward()
print('points.grad =')
print(points.grad)
# points.grad = 
# [[     1.,     1.,     1. ]
#  [ 0.3333, 0.3333, 0.3333 ]
#  [ 0.3333, 0.3333, 0.3333 ]
#  [ 0.3333, 0.3333, 0.3333 ]
#  [     1.,     1.,     1. ]]

```




