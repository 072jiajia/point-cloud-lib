# point-cloud-lib


```
conda create -n pclib python=3.7
source activate pclib
conda install -c bioconda google-sparsehash
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
CUDA_HOME=/usr/local/cuda-10.2
cd lib/point_grouping
python setup.py install
```
