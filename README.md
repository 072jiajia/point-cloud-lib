# point-cloud-lib


```
conda create -n pclib python=3.7
source activate pclib
conda install -c bioconda google-sparsehash
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
CUDA_HOME=/usr/local/cuda-10.2
cd lib
python setup.py install
```

```
lib
 +- setup.py
 +- src
 | +- __utils__.h # define macros
 | +- __kernels__.cu # include kernel files
 | +- __bindings__.cpp # for creating Python bindings
 | +- {op_name}_kernel.cu
```
