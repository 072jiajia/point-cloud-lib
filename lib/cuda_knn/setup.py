from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='knn',
    ext_modules=[
        CUDAExtension('knn', [
            'src/interpolate_kernel.cu',
            'src/interpolate.cpp'
        ], extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)