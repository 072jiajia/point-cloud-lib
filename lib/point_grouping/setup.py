from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='point_grouping',
    ext_modules=[
        CUDAExtension('point_grouping', [
            'src/pointgroup_ops_api.cpp',
            'src/pointgroup_ops.cpp',
            'src/cuda.cu'
        ], extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)