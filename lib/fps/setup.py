from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pclib',
    ext_modules=[
        CUDAExtension('pclib', [
            # 'src/fps_kernel.cu',
            # 'src/fps_knn_kernel.cu',
            # 'src/fps_knn.cpp',

            # 'src/count_elements.cu',
            # 'src/max_grouping.cu',
            # 'src/fps_group_kernel.cu',
            # 'src/fps_index_kernel.cu',
            # 'src/knn_index_kernel.cu',
            # 'src/grouping.cu',

            'src/__kernels__.cu',
            'src/__bindings__.cpp',
        ], extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)