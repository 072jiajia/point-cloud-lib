from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fps',
    ext_modules=[
        CUDAExtension('fps', [
            # 'src/fps_kernel.cu',
            # 'src/fps_knn_kernel.cu',
            # 'src/fps_knn.cpp',
            'src/fps_group_kernel.cu',
            'src/fps_index_kernel.cu',
            'src/knn_index_kernel.cu',
            'src/bindings.cpp',
        ], extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)