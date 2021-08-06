from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fps',
    ext_modules=[
        CUDAExtension('fps', [
            # 'src/fps_kernel.cu',
            # 'src/fps.cpp',
            # 'src/fps_knn_kernel.cu',
            # 'src/fps_knn.cpp',
            'src/knn_kernel.cu',
            'src/knn.cpp',
        ], extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)