from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fps',
    ext_modules=[
        CUDAExtension('fps', [
            'src/fps_kernel.cu',
            'src/fps.cpp',
        ], extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)