from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pclib',
    ext_modules=[
        CUDAExtension('pclib', [
            'src/__kernels__.cu',
            'src/__bindings__.cpp',
        ], extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)