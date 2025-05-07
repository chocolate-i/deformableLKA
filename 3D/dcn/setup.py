#!/usr/bin/env python

import os
import glob
import torch
import torch.utils.cpp_extension

# 导入必要的扩展类
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import BuildExtension

# 修改为系统中实际的CUDA路径
cuda_path = '/usr/lib/nvidia-cuda-toolkit'
os.environ['CUDA_HOME'] = cuda_path
torch.utils.cpp_extension.CUDA_HOME = cuda_path

# 确保CUDA_HOME可用于后续代码
CUDA_HOME = torch.utils.cpp_extension.CUDA_HOME

from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    # 使用系统中存在的CUDA
    if torch.cuda.is_available():
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        raise NotImplementedError('Cuda is not available')

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "D3D",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

setup(
    name="D3D",
    version="1.0",
    author="yingxinyi",
    url="https://github.com/XinyiYing/D3Dnet",
    description="Deformable 3D Convolutional Network",
    packages=find_packages(exclude=("configs", "tests",)),
    # install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
