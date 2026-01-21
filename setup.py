import os
import glob
import platform
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import torch

def get_extensions():
    # Source file detection
    cpp_sources = glob.glob(os.path.join("cpp", "*.cpp"))
    cuda_sources = glob.glob(os.path.join("cpp", "*.cu"))

    sources = cpp_sources

    # Check for CUDA availability and add CUDA sources if present/available
    use_cuda = False
    if torch.cuda.is_available() and len(cuda_sources) > 0:
         sources += cuda_sources
         use_cuda = True
         print("Compiling with CUDA support")
    else:
         print("Compiling for CPU only")

    # Compilation flags
    extra_compile_args = {"cxx": []}

    if platform.system() == "Windows":
        extra_compile_args["cxx"].append("/O2")
    else:
        extra_compile_args["cxx"].append("-O3")
        extra_compile_args["cxx"].append("-std=c++17")

    if use_cuda:
        extra_compile_args["nvcc"] = ["-O3"]
        extra_compile_args["cxx"].append("-DWITH_CUDA")

    # Include and Library paths
    include_dirs = [os.path.abspath("include")]
    library_dirs = [os.path.abspath("lib")]

    # Select extension type
    Extension = CUDAExtension if use_cuda else CppExtension

    ext_modules = [
        Extension(
            name="custom_ops",
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

setup(
    name="custom_ops",
    version="0.1.0",
    author="User",
    description="A template for PyTorch C++ Extension",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
