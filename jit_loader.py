import torch
from torch.utils.cpp_extension import load
import os
import glob
import platform

def load_extension():
    # Source detection
    cpp_sources = glob.glob(os.path.join("cpp", "*.cpp"))
    cuda_sources = glob.glob(os.path.join("cpp", "*.cu"))

    sources = cpp_sources
    with_cuda = False
    if torch.cuda.is_available() and len(cuda_sources) > 0:
        sources += cuda_sources
        with_cuda = True

    # Compilation flags
    extra_cflags = []
    if platform.system() == "Windows":
        extra_cflags.append("/O2")
    else:
        extra_cflags.append("-O3")
        extra_cflags.append("-std=c++17")

    extra_cuda_cflags = ["-O3"] if with_cuda else []

    if with_cuda:
        extra_cflags.append("-DWITH_CUDA")

    # Paths
    cwd = os.path.dirname(os.path.abspath(__file__))
    include_dirs = [os.path.join(cwd, "include")]

    # Load extension
    module = load(
        name="custom_ops_jit",
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=include_dirs,
        # library_dirs=[os.path.join(cwd, "lib")], # load() doesn't always support library_dirs gracefully in all versions, but extra_ldflags can be used if needed.
        verbose=True,
        with_cuda=with_cuda
    )
    return module

if __name__ == "__main__":
    # Test loading
    ops = load_extension()
    print("JIT Loading successful!")
    print(dir(ops))
