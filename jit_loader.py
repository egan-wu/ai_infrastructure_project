import torch
from torch.utils.cpp_extension import load
import os
import glob
import platform
import shutil

def load_extension():
    # Use absolute paths
    cwd = os.path.dirname(os.path.abspath(__file__))
    cpp_dir = os.path.join(cwd, "cpp")

    # Source detection
    cpp_sources = glob.glob(os.path.join(cpp_dir, "*.cpp"))
    cuda_sources = glob.glob(os.path.join(cpp_dir, "*.cu"))

    # Exclude the standalone daemon from the python extension
    daemon_src = os.path.join(cpp_dir, "npu_daemon.cpp")
    if daemon_src in cpp_sources:
        cpp_sources.remove(daemon_src)

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

    # Include paths
    include_dirs = [os.path.join(cwd, "include")]

    # Force rebuild by using a new name or cleaning
    # We will use a dynamic name based on content hash or just random/timestamp if needed
    # But for now, let's just use "custom_ops_v2"

    module = load(
        name="custom_ops_v2",
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=include_dirs,
        verbose=True,
        with_cuda=with_cuda
    )
    return module

if __name__ == "__main__":
    # Test loading
    ops = load_extension()
    print("JIT Loading successful!")
    print(dir(ops))
