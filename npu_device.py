import torch
import sys
import os

# Robust import logic for custom_ops
custom_ops = None

# 1. Try importing the installed/local static extension
try:
    import custom_ops as _static_ops
    # 2. Check for staleness (ensure it has the new API)
    if hasattr(_static_ops, 'h2d_copy'):
        custom_ops = _static_ops
    else:
        print("[NPU Device] Warning: Found 'custom_ops' but it is missing 'h2d_copy'. It might be stale.")
        print("[NPU Device] Will attempt to JIT compile the latest extension.")
except ImportError:
    pass

# 3. Fallback to JIT compilation
if custom_ops is None:
    print("[NPU Device] 'custom_ops' not found or stale. JIT compiling extension...")
    try:
        # Ensure we can import jit_loader from the current directory
        # (Assuming the script is run from the repo root)
        sys.path.append(os.getcwd())
        from jit_loader import load_extension
        custom_ops = load_extension()
    except Exception as e:
        print(f"[NPU Device] Error: Failed to load extension via JIT. {e}")
        print("[NPU Device] Please ensure you have run 'python setup.py build_ext --inplace' or have a working compiler.")
        raise ImportError("Could not load custom_ops")

class NPUTensor:
    def __init__(self, shape, offset, dtype=torch.float32):
        self.shape = shape
        self.offset = offset
        self.dtype = dtype
        self.numel = 1
        for s in shape:
            self.numel *= s

    def to_cpu(self):
        """Moves data from NPU to CPU."""
        # Create empty tensor on CPU
        dst = torch.empty(self.shape, dtype=self.dtype)
        # Perform D2H copy
        custom_ops.d2h_copy(dst, self.offset)
        return dst

    def __repr__(self):
        return f"NPUTensor(offset={self.offset}, shape={self.shape}, dtype={self.dtype})"

def to_npu(tensor, offset):
    """Moves data from CPU to NPU at specific offset."""
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")

    # Perform H2D copy
    custom_ops.h2d_copy(tensor, offset)
    return NPUTensor(tensor.shape, offset, tensor.dtype)

def add(npu_tensor, scalar, out_offset):
    """Performs addition on NPU.
       Returns a new NPUTensor pointing to out_offset.
    """
    if not isinstance(npu_tensor, NPUTensor):
        raise ValueError("Input must be an NPUTensor")

    # Trigger NPU Compute
    custom_ops.npu_add(npu_tensor.offset, out_offset, npu_tensor.numel, scalar)

    return NPUTensor(npu_tensor.shape, out_offset, npu_tensor.dtype)
