import torch
import custom_ops

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
