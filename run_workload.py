import torch
import sys
import os
import time
import numpy as np

# Wrapper class for X_NPU Tensors
class XNPUTensor:
    def __init__(self, shape, dtype=torch.float32, ops_module=None):
        self.ops = ops_module
        self.shape = shape
        self.dtype = dtype
        self.numel = 1
        for s in shape: self.numel *= s
        self.element_size = 4 # Float32
        self.nbytes = self.numel * self.element_size

        # Allocate on Driver (Device 0 by default)
        self.ptr = self.ops.npu_malloc(0, self.nbytes)
        # print(f"[XNPUTensor] Allocated {self.nbytes} bytes at virtual address {hex(self.ptr)}")

    def __del__(self):
        if hasattr(self, 'ptr') and self.ptr:
            # print(f"[XNPUTensor] Freeing {hex(self.ptr)}")
            self.ops.npu_free(0, self.ptr)

    def copy_from(self, cpu_tensor):
        if cpu_tensor.numel() != self.numel:
            raise ValueError("Size mismatch in copy_from")
        t = cpu_tensor.contiguous().to(dtype=torch.float32)
        self.ops.npu_h2d(0, t, self.ptr)

    def copy_to_cpu(self):
        cpu_t = torch.empty(self.shape, dtype=self.dtype)
        self.ops.npu_d2h(0, cpu_t, self.ptr)
        return cpu_t

def verify_op(ops, op_name, op_code, op_func, size=1024):
    print(f"\n--- Verifying {op_name} ---")

    # 1. Generate Data
    t1 = torch.randn(size)
    t2 = torch.randn(size)
    golden = op_func(t1, t2)

    # 2. Allocate NPU
    x1 = XNPUTensor(t1.shape, ops_module=ops)
    x2 = XNPUTensor(t2.shape, ops_module=ops)
    x_out = XNPUTensor(t1.shape, ops_module=ops)

    # 3. H2D
    x1.copy_from(t1)
    x2.copy_from(t2)

    # 4. Compute
    ops.npu_compute(0, op_code, x1.ptr, x2.ptr, x_out.ptr, t1.numel(), 0.0)

    # 5. D2H
    res = x_out.copy_to_cpu()

    # 6. Check
    if torch.allclose(res, golden, atol=1e-4):
        print(f"✅ {op_name} PASSED")
        return True
    else:
        print(f"❌ {op_name} FAILED")
        print(f"   Expected: {golden[:5]}")
        print(f"   Got:      {res[:5]}")
        return False

def main():
    print("=== NPU Client Workload (Verification Mode) ===")

    try:
        sys.path.append(os.getcwd())
        from jit_loader import load_extension
        ops = load_extension()
        ops.init()
        print("[Client] Extension Loaded.")
    except Exception as e:
        print(f"Failed to load extension: {e}")
        sys.exit(1)

    try:
        # 1. H2D / D2H Loopback Test
        print("\n[Test 1] H2D -> D2H Loopback...")
        t_in = torch.randn(1024)
        x_in = XNPUTensor(t_in.shape, ops_module=ops)
        x_in.copy_from(t_in)
        t_out = x_in.copy_to_cpu()

        if torch.allclose(t_in, t_out):
            print("✅ H2D/D2H Loopback PASSED")
        else:
            print("❌ H2D/D2H Loopback FAILED")
            sys.exit(1)

        # 2. Verify Ops
        verify_op(ops, "ADD", "OP_ADD", torch.add)
        verify_op(ops, "SUB", "OP_SUB", torch.sub)
        verify_op(ops, "MUL", "OP_MUL", torch.mul)

        print("\n=== All Tests Completed ===")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
