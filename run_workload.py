import torch
import sys
import os
import time

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
        print(f"[XNPUTensor] Allocated {self.nbytes} bytes at virtual address {hex(self.ptr)}")

    def __del__(self):
        if hasattr(self, 'ptr') and self.ptr:
            print(f"[XNPUTensor] Freeing {hex(self.ptr)}")
            self.ops.npu_free(0, self.ptr)

    def copy_from(self, cpu_tensor):
        # H2D
        if cpu_tensor.numel() != self.numel:
            raise ValueError("Size mismatch in copy_from")
        self.ops.npu_h2d(0, cpu_tensor, self.ptr) # Ops expects virtual pointer now?
        # Check ops.cpp: npu_h2d expects offset.
        # Wait, our Allocator returns virtual address (shm_base + offset).
        # But npu_malloc in ops.cpp calls client->request_allocate which returns OFFSET.
        # Let's check ops.cpp again.
        pass

    def copy_to_cpu(self):
        cpu_t = torch.empty(self.shape, dtype=self.dtype)
        # D2H
        self.ops.npu_d2h(0, cpu_t, self.ptr)
        return cpu_t

def main():
    print("=== NPU Client Workload (Manual Wrapper) ===")

    try:
        sys.path.append(os.getcwd())
        from jit_loader import load_extension
        ops = load_extension()
        ops.init() # Connects client
        print("[Client] Extension Loaded.")
    except Exception as e:
        print(f"Failed to load extension: {e}")
        sys.exit(1)

    try:
        # 1. Create Wrapper
        print("\n[Test 1] Allocate XNPUTensor...")
        xnpu_t = XNPUTensor((1024,), ops_module=ops)

        # 2. H2D
        print("\n[Test 2] H2D Copy...")
        cpu_data = torch.randn(1024, dtype=torch.float32)
        # We need to ensure npu_h2d accepts the handle returned by npu_malloc
        # In ops.cpp: npu_malloc returns offset? No, check below.
        ops.npu_h2d(0, cpu_data, xnpu_t.ptr)
        print("✅ H2D Complete.")

        # 3. D2H
        print("\n[Test 3] D2H Copy...")
        res = torch.zeros(1024, dtype=torch.float32)
        ops.npu_d2h(0, res, xnpu_t.ptr)

        if torch.allclose(cpu_data, res):
            print("✅ Data Verification Passed!")
        else:
            print("❌ Data Verification Failed!")
            print(f"Original: {cpu_data[0]}")
            print(f"Returned: {res[0]}")

        # 4. Lifecycle
        print("\n[Test 4] Explicit Deletion...")
        del xnpu_t
        print("✅ Tensor deleted.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
