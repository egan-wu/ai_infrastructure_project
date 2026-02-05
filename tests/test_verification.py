import torch
import sys
import os
import shutil
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

        self.ptr = self.ops.npu_malloc(0, self.nbytes)

    def __del__(self):
        if hasattr(self, 'ptr') and self.ptr:
            self.ops.npu_free(0, self.ptr)

    def copy_from(self, cpu_tensor):
        if cpu_tensor.numel() != self.numel:
            raise ValueError("Size mismatch in copy_from")
        # Ensure contiguous and float32
        t = cpu_tensor.contiguous().to(dtype=torch.float32)
        self.ops.npu_h2d(0, t, self.ptr)

    def copy_to_cpu(self):
        cpu_t = torch.empty(self.shape, dtype=self.dtype)
        self.ops.npu_d2h(0, cpu_t, self.ptr)
        return cpu_t

class VerificationRunner:
    def __init__(self):
        self.golden_dir = "./cpu_golden"
        if os.path.exists(self.golden_dir):
            shutil.rmtree(self.golden_dir)
        os.makedirs(self.golden_dir)

        # Load Extension
        sys.path.append(os.getcwd())
        from jit_loader import load_extension
        self.ops = load_extension()
        self.ops.init()

    def generate_golden(self, name, inputs, op_func):
        path = os.path.join(self.golden_dir, name)
        os.makedirs(path, exist_ok=True)

        # Save Inputs
        for i, t in enumerate(inputs):
            torch.save(t, os.path.join(path, f"input_{i}.pt"))

        # Compute Golden
        res = op_func(*inputs)
        torch.save(res, os.path.join(path, "golden.pt"))
        return res

    def run_npu_test(self, name, inputs, op_code):
        print(f"\n--- Running Test: {name} ---")

        # 1. Alloc Inputs
        npu_inputs = []
        for t in inputs:
            xt = XNPUTensor(t.shape, ops_module=self.ops)
            xt.copy_from(t)
            npu_inputs.append(xt)

        # 2. Alloc Output
        # Assume output shape same as input 0 for now (elementwise)
        out_shape = inputs[0].shape
        if op_code == "OP_MATMUL":
            # (M, K) x (K, N) -> (M, N)
            M = inputs[0].size(0)
            N = inputs[1].size(1)
            out_shape = (M, N)

        npu_out = XNPUTensor(out_shape, ops_module=self.ops)

        # 3. Execute
        # npu_compute(dev, op, src1, src2, dst, size, scalar)
        src1 = npu_inputs[0].ptr
        src2 = npu_inputs[1].ptr if len(npu_inputs) > 1 else 0

        size_param = 0
        if op_code == "OP_MATMUL":
            size_param = inputs[0].numel()
        else:
            size_param = inputs[0].numel()

        self.ops.npu_compute(0, op_code, src1, src2, npu_out.ptr, size_param, 0.0)

        # 4. Readback
        res = npu_out.copy_to_cpu()

        # 5. Load Golden
        golden = torch.load(os.path.join(self.golden_dir, name, "golden.pt"))

        # 6. Compare
        if torch.allclose(res, golden, atol=1e-4):
            print(f"✅ PASSED: {name}")
        else:
            print(f"❌ FAILED: {name}")
            print(f"   Expected: {golden.flatten()[:5]}")
            print(f"   Got:      {res.flatten()[:5]}")

def main():
    runner = VerificationRunner()

    # Test 1: ADD
    t1 = torch.randn(1024)
    t2 = torch.randn(1024)
    runner.generate_golden("test_add", [t1, t2], torch.add)
    runner.run_npu_test("test_add", [t1, t2], "OP_ADD")

    # Test 2: SUB
    runner.generate_golden("test_sub", [t1, t2], torch.sub)
    runner.run_npu_test("test_sub", [t1, t2], "OP_SUB")

    # Test 3: MUL
    runner.generate_golden("test_mul", [t1, t2], torch.mul)
    runner.run_npu_test("test_mul", [t1, t2], "OP_MUL")

if __name__ == "__main__":
    main()
