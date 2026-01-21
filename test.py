import torch
import time

def test_implementation(ops_module, device="cpu"):
    print(f"Testing on {device}...")

    # Setup inputs
    h, w = 100, 100
    a = torch.randn(h, w, dtype=torch.float32, device=device)
    b = torch.randn(h, w, dtype=torch.float32, device=device)
    alpha = 0.5
    beta = 2.0

    # Run custom op
    # Note: TensorAccessor in our C++ code enforces 2D float32
    try:
        c_custom = ops_module.weighted_sum(a, b, alpha, beta)
    except Exception as e:
        print(f"Execution failed: {e}")
        return False

    # Run PyTorch reference
    c_ref = alpha * a + beta * b

    print("Expected Answer:")
    print(c_ref)
    print("-----")
    print("Custom Op Answer:")
    print(c_custom)

    # Verify
    if torch.allclose(c_custom, c_ref, atol=1e-5):
        print(f"✅ Correctness check passed on {device}!")
        return True
    else:
        print(f"❌ Correctness check failed on {device}!")
        diff = (c_custom - c_ref).abs().max()
        print(f"Max difference: {diff}")
        return False

def main():
    print("=== Testing JIT Compilation ===")
    try:
        import jit_loader
        jit_module = jit_loader.load_extension()
        test_implementation(jit_module, "cpu")
        if torch.cuda.is_available():
            test_implementation(jit_module, "cuda")
    except Exception as e:
        print(f"JIT loading failed: {e}")

    print("\n=== Testing AOT Compilation ===")
    try:
        import custom_ops
        test_implementation(custom_ops, "cpu")
        if torch.cuda.is_available():
            test_implementation(custom_ops, "cuda")
    except ImportError:
        print("AOT module 'custom_ops' not found. Run 'python setup.py install' or 'python setup.py build_ext --inplace' first.")
    except Exception as e:
        print(f"AOT testing failed: {e}")

if __name__ == "__main__":
    main()
