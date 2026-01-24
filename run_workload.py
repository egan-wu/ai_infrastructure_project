import torch
import sys
import os
import time

def main():
    print("=== NPU Workload Runner ===")

    # Load Extension
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
        # Initialize Devices (Connect to existing Daemons)
        print("[Client] Connecting to NPU 0...")
        ops.init_device(0)
        print("[Client] Connecting to NPU 1...")
        ops.init_device(1)

        # 1. Allocate Memory
        size_elements = 1024
        size_bytes = size_elements * 4 # float

        print("\n[Step 1] Allocating Memory...")
        addr0_in = ops.npu_malloc(0, size_bytes)
        addr0_out = ops.npu_malloc(0, size_bytes)
        addr1_in = ops.npu_malloc(1, size_bytes)

        print(f"  Dev0 In: {hex(addr0_in)}")
        print(f"  Dev0 Out: {hex(addr0_out)}")
        print(f"  Dev1 In: {hex(addr1_in)}")

        # 2. H2D
        print("\n[Step 2] Host to Device (Dev 0)...")
        t_in = torch.ones(size_elements, dtype=torch.float32) * 10.0
        ops.npu_h2d(0, t_in, addr0_in)

        # 3. Compute on Dev 0
        print("\n[Step 3] Compute on Dev 0 (Add)...")
        # 10 + 10 = 20
        ops.npu_compute(0, "OP_ADD", addr0_in, addr0_in, addr0_out, size_elements, 0.0)

        # 4. D2D (Dev 0 -> Dev 1)
        print("\n[Step 4] Device to Device (Dev 0 -> Dev 1)...")
        ops.npu_d2d(0, addr0_out, 1, addr1_in, size_bytes)

        # 5. D2H (Dev 1 -> Host)
        print("\n[Step 5] Device to Host (Dev 1)...")
        t_out = torch.zeros(size_elements, dtype=torch.float32)
        ops.npu_d2h(1, t_out, addr1_in)

        # 6. Verify
        print(f"\nResult Sample: {t_out[0:5]}")
        expected = 20.0
        if torch.all(t_out == expected):
            print("✅ SUCCESS: Workflow Complete.")
        else:
            print("❌ FAILURE: Incorrect results.")
            print(f"Expected: {expected}, Got: {t_out[0]}")

        # Cleanup Memory (Daemons keep running)
        print("\n[Step 6] Freeing Memory...")
        ops.npu_free(0, addr0_in)
        ops.npu_free(0, addr0_out)
        ops.npu_free(1, addr1_in)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
