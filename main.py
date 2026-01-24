import torch
import sys
import os
import time
import subprocess
import platform

def compile_daemon():
    print("[Test] Compiling NPU Daemon...")
    if platform.system() == "Windows":
        # Assumes MSVC
        cmd = "cl /EHsc /O2 cpp/npu_daemon.cpp /Fe:npu_daemon.exe"
    else:
        cmd = "g++ -O3 -pthread cpp/npu_daemon.cpp -o npu_daemon -lrt"

    ret = os.system(cmd)
    if ret != 0:
        print("Failed to compile daemon. Ensure C++ compiler is in path.")
        sys.exit(1)

def start_daemon(device_id):
    print(f"[Test] Starting NPU Daemon {device_id}...")
    exe = "./npu_daemon" if platform.system() != "Windows" else "npu_daemon.exe"
    # Pass device_id as argument
    proc = subprocess.Popen([exe, str(device_id)])
    return proc

def main():
    compile_daemon()

    # Start Daemons for Device 0 and Device 1
    d0 = start_daemon(0)
    d1 = start_daemon(1)

    # Give them time to initialize SHM
    time.sleep(1)

    try:
        # Load Extension
        sys.path.append(os.getcwd())
        from jit_loader import load_extension
        ops = load_extension()
        ops.init()
        print("Extension Loaded.")

        # Initialize Devices (Connects to SHM)
        ops.init_device(0)
        ops.init_device(1)

        # 1. Allocate Memory
        # We handle memory manually using the new driver API
        size_elements = 1024
        size_bytes = size_elements * 4 # float

        print("\n[Step 1] Allocating Memory...")
        addr0_in = ops.npu_malloc(0, size_bytes)
        addr0_out = ops.npu_malloc(0, size_bytes)
        addr1_in = ops.npu_malloc(1, size_bytes) # We will copy result here

        print(f"  Dev0 In: {hex(addr0_in)}")
        print(f"  Dev0 Out: {hex(addr0_out)}")
        print(f"  Dev1 In: {hex(addr1_in)}")

        # 2. H2D
        print("\n[Step 2] Host to Device (Dev 0)...")
        t_in = torch.ones(size_elements, dtype=torch.float32) * 10.0
        ops.npu_h2d(0, t_in, addr0_in)

        # 3. Compute on Dev 0
        print("\n[Step 3] Compute on Dev 0 (Add)...")
        # OP_ADD: src1, src2, dst, size, scalar
        # We add the input to itself: 10 + 10 = 20
        ops.npu_compute(0, "OP_ADD", addr0_in, addr0_in, addr0_out, size_elements, 0.0)

        # 4. D2D (Dev 0 -> Dev 1)
        print("\n[Step 4] Device to Device (Dev 0 -> Dev 1)...")
        # Copy result from Dev 0 Out to Dev 1 In
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

        # Cleanup
        print("\n[Step 7] Cleanup...")
        ops.npu_free(0, addr0_in)
        ops.npu_free(0, addr0_out)
        ops.npu_free(1, addr1_in)

        # Send Exit Signal to Daemons
        ops.npu_compute(0, "OP_EXIT", 0, 0, 0, 0, 0.0)
        ops.npu_compute(1, "OP_EXIT", 0, 0, 0, 0, 0.0)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Stopping Daemons...")
        # Give them a moment to process EXIT if sent
        time.sleep(0.5)
        d0.terminate()
        d1.terminate()
        try:
            d0.wait(timeout=1)
            d1.wait(timeout=1)
        except:
            d0.kill()
            d1.kill()

if __name__ == "__main__":
    main()
