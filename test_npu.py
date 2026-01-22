import torch
import time
import subprocess
import sys
import os
import platform
# We need to make sure we can import custom_ops.
# If built in-place, it should be in the current directory.
try:
    import custom_ops
except ImportError:
    # Try adding build directory if needed, but usually setup.py build_ext --inplace is used.
    pass

import npu_device

def main():
    print("=== Testing Cross-Process NPU Simulation (Workflow) ===")

    # 1. Start Daemon (Linux only for automated test)
    daemon_proc = None
    if platform.system() != "Windows":
        print("[Test] Compiling NPU Daemon (Linux)...")
        # Ensure we compile with -pthread and -lrt
        ret = os.system("g++ -O3 -pthread cpp/npu_daemon.cpp -o npu_daemon -lrt")
        if ret != 0:
            print("Failed to compile daemon")
            sys.exit(1)

        print("[Test] Starting NPU Daemon...")
        daemon_proc = subprocess.Popen(["./npu_daemon"])
        time.sleep(1) # Wait for startup
    else:
        print("[Test] Assuming npu_daemon.exe is running separately (Windows)...")

    try:
        # 3. Prepare Data
        x_cpu = torch.ones(5, 5, dtype=torch.float32)
        print(f"[Client] Input Tensor (CPU):\n{x_cpu}")

        # 4. H2D
        offset_in = 4096
        print(f"[Client] Moving to NPU (Offset {offset_in})...")
        x_npu = npu_device.to_npu(x_cpu, offset_in)
        print(f"[Client] NPU Tensor: {x_npu}")

        # 5. Compute
        offset_out = 8192
        scalar = 10.0
        print(f"[Client] Computing Add (Scalar {scalar}) -> Offset {offset_out}...")
        y_npu = npu_device.add(x_npu, scalar, offset_out)
        print(f"[Client] Result NPU Tensor: {y_npu}")

        # 6. D2H
        print("[Client] Moving result back to CPU...")
        y_cpu = y_npu.to_cpu()
        print(f"[Client] Result Tensor (CPU):\n{y_cpu}")

        # 7. Verify
        expected = x_cpu + scalar
        if torch.allclose(y_cpu, expected):
            print("✅ NPU Verification Passed!")
        else:
            print("❌ NPU Verification Failed!")
            print(f"Expected:\n{expected}")

        # 8. Exit NPU
        print("[Client] Sending Exit Signal...")
        custom_ops.npu_exit()

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if daemon_proc:
            print("[Test] Waiting for Daemon to exit...")
            # We expect it to exit gracefully
            try:
                daemon_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("[Test] Daemon did not exit in time, killing...")
                daemon_proc.terminate()
            print("[Test] Daemon exited.")

if __name__ == "__main__":
    main()
