import torch
import time
import subprocess
import sys
import os
import platform

def main():
    print("=== Testing Cross-Process NPU Simulation ===")

    # 1. Start Daemon (Linux only for automated test)
    # On Windows, the user must run npu_daemon.exe manually as per instructions.
    daemon_proc = None
    if platform.system() != "Windows":
        print("[Test] Compiling NPU Daemon (Linux)...")
        ret = os.system("g++ -O3 -pthread cpp/npu_daemon.cpp -o npu_daemon -lrt")
        if ret != 0:
            print("Failed to compile daemon")
            sys.exit(1)

        print("[Test] Starting NPU Daemon...")
        # Start in background
        daemon_proc = subprocess.Popen(["./npu_daemon"])
        time.sleep(1) # Wait for startup
    else:
        print("[Test] Assuming npu_daemon.exe is running separately (Windows)...")

    try:
        # 2. Load Extension
        # We need to rebuild/install first if code changed
        # Assuming setup.py install or build_ext --inplace was run
        import custom_ops

        # 3. Prepare Data
        # SHM offset (must be after the control struct, e.g., 4096 bytes)
        shm_offset = 4096
        scalar = 10.0

        x = torch.ones(5, 5, dtype=torch.float32)
        print(f"[Client] Input Tensor:\n{x}")

        # 4. Call NPU
        print(f"[Client] Calling NPU with scalar={scalar}...")
        start_t = time.time()
        y = custom_ops.call_npu(x, scalar, shm_offset)
        end_t = time.time()

        print(f"[Client] Call took {end_t - start_t:.4f}s")
        print(f"[Client] Result Tensor:\n{y}")

        # 5. Verify
        expected = x + scalar
        if torch.allclose(y, expected):
            print("✅ NPU Verification Passed!")
        else:
            print("❌ NPU Verification Failed!")
            print(f"Expected:\n{expected}")

    except ImportError:
        print("Could not import custom_ops. Make sure to build the extension first.")
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        # Cleanup
        if daemon_proc:
            print("[Test] Killing NPU Daemon...")
            daemon_proc.terminate()
            daemon_proc.wait()

if __name__ == "__main__":
    main()
