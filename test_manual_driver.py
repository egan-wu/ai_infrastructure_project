import torch
import sys
import os
import time
import subprocess
import platform
import types

def main():
    print("=== Testing Manual Driver Simulation ===")

    # 1. Start Daemon
    daemon_proc = None
    if platform.system() != "Windows":
        print("[Test] Compiling NPU Daemon...")
        ret = os.system("g++ -O3 -pthread cpp/npu_daemon.cpp -o npu_daemon -lrt")
        if ret != 0:
            print("Failed to compile daemon")
            sys.exit(1)

        print("[Test] Starting NPU Daemon...")
        daemon_proc = subprocess.Popen(["./npu_daemon"])
        time.sleep(1)
    else:
        print("[Test] Assuming npu_daemon.exe is running...")

    try:
        # 2. Load Extension
        sys.path.append(os.getcwd())
        from jit_loader import load_extension
        custom_ops = load_extension()
        custom_ops.init()

        torch.x_tpu = types.ModuleType("torch.x_tpu")
        sys.modules["torch.x_tpu"] = torch.x_tpu

        print("Extension Loaded.")

        # 3. Setup Tensors
        # Determine device name
        # Force fallback to privateuseone:0 to avoid linkage string check issues
        print("⚠️ Forcing use of 'privateuseone:0' to bypass alias linkage check")
        device_name = "privateuseone:0"

        size = 4
        a_cpu = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
        b_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

        print("\n--- Allocation & H2D (PyTorch Managed) ---")
        a = a_cpu.to(device_name)
        b = b_cpu.to(device_name)
        out = torch.empty_like(a).to(device_name)

        # print(a) # Might crash if backend not fully linked for printing
        print(f"a_ptr:   {hex(a.data_ptr())}")
        print(f"b_ptr:   {hex(b.data_ptr())}")
        print(f"out_ptr: {hex(out.data_ptr())}")

        # 4. Manual ADD
        print("\n--- Manual OP_ADD ---")
        # npu_ioctl(op_code, src1, src2, dst, size)
        custom_ops.npu_ioctl("OP_ADD", a.data_ptr(), b.data_ptr(), out.data_ptr(), size)

        # Sync
        custom_ops.npu_wait()

        # Verify
        res_cpu = out.to("cpu")
        print(f"Result (ADD): {res_cpu}")
        assert torch.allclose(res_cpu, a_cpu + b_cpu), "Manual ADD failed"
        print("✅ Manual ADD Passed")

        # 5. Manual SUB
        print("\n--- Manual OP_SUB ---")
        custom_ops.npu_ioctl("OP_SUB", a.data_ptr(), b.data_ptr(), out.data_ptr(), size)
        custom_ops.npu_wait()

        res_cpu = out.to("cpu")
        print(f"Result (SUB): {res_cpu}")
        assert torch.allclose(res_cpu, a_cpu - b_cpu), "Manual SUB failed"
        print("✅ Manual SUB Passed")

        # 6. Manual EXIT
        print("\n--- Cleanup ---")
        custom_ops.npu_ioctl("OP_EXIT", 0, 0, 0, 0)

    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if daemon_proc:
            print("[Test] Stopping Daemon...")
            try:
                daemon_proc.terminate()
                daemon_proc.wait(timeout=2)
            except:
                daemon_proc.kill()

if __name__ == "__main__":
    main()
