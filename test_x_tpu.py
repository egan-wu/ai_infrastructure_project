import torch
import sys
import os
import time
import subprocess
import platform
import types

def main():
    print("=== Testing X_TPU PyTorch Integration ===")

    # 1. Start Daemon FIRST
    daemon_proc = None
    if platform.system() != "Windows":
        print("[Test] Compiling NPU Daemon (Linux)...")
        ret = os.system("g++ -O3 -pthread cpp/npu_daemon.cpp -o npu_daemon -lrt")
        if ret != 0:
            print("Failed to compile daemon")
            sys.exit(1)

        print("[Test] Starting NPU Daemon...")
        daemon_proc = subprocess.Popen(["./npu_daemon"])
        time.sleep(1)
    else:
        print("[Test] Assuming npu_daemon.exe is running separately (Windows)...")

    try:
        # 2. Load Extension
        custom_ops = None
        try:
            import custom_ops as _static_ops
            if hasattr(_static_ops, 'init'):
                custom_ops = _static_ops
            else:
                print("Static custom_ops stale.")
        except ImportError:
            pass

        if custom_ops is None:
            sys.path.append(os.getcwd())
            from jit_loader import load_extension
            custom_ops = load_extension()

        # 3. Initialize Backend (Must happen AFTER daemon is running for SHM connection)
        custom_ops.init()

        # Hack: Register dummy module
        torch.x_tpu = types.ModuleType("torch.x_tpu")
        sys.modules["torch.x_tpu"] = torch.x_tpu

        # 4. Test Device Registration
        print("\n--- Test 1: Device Registration ---")
        try:
            d = torch.device("x_tpu")
            print(f"✅ Created device: {d}")
        except RuntimeError as e:
            print(f"❌ Failed to create device 'x_tpu': {e}")
            d = torch.device("privateuseone:0")
            print(f"⚠️  Fallback to: {d}")

        # 5. Test .to("x_tpu") (Allocation + H2D)
        print("\n--- Test 2: Allocation & H2D ---")
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

        try:
            x_dev = x.to(d)
        except RuntimeError as e:
            print(f"Warning: .to(device) failed: {e}")
            x_dev = x.to("privateuseone:0")

        print(f"✅ x_dev: {x_dev}")
        print(f"   Device: {x_dev.device}")

        # 6. Test Native Ops
        print("\n--- Test 3: Native Operations ---")

        # Add
        y_dev = torch.add(x_dev, x_dev)
        print("Executing Add...")
        y_cpu = y_dev.to("cpu")
        print(f"   Result Add (CPU): \n{y_cpu}")
        assert torch.allclose(y_cpu, x * 2), "Add failed"

        # Mul
        z_dev = torch.mul(x_dev, x_dev)
        print("Executing Mul...")
        z_cpu = z_dev.to("cpu")
        print(f"   Result Mul (CPU): \n{z_cpu}")
        assert torch.allclose(z_cpu, x * x), "Mul failed"

        # MM
        print("Executing MM...")
        mm_dev = torch.mm(x_dev, x_dev)
        mm_cpu = mm_dev.to("cpu")
        print(f"   Result MM (CPU): \n{mm_cpu}")
        expected_mm = torch.mm(x, x)
        assert torch.allclose(mm_cpu, expected_mm), "MM failed"
        print("✅ Native Ops Passed!")

        # 7. Test Fallback
        print("\n--- Test 4: Fallback Mechanism (Sub) ---")
        print("Calling torch.sub (should warn and fallback)...")
        sub_dev = torch.sub(x_dev, x_dev)
        print(f"   Result Sub (on device?): {sub_dev.device}")
        sub_cpu = sub_dev.to("cpu")
        print(f"   Result Sub (CPU): \n{sub_cpu}")
        assert torch.allclose(sub_cpu, x - x), "Fallback Sub failed"
        print(f"✅ Fallback Passed! Device: {sub_dev.device}")

        # 8. Exit
        print("\n--- Cleanup ---")
        custom_ops.npu_exit()

    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if daemon_proc:
            print("[Test] Stopping Daemon...")
            daemon_proc.terminate()
            try:
                daemon_proc.wait(timeout=2)
            except:
                daemon_proc.kill()

if __name__ == "__main__":
    main()
