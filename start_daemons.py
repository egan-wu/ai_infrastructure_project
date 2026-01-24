import os
import sys
import subprocess
import platform
import time

def compile_binaries():
    print("[Manager] Compiling Drivers & Daemons...")

    # Flags
    if platform.system() == "Windows":
        # MSVC
        cmd_driver = "cl /EHsc /O2 cpp/npu_driver.cpp /Fe:npu_driver.exe"
        cmd_daemon = "cl /EHsc /O2 cpp/npu_daemon.cpp /Fe:npu_daemon.exe"
    else:
        # GCC
        cmd_driver = "g++ -O3 -pthread cpp/npu_driver.cpp -o npu_driver -lrt"
        cmd_daemon = "g++ -O3 -pthread cpp/npu_daemon.cpp -o npu_daemon -lrt"

    if os.system(cmd_driver) != 0:
        print("Failed to compile npu_driver")
        sys.exit(1)

    if os.system(cmd_daemon) != 0:
        print("Failed to compile npu_daemon")
        sys.exit(1)

def main():
    compile_binaries()

    print("\n[Manager] Launching NPU Driver Daemon (Resource Manager)...")
    exe_driver = "./npu_driver" if platform.system() != "Windows" else "npu_driver.exe"
    p_driver = subprocess.Popen([exe_driver])

    # Give it time to create SHM
    time.sleep(1)

    print("[Manager] Launching NPU Hardware Daemon (Compute Unit)...")
    exe_daemon = "./npu_daemon" if platform.system() != "Windows" else "npu_daemon.exe"
    p_daemon = subprocess.Popen([exe_daemon])

    print("\n[Manager] System Online. Press Ctrl+C to shutdown.\n")

    try:
        p_driver.wait()
        p_daemon.wait()
    except KeyboardInterrupt:
        print("\n[Manager] Shutting down...")
    finally:
        try: p_daemon.terminate()
        except: pass
        try: p_driver.terminate()
        except: pass
        print("[Manager] Stopped.")

if __name__ == "__main__":
    main()
