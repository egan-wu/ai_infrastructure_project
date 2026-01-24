import os
import sys
import subprocess
import platform
import time
import signal

def compile_daemon():
    print("[Manager] Compiling NPU Daemon...")
    if platform.system() == "Windows":
        cmd = "cl /EHsc /O2 cpp/npu_daemon.cpp /Fe:npu_daemon.exe"
    else:
        cmd = "g++ -O3 -pthread cpp/npu_daemon.cpp -o npu_daemon -lrt"

    ret = os.system(cmd)
    if ret != 0:
        print("Failed to compile daemon.")
        sys.exit(1)

def start_daemon(device_id):
    print(f"[Manager] Starting NPU Daemon {device_id}...")
    exe = "./npu_daemon" if platform.system() != "Windows" else "npu_daemon.exe"
    proc = subprocess.Popen([exe, str(device_id)])
    return proc

def main():
    compile_daemon()

    processes = []

    # Launch 2 Devices
    try:
        p0 = start_daemon(0)
        processes.append(p0)

        p1 = start_daemon(1)
        processes.append(p1)

        print("\n[Manager] NPU Cluster is running.")
        print("[Manager] Press Ctrl+C to stop all devices.\n")

        # Keep alive
        while True:
            time.sleep(1)
            # Check if processes are still alive
            if p0.poll() is not None or p1.poll() is not None:
                print("[Manager] A daemon exited unexpectedly!")
                break

    except KeyboardInterrupt:
        print("\n[Manager] Shutting down cluster...")
    finally:
        for p in processes:
            try:
                p.terminate()
            except:
                pass
            p.wait()
        print("[Manager] Stopped.")

if __name__ == "__main__":
    main()
