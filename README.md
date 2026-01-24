# PyTorch NPU Driver Simulation (Three-Tier Architecture)

This project simulates a professional **Kernel/User-mode Driver Model** for a custom NPU, integrated into PyTorch.

## Architecture

1.  **Driver Daemon (`npu_driver`)**:
    -   Acts as the "Kernel Resource Manager".
    -   Owns the **2GB Shared Memory** block (the hardware BAR).
    -   Manages memory allocation via a First-Fit allocator.
    -   Listens on a **Named Pipe** (Windows) or **Unix Socket** (Linux) for IPC.
    -   Automatically reclaims memory if a client process crashes.

2.  **Hardware Daemon (`npu_daemon`)**:
    -   Simulates the NPU Compute Unit.
    -   Attaches to the Shared Memory created by the Driver to access data.

3.  **Client Extension (`x_npu` Backend)**:
    -   PyTorch C++ Extension loaded via JIT.
    -   Implements a custom memory client that forwards `allocate/free` requests to the Driver Daemon via IPC.
    -   Maps the Shared Memory into its own address space for direct access (simulating BAR mapping).

## Prerequisites

-   Python 3.9+
-   PyTorch 2.0+
-   C++ Compiler (MSVC or GCC)
-   Ninja Build System

## Usage

### 1. Start the System (Driver & Hardware)
Open a terminal and run the cluster manager:
```bash
python start_daemons.py
```
*Output: Launches `npu_driver` (Alloc Server) and `npu_daemon` (Hardware Sim).*

### 2. Run the Client Workload
Open a second terminal and run the Python script:
```bash
python run_workload.py
```
This script demonstrates:
-   Connecting to the Driver IPC.
-   Allocating memory using `XNPUTensor` (wrapper for manual driver API).
-   Direct Data Movement (H2D/D2H) via shared memory copy.
-   Robustness: If the script crashes, the Driver Daemon automatically cleans up allocations.

## API Note
Due to current PyTorch environment constraints with `PrivateUse1` backend registration, this project uses a Manual Driver API instead of `torch.device("x_npu")`.
-   **Allocation**: `ops.npu_malloc(device_id, size)` -> Returns virtual pointer.
-   **Data Transfer**: `ops.npu_h2d` / `ops.npu_d2h`.
-   **Wrapper**: `XNPUTensor` class encapsulates lifecycle management.

## Windows Specifics
-   Uses `CreateFileMapping` / `MapViewOfFile`.
-   Uses `CreateNamedPipe` / `CallNamedPipe`.
-   Shared Memory Name: `Local\X_NPU_SHM`
-   Pipe Name: `\\.\pipe\x_npu_driver`
