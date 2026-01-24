# PyTorch C++ Extension Template & Multi-NPU Simulation

This project provides a robust template for PyTorch C++ Extensions and features a **Cross-Process Multi-NPU Simulation Framework**. It demonstrates how to integrate custom hardware accelerators into PyTorch using Shared Memory (IPC) for host-device communication.

## Project Structure

```
.
├── cpp/
│   ├── bindings.cpp          # Pybind11 module definitions
│   ├── ops.cpp               # Host Driver & PyTorch Logic
│   ├── ops.h                 # Header declarations
│   ├── npu_daemon.cpp        # Standalone NPU Simulator Daemon (Device)
│   ├── npu_protocol.h        # Shared Memory Protocol
│   ├── npu_mem_manager.h     # Host-side Memory Allocator
│   └── npu_device_manager.h  # Global Device Manager
├── setup.py                  # AOT compilation script
├── jit_loader.py             # JIT compilation script
├── start_daemons.py          # Script to launch NPU Cluster
├── run_workload.py           # Script to execute NPU jobs
└── test.py                   # Basic Functional Tests
```

## Features

-   **Multi-NPU Support**: Simulates multiple independent NPU devices, each with its own Shared Memory address space.
-   **Global Resource Manager**: Centralized `NPUDeviceManager` handles device connections and context.
-   **Memory Management**: Custom `NPUMemoryAllocator` implementing a coalescing free-list strategy for managing device memory (16MB per device).
-   **P2P Communication**: Supports `npu_d2d` (Device-to-Device) copies simulated via Host DMA.
-   **Manual Driver API**: Exposes low-level driver controls to Python (`malloc`, `free`, `h2d`, `d2h`, `compute`).

## Prerequisites

-   **Python**: 3.9+
-   **PyTorch**: 2.0+
-   **C++ Compiler**:
    -   Linux: GCC/G++
    -   Windows: MSVC
-   **Build Tool**: Ninja (recommended)

```bash
pip install torch ninja
```

## Running the Multi-NPU Simulation

The simulation is split into two parts: the "Cluster Manager" that runs the devices, and the "Client" that submits workloads.

### Step 1: Start the NPU Cluster
Open a terminal and run:
```bash
python start_daemons.py
```
This will compile and launch two NPU Daemon processes (Device 0 and Device 1). Leave this running.

### Step 2: Run the Workload
Open a second terminal and run:
```bash
python run_workload.py
```
This script will:
1.  Connect to the running NPU Daemons.
2.  Allocate memory on Device 0 and Device 1.
3.  Transfer data to Device 0.
4.  Execute a Compute Op (ADD) on Device 0.
5.  Transfer the result from Device 0 to Device 1 (P2P).
6.  Retrieve the final result from Device 1 to Host.
7.  Verify correctness.

## Python API Reference

The extension exposes the following API under the `custom_ops` module (loaded via JIT or AOT).

### Management
-   `init_device(device_id)`: Initializes connection to a specific NPU device.

### Memory
-   `npu_malloc(device_id, size)`: Allocates memory on the specified device. Returns a virtual handle.
-   `npu_free(device_id, addr)`: Frees the memory at the given handle.

### Data Movement
-   `npu_h2d(device_id, tensor, dst_addr)`: Copies data from a Host Tensor to NPU memory.
-   `npu_d2h(device_id, tensor, src_addr)`: Copies data from NPU memory to a Host Tensor.
-   `npu_d2d(src_id, src_addr, dst_id, dst_addr, size)`: Copies data between two NPU devices.

### Execution
-   `npu_compute(device_id, opcode, src1, src2, dst, size, scalar)`: Submits a command to the NPU.
    -   **Opcodes**: `OP_ADD`, `OP_SUB`, `OP_MUL`, `OP_MATMUL`, `OP_EXIT`.

## Architecture Details

1.  **Device Context**: Each `NPUDeviceContext` manages a `SharedMemoryHandler` and an `NPUMemoryAllocator`.
2.  **Handles**: Pointers returned to Python are 64-bit encoded handles containing both the `device_id` (high 32 bits) and the `offset` (low 32 bits).
3.  **Daemon**: The `npu_daemon` is a standalone C++ process. It accepts a `device_id` argument to determine which Shared Memory segment to attach to (e.g., `Local\NPU_SHM_0`).
