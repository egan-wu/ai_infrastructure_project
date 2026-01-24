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
├── main.py                   # Multi-NPU Simulation Workflow
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

The `main.py` script demonstrates a full workflow involving two NPU devices.

1.  **Functionality**:
    -   Compiles and starts two `npu_daemon` processes (Device 0 and Device 1).
    -   Allocates memory on both devices.
    -   Transfers data to NPU 0 (H2D).
    -   Executes an `ADD` operation on NPU 0.
    -   Transfers the result from NPU 0 to NPU 1 (P2P/D2D).
    -   Retrieves the result from NPU 1 to Host (D2H).
    -   Verifies correctness.

2.  **Run**:
    ```bash
    python main.py
    ```

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
