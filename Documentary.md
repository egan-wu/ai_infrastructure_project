# Project Documentation: Cross-Process NPU Simulation Framework

## 1. Codebase Overview

This project implements a custom PyTorch C++ extension that demonstrates a hybrid execution model. It supports standard CPU/CUDA operations alongside a novel **Cross-Process NPU Simulation**. The NPU (Neural Processing Unit) is simulated as a standalone daemon process that communicates with the PyTorch extension via **Shared Memory (IPC)**.

### File Structure
```
.
├── cpp/
│   ├── bindings.cpp     # Pybind11 module definitions (Python <-> C++ glue)
│   ├── ops.cpp          # CPU implementation and NPU Client logic
│   ├── ops.h            # Header declarations
│   ├── npu_daemon.cpp   # Standalone NPU Simulator Daemon (Server)
│   ├── npu_protocol.h   # Shared Memory IPC Protocol definition
│   └── cuda_ops.cu      # CUDA kernel implementations
├── setup.py             # AOT compilation script
├── jit_loader.py        # JIT compilation script
├── test_npu.py          # NPU verification script
└── test.py              # General functional test script
```

## 2. Functionality Supporting

The framework provides three main modes of execution:

1.  **CPU Execution**: Efficient element-wise operations using `torch::TensorAccessor`.
2.  **CUDA Execution**: GPU acceleration using custom CUDA kernels (automatically enabled if hardware is present).
3.  **NPU Simulation**: A manual driver simulation where the PyTorch extension acts as the "Host Driver" and an external process acts as the "Device". Data transfer is handled via Shared Memory, mimicking DMA (Direct Memory Access).

## 3. Supported Operations

The following operations are exposed to Python via the `custom_ops` module:

### 3.1 `weighted_sum`
Computes $C = \alpha \times A + \beta \times B$.
-   **Backend**: CPU (default) or CUDA (if inputs are on GPU).
-   **Arguments**:
    -   `a` (Tensor): Input tensor A.
    -   `b` (Tensor): Input tensor B.
    -   `alpha` (float): Weight for A.
    -   `beta` (float): Weight for B.

### 3.2 `call_npu`
Offloads a computation to the simulated NPU.
-   **Backend**: Simulated NPU (via Shared Memory).
-   **Logic**: `Output = Input + Scalar` (Simulated "Add" Op).
-   **Arguments**:
    -   `input` (Tensor): CPU Float32 tensor.
    -   `scalar` (float): Value to add.
    -   `offset` (int): Offset in Shared Memory to use for data transfer.

---

## 4. Usage Example

### Python Code
```python
import torch
import custom_ops

# --- 1. Weighted Sum (CPU/CUDA) ---
a = torch.randn(5, 5)
b = torch.randn(5, 5)
res = custom_ops.weighted_sum(a, b, 0.5, 2.0)
print(res)

# --- 2. NPU Simulation ---
# Note: 'npu_daemon' must be running in the background!
data = torch.ones(5, 5)
# Offload to NPU: Adds 10.0 to every element
npu_res = custom_ops.call_npu(data, 10.0, 4096)
print(npu_res)
```

## 5. Development Steps (Logical Breakdown)

The project was implemented in the following logical stages:

### Step 1: Protocol Design (`npu_protocol.h`)
-   Defined the **Shared Memory Protocol**.
-   Created the `NPUControl` structure to manage instruction dispatch (Opcode, Size, Offset, Scalar).
-   Implemented synchronization flags (`host_ready`, `device_done`) for handshake.
-   Built the `SharedMemoryHandler` class to abstract platform-specific IPC (Windows `CreateFileMapping` vs. Linux `shm_open`).

### Step 2: NPU Daemon Implementation (`npu_daemon.cpp`)
-   Built a standalone executable to act as the Device.
-   **Initialization**: Creates the Shared Memory segment (Server Mode).
-   **Loop**: Continuously polls the `host_ready` flag.
-   **Execution**: When a command is received, it reads data from the shared buffer (using the provided offset), performs the addition, and signals completion via `device_done`.

### Step 3: PyTorch Extension & Client (`ops.cpp`)
-   Implemented the **Host Driver** logic in `call_npu`.
-   **DMA Simulation**: `memcpy` from Tensor data to Shared Memory.
-   **Dispatch**: Writes the command to the `NPUControl` struct and sets `host_ready`.
-   **Synchronization**: Spins/Waits for `device_done` (timeout protected).
-   **Writeback**: Copies the result from Shared Memory back to a new Tensor.

### Step 4: Build System & Verification (`setup.py`, `test_npu.py`)
-   Configured `setup.py` to compile C++ and CUDA sources while excluding the standalone daemon.
-   Wrote `test_npu.py` to automate the end-to-end workflow: compile daemon -> start daemon -> run Python test -> kill daemon.

---

## 6. Detailed Running Instructions

### Prerequisites
-   Python 3.9+
-   PyTorch 2.0+
-   C++ Compiler (MSVC for Windows, G++ for Linux)

### Linux
1.  **Compile the Daemon**:
    ```bash
    g++ -O3 -pthread cpp/npu_daemon.cpp -o npu_daemon -lrt
    ```
2.  **Install the Extension**:
    ```bash
    python setup.py install
    ```
3.  **Run the Verification**:
    ```bash
    # Automatically handles daemon start/stop
    python test_npu.py
    ```

### Windows
1.  **Compile the Daemon**:
    Open "x64 Native Tools Command Prompt" and run:
    ```cmd
    cl.exe /EHsc /O2 cpp/npu_daemon.cpp /Fe:npu_daemon.exe
    ```
2.  **Install the Extension**:
    ```cmd
    python setup.py install
    ```
3.  **Run the Verification**:
    Start the daemon manually in one terminal:
    ```cmd
    npu_daemon.exe
    ```
    Run the test in another terminal:
    ```cmd
    python test_npu.py
    ```

---

## 7. Functions Review

### 7.1 `SharedMemoryHandler` (C++ Class)
-   **Purpose**: Manages the life cycle of the shared memory block.
-   **Key Mechanism**:
    -   **Windows**: Uses `CreateFileMapping` (Server) or `OpenFileMapping` (Client) and `MapViewOfFile`.
    -   **Linux**: Uses `shm_open`, `ftruncate`, and `mmap`.
-   **Memory Layout**: The first `sizeof(NPUControl)` bytes are reserved for the control struct. The rest is the raw data buffer.

### 7.2 `call_npu` (C++ Function)
-   **Signature**: `torch::Tensor call_npu(torch::Tensor input, float scalar, uint64_t offset)`
-   **Workflow**:
    1.  **Validate**: checks for contiguous float32 input.
    2.  **Connect**: Attaches to existing SHM region.
    3.  **Copy (H2D)**: Copies input tensor bytes to `SHM_Base + Offset`.
    4.  **Signal**: Sets `ctrl->opcode=1`, `ctrl->scalar=scalar`, `ctrl->host_ready=true`.
    5.  **Wait**: Blocks until `ctrl->device_done` is true.
    6.  **Copy (D2H)**: Reads result from SHM back to output tensor.

### 7.3 `NPUCore::compute` (Daemon Function)
-   **Logic**: Simple in-place addition.
    ```cpp
    for (size_t i = 0; i < count; ++i) {
        data[i] += scalar;
    }
    ```
-   **Addressing**: It receives a raw pointer calculated as `(char*)SHM_Base + Command_Offset`. This simulates how a real NPU would access physical memory addresses provided by the driver.

### 7.4 `weighted_sum` (C++ Function)
-   **Dispatcher**: Checks `tensor.device()`.
    -   If **CUDA**: Calls `weighted_sum_cuda` (wrapper for `weighted_sum_kernel`).
    -   If **CPU**: Calls `weighted_sum_cpu` (uses `TensorAccessor` for efficient looping).
