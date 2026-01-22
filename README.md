# PyTorch C++ Extension Template

This project is a robust template for creating PyTorch C++ Extensions. It demonstrates how to integrate custom C++ and CUDA kernels into PyTorch using both Ahead-of-Time (AOT) compilation and Just-In-Time (JIT) compilation.

## Project Structure

```
.
├── cpp/
│   ├── bindings.cpp    # Pybind11 module definitions
│   ├── ops.cpp         # CPU implementation using TensorAccessor
│   ├── ops.h           # Header declarations
│   ├── npu_daemon.cpp  # Standalone NPU Simulator Daemon
│   ├── npu_protocol.h  # Shared Memory IPC Protocol
│   └── cuda_ops.cu     # CUDA kernel skeleton
├── include/            # Directory for external headers
├── lib/                # Directory for external libraries
├── setup.py            # Script for AOT compilation
├── jit_loader.py       # Script for JIT loading
└── test.py             # Verification script
```

## Prerequisites

- **Python**: 3.9+
- **PyTorch**: 2.0+
- **C++ Compiler**:
  - Linux: GCC/G++
  - macOS: Clang
  - Windows: MSVC
- **Build Tool**: Ninja (recommended for faster builds)

```bash
pip install torch ninja
```

## Usage

### 1. Testing (Recommended First Step)

The `test.py` script demonstrates both JIT and AOT workflows and verifies the correctness of the custom operator.

```bash
python test.py
```

**Expected Output (CPU):**
```
=== Testing JIT Compilation ===
... (compilation logs) ...
Testing on cpu...
✅ Correctness check passed on cpu!

=== Testing AOT Compilation ===
Testing on cpu...
✅ Correctness check passed on cpu!
```
*(Note: If you haven't installed the AOT module yet, the AOT section might warn that the module is not found.)*

### 2. Ahead-of-Time (AOT) Compilation

This method compiles the extension once and installs it as a standard Python package. Ideal for production.

**Build and Install:**
```bash
python setup.py install
```

**Build In-place (for development):**
```bash
python setup.py build_ext --inplace
```

**Usage in Python:**
```python
import torch
import custom_ops

a = torch.randn(10, 10)
b = torch.randn(10, 10)
result = custom_ops.weighted_sum(a, b, 0.5, 2.0)
```

### 3. Just-In-Time (JIT) Compilation

This method compiles the C++ code at runtime. Ideal for rapid prototyping and development.

**Usage in Python:**
```python
from jit_loader import load_extension

custom_ops = load_extension()
# Now you can use it just like the AOT module
```

## Cross-Process NPU Simulation

This project includes a simulation of an NPU (Neural Processing Unit) running as a separate process (Daemon). The PyTorch extension communicates with this daemon using **Shared Memory**.

### Architecture
- **Protocol**: Defined in `cpp/npu_protocol.h` (Opcode, Size, Offset, Handshake Flags).
- **Daemon (`cpp/npu_daemon.cpp`)**: Allocates shared memory (Server), waits for commands, performs a dummy computation, and signals completion.
- **Client (`cpp/ops.cpp`)**: Connects to shared memory (Client), copies input tensor to SHM, triggers the daemon, waits for completion, and copies the result back.

### Running the Simulation

#### 1. Compile the Daemon

**Windows (MSVC):**
Open "x64 Native Tools Command Prompt" and run:
```cmd
cl.exe /EHsc /O2 cpp/npu_daemon.cpp /Fe:npu_daemon.exe
```

**Linux:**
```bash
g++ -O3 -pthread cpp/npu_daemon.cpp -o npu_daemon -lrt
```

#### 2. Run the Test

The `test_npu.py` script automates the verification. On Linux, it compiles and starts the daemon automatically. **On Windows, you must start `npu_daemon.exe` manually before running the python script.**

```bash
python test_npu.py
```

**Example Output:**
```
[NPU Daemon] Starting...
[NPU Core] Initialized.
[NPU Daemon] Waiting for commands...
...
[Client] Calling NPU with scalar=10.0...
[Client] Call took 0.0014s
✅ NPU Verification Passed!
```

## Example Logic

The template implements a **Weighted Element-wise Sum**:
$$C = \alpha \times A + \beta \times B$$

It showcases:
- **`torch::TensorAccessor`**: Efficient element-wise access on CPU.
- **Input Validation**: Using `TORCH_CHECK` to ensure tensor shapes and types match.
- **Device Dispatch**: Automatically routing execution to CPU or CUDA implementations based on input tensor device.

## CUDA Support

The project is pre-configured for CUDA.
- If a CUDA-capable GPU and the `nvcc` compiler are detected, `setup.py` and `jit_loader.py` will automatically compile `cpp/cuda_ops.cu` and define `-DWITH_CUDA`.
- The `weighted_sum` function checks the device of the input tensors and calls the appropriate kernel.

To add your own CUDA logic:
1.  Implement your kernel in `cpp/cuda_ops.cu`.
2.  Expose the function in `cpp/ops.h`.
3.  Call it from the dispatcher in `cpp/ops.cpp`.
