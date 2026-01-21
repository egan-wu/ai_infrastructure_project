#include "ops.h"
#include "npu_protocol.h"
#include <iostream>
#include <thread>
#include <chrono>

// CPU Implementation using TensorAccessor
torch::Tensor weighted_sum_cpu(torch::Tensor a, torch::Tensor b, float alpha, float beta) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensors must have the same size");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "Tensors must have the same dtype");

    auto result = torch::empty_like(a);

    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "Only float32 supported for this example");
    TORCH_CHECK(a.dim() == 2, "This example supports 2D tensors");

    auto a_acc = a.accessor<float, 2>();
    auto b_acc = b.accessor<float, 2>();
    auto res_acc = result.accessor<float, 2>();

    for(int i = 0; i < a.size(0); i++) {
        for(int j = 0; j < a.size(1); j++) {
            res_acc[i][j] = alpha * a_acc[i][j] + beta * b_acc[i][j];
        }
    }

    return result;
}

torch::Tensor weighted_sum(torch::Tensor a, torch::Tensor b, float alpha, float beta) {
    TORCH_CHECK(a.device() == b.device(), "Tensors must be on the same device");
    if (a.device().is_cuda()) {
        #ifdef WITH_CUDA
        return weighted_sum_cuda(a, b, alpha, beta);
        #else
        TORCH_CHECK(false, "CUDA support was not compiled in.");
        #endif
    } else {
        return weighted_sum_cpu(a, b, alpha, beta);
    }
}

// NPU Client Implementation
torch::Tensor call_npu(torch::Tensor input, float scalar, uint64_t offset) {
    // 1. Validation
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Only float32 supported for NPU");
    TORCH_CHECK(input.device().is_cpu(), "Input must be on CPU for this simulation");

    // 2. Connect to Shared Memory (Client Mode)
    // Note: In a real app, you might want to keep this connection open (static or singleton)
    // instead of opening/closing every call. For simulation, this is fine.
    SharedMemoryHandler shm(SHM_NAME, SHM_SIZE, false);
    if (!shm.is_valid()) {
        TORCH_CHECK(false, "Failed to connect to NPU Shared Memory. Is the daemon running?");
    }

    NPUControl* ctrl = shm.get_control();

    // Check if device is busy (simple check)
    if (ctrl->host_ready) {
        TORCH_CHECK(false, "NPU is busy (host_ready is still true).");
    }

    // 3. DMA H2D (Simulated copy to SHM)
    // Calculate destination address
    char* base_addr = static_cast<char*>(shm.buffer);
    if (offset + input.nbytes() > SHM_SIZE) {
        TORCH_CHECK(false, "Shared memory overflow. Offset + Size > SHM_SIZE");
    }

    void* dest_ptr = base_addr + offset;
    std::memcpy(dest_ptr, input.data_ptr(), input.nbytes());

    // 4. Trigger NPU
    ctrl->opcode = 1; // Dummy opcode
    ctrl->address_offset = offset;
    ctrl->size = input.numel();
    ctrl->scalar = scalar;

    // Reset done flag before triggering
    ctrl->device_done = false;

    // Memory barrier
    #ifdef _WIN32
    MemoryBarrier();
    #else
    __sync_synchronize();
    #endif

    ctrl->host_ready = true; // Signal NPU

    // 5. Wait for NPU
    // Timeout loop
    int timeout_ms = 5000;
    int elapsed = 0;
    while (!ctrl->device_done) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        elapsed++;
        if (elapsed > timeout_ms) {
            TORCH_CHECK(false, "NPU timed out.");
        }
    }

    // 6. DMA D2H (Copy result back)
    auto result = torch::empty_like(input);
    std::memcpy(result.data_ptr(), dest_ptr, input.nbytes());

    return result;
}
