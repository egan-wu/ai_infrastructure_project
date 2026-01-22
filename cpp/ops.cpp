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

// ==========================================
// NPU Driver Simulation Implementation
// ==========================================

// Helper to manage SHM connection
class NPUConnection {
public:
    SharedMemoryHandler shm;

    NPUConnection() : shm(SHM_NAME, SHM_SIZE, false) {
        if (!shm.is_valid()) {
            std::cerr << "Warning: Failed to connect to NPU Shared Memory." << std::endl;
        }
    }

    void check_connection() {
        if (!shm.is_valid()) {
             TORCH_CHECK(false, "Not connected to NPU Daemon.");
        }
    }

    // Single command dispatch
    void send_command(OpCode op, uint64_t src, uint64_t dst, uint32_t size, float scalar) {
        check_connection();
        NPUControl* ctrl = shm.get_control();

        // Busy wait if host_ready is true (shouldn't happen if single threaded host)
        // But in case of race or previous crash:
        int safety = 0;
        while(ctrl->host_ready && safety < 1000) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            safety++;
        }
        if (ctrl->host_ready) {
             TORCH_CHECK(false, "NPU is stuck in busy state (host_ready=true).");
        }

        ctrl->opcode = op;
        ctrl->src_offset = src;
        ctrl->dst_offset = dst;
        ctrl->size = size;
        ctrl->scalar = scalar;

        ctrl->device_done = false;

        #ifdef _WIN32
        MemoryBarrier();
        #else
        __sync_synchronize();
        #endif

        ctrl->host_ready = true;

        // Wait for Ack
        // Timeout 5s
        int timeout_ms = 5000;
        int elapsed = 0;
        while (!ctrl->device_done) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            elapsed++;
            if (elapsed > timeout_ms) {
                TORCH_CHECK(false, "NPU timed out waiting for Ack.");
            }
        }
    }
};

// Singleton connection
static NPUConnection& get_conn() {
    static NPUConnection conn;
    return conn;
}

void h2d_copy(torch::Tensor src, uint64_t offset) {
    TORCH_CHECK(src.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(src.scalar_type() == torch::kFloat32, "Only float32 supported for NPU");
    TORCH_CHECK(src.device().is_cpu(), "Input must be on CPU");

    auto& conn = get_conn();
    conn.check_connection();

    // 1. Perform Copy
    char* base_addr = static_cast<char*>(conn.shm.buffer);
    if (offset + src.nbytes() > SHM_SIZE) {
        TORCH_CHECK(false, "Shared memory overflow.");
    }

    void* dest_ptr = base_addr + offset;
    std::memcpy(dest_ptr, src.data_ptr(), src.nbytes());

    // 2. Notify Daemon
    conn.send_command(OP_H2D_COPY, offset, 0, src.numel(), 0.0f);
}

void d2h_copy(torch::Tensor dst, uint64_t offset) {
    TORCH_CHECK(dst.is_contiguous(), "Dst tensor must be contiguous");
    TORCH_CHECK(dst.scalar_type() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(dst.device().is_cpu(), "Dst tensor must be on CPU");

    auto& conn = get_conn();
    conn.check_connection();

    // 1. Notify Daemon (Request Read Access/Sync)
    conn.send_command(OP_D2H_COPY, offset, 0, dst.numel(), 0.0f);

    // 2. Perform Copy
    char* base_addr = static_cast<char*>(conn.shm.buffer);
    if (offset + dst.nbytes() > SHM_SIZE) {
        TORCH_CHECK(false, "Shared memory overflow.");
    }

    void* src_ptr = base_addr + offset;
    std::memcpy(dst.data_ptr(), src_ptr, dst.nbytes());
}

void npu_add(uint64_t src_offset, uint64_t dst_offset, uint32_t size, float scalar) {
    auto& conn = get_conn();
    conn.send_command(OP_COMPUTE_ADD, src_offset, dst_offset, size, scalar);
}

void npu_exit() {
    auto& conn = get_conn();
    // We send exit, but maybe we don't wait for ack if it shuts down too fast?
    // But daemon logic: Ack then Exit. So we should wait.
    conn.send_command(OP_EXIT, 0, 0, 0, 0.0f);
}
