#include "npu_protocol.h"
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>

// This class simulates the NPU Core.
class NPUCore {
public:
    NPUCore() {
        std::cout << "[NPU Core] Initialized." << std::endl;
    }

    void compute_add(float* src1, float* dst, size_t count, float scalar) {
        // If scalar is used, we add scalar. If src2 was supported, we would use it.
        // For simplicity, let's assume this is src1 + scalar
        for (size_t i = 0; i < count; ++i) {
            dst[i] = src1[i] + scalar;
        }
    }

    // Element-wise addition of two tensors
    void compute_add_tensors(float* src1, float* src2, float* dst, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            dst[i] = src1[i] + src2[i];
        }
    }

    void compute_mul(float* src1, float* src2, float* dst, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            dst[i] = src1[i] * src2[i];
        }
    }

    // Naive Matrix Multiplication: (M x K) * (K x N) -> (M x N)
    void compute_matmul(float* src1, float* src2, float* dst, uint32_t M, uint32_t K, uint32_t N) {
        // Initialize dst to 0
        for (uint32_t i = 0; i < M * N; ++i) {
            dst[i] = 0.0f;
        }

        for (uint32_t m = 0; m < M; ++m) {
            for (uint32_t n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < K; ++k) {
                    // src1 is row-major: [m, k] -> m * K + k
                    // src2 is row-major: [k, n] -> k * N + n
                    sum += src1[m * K + k] * src2[k * N + n];
                }
                dst[m * N + n] = sum;
            }
        }
    }
};

int main() {
    std::cout << "[NPU Daemon] Starting..." << std::endl;

    SharedMemoryHandler shm(SHM_NAME, SHM_SIZE, true);

    if (!shm.is_valid()) {
        std::cerr << "[NPU Daemon] Failed to initialize Shared Memory." << std::endl;
        return 1;
    }

    NPUControl* ctrl = shm.get_control();

    // Initialize flags
    ctrl->host_ready = false;
    ctrl->device_done = false;

    NPUCore core;

    std::cout << "[NPU Daemon] Waiting for commands..." << std::endl;

    bool running = true;
    while (running) {
        if (ctrl->host_ready) {

            char* base_addr = static_cast<char*>(shm.buffer);

            switch (ctrl->opcode) {
                case OP_H2D_COPY:
                    std::cout << "[NPU Daemon] OP_H2D_COPY: Host copied data to Device." << std::endl;
                    break;

                case OP_D2H_COPY:
                    std::cout << "[NPU Daemon] OP_D2H_COPY: Host requested data from Device." << std::endl;
                    break;

                case OP_COMPUTE_ADD: {
                    // Decide if scalar or tensor add based on src_offset_2
                    // Logic: if src_offset_2 is 0 (or same as src1 in some designs, but let's assume 0 implies unused for now?)
                    // Actually, let's look at how we will invoke it.
                    // To keep it simple: If scalar != 0, we do scalar add. If scalar == 0, we do tensor add?
                    // No, that's ambiguous. Let's assume if src_offset_2 is valid (e.g. > 0), it's tensor add.
                    // But 0 is a valid offset (technically).
                    // Let's assume the Host sets a convention.
                    // For this iteration, let's treat OP_COMPUTE_ADD as Tensor + Tensor if src_offset_2 != src_offset_1 (if we support broadcasting 0??)
                    // Let's just implement Tensor + Tensor for now as requested by aten::add.Tensor.
                    // If scalar is needed, we might need a separate opcode or flag.
                    // Wait, the previous implementation was scalar add.
                    // We need to support both or update. The requirement is `aten::add.Tensor`.

                    size_t bytes_needed = ctrl->size_1 * sizeof(float);
                    // Bounds check
                    if (ctrl->src_offset_1 + bytes_needed > SHM_SIZE ||
                        ctrl->src_offset_2 + bytes_needed > SHM_SIZE ||
                        ctrl->dst_offset + bytes_needed > SHM_SIZE) {
                         std::cerr << "[NPU Daemon] Error: Memory access out of bounds!" << std::endl;
                    } else {
                        float* src1 = reinterpret_cast<float*>(base_addr + ctrl->src_offset_1);
                        float* src2 = reinterpret_cast<float*>(base_addr + ctrl->src_offset_2);
                        float* dst = reinterpret_cast<float*>(base_addr + ctrl->dst_offset);

                        std::cout << "[NPU Daemon] OP_ADD (Tensor): Size=" << ctrl->size_1 << std::endl;
                        core.compute_add_tensors(src1, src2, dst, ctrl->size_1);
                    }
                    break;
                }

                case OP_COMPUTE_MUL: {
                    size_t bytes_needed = ctrl->size_1 * sizeof(float);
                    if (ctrl->src_offset_1 + bytes_needed > SHM_SIZE ||
                        ctrl->src_offset_2 + bytes_needed > SHM_SIZE ||
                        ctrl->dst_offset + bytes_needed > SHM_SIZE) {
                         std::cerr << "[NPU Daemon] Error: Memory access out of bounds!" << std::endl;
                    } else {
                        float* src1 = reinterpret_cast<float*>(base_addr + ctrl->src_offset_1);
                        float* src2 = reinterpret_cast<float*>(base_addr + ctrl->src_offset_2);
                        float* dst = reinterpret_cast<float*>(base_addr + ctrl->dst_offset);

                        std::cout << "[NPU Daemon] OP_MUL: Size=" << ctrl->size_1 << std::endl;
                        core.compute_mul(src1, src2, dst, ctrl->size_1);
                    }
                    break;
                }

                case OP_COMPUTE_MATMUL: {
                    // src1: MxK, src2: KxN, dst: MxN
                    uint32_t M = ctrl->size_1;
                    uint32_t K = ctrl->size_2;
                    uint32_t N = ctrl->size_3;

                    size_t size_1_bytes = M * K * sizeof(float);
                    size_t size_2_bytes = K * N * sizeof(float);
                    size_t size_dst_bytes = M * N * sizeof(float);

                    if (ctrl->src_offset_1 + size_1_bytes > SHM_SIZE ||
                        ctrl->src_offset_2 + size_2_bytes > SHM_SIZE ||
                        ctrl->dst_offset + size_dst_bytes > SHM_SIZE) {
                         std::cerr << "[NPU Daemon] Error: Memory access out of bounds!" << std::endl;
                    } else {
                        float* src1 = reinterpret_cast<float*>(base_addr + ctrl->src_offset_1);
                        float* src2 = reinterpret_cast<float*>(base_addr + ctrl->src_offset_2);
                        float* dst = reinterpret_cast<float*>(base_addr + ctrl->dst_offset);

                        std::cout << "[NPU Daemon] OP_MATMUL: " << M << "x" << K << " * " << K << "x" << N << std::endl;
                        core.compute_matmul(src1, src2, dst, M, K, N);
                    }
                    break;
                }

                case OP_EXIT:
                    std::cout << "[NPU Daemon] OP_EXIT: Shutting down..." << std::endl;
                    running = false;
                    break;

                default:
                    std::cerr << "[NPU Daemon] Unknown Opcode: " << ctrl->opcode << std::endl;
                    break;
            }

            ctrl->host_ready = false;

            #ifdef _WIN32
            MemoryBarrier();
            #else
            __sync_synchronize();
            #endif

            ctrl->device_done = true;

            if (!running) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::cout << "[NPU Daemon] Exited gracefully." << std::endl;
    return 0;
}
