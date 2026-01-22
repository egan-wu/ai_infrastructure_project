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

    void compute_add(float* src, float* dst, size_t count, float scalar) {
        // Simulate computation reading from src and writing to dst
        for (size_t i = 0; i < count; ++i) {
            dst[i] = src[i] + scalar;
        }
    }
};

int main() {
    std::cout << "[NPU Daemon] Starting..." << std::endl;

    // Create Shared Memory (Server Mode)
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
        // Poll for host_ready
        if (ctrl->host_ready) {

            // Get base address for offset calculations
            char* base_addr = static_cast<char*>(shm.buffer);

            switch (ctrl->opcode) {
                case OP_H2D_COPY:
                    std::cout << "[NPU Daemon] OP_H2D_COPY: Host copied data to Device." << std::endl;
                    // In a real device, this might trigger a DMA engine or update page tables.
                    // Here we just acknowledge.
                    break;

                case OP_D2H_COPY:
                    std::cout << "[NPU Daemon] OP_D2H_COPY: Host requested data from Device." << std::endl;
                    // Acknowledge that device memory is ready to be read.
                    break;

                case OP_COMPUTE_ADD: {
                    std::cout << "[NPU Daemon] OP_COMPUTE_ADD: Size=" << ctrl->size
                              << ", SrcOffset=" << ctrl->src_offset
                              << ", DstOffset=" << ctrl->dst_offset
                              << ", Scalar=" << ctrl->scalar << std::endl;

                    size_t bytes_needed = ctrl->size * sizeof(float);
                    if (ctrl->src_offset + bytes_needed > SHM_SIZE ||
                        ctrl->dst_offset + bytes_needed > SHM_SIZE) {
                        std::cerr << "[NPU Daemon] Error: Memory access out of bounds! SHM_SIZE=" << SHM_SIZE << std::endl;
                    } else {
                        float* src_ptr = reinterpret_cast<float*>(base_addr + ctrl->src_offset);
                        float* dst_ptr = reinterpret_cast<float*>(base_addr + ctrl->dst_offset);

                        core.compute_add(src_ptr, dst_ptr, ctrl->size, ctrl->scalar);
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

            // Handshake: Signal completion and reset host flag
            ctrl->host_ready = false;

            // Memory barrier
            #ifdef _WIN32
            MemoryBarrier();
            #else
            __sync_synchronize();
            #endif

            ctrl->device_done = true;

            if (!running) {
                break; // Exit loop immediately after ack
            }
        }

        // Sleep to reduce CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::cout << "[NPU Daemon] Exited gracefully." << std::endl;
    return 0;
}
