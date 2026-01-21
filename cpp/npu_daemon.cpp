#include "npu_protocol.h"
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>

// This class simulates the NPU Core.
// In a real scenario, this would interface with a driver or a static library.
class NPUCore {
public:
    NPUCore() {
        std::cout << "[NPU Core] Initialized." << std::endl;
    }

    // Placeholder for linking a static library
    // void link_hardware_driver() { ... }

    void compute(float* data, size_t count, float scalar) {
        // Simulate computation
        for (size_t i = 0; i < count; ++i) {
            data[i] += scalar;
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
            std::cout << "[NPU Daemon] Received Opcode: " << ctrl->opcode
                      << ", Size: " << ctrl->size
                      << ", Offset: " << ctrl->address_offset
                      << ", Scalar: " << ctrl->scalar << std::endl;

            // Get pointer to data based on offset relative to the START of the buffer
            // The protocol defines address_offset as offset from the base of SHM
            char* base_addr = static_cast<char*>(shm.buffer);
            float* data_ptr = reinterpret_cast<float*>(base_addr + ctrl->address_offset);

            // Perform computation
            core.compute(data_ptr, ctrl->size, ctrl->scalar);

            // Handshake: Signal completion and reset host flag
            // Order matters: clear host_ready, then set device_done.
            // Client waits for device_done=true.

            ctrl->host_ready = false;

            // Memory barrier could be needed here in strict systems,
            // but volatile helps for simple simulation.
            #ifdef _WIN32
            MemoryBarrier();
            #else
            __sync_synchronize();
            #endif

            ctrl->device_done = true;

            std::cout << "[NPU Daemon] Task completed." << std::endl;
        }

        // Sleep to reduce CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return 0;
}
