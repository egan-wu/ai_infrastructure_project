#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <cstring>
#include "npu_driver_protocol.h"
#include "npu_protocol.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

// Helper to print float array
void debug_print(const char* label, float* data, size_t count) {
    size_t n = count < 5 ? count : 5;
    std::cout << "[NPU Hardware] " << label << ": [";
    for (size_t i = 0; i < n; ++i) {
        std::cout << data[i] << (i < n-1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
}

class NPUCore {
public:
    void compute_add_tensors(float* src1, float* src2, float* dst, size_t count) {
        debug_print("ADD Input 1", src1, count);
        debug_print("ADD Input 2", src2, count);
        for (size_t i = 0; i < count; ++i) dst[i] = src1[i] + src2[i];
        debug_print("ADD Output", dst, count);
    }

    void compute_sub_tensors(float* src1, float* src2, float* dst, size_t count) {
        debug_print("SUB Input 1", src1, count);
        debug_print("SUB Input 2", src2, count);
        for (size_t i = 0; i < count; ++i) dst[i] = src1[i] - src2[i];
        debug_print("SUB Output", dst, count);
    }

    void compute_mul(float* src1, float* src2, float* dst, size_t count) {
        debug_print("MUL Input 1", src1, count);
        debug_print("MUL Input 2", src2, count);
        for (size_t i = 0; i < count; ++i) dst[i] = src1[i] * src2[i];
        debug_print("MUL Output", dst, count);
    }

    void compute_matmul(float* src1, float* src2, float* dst, uint32_t M, uint32_t K, uint32_t N) {
        // ... (Matmul debug omitted for now as we test ADD)
    }
};

int main() {
    std::cout << "[NPU Hardware] Starting..." << std::endl;

    void* shm_base = nullptr;

#ifdef _WIN32
    HANDLE hMapFile = OpenFileMappingA(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, NPU_SHM_NAME);
    if (hMapFile == NULL) return 1;
    shm_base = MapViewOfFile(hMapFile, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);
#else
    int fd = shm_open(NPU_SHM_NAME, O_RDWR, 0666);
    if (fd == -1) {
        // Wait for driver?
        std::cerr << "Waiting for SHM..." << std::endl;
        while(fd == -1) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            fd = shm_open(NPU_SHM_NAME, O_RDWR, 0666);
        }
    }
    shm_base = mmap(0, NPU_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
#endif

    if (!shm_base || shm_base == (void*)-1) return 1;

    NPUControl* ctrl = static_cast<NPUControl*>(shm_base);
    NPUCore core;

    std::cout << "[NPU Hardware] Connected. Checking Magic..." << std::endl;

    // Check Magic
    if (ctrl->magic != 0xCAFEBABE) {
        std::cerr << "[NPU Hardware] ERROR: Invalid Magic Number: 0x" << std::hex << ctrl->magic << std::endl;
        // Maybe driver hasn't initialized yet? Wait.
    } else {
        std::cout << "[NPU Hardware] Magic OK: 0x" << std::hex << ctrl->magic << std::endl;
    }

    std::cout << "[NPU Hardware] Idle loop..." << std::endl;

    while (true) {
        // Atomic Load with Acquire
        if (ctrl->host_ready.load(std::memory_order_acquire) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire); // Extra safety

            char* base_addr = static_cast<char*>(shm_base);
            std::cout << "[NPU Hardware] Command Received: " << ctrl->opcode << std::endl;

            switch (ctrl->opcode) {
                case OP_COMPUTE_ADD: {
                    float* src1 = reinterpret_cast<float*>(base_addr + ctrl->src_offset_1);
                    float* src2 = reinterpret_cast<float*>(base_addr + ctrl->src_offset_2);
                    float* dst = reinterpret_cast<float*>(base_addr + ctrl->dst_offset);
                    core.compute_add_tensors(src1, src2, dst, ctrl->size_1);
                    break;
                }
                case OP_COMPUTE_SUB: {
                    float* src1 = reinterpret_cast<float*>(base_addr + ctrl->src_offset_1);
                    float* src2 = reinterpret_cast<float*>(base_addr + ctrl->src_offset_2);
                    float* dst = reinterpret_cast<float*>(base_addr + ctrl->dst_offset);
                    core.compute_sub_tensors(src1, src2, dst, ctrl->size_1);
                    break;
                }
                case OP_COMPUTE_MUL: {
                    float* src1 = reinterpret_cast<float*>(base_addr + ctrl->src_offset_1);
                    float* src2 = reinterpret_cast<float*>(base_addr + ctrl->src_offset_2);
                    float* dst = reinterpret_cast<float*>(base_addr + ctrl->dst_offset);
                    core.compute_mul(src1, src2, dst, ctrl->size_1);
                    break;
                }
                case OP_EXIT:
                    return 0;
            }

            // Signal Done
            ctrl->host_ready.store(0, std::memory_order_release);
            std::atomic_thread_fence(std::memory_order_release);
            ctrl->device_done.store(1, std::memory_order_release);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    return 0;
}
