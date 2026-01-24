#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <cstring>
#include "npu_driver_protocol.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

// This NPU Daemon acts as the HARDWARE.
// It connects to the Shared Memory created by the DRIVER.
// Ideally, it would have its own command queue (PCIe BAR registers), but for this task
// we focus on memory. We will just attach to SHM to prove we see the same memory.

int main() {
    std::cout << "[NPU Hardware] Starting..." << std::endl;

    void* shm_base = nullptr;

#ifdef _WIN32
    HANDLE hMapFile = OpenFileMappingA(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, NPU_SHM_NAME);
    if (hMapFile == NULL) {
        std::cerr << "[NPU Hardware] Failed to open SHM. Is Driver running?" << std::endl;
        return 1;
    }
    shm_base = MapViewOfFile(hMapFile, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);
#else
    int fd = shm_open(NPU_SHM_NAME, O_RDWR, 0666);
    if (fd == -1) {
        std::cerr << "[NPU Hardware] Failed to open SHM. Is Driver running?" << std::endl;
        return 1;
    }
    shm_base = mmap(0, NPU_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
#endif

    if (!shm_base || shm_base == (void*)-1) {
        std::cerr << "[NPU Hardware] Failed to map SHM." << std::endl;
        return 1;
    }

    std::cout << "[NPU Hardware] Connected to 2GB VRAM." << std::endl;
    std::cout << "[NPU Hardware] Idle loop (Ctrl+C to stop)..." << std::endl;

    // In a real scenario, this loop would poll a Command Queue in SHM
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}
