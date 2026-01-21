#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include <atomic>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

// Shared Memory Constants
// On Windows, the name is typically "Local\\Name". On Linux, it's "/Name".
#ifdef _WIN32
static const char* SHM_NAME = "Local\\NPU_SHM";
#else
static const char* SHM_NAME = "/npu_shm";
#endif
static const size_t SHM_SIZE = 16 * 1024 * 1024; // 16 MB

// Protocol Structure
struct NPUControl {
    uint32_t opcode;           // Instruction ID
    uint64_t address_offset;   // Start address of data within shared buffer (bytes)
    uint32_t size;             // Number of float elements
    float scalar;              // Parameter for computation

    // Synchronization flags
    // volatile is used to prevent compiler optimization, but atomic is better for portable concurrency.
    // However, for raw SHM structs without C++ standard library shared layout, volatile bool or atomic_bool (if standard layout) is common.
    // We will use volatile bool for simplicity in this C-style struct, but enforce memory barriers in code if needed.
    volatile bool host_ready;
    volatile bool device_done;
};

// Cross-Platform Shared Memory Handler
class SharedMemoryHandler {
public:
    void* buffer = nullptr;
    size_t size = 0;
    bool is_server = false;
    std::string name;

#ifdef _WIN32
    HANDLE hMapFile = NULL;
#else
    int shm_fd = -1;
#endif

    SharedMemoryHandler(const std::string& shm_name, size_t shm_size, bool server)
        : name(shm_name), size(shm_size), is_server(server) {

#ifdef _WIN32
        if (is_server) {
            hMapFile = CreateFileMappingA(
                INVALID_HANDLE_VALUE,    // use paging file
                NULL,                    // default security
                PAGE_READWRITE,          // read/write access
                0,                       // maximum object size (high-order DWORD)
                (DWORD)size,             // maximum object size (low-order DWORD)
                name.c_str());           // name of mapping object
        } else {
            hMapFile = OpenFileMappingA(
                FILE_MAP_ALL_ACCESS,   // read/write access
                FALSE,                 // do not inherit the name
                name.c_str());         // name of mapping object
        }

        if (hMapFile == NULL) {
            std::cerr << "Could not create/open file mapping object (" << GetLastError() << ")." << std::endl;
            return;
        }

        buffer = MapViewOfFile(
            hMapFile,            // handle to map object
            FILE_MAP_ALL_ACCESS, // read/write permission
            0,
            0,
            size);

        if (buffer == NULL) {
            std::cerr << "Could not map view of file (" << GetLastError() << ")." << std::endl;
            CloseHandle(hMapFile);
            return;
        }
#else
        // POSIX Implementation
        if (is_server) {
            shm_fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
            if (shm_fd == -1) {
                perror("shm_open");
                return;
            }
            if (ftruncate(shm_fd, size) == -1) {
                perror("ftruncate");
                return;
            }
        } else {
            shm_fd = shm_open(name.c_str(), O_RDWR, 0666);
            if (shm_fd == -1) {
                // Ideally client waits for server, but here we just fail
                perror("shm_open client");
                return;
            }
        }

        buffer = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (buffer == MAP_FAILED) {
            perror("mmap");
            return;
        }
#endif
    }

    ~SharedMemoryHandler() {
#ifdef _WIN32
        if (buffer) UnmapViewOfFile(buffer);
        if (hMapFile) CloseHandle(hMapFile);
#else
        if (buffer != MAP_FAILED) munmap(buffer, size);
        if (shm_fd != -1) close(shm_fd);
        if (is_server) shm_unlink(name.c_str());
#endif
    }

    bool is_valid() const { return buffer != nullptr; }

    NPUControl* get_control() {
        return static_cast<NPUControl*>(buffer);
    }

    void* get_data_start() {
        // Data starts after the control struct
        // We align to 64 bytes for good measure
        size_t offset = sizeof(NPUControl);
        if (offset % 64 != 0) {
            offset += 64 - (offset % 64);
        }
        return static_cast<char*>(buffer) + offset;
    }
};
