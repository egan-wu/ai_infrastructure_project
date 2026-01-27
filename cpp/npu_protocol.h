#pragma once

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <iostream>
#include <string>

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
static const size_t SHM_SIZE = 16 * 1024 * 1024; // 16 MB

inline std::string get_shm_name(int device_id) {
    std::string base;
#ifdef _WIN32
    base = "Local\\NPU_SHM_";
#else
    base = "/npu_shm_";
#endif
    return base + std::to_string(device_id);
}

// OpCodes
enum OpCode : uint32_t {
    OP_H2D_COPY = 1,
    OP_D2H_COPY = 2,
    OP_COMPUTE_ADD = 3,
    OP_COMPUTE_MUL = 4,
    OP_COMPUTE_MATMUL = 5,
    OP_EXIT = 6,
    OP_COMPUTE_SUB = 7
};

// Protocol Structure
// Aligned to 64 bytes to prevent False Sharing
struct alignas(64) NPUControl {
    uint32_t magic;            // 0xCAFEBABE
    uint32_t opcode;           // Instruction ID
    uint64_t src_offset_1;     // Start address of input data 1 in SHM
    uint64_t src_offset_2;     // Start address of input data 2 in SHM (if needed)
    uint64_t dst_offset;       // Destination address for the output in SHM

    // Dimensions for MatMul / Element-wise
    uint32_t size_1;           // Total elements (for elementwise) or M (for MxK)
    uint32_t size_2;           // K (for MxK * KxN)
    uint32_t size_3;           // N (for KxN)

    float scalar;              // Parameter for scalar computation (legacy/optional)

    // Synchronization Flags
    // We use atomic uint32_t to ensure cross-process visibility with correct memory ordering
    std::atomic<uint32_t> host_ready;
    std::atomic<uint32_t> device_done;
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
        : size(shm_size), is_server(server), name(shm_name) {

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
            // Only server unlinks
            shm_unlink(name.c_str());
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

    // Data start address helper
    void* get_data_start() {
        size_t offset = sizeof(NPUControl);
        if (offset % 64 != 0) {
            offset += 64 - (offset % 64);
        }
        return static_cast<char*>(buffer) + offset;
    }
};
