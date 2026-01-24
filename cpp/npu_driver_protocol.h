#pragma once
#include <cstdint>
#include <cstddef>

// Protocol Constants
#ifdef _WIN32
static const char* NPU_DRIVER_PIPE = "\\\\.\\pipe\\x_npu_driver";
static const char* NPU_SHM_NAME = "Local\\X_NPU_SHM";
#else
static const char* NPU_DRIVER_PIPE = "/tmp/x_npu_driver.sock";
static const char* NPU_SHM_NAME = "/x_npu_shm";
#endif

static const size_t NPU_SHM_SIZE = 2ULL * 1024 * 1024 * 1024; // 2 GB

// Message Types
enum class DriverMsgType : uint32_t {
    ALLOC_REQ = 1,
    ALLOC_RESP = 2,
    FREE_REQ = 3,
    FREE_RESP = 4,
    ERROR_RESP = 5
};

// Structures (Packed for safety)
#pragma pack(push, 1)

struct DriverHeader {
    DriverMsgType type;
    uint32_t payload_size;
};

struct AllocReq {
    size_t size;
    // Potentially alignment or other flags
};

struct AllocResp {
    uint64_t offset; // Offset within SHM
};

struct FreeReq {
    uint64_t offset;
};

struct FreeResp {
    bool success;
};

#pragma pack(pop)
