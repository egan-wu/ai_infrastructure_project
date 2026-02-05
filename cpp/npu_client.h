#pragma once
#include <cstdint>
#include <mutex>
#include "npu_driver_protocol.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

class NPUClient {
private:
#ifdef _WIN32
    HANDLE hPipe;
    HANDLE hMapFile;
#else
    int sock_fd;
    int shm_fd;
#endif
    void* shm_base;
    std::mutex msg_mutex;

public:
    NPUClient();
    ~NPUClient();

    void* get_base_ptr();
    uint64_t request_allocate(size_t size);
    void request_free(uint64_t offset);
};
