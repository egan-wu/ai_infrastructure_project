#include "npu_client.h"
#include <stdexcept>
#include <iostream>

#ifdef _WIN32

NPUClient::NPUClient() {
    // 1. Connect to Pipe
    // Try loop if busy
    while (true) {
        hPipe = CreateFileA(
            NPU_DRIVER_PIPE,
            GENERIC_READ | GENERIC_WRITE,
            0, NULL, OPEN_EXISTING, 0, NULL);

        if (hPipe != INVALID_HANDLE_VALUE) break;
        if (GetLastError() != ERROR_PIPE_BUSY) {
            throw std::runtime_error("Could not connect to NPU Driver Pipe.");
        }
        WaitNamedPipeA(NPU_DRIVER_PIPE, 2000);
    }

    // Set message mode
    DWORD mode = PIPE_READMODE_MESSAGE;
    SetNamedPipeHandleState(hPipe, &mode, NULL, NULL);

    // 2. Map Shared Memory (Client Mode)
    hMapFile = OpenFileMappingA(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, NPU_SHM_NAME);
    if (hMapFile == NULL) {
        CloseHandle(hPipe);
        throw std::runtime_error("Could not open NPU Shared Memory.");
    }

    shm_base = MapViewOfFile(hMapFile, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);
    if (shm_base == NULL) {
        CloseHandle(hMapFile);
        CloseHandle(hPipe);
        throw std::runtime_error("Could not map NPU Shared Memory.");
    }
}

NPUClient::~NPUClient() {
    if (shm_base) UnmapViewOfFile(shm_base);
    if (hMapFile) CloseHandle(hMapFile);
    if (hPipe != INVALID_HANDLE_VALUE) CloseHandle(hPipe);
}

uint64_t NPUClient::request_allocate(size_t size) {
    std::lock_guard<std::mutex> lock(msg_mutex);

    DriverHeader head = {DriverMsgType::ALLOC_REQ, sizeof(AllocReq)};
    AllocReq req = {size};

    DWORD written;
    if (!WriteFile(hPipe, &head, sizeof(head), &written, NULL) ||
        !WriteFile(hPipe, &req, sizeof(req), &written, NULL)) {
        throw std::runtime_error("Failed to send Alloc Req");
    }

    DriverHeader r_head;
    DWORD read;
    if (!ReadFile(hPipe, &r_head, sizeof(r_head), &read, NULL)) {
        throw std::runtime_error("Failed to read Alloc Resp Header");
    }

    if (r_head.type == DriverMsgType::ERROR_RESP) {
        throw std::runtime_error("NPU Driver reported OOM");
    }

    if (r_head.type == DriverMsgType::ALLOC_RESP) {
        AllocResp resp;
        ReadFile(hPipe, &resp, sizeof(resp), &read, NULL);
        return resp.offset;
    }

    throw std::runtime_error("Unexpected response from driver");
}

void NPUClient::request_free(uint64_t offset) {
    std::lock_guard<std::mutex> lock(msg_mutex);

    DriverHeader head = {DriverMsgType::FREE_REQ, sizeof(FreeReq)};
    FreeReq req = {offset};

    DWORD written;
    WriteFile(hPipe, &head, sizeof(head), &written, NULL);
    WriteFile(hPipe, &req, sizeof(req), &written, NULL);

    // Wait for ack
    DriverHeader r_head;
    DWORD read;
    ReadFile(hPipe, &r_head, sizeof(r_head), &read, NULL);
}

#else
#include <sys/mman.h>
#include <fcntl.h>

NPUClient::NPUClient() {
    // 1. Connect to Socket
    sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, NPU_DRIVER_PIPE, sizeof(addr.sun_path)-1);

    if (connect(sock_fd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        throw std::runtime_error("Could not connect to NPU Driver Socket.");
    }

    // 2. Map Shared Memory
    shm_fd = shm_open(NPU_SHM_NAME, O_RDWR, 0666);
    if (shm_fd == -1) {
        close(sock_fd);
        throw std::runtime_error("Could not open NPU Shared Memory.");
    }

    shm_base = mmap(0, NPU_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_base == MAP_FAILED) {
        close(shm_fd);
        close(sock_fd);
        throw std::runtime_error("Could not map NPU Shared Memory.");
    }
}

NPUClient::~NPUClient() {
    if (shm_base != MAP_FAILED) munmap(shm_base, NPU_SHM_SIZE);
    if (shm_fd != -1) close(shm_fd);
    if (sock_fd != -1) close(sock_fd);
}

uint64_t NPUClient::request_allocate(size_t size) {
    std::lock_guard<std::mutex> lock(msg_mutex);

    DriverHeader head = {DriverMsgType::ALLOC_REQ, sizeof(AllocReq)};
    AllocReq req = {size};

    send(sock_fd, &head, sizeof(head), 0);
    send(sock_fd, &req, sizeof(req), 0);

    DriverHeader r_head;
    recv(sock_fd, &r_head, sizeof(r_head), MSG_WAITALL);

    if (r_head.type == DriverMsgType::ERROR_RESP) {
        throw std::runtime_error("NPU Driver reported OOM");
    }

    if (r_head.type == DriverMsgType::ALLOC_RESP) {
        AllocResp resp;
        recv(sock_fd, &resp, sizeof(resp), MSG_WAITALL);
        return resp.offset;
    }
    throw std::runtime_error("Unexpected response from driver");
}

void NPUClient::request_free(uint64_t offset) {
    std::lock_guard<std::mutex> lock(msg_mutex);

    DriverHeader head = {DriverMsgType::FREE_REQ, sizeof(FreeReq)};
    FreeReq req = {offset};

    send(sock_fd, &head, sizeof(head), 0);
    send(sock_fd, &req, sizeof(req), 0);

    DriverHeader r_head;
    recv(sock_fd, &r_head, sizeof(r_head), MSG_WAITALL);
}

#endif

void* NPUClient::get_base_ptr() {
    return shm_base;
}
