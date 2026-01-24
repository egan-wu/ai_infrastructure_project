#include "npu_driver_protocol.h"
#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <cstring>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#endif

// ==========================================
// 1. Memory Allocator (First-Fit)
// ==========================================
struct Block {
    uint64_t offset;
    size_t size;
    bool is_free;
};

class NPUResourceManager {
private:
    std::list<Block> blocks;
    std::mutex mutex;

    // Map client_id -> list of allocated offsets
    // Used for cleanup on disconnect
    std::map<uint64_t, std::vector<uint64_t>> client_allocations;

public:
    NPUResourceManager(size_t total_size) {
        // Initial block covers everything
        // Reserve first few bytes for control/metadata if needed?
        // For now, start at 0.
        blocks.push_back({0, total_size, true});
    }

    uint64_t allocate(size_t size, uint64_t client_id) {
        std::lock_guard<std::mutex> lock(mutex);

        // Alignment (64 bytes)
        size_t aligned = size;
        if (aligned % 64 != 0) aligned += 64 - (aligned % 64);

        for (auto it = blocks.begin(); it != blocks.end(); ++it) {
            if (it->is_free && it->size >= aligned) {
                // Found block
                uint64_t offset = it->offset;

                if (it->size > aligned) {
                    // Split
                    Block new_block = {offset + aligned, it->size - aligned, true};
                    it->size = aligned;
                    it->is_free = false;
                    blocks.insert(std::next(it), new_block);
                } else {
                    // Exact fit
                    it->is_free = false;
                }

                client_allocations[client_id].push_back(offset);
                return offset;
            }
        }

        throw std::runtime_error("OOM");
    }

    bool free(uint64_t offset, uint64_t client_id) {
        std::lock_guard<std::mutex> lock(mutex);

        // Locate block
        auto it = blocks.begin();
        for (; it != blocks.end(); ++it) {
            if (it->offset == offset) break;
        }

        if (it == blocks.end() || it->is_free) return false;

        it->is_free = true;

        // Remove from client tracking
        auto& allocs = client_allocations[client_id];
        auto a_it = std::find(allocs.begin(), allocs.end(), offset);
        if (a_it != allocs.end()) allocs.erase(a_it);

        // Coalesce Next
        auto next = std::next(it);
        if (next != blocks.end() && next->is_free) {
            it->size += next->size;
            blocks.erase(next);
        }

        // Coalesce Prev
        if (it != blocks.begin()) {
            auto prev = std::prev(it);
            if (prev->is_free) {
                prev->size += it->size;
                blocks.erase(it);
            }
        }
        return true;
    }

    void cleanup_client(uint64_t client_id) {
        // Can't hold main mutex while calling free because free takes mutex?
        // Actually free takes mutex. We need to be careful.
        // Copy the list of allocations first.

        std::vector<uint64_t> to_free;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (client_allocations.count(client_id)) {
                to_free = client_allocations[client_id];
                client_allocations.erase(client_id); // Drop tracking now
            }
        }

        if (!to_free.empty()) {
            std::cout << "[Driver] Cleaning up " << to_free.size() << " allocations for client " << client_id << std::endl;
            for (uint64_t off : to_free) {
                // Manually free internal logic avoiding double tracking removal issues
                // Or just modify free to handle untracked?
                // Let's just implement internal free logic helper
                internal_free(off);
            }
        }
    }

    // Unsafe internal free (expects no mutex held? No, wait.
    // We released mutex above. So we can just lock again.
    void internal_free(uint64_t offset) {
         std::lock_guard<std::mutex> lock(mutex);
         auto it = blocks.begin();
         for (; it != blocks.end(); ++it) {
             if (it->offset == offset) break;
         }
         if (it == blocks.end() || it->is_free) return;

         it->is_free = true;
         // Merging logic same as above
         auto next = std::next(it);
         if (next != blocks.end() && next->is_free) {
             it->size += next->size;
             blocks.erase(next);
         }
         if (it != blocks.begin()) {
             auto prev = std::prev(it);
             if (prev->is_free) {
                 prev->size += it->size;
                 blocks.erase(it);
             }
         }
    }
};

static NPUResourceManager manager(NPU_SHM_SIZE);

// ==========================================
// 2. Platform Specific Server
// ==========================================

#ifdef _WIN32
void handle_client(HANDLE hPipe, uint64_t client_id) {
    std::cout << "[Driver] Client " << client_id << " connected." << std::endl;

    while (true) {
        DriverHeader header;
        DWORD bytesRead;
        BOOL success = ReadFile(hPipe, &header, sizeof(header), &bytesRead, NULL);

        if (!success || bytesRead == 0) break; // Disconnect

        if (header.type == DriverMsgType::ALLOC_REQ) {
            AllocReq req;
            ReadFile(hPipe, &req, sizeof(req), &bytesRead, NULL);

            AllocResp resp;
            try {
                resp.offset = manager.allocate(req.size, client_id);
                // Send Header + Resp
                DriverHeader r_head = {DriverMsgType::ALLOC_RESP, sizeof(resp)};
                DWORD written;
                WriteFile(hPipe, &r_head, sizeof(r_head), &written, NULL);
                WriteFile(hPipe, &resp, sizeof(resp), &written, NULL);
                std::cout << "[Driver] Allocated " << req.size << " bytes at " << resp.offset << std::endl;
            } catch(...) {
                DriverHeader r_head = {DriverMsgType::ERROR_RESP, 0};
                DWORD written;
                WriteFile(hPipe, &r_head, sizeof(r_head), &written, NULL);
            }

        } else if (header.type == DriverMsgType::FREE_REQ) {
            FreeReq req;
            ReadFile(hPipe, &req, sizeof(req), &bytesRead, NULL);
            manager.free(req.offset, client_id);
            // No response needed? Or ack? Let's ack.
            DriverHeader r_head = {DriverMsgType::FREE_RESP, 0};
            DWORD written;
            WriteFile(hPipe, &r_head, sizeof(r_head), &written, NULL);
            std::cout << "[Driver] Freed offset " << req.offset << std::endl;
        }
    }

    std::cout << "[Driver] Client " << client_id << " disconnected." << std::endl;
    manager.cleanup_client(client_id);
    DisconnectNamedPipe(hPipe);
    CloseHandle(hPipe);
}

void run_server() {
    uint64_t next_id = 1;

    // Create Shared Memory
    HANDLE hMapFile = CreateFileMappingA(
        INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
        (DWORD)(NPU_SHM_SIZE >> 32), (DWORD)(NPU_SHM_SIZE & 0xFFFFFFFF),
        NPU_SHM_NAME);

    if (hMapFile == NULL) {
        std::cerr << "Failed to create SHM: " << GetLastError() << std::endl;
        return;
    }
    std::cout << "[Driver] Shared Memory Created (2GB)." << std::endl;

    while (true) {
        HANDLE hPipe = CreateNamedPipeA(
            NPU_DRIVER_PIPE,
            PIPE_ACCESS_DUPLEX,
            PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
            PIPE_UNLIMITED_INSTANCES,
            512, 512, 0, NULL);

        if (hPipe == INVALID_HANDLE_VALUE) {
            std::cerr << "CreateNamedPipe failed." << std::endl;
            return;
        }

        if (ConnectNamedPipe(hPipe, NULL) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED)) {
            std::thread t(handle_client, hPipe, next_id++);
            t.detach();
        } else {
            CloseHandle(hPipe);
        }
    }
}
#else
void handle_client(int client_sock, uint64_t client_id) {
    std::cout << "[Driver] Client " << client_id << " connected." << std::endl;

    while (true) {
        DriverHeader header;
        ssize_t bytes = recv(client_sock, &header, sizeof(header), MSG_WAITALL);
        if (bytes <= 0) break;

        if (header.type == DriverMsgType::ALLOC_REQ) {
            AllocReq req;
            recv(client_sock, &req, sizeof(req), MSG_WAITALL);

            AllocResp resp;
            try {
                resp.offset = manager.allocate(req.size, client_id);
                DriverHeader r_head = {DriverMsgType::ALLOC_RESP, sizeof(resp)};
                send(client_sock, &r_head, sizeof(r_head), 0);
                send(client_sock, &resp, sizeof(resp), 0);
                std::cout << "[Driver] Allocated " << req.size << " bytes at " << resp.offset << std::endl;
            } catch(...) {
                DriverHeader r_head = {DriverMsgType::ERROR_RESP, 0};
                send(client_sock, &r_head, sizeof(r_head), 0);
            }
        } else if (header.type == DriverMsgType::FREE_REQ) {
            FreeReq req;
            recv(client_sock, &req, sizeof(req), MSG_WAITALL);
            manager.free(req.offset, client_id);

            DriverHeader r_head = {DriverMsgType::FREE_RESP, 0};
            send(client_sock, &r_head, sizeof(r_head), 0);
            std::cout << "[Driver] Freed offset " << req.offset << std::endl;
        }
    }

    std::cout << "[Driver] Client " << client_id << " disconnected." << std::endl;
    manager.cleanup_client(client_id);
    close(client_sock);
}

void run_server() {
    uint64_t next_id = 1;

    // Create SHM
    shm_unlink(NPU_SHM_NAME);
    int fd = shm_open(NPU_SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (fd == -1) { perror("shm_open"); return; }
    if (ftruncate(fd, NPU_SHM_SIZE) == -1) { perror("ftruncate"); return; }
    std::cout << "[Driver] Shared Memory Created (2GB)." << std::endl;

    // Create Socket
    unlink(NPU_DRIVER_PIPE);
    int server_sock = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, NPU_DRIVER_PIPE, sizeof(addr.sun_path)-1);

    if (bind(server_sock, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        perror("bind"); return;
    }
    if (listen(server_sock, 5) == -1) {
        perror("listen"); return;
    }

    while (true) {
        int client_sock = accept(server_sock, NULL, NULL);
        if (client_sock != -1) {
            std::thread t(handle_client, client_sock, next_id++);
            t.detach();
        }
    }
}
#endif

int main() {
    std::cout << "=== NPU Driver Daemon Started ===" << std::endl;
    run_server();
    return 0;
}
