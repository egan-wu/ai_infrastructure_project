#include "npu_driver_protocol.h"
#include "npu_protocol.h" // For NPUControl struct
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
    std::map<uint64_t, std::vector<uint64_t>> client_allocations;

public:
    NPUResourceManager(size_t total_size) {
        // Reserve first 64KB for System Control to be safe
        size_t reserved = 65536;
        if (total_size <= reserved) {
            reserved = 0;
        }

        // Initial block starts after reserved area
        blocks.push_back({reserved, total_size - reserved, true});
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

        auto it = blocks.begin();
        for (; it != blocks.end(); ++it) {
            if (it->offset == offset) break;
        }

        if (it == blocks.end() || it->is_free) return false;

        it->is_free = true;

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
        std::vector<uint64_t> to_free;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (client_allocations.count(client_id)) {
                to_free = client_allocations[client_id];
                client_allocations.erase(client_id);
            }
        }

        if (!to_free.empty()) {
            std::cout << "[Driver] Cleaning up " << to_free.size() << " allocations for client " << client_id << std::endl;
            for (uint64_t off : to_free) {
                internal_free(off);
            }
        }
    }

    void internal_free(uint64_t offset) {
         std::lock_guard<std::mutex> lock(mutex);
         auto it = blocks.begin();
         for (; it != blocks.end(); ++it) {
             if (it->offset == offset) break;
         }
         if (it == blocks.end() || it->is_free) return;

         it->is_free = true;
         // Merging logic
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

void initialize_shm(void* addr) {
    if (!addr || addr == (void*)-1) return;
    NPUControl* ctrl = static_cast<NPUControl*>(addr);

    // Initialize Control Structure
    // Using placement new to ensure atomics are constructed properly
    new (ctrl) NPUControl();

    ctrl->magic = 0xCAFEBABE;
    ctrl->host_ready.store(0);
    ctrl->device_done.store(0);

    std::cout << "[Driver] Initialized SHM Control Struct at offset 0. Magic: 0x" << std::hex << ctrl->magic << std::endl;
}

#ifdef _WIN32
// ... (Windows implementation omitted for brevity, logic mirrors Linux)
// Just ensuring initialization happens after MapViewOfFile
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

    void* base = mmap(0, NPU_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base != MAP_FAILED) {
        initialize_shm(base);
        // munmap(base, NPU_SHM_SIZE); // Keep it open? No need for driver to keep mapped if only Allocator logic uses metadata
        // But we initialized it.
    }

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
