#include "ops.h"
#include "npu_client.h"
#include "npu_protocol.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <ATen/InferSize.h>

// ==========================================
// 1. Singleton Client
// ==========================================

static std::unique_ptr<NPUClient> global_client;
static std::once_flag init_flag;

static NPUClient* get_client() {
    std::call_once(init_flag, []() {
        try {
            global_client = std::make_unique<NPUClient>();
            // Verify Magic
            NPUControl* ctrl = static_cast<NPUControl*>(global_client->get_base_ptr());
            if (ctrl->magic != 0xCAFEBABE) {
                std::cerr << "[x_npu] WARNING: Invalid Magic Number in SHM: 0x" << std::hex << ctrl->magic << std::endl;
            } else {
                std::cout << "[x_npu] Connected to NPU Driver. Magic OK." << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[x_npu] FATAL: " << e.what() << std::endl;
        }
    });
    return global_client.get();
}

// ==========================================
// 2. Allocator
// ==========================================

static void deleteNPU(void* ptr) {
    if (!ptr) return;
    NPUClient* client = get_client();
    if (!client) return;

    char* base = static_cast<char*>(client->get_base_ptr());
    char* target = static_cast<char*>(ptr);

    if (target < base || target >= base + NPU_SHM_SIZE) return;

    uint64_t offset = target - base;
    try {
        client->request_free(offset);
    } catch(...) {}
}

class NPUAllocator : public c10::Allocator {
public:
    c10::DataPtr allocate(size_t n) override {
        NPUClient* client = get_client();
        if (!client) TORCH_CHECK(false, "NPU Driver not connected.");

        uint64_t offset = client->request_allocate(n);
        void* ptr = static_cast<char*>(client->get_base_ptr()) + offset;

        return {ptr, ptr, &deleteNPU, c10::Device(c10::DeviceType::PrivateUse1, 0)};
    }

    c10::DeleterFnPtr raw_deleter() const override {
        return &deleteNPU;
    }

    void copy_data(void* dest, const void* src, std::size_t count) const override {
        std::memcpy(dest, src, count);
    }
};

static NPUAllocator global_npu_allocator;

// ==========================================
// 3. API Implementation
// ==========================================

void init_x_tpu_extension() {
    c10::register_privateuse1_backend("x_npu");
    c10::SetAllocator(c10::DeviceType::PrivateUse1, &global_npu_allocator);
    std::cout << "[x_npu] Extension Initialized (Backend: x_npu)." << std::endl;
}

uint64_t npu_malloc(int device_id, size_t size) {
    NPUClient* client = get_client();
    return client->request_allocate(size);
}

void npu_free(int device_id, uint64_t addr) {
    NPUClient* client = get_client();
    client->request_free(addr);
}

void npu_h2d(int device_id, torch::Tensor src, uint64_t dst_offset) {
    NPUClient* client = get_client();
    void* dst_ptr = static_cast<char*>(client->get_base_ptr()) + dst_offset;
    std::memcpy(dst_ptr, src.data_ptr(), src.nbytes());
}

void npu_d2h(int device_id, torch::Tensor dst, uint64_t src_offset) {
    NPUClient* client = get_client();
    void* src_ptr = static_cast<char*>(client->get_base_ptr()) + src_offset;
    std::memcpy(dst.data_ptr(), src_ptr, dst.nbytes());
}

void npu_compute(int device_id, const std::string& op_code, uint64_t src1, uint64_t src2, uint64_t dst, uint32_t size, float scalar) {
    NPUClient* client = get_client();
    void* base = client->get_base_ptr();
    NPUControl* ctrl = static_cast<NPUControl*>(base);

    // Wait for Ready (Spinlock on host_ready == 0)
    int timeout = 0;
    while(ctrl->host_ready.load(std::memory_order_acquire) != 0) {
        std::this_thread::yield();
        timeout++;
        if (timeout > 10000000) {
            std::cerr << "[x_npu] Timeout waiting for NPU idle." << std::endl;
            return;
        }
    }

    if (op_code == "OP_ADD") ctrl->opcode = OP_COMPUTE_ADD;
    else if (op_code == "OP_SUB") ctrl->opcode = OP_COMPUTE_SUB;
    else if (op_code == "OP_MUL") ctrl->opcode = OP_COMPUTE_MUL;
    else if (op_code == "OP_MATMUL") ctrl->opcode = OP_COMPUTE_MATMUL;
    else if (op_code == "OP_EXIT") ctrl->opcode = OP_EXIT;
    else return;

    ctrl->src_offset_1 = src1;
    ctrl->src_offset_2 = src2;
    ctrl->dst_offset = dst;
    ctrl->size_1 = size;
    ctrl->scalar = scalar;

    // Reset Device Done
    ctrl->device_done.store(0, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_release); // Ensure data writes visible

    // Signal Host Ready
    ctrl->host_ready.store(1, std::memory_order_release);

    // Wait for Completion
    int timeout_wait = 0;
    while(ctrl->device_done.load(std::memory_order_acquire) == 0) {
        std::this_thread::yield();
        timeout_wait++;
        if (timeout_wait > 50000000) {
             std::cerr << "[x_npu] Timeout waiting for NPU execution." << std::endl;
             return;
        }
    }
}

void npu_d2d(int src_dev, uint64_t src_addr, int dst_dev, uint64_t dst_addr, size_t size) {
    NPUClient* client = get_client();
    char* base = static_cast<char*>(client->get_base_ptr());
    std::memcpy(base + dst_addr, base + src_addr, size);
}

// PyTorch Ops Stubs
at::Tensor npu_empty(at::IntArrayRef size, std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout, std::optional<at::Device> device, std::optional<bool> pin_memory, std::optional<at::MemoryFormat> memory_format) { return at::Tensor(); }
at::Tensor npu_empty_strided(at::IntArrayRef size, at::IntArrayRef stride, std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout, std::optional<at::Device> device, std::optional<bool> pin_memory) { return at::Tensor(); }
at::Tensor npu_copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) { return dst; }
at::Tensor npu_copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) { return dst; }
void npu_ioctl(const std::string& op_code, uintptr_t src1, uintptr_t src2, uintptr_t dst, uint32_t size) {}
void npu_wait() {}
void h2d_copy(torch::Tensor src, uint64_t offset) {}
void d2h_copy(torch::Tensor dst, uint64_t offset) {}
void npu_add_legacy(uint64_t src, uint64_t dst, uint32_t size, float scalar) {}
void npu_exit() {}
at::Tensor npu_view(const at::Tensor& self, at::IntArrayRef size) { return self; }

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {}
void npu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {}
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&npu_fallback>());
}
