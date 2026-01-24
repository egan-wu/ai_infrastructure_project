#include "ops.h"
#include "npu_client.h"
#include "npu_protocol.h" // Keep for OpCodes if needed
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
            std::cout << "[x_npu] Connected to NPU Driver." << std::endl;
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
    // Pointer is Virtual Address (shm_base + offset)
    // We need to recover offset
    NPUClient* client = get_client();
    if (!client) return;

    char* base = static_cast<char*>(client->get_base_ptr());
    char* target = static_cast<char*>(ptr);

    // Validate range
    if (target < base || target >= base + NPU_SHM_SIZE) {
        std::cerr << "[x_npu] Error: Freeing invalid pointer" << std::endl;
        return;
    }

    uint64_t offset = target - base;
    try {
        client->request_free(offset);
    } catch(...) {
        std::cerr << "[x_npu] Warning: Failed to free memory" << std::endl;
    }
}

class NPUAllocator : public c10::Allocator {
public:
    c10::DataPtr allocate(size_t n) override {
        NPUClient* client = get_client();
        if (!client) {
             TORCH_CHECK(false, "NPU Driver not connected.");
        }

        uint64_t offset = client->request_allocate(n);
        void* ptr = static_cast<char*>(client->get_base_ptr()) + offset;

        return {ptr, ptr, &deleteNPU, c10::Device(c10::DeviceType::PrivateUse1, 0)};
    }

    c10::DeleterFnPtr raw_deleter() const override {
        return &deleteNPU;
    }

    void copy_data(void* dest, const void* src, std::size_t count) const override {
        // Standard memcpy works because we have mapped the SHM into our address space
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

// Legacy wrappers to maintain compatibility if needed, but mostly redirected to new logic
void init_device(int device_id) {
    // No-op in single-driver model for now, or ensure client init
    get_client();
}

uint64_t npu_malloc(int device_id, size_t size) {
    // Returns offset for legacy API compatibility
    NPUClient* client = get_client();
    return client->request_allocate(size);
}

void npu_free(int device_id, uint64_t addr) {
    // Expects offset
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

// Stub for compute - requires connecting to NPU Daemon (Command Processor)
// For this task (Memory Management), we focus on Allocator.
// Ideally, NPU Daemon would *also* attach to the same SHM.
// We can use the old `SharedMemoryHandler` logic or update it to use the new Driver SHM Name.
// But for now, let's just make sure memory works.
void npu_compute(int device_id, const std::string& op_code, uint64_t src1, uint64_t src2, uint64_t dst, uint32_t size, float scalar) {
     // Not fully implemented in this refactor cycle as focus is on Driver/Allocator
     std::cout << "[x_npu] Compute requested: " << op_code << std::endl;
}

// P2P
void npu_d2d(int src_dev, uint64_t src_addr, int dst_dev, uint64_t dst_addr, size_t size) {
    // Both are offsets in the same 2GB space
    NPUClient* client = get_client();
    char* base = static_cast<char*>(client->get_base_ptr());
    std::memcpy(base + dst_addr, base + src_addr, size);
}


// ==========================================
// 4. PyTorch Ops
// ==========================================

at::Tensor npu_empty(at::IntArrayRef size, std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout, std::optional<at::Device> device, std::optional<bool> pin_memory, std::optional<at::MemoryFormat> memory_format) {

    int64_t nelement = 1;
    for (auto s : size) nelement *= s;
    size_t bytes = nelement * sizeof(float); // Simplified for float

    auto data_ptr = global_npu_allocator.allocate(bytes);

    auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        bytes,
        std::move(data_ptr),
        &global_npu_allocator,
        true
    );

    auto tensor = at::detail::make_tensor<c10::TensorImpl>(
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        c10::scalarTypeToTypeMeta(dtype.value_or(at::kFloat)),
        c10::Device(c10::DeviceType::PrivateUse1, 0)
    );

    tensor.unsafeGetTensorImpl()->set_storage_keep_dtype(std::move(storage_impl));
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);

    return tensor;
}

// Matching signature for aten::empty_strided
at::Tensor npu_empty_strided(at::IntArrayRef size, at::IntArrayRef stride, std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout, std::optional<at::Device> device, std::optional<bool> pin_memory) {
    return npu_empty(size, dtype, layout, device, pin_memory, std::nullopt);
}

at::Tensor npu_copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
    if (self.nbytes() != dst.nbytes()) return dst;

    void* src_ptr = self.data_ptr();
    void* dst_ptr = dst.data_ptr();

    if (src_ptr && dst_ptr) {
        std::memcpy(dst_ptr, src_ptr, self.nbytes());
    }
    // Return by value to match schema
    return dst;
}

at::Tensor npu_copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
    npu_copy_from(self, dst, false);
    // Return by value
    return dst;
}

// Legacy Stubs
void npu_ioctl(const std::string& op_code, uintptr_t src1, uintptr_t src2, uintptr_t dst, uint32_t size) {}
void npu_wait() {}
void h2d_copy(torch::Tensor src, uint64_t offset) {}
void d2h_copy(torch::Tensor dst, uint64_t offset) {}
void npu_add_legacy(uint64_t src, uint64_t dst, uint32_t size, float scalar) {}
void npu_exit() {}

at::Tensor npu_view(const at::Tensor& self, at::IntArrayRef size) {
    auto inferred_size = at::infer_size(size, self.numel());
    auto alias = at::detail::make_tensor<c10::TensorImpl>(
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        self.dtype(),
        self.device()
    );
    alias.unsafeGetTensorImpl()->set_storage_keep_dtype(self.storage());
    alias.unsafeGetTensorImpl()->set_storage_offset(self.storage_offset());
    alias.unsafeGetTensorImpl()->set_sizes_contiguous(inferred_size);
    return alias;
}

// Registration
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", &npu_empty);
    m.impl("empty_strided", &npu_empty_strided); // Re-use
    m.impl("view", &npu_view);
    // Explicitly unregister copy_ logic if present or just use what we have.
    // The issue is mismatch signature.
    // _copy_from: (Tensor self, Tensor dst, bool non_blocking=False) -> Tensor
    // Our C++: (Tensor& self, Tensor& dst, bool non_blocking)
    // Wait, PyTorch schema says: (Tensor self, Tensor dst, bool non_blocking=False) -> Tensor
    // But self and dst are Tensor, not Tensor& in the schema?
    // C++ signature mapping usually allows const Tensor&.
    // The issue is the return type. Schema says Tensor, we return Tensor&.
    // Let's fix the return type to match exactly what PyTorch expects: 'at::Tensor' (by value)

    m.impl("_copy_from", &npu_copy_from);
    m.impl("_copy_from_and_resize", &npu_copy_from_and_resize);
}

// Register generic copy for to("x_npu") to work
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // We rely on _copy_from for PyTorch < 2.4 / PrivateUse1
    // Avoiding direct copy_ impl if signatures mismatch.
}

void npu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
     // Simplified fallback
     auto& arguments = *stack;
    for (size_t i = 0; i < arguments.size(); ++i) {
        if (arguments[i].isTensor()) {
            at::Tensor t = arguments[i].toTensor();
            if (t.defined() && t.device().type() == c10::DeviceType::PrivateUse1) {
                at::Tensor cpu_t = at::empty_like(t, at::TensorOptions().device(c10::kCPU));
                npu_copy_from(t, cpu_t, false);
                arguments[i] = cpu_t;
            }
        }
    }
    op.callBoxed(stack);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&npu_fallback>());
}
