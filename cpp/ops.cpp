#include "ops.h"
#include "npu_protocol.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <deque>
#include <algorithm>
#include <map>
#include <ATen/InferSize.h>

// ==========================================
// 1. Shared Memory Connection & Command Buffer
// ==========================================

class NPUConnection {
public:
    SharedMemoryHandler shm;
    std::mutex cmd_mutex;

    NPUConnection() : shm(SHM_NAME, SHM_SIZE, false) {
        if (!shm.is_valid()) {
            std::cerr << "[x_tpu] FATAL: Failed to connect to NPU Daemon shared memory." << std::endl;
        }
    }

    void check_connection() {
        if (!shm.is_valid()) {
             TORCH_CHECK(false, "Not connected to NPU Daemon.");
        }
    }

    // Synchronous Dispatch
    void submit_command(OpCode op, uint64_t src1, uint64_t src2, uint64_t dst,
                        uint32_t s1, uint32_t s2, uint32_t s3, float scalar = 0.0f) {
        std::lock_guard<std::mutex> lock(cmd_mutex);
        check_connection();
        NPUControl* ctrl = shm.get_control();

        int safety = 0;
        while(ctrl->host_ready && safety < 10000) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            safety++;
        }
        if (ctrl->host_ready) {
             TORCH_CHECK(false, "NPU is stuck.");
        }

        ctrl->opcode = op;
        ctrl->src_offset_1 = src1;
        ctrl->src_offset_2 = src2;
        ctrl->dst_offset = dst;
        ctrl->size_1 = s1;
        ctrl->size_2 = s2;
        ctrl->size_3 = s3;
        ctrl->scalar = scalar;

        ctrl->device_done = false;

        #ifdef _WIN32
        MemoryBarrier();
        #else
        __sync_synchronize();
        #endif

        ctrl->host_ready = true;

        int timeout_ms = 5000;
        int elapsed = 0;
        while (!ctrl->device_done) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            elapsed++;
            if (elapsed > timeout_ms) {
                TORCH_CHECK(false, "NPU timed out.");
            }
        }
    }
};

static NPUConnection& get_conn() {
    static NPUConnection conn;
    return conn;
}

// Helper to inspect memory
void debug_print_shm(uint64_t offset, size_t count, const std::string& label) {
    if (!get_conn().shm.is_valid()) return;
    char* base = static_cast<char*>(get_conn().shm.buffer);
    float* data = reinterpret_cast<float*>(base + offset);
    std::cout << "[DEBUG SHM] " << label << " (Offset " << offset << ", " << count << " floats): [";
    for (size_t i = 0; i < count; ++i) {
        std::cout << data[i] << (i < count - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
}

// ==========================================
// 2. Linear Allocator with Simple Reuse
// ==========================================

struct Block {
    uint64_t offset;
    size_t size;
};

class NPUAllocatorState {
private:
    static constexpr uint64_t max_size = SHM_SIZE;
    static constexpr uint64_t start_offset = 256;

    uint64_t head_offset;
    std::vector<Block> freed_blocks;
    std::map<uint64_t, size_t> alloc_map;
    std::mutex mutex;

public:
    NPUAllocatorState() : head_offset(start_offset) {}

    void* allocate(size_t n) {
        std::lock_guard<std::mutex> lock(mutex);

        size_t aligned_n = n;
        if (aligned_n % 64 != 0) aligned_n += 64 - (aligned_n % 64);

        uint64_t offset = 0;
        bool reused = false;

        for (auto it = freed_blocks.begin(); it != freed_blocks.end(); ++it) {
            if (it->size >= aligned_n) {
                offset = it->offset;
                freed_blocks.erase(it);
                reused = true;
                break;
            }
        }

        if (!reused) {
            if (head_offset + aligned_n > max_size) {
                 TORCH_CHECK(false, "[x_tpu] Out of Memory! (16MB limit)");
            }
            offset = head_offset;
            head_offset += aligned_n;
        }

        alloc_map[offset] = aligned_n;

        // Return offset disguised as pointer
        return reinterpret_cast<void*>(offset);
    }

    void free(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex);
        uint64_t offset = reinterpret_cast<uint64_t>(ptr);

        if (alloc_map.count(offset)) {
            size_t size = alloc_map[offset];
            freed_blocks.push_back({offset, size});
            alloc_map.erase(offset);
        }
    }
};

static NPUAllocatorState& get_allocator_state() {
    static NPUAllocatorState state;
    return state;
}

static void deleteNPU(void* ptr) {
    get_allocator_state().free(ptr);
}

class NPUAllocator : public c10::Allocator {
public:
    c10::DataPtr allocate(size_t n) override {
        void* ptr = get_allocator_state().allocate(n);
        return {ptr, ptr, &deleteNPU, c10::Device(c10::DeviceType::PrivateUse1, 0)};
    }

    c10::DeleterFnPtr raw_deleter() const override {
        return &deleteNPU;
    }

    void copy_data(void* dest, const void* src, std::size_t count) const override {
        // This is called by PyTorch for CPU-CPU copies, but for us?
        // If one of them is NPU, we need to handle it.
        // But c10::Allocator::copy_data is rarely called directly for devices?
        // Safe implementation assuming virtual pointers if called?
        // But our pointers are offsets.
        // This method is generally for CPU allocators.
        std::memcpy(dest, src, count);
    }
};

static NPUAllocator global_npu_allocator;

// ==========================================
// 3. Helpers
// ==========================================

uint64_t get_offset(const at::Tensor& t) {
    // DataPtr is already the offset
    return reinterpret_cast<uint64_t>(t.data_ptr());
}

void* resolve_ptr(void* ptr) {
    if (!ptr) return nullptr;
    char* base = static_cast<char*>(get_conn().shm.buffer);
    uint64_t offset = reinterpret_cast<uint64_t>(ptr);
    return base + offset;
}

// ==========================================
// 4. Manual Driver API
// ==========================================

void npu_ioctl(const std::string& op_code, uintptr_t src1, uintptr_t src2, uintptr_t dst, uint32_t size) {
    OpCode code;
    if (op_code == "OP_ADD") code = OP_COMPUTE_ADD;
    else if (op_code == "OP_SUB") code = OP_COMPUTE_SUB;
    else if (op_code == "OP_MUL") code = OP_COMPUTE_MUL;
    else if (op_code == "OP_MATMUL") code = OP_COMPUTE_MATMUL;
    else if (op_code == "OP_H2D_COPY") code = OP_H2D_COPY;
    else if (op_code == "OP_D2H_COPY") code = OP_D2H_COPY;
    else if (op_code == "OP_EXIT") code = OP_EXIT;
    else {
         std::cerr << "[x_tpu] Unknown OpCode string: " << op_code << std::endl;
         return;
    }

    // Input pointers are already offsets
    get_conn().submit_command(code, (uint64_t)src1, (uint64_t)src2, (uint64_t)dst, size, 0, 0);
}

void npu_wait() {
    get_conn().check_connection();
}


// ==========================================
// 5. Kernel Implementations
// ==========================================

at::Tensor npu_empty(at::IntArrayRef size, std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout, std::optional<at::Device> device, std::optional<bool> pin_memory, std::optional<at::MemoryFormat> memory_format) {

    int64_t nelement = 1;
    for (auto s : size) nelement *= s;
    size_t bytes = nelement * sizeof(float);

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

at::Tensor npu_empty_strided(at::IntArrayRef size, at::IntArrayRef stride, std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout, std::optional<at::Device> device, std::optional<bool> pin_memory) {
    return npu_empty(size, dtype, layout, device, pin_memory, std::nullopt);
}

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

at::Tensor npu_copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
    bool dst_is_npu = dst.device().type() == c10::DeviceType::PrivateUse1;
    bool src_is_npu = self.device().type() == c10::DeviceType::PrivateUse1;

    size_t nbytes = self.nbytes();
    if (nbytes == 0) return dst;

    if (dst_is_npu && !src_is_npu) {
        // H2D
        void* src_ptr = self.data_ptr();
        void* dst_ptr = resolve_ptr(dst.data_ptr()); // Resolve offset to pointer

        if (dst_ptr == nullptr || src_ptr == nullptr) {
             std::cerr << "FATAL: Null pointer in H2D copy" << std::endl;
             return dst;
        }

        std::memcpy(dst_ptr, src_ptr, nbytes);
        get_conn().submit_command(OP_H2D_COPY, get_offset(dst), 0, 0, 0, 0, 0);

    } else if (src_is_npu && !dst_is_npu) {
        // D2H
        get_conn().submit_command(OP_D2H_COPY, get_offset(self), 0, 0, 0, 0, 0);

        void* src_ptr = resolve_ptr(self.data_ptr()); // Resolve offset to pointer
        void* dst_ptr = dst.data_ptr();

        if (dst_ptr == nullptr || src_ptr == nullptr) {
             std::cerr << "FATAL: Null pointer in D2H copy" << std::endl;
             return dst;
        }

        std::memcpy(dst_ptr, src_ptr, nbytes);

    } else if (src_is_npu && dst_is_npu) {
         // D2D (internal copy)
         void* src_ptr = resolve_ptr(self.data_ptr());
         void* dst_ptr = resolve_ptr(dst.data_ptr());
         std::memcpy(dst_ptr, src_ptr, self.nbytes());
    }

    return dst;
}

at::Tensor npu_copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
    return npu_copy_from(self, dst, false);
}

at::Tensor npu_add(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    auto out = at::empty_like(self);
    // Convert alpha to float
    float scalar = alpha.to<float>();

    get_conn().submit_command(OP_COMPUTE_ADD,
                              get_offset(self), get_offset(other), get_offset(out),
                              self.numel(), 0, 0, scalar);
    return out;
}

at::Tensor npu_mul(const at::Tensor& self, const at::Tensor& other) {
    auto out = at::empty_like(self);
    get_conn().submit_command(OP_COMPUTE_MUL,
                              get_offset(self), get_offset(other), get_offset(out),
                              self.numel(), 0, 0);
    return out;
}

at::Tensor npu_mm(const at::Tensor& self, const at::Tensor& other) {
    int64_t M = self.size(0);
    int64_t K = self.size(1);
    int64_t N = other.size(1);

    auto out = at::empty({M, N}, self.options());
    get_conn().submit_command(OP_COMPUTE_MATMUL,
                              get_offset(self), get_offset(other), get_offset(out),
                              M, K, N);
    return out;
}

// Legacy
void h2d_copy(torch::Tensor src, uint64_t offset) {
    // Offset is passed directly
    char* base = static_cast<char*>(get_conn().shm.buffer);
    if (offset + src.nbytes() > SHM_SIZE) return;
    std::memcpy(base + offset, src.data_ptr(), src.nbytes());
    get_conn().submit_command(OP_H2D_COPY, offset, 0, 0, 0, 0, 0);
}

void d2h_copy(torch::Tensor dst, uint64_t offset) {
    get_conn().submit_command(OP_D2H_COPY, offset, 0, 0, 0, 0, 0);
    char* base = static_cast<char*>(get_conn().shm.buffer);
    std::memcpy(dst.data_ptr(), base + offset, dst.nbytes());
}

void npu_add_legacy(uint64_t src, uint64_t dst, uint32_t size, float scalar) {
     get_conn().submit_command(OP_COMPUTE_ADD, src, 0, dst, size, 0, 0);
}

void npu_exit() {
    get_conn().submit_command(OP_EXIT, 0, 0, 0, 0, 0, 0);
}

// ==========================================
// 6. Registration
// ==========================================

void npu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    auto& arguments = *stack;
    for (size_t i = 0; i < arguments.size(); ++i) {
        if (arguments[i].isTensor()) {
            at::Tensor t = arguments[i].toTensor();
            if (t.defined() && t.device().type() == c10::DeviceType::PrivateUse1) {
                // Direct copy
                at::Tensor cpu_t = at::empty_like(t, at::TensorOptions().device(c10::kCPU));
                npu_copy_from(t, cpu_t, false);
                arguments[i] = cpu_t;
            }
        }
    }

    op.callBoxed(stack);

    for (size_t i = 0; i < arguments.size(); ++i) {
        if (arguments[i].isTensor()) {
            at::Tensor t = arguments[i].toTensor();
            if (t.defined() && t.device().is_cpu()) {
                at::Tensor npu_t = npu_empty(t.sizes(), t.scalar_type(), t.layout(),
                                             c10::Device(c10::DeviceType::PrivateUse1, 0),
                                             false, std::nullopt);
                npu_copy_from(t, npu_t, false);
                arguments[i] = npu_t;
            }
        }
    }
}

void init_x_tpu_extension() {
    c10::register_privateuse1_backend("x_tpu");
    c10::SetAllocator(c10::DeviceType::PrivateUse1, &global_npu_allocator);
    std::cout << "[x_tpu] Extension Initialized." << std::endl;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", &npu_empty);
    m.impl("empty_strided", &npu_empty_strided);
    m.impl("view", &npu_view);
    m.impl("_copy_from", &npu_copy_from);
    m.impl("_copy_from_and_resize", &npu_copy_from_and_resize);

    m.impl("add.Tensor", static_cast<at::Tensor (*)(const at::Tensor&, const at::Tensor&, const at::Scalar&)>(&npu_add));
    m.impl("mul.Tensor", static_cast<at::Tensor (*)(const at::Tensor&, const at::Tensor&)>(&npu_mul));
    m.impl("mm", static_cast<at::Tensor (*)(const at::Tensor&, const at::Tensor&)>(&npu_mm));
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&npu_fallback>());
}
