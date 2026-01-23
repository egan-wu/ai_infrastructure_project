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
                        uint32_t s1, uint32_t s2, uint32_t s3) {
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

// ==========================================
// 2. Linear Allocator with Simple Reuse
// ==========================================

struct Block {
    uint64_t offset;
    size_t size;
};

class NPUAllocatorState {
private:
    uint64_t head_offset;
    const uint64_t max_size = SHM_SIZE;
    const uint64_t start_offset = 256;
    std::vector<Block> freed_blocks;
    std::map<void*, size_t> alloc_map;
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

        if (!get_conn().shm.is_valid()) {
             TORCH_CHECK(false, "Shared Memory invalid during allocation.");
        }
        void* ptr = static_cast<char*>(get_conn().shm.buffer) + offset;
        alloc_map[ptr] = aligned_n;

        // Detailed log kept for safety, but reduced verbosity elsewhere
        // std::cout << "[x_tpu Alloc] Bytes: " << n << ", Offset: " << offset << std::endl;

        return ptr;
    }

    void free(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex);
        if (alloc_map.count(ptr)) {
            size_t size = alloc_map[ptr];
            if (get_conn().shm.is_valid()) {
                char* base = static_cast<char*>(get_conn().shm.buffer);
                uint64_t offset = static_cast<char*>(ptr) - base;
                freed_blocks.push_back({offset, size});
            }
            alloc_map.erase(ptr);
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
        std::memcpy(dest, src, count);
    }
};

static NPUAllocator global_npu_allocator;

// ==========================================
// 3. Helpers
// ==========================================

uint64_t get_offset(const at::Tensor& t) {
    void* ptr = t.data_ptr();
    if (!ptr) {
        // Can be null if size is 0 or undefined
        return 0;
    }
    char* base = static_cast<char*>(get_conn().shm.buffer);
    return static_cast<char*>(ptr) - base;
}

// ==========================================
// 4. Kernel Implementations
// ==========================================

at::Tensor npu_empty(at::IntArrayRef size, std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout, std::optional<at::Device> device, std::optional<bool> pin_memory, std::optional<at::MemoryFormat> memory_format) {

    int64_t nelement = 1;
    for (auto s : size) nelement *= s;
    size_t bytes = nelement * sizeof(float);

    auto data_ptr = global_npu_allocator.allocate(bytes);

    // Create StorageImpl using c10::Storage wrapper to ensure proper lifecycle
    auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        bytes,
        std::move(data_ptr),
        &global_npu_allocator,
        true
    );

    // Create TensorImpl
    auto tensor = at::detail::make_tensor<c10::TensorImpl>(
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        c10::scalarTypeToTypeMeta(dtype.value_or(at::kFloat)),
        c10::Device(c10::DeviceType::PrivateUse1, 0)
    );

    // Link Storage
    tensor.unsafeGetTensorImpl()->set_storage_keep_dtype(std::move(storage_impl));

    // Set Sizes
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);

    return tensor;
}

at::Tensor npu_empty_strided(at::IntArrayRef size, at::IntArrayRef stride, std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout, std::optional<at::Device> device, std::optional<bool> pin_memory) {
    // For now, we assume contiguous behavior for PrivateUse1 simple simulation
    // Ideally we should verify strides match contiguous layout or support strides in allocator
    return npu_empty(size, dtype, layout, device, pin_memory, std::nullopt);
}

// Native View Implementation (Metadata Alias)
at::Tensor npu_view(const at::Tensor& self, at::IntArrayRef size) {
    // Infer size (handle -1)
    auto inferred_size = at::infer_size(size, self.numel());

    // Create an alias
    auto alias = at::detail::make_tensor<c10::TensorImpl>(
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        self.dtype(),
        self.device()
    );

    // Share storage
    alias.unsafeGetTensorImpl()->set_storage_keep_dtype(self.storage());
    alias.unsafeGetTensorImpl()->set_storage_offset(self.storage_offset());

    // Set resolved size
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
        void* dst_ptr = dst.data_ptr();

        if (dst_ptr == nullptr || src_ptr == nullptr) {
             // If we reach here with valid nbytes > 0, it is fatal
             std::cerr << "FATAL: Null pointer in H2D copy" << std::endl;
             return dst;
        }

        std::memcpy(dst_ptr, src_ptr, nbytes);
        get_conn().submit_command(OP_H2D_COPY, get_offset(dst), 0, 0, 0, 0, 0);

    } else if (src_is_npu && !dst_is_npu) {
        // D2H
        get_conn().submit_command(OP_D2H_COPY, get_offset(self), 0, 0, 0, 0, 0);

        void* src_ptr = self.data_ptr();
        void* dst_ptr = dst.data_ptr();

        if (dst_ptr == nullptr || src_ptr == nullptr) {
             std::cerr << "FATAL: Null pointer in D2H copy" << std::endl;
             return dst;
        }

        std::memcpy(dst_ptr, src_ptr, nbytes);

    } else if (src_is_npu && dst_is_npu) {
         // D2D
         std::memcpy(dst.data_ptr(), self.data_ptr(), self.nbytes());
    }

    return dst;
}

at::Tensor npu_copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
    return npu_copy_from(self, dst, false);
}

at::Tensor npu_add(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    auto out = at::empty_like(self);
    get_conn().submit_command(OP_COMPUTE_ADD,
                              get_offset(self), get_offset(other), get_offset(out),
                              self.numel(), 0, 0);
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
// 5. Registration
// ==========================================

void npu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    // Only log if verbose debug needed, otherwise it clogs output during standard fallbacks like view/print
    // std::cout << "[x_tpu Warning] Operator " << op.schema().operator_name() << " is not implemented. Falling back to CPU." << std::endl;

    auto& arguments = *stack;
    for (size_t i = 0; i < arguments.size(); ++i) {
        if (arguments[i].isTensor()) {
            at::Tensor t = arguments[i].toTensor();
            if (t.defined() && t.device().type() == c10::DeviceType::PrivateUse1) {
                arguments[i] = t.cpu();
            }
        }
    }

    op.callBoxed(stack);

    for (size_t i = 0; i < arguments.size(); ++i) {
        if (arguments[i].isTensor()) {
            at::Tensor t = arguments[i].toTensor();
            if (t.defined() && t.device().is_cpu()) {
                arguments[i] = t.to(c10::Device(c10::DeviceType::PrivateUse1, 0));
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
    m.impl("view", &npu_view); // Native view
    m.impl("_copy_from", &npu_copy_from);
    m.impl("_copy_from_and_resize", &npu_copy_from_and_resize);

    m.impl("add.Tensor", static_cast<at::Tensor (*)(const at::Tensor&, const at::Tensor&, const at::Scalar&)>(&npu_add));
    m.impl("mul.Tensor", static_cast<at::Tensor (*)(const at::Tensor&, const at::Tensor&)>(&npu_mul));
    m.impl("mm", static_cast<at::Tensor (*)(const at::Tensor&, const at::Tensor&)>(&npu_mm));
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&npu_fallback>());
}
