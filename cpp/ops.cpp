#include "ops.h"
#include "npu_device_manager.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <ATen/InferSize.h>

// ==========================================
// 1. Helpers
// ==========================================

// Handle encoding: [DeviceID (32b)] [Offset (32b)]
// This allows a single 64-bit pointer to carry both device and offset info.
inline uint64_t encode_handle(int device_id, uint64_t offset) {
    return (static_cast<uint64_t>(device_id) << 32) | (offset & 0xFFFFFFFF);
}

inline std::pair<int, uint64_t> decode_handle(void* ptr) {
    uint64_t handle = reinterpret_cast<uint64_t>(ptr);
    int dev = handle >> 32;
    uint64_t off = handle & 0xFFFFFFFF;
    return {dev, off};
}

inline std::pair<int, uint64_t> decode_handle_u64(uint64_t handle) {
    int dev = handle >> 32;
    uint64_t off = handle & 0xFFFFFFFF;
    return {dev, off};
}

void* resolve_host_ptr(void* ptr) {
    if (!ptr) return nullptr;
    auto [dev, off] = decode_handle(ptr);
    auto ctx = NPUDeviceManager::get_instance().get_device(dev);
    if (!ctx) return nullptr;
    return static_cast<char*>(ctx->get_base_ptr()) + off;
}

void submit_command(NPUDeviceContext* ctx, OpCode op, uint64_t src1, uint64_t src2, uint64_t dst,
                    uint32_t s1, uint32_t s2, uint32_t s3, float scalar = 0.0f) {
    std::lock_guard<std::mutex> lock(ctx->cmd_mutex);
    NPUControl* ctrl = ctx->get_control();

    int safety = 0;
    while(ctrl->host_ready && safety < 10000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        safety++;
    }
    if (ctrl->host_ready) {
         TORCH_CHECK(false, "NPU Device ", ctx->device_id, " is stuck.");
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
            TORCH_CHECK(false, "NPU Device ", ctx->device_id, " timed out.");
        }
    }
}

// ==========================================
// 2. Allocator
// ==========================================

static void deleteNPU(void* ptr) {
    if (!ptr) return;
    auto [dev, off] = decode_handle(ptr);
    auto ctx = NPUDeviceManager::get_instance().get_device(dev);
    if (ctx) {
        ctx->allocator->free(off);
    }
}

class NPUAllocator : public c10::Allocator {
public:
    c10::DataPtr allocate(size_t n) override {
        // Use the thread-local current device index set by PyTorch or user
        int dev_id = NPUDeviceManager::current_device_index();

        // Auto-init/connect
        auto ctx = NPUDeviceManager::get_instance().get_device(dev_id);
        if (!ctx) {
             NPUDeviceManager::get_instance().init_device(dev_id);
             ctx = NPUDeviceManager::get_instance().get_device(dev_id);
             if (!ctx) TORCH_CHECK(false, "Failed to get NPU device ", dev_id);
        }

        uint64_t offset = ctx->allocator->allocate(n);
        uint64_t handle = encode_handle(dev_id, offset);
        void* ptr = reinterpret_cast<void*>(handle);

        return {ptr, ptr, &deleteNPU, c10::Device(c10::DeviceType::PrivateUse1, dev_id)};
    }

    c10::DeleterFnPtr raw_deleter() const override {
        return &deleteNPU;
    }

    void copy_data(void* dest, const void* src, std::size_t count) const override {
        void* real_dst = resolve_host_ptr(dest);
        void* real_src = resolve_host_ptr(const_cast<void*>(src));

        void* d = real_dst ? real_dst : dest;
        const void* s = real_src ? real_src : src;

        std::memcpy(d, s, count);
    }
};

static NPUAllocator global_npu_allocator;

// ==========================================
// 3. API Implementation
// ==========================================

void init_x_tpu_extension() {
    c10::register_privateuse1_backend("x_tpu");
    c10::SetAllocator(c10::DeviceType::PrivateUse1, &global_npu_allocator);
    std::cout << "[x_tpu] Extension Initialized." << std::endl;
}

void init_device(int device_id) {
    NPUDeviceManager::get_instance().init_device(device_id);
}

uint64_t npu_malloc(int device_id, size_t size) {
    auto ctx = NPUDeviceManager::get_instance().get_device(device_id);
    if (!ctx) {
        NPUDeviceManager::get_instance().init_device(device_id);
        ctx = NPUDeviceManager::get_instance().get_device(device_id);
    }
    uint64_t off = ctx->allocator->allocate(size);
    return encode_handle(device_id, off);
}

void npu_free(int device_id, uint64_t addr) {
    auto [dev, off] = decode_handle_u64(addr);
    auto ctx = NPUDeviceManager::get_instance().get_device(dev); // Use device from handle
    if (ctx) ctx->allocator->free(off);
}

void npu_h2d(int device_id, torch::Tensor src, uint64_t dst_addr) {
    auto [dev, off] = decode_handle_u64(dst_addr);
    TORCH_CHECK(dev == device_id, "Device mismatch in h2d");

    auto ctx = NPUDeviceManager::get_instance().get_device(dev);
    void* host_dst = static_cast<char*>(ctx->get_base_ptr()) + off;

    std::memcpy(host_dst, src.data_ptr(), src.nbytes());
    submit_command(ctx, OP_H2D_COPY, off, 0, 0, 0, 0, 0);
}

void npu_d2h(int device_id, torch::Tensor dst, uint64_t src_addr) {
    auto [dev, off] = decode_handle_u64(src_addr);
    TORCH_CHECK(dev == device_id, "Device mismatch in d2h");

    auto ctx = NPUDeviceManager::get_instance().get_device(dev);
    submit_command(ctx, OP_D2H_COPY, off, 0, 0, 0, 0, 0);

    void* host_src = static_cast<char*>(ctx->get_base_ptr()) + off;
    std::memcpy(dst.data_ptr(), host_src, dst.nbytes());
}

void npu_d2d(int src_dev, uint64_t src_addr, int dst_dev, uint64_t dst_addr, size_t size) {
    auto [s_dev, s_off] = decode_handle_u64(src_addr);
    auto [d_dev, d_off] = decode_handle_u64(dst_addr);

    TORCH_CHECK(s_dev == src_dev, "Source device mismatch");
    TORCH_CHECK(d_dev == dst_dev, "Destination device mismatch");

    auto src_ctx = NPUDeviceManager::get_instance().get_device(src_dev);
    auto dst_ctx = NPUDeviceManager::get_instance().get_device(dst_dev);

    void* p_src = static_cast<char*>(src_ctx->get_base_ptr()) + s_off;
    void* p_dst = static_cast<char*>(dst_ctx->get_base_ptr()) + d_off;

    // Simulate DMA copy
    std::memcpy(p_dst, p_src, size);

    // We do NOT send a command to Daemon for P2P currently (as per instructions)
    // Or we could send a log op? But instructions said "Host driver will act as DMA orchestrator... without Daemon-to-Daemon direct communication"
}

void npu_compute(int device_id, const std::string& op_code, uint64_t src1, uint64_t src2, uint64_t dst, uint32_t size, float scalar) {
    auto ctx = NPUDeviceManager::get_instance().get_device(device_id);

    OpCode code;
    bool is_exit = false;

    if (op_code == "OP_ADD") code = OP_COMPUTE_ADD;
    else if (op_code == "OP_SUB") code = OP_COMPUTE_SUB;
    else if (op_code == "OP_MUL") code = OP_COMPUTE_MUL;
    else if (op_code == "OP_MATMUL") code = OP_COMPUTE_MATMUL; // Caller must handle size/dims packing
    else if (op_code == "OP_H2D_COPY") code = OP_H2D_COPY;
    else if (op_code == "OP_D2H_COPY") code = OP_D2H_COPY;
    else if (op_code == "OP_EXIT") { code = OP_EXIT; is_exit = true; }
    else TORCH_CHECK(false, "Unknown OpCode");

    uint64_t off1 = 0, off2 = 0, off_dst = 0;

    if (!is_exit) {
        // Decode handles to offsets
        auto [d1, o1] = decode_handle_u64(src1);
        auto [d2, o2] = decode_handle_u64(src2);
        auto [d3, o3] = decode_handle_u64(dst);

        TORCH_CHECK(d1 == device_id, "Src1 not on computation device");
        if (src2 != 0) TORCH_CHECK(d2 == device_id, "Src2 not on computation device");
        if (dst != 0) TORCH_CHECK(d3 == device_id, "Dst not on computation device");

        off1 = o1;
        off2 = o2;
        off_dst = o3;
    }

    submit_command(ctx, code, off1, off2, off_dst, size, 0, 0, scalar);
}

// ==========================================
// 4. PyTorch Ops
// ==========================================

at::Tensor npu_empty(at::IntArrayRef size, std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout, std::optional<at::Device> device, std::optional<bool> pin_memory, std::optional<at::MemoryFormat> memory_format) {
    // Set thread-local device index
    if (device.has_value()) {
        NPUDeviceManager::current_device_index() = device->index();
    }

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
        c10::Device(c10::DeviceType::PrivateUse1, NPUDeviceManager::current_device_index())
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

    if (dst_is_npu && !src_is_npu) {
        // H2D
        auto [dev, off] = decode_handle(dst.data_ptr());
        auto ctx = NPUDeviceManager::get_instance().get_device(dev);
        if(!ctx) return dst;

        void* dst_ptr = static_cast<char*>(ctx->get_base_ptr()) + off;
        std::memcpy(dst_ptr, self.data_ptr(), self.nbytes());
        submit_command(ctx, OP_H2D_COPY, off, 0, 0, 0, 0, 0);

    } else if (src_is_npu && !dst_is_npu) {
        // D2H
        auto [dev, off] = decode_handle(self.data_ptr());
        auto ctx = NPUDeviceManager::get_instance().get_device(dev);
        if(!ctx) return dst;

        submit_command(ctx, OP_D2H_COPY, off, 0, 0, 0, 0, 0);
        void* src_ptr = static_cast<char*>(ctx->get_base_ptr()) + off;
        std::memcpy(dst.data_ptr(), src_ptr, self.nbytes());

    } else if (src_is_npu && dst_is_npu) {
        // D2D (internal copy)
        auto [s_dev, s_off] = decode_handle(self.data_ptr());
        auto [d_dev, d_off] = decode_handle(dst.data_ptr());

        // Use Host DMA (memcpy)
        auto s_ctx = NPUDeviceManager::get_instance().get_device(s_dev);
        auto d_ctx = NPUDeviceManager::get_instance().get_device(d_dev);

        void* s_ptr = static_cast<char*>(s_ctx->get_base_ptr()) + s_off;
        void* d_ptr = static_cast<char*>(d_ctx->get_base_ptr()) + d_off;

        std::memcpy(d_ptr, s_ptr, self.nbytes());
    }

    return dst;
}

at::Tensor npu_copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
    return npu_copy_from(self, dst, false);
}

at::Tensor npu_add(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    auto dev = self.device();
    NPUDeviceManager::current_device_index() = dev.index(); // Ensure context
    auto out = at::empty_like(self);

    auto [d1, off1] = decode_handle(self.data_ptr());
    auto [d2, off2] = decode_handle(other.data_ptr());
    auto [d3, off_out] = decode_handle(out.data_ptr());

    // Assume all on same device
    auto ctx = NPUDeviceManager::get_instance().get_device(d1);
    submit_command(ctx, OP_COMPUTE_ADD, off1, off2, off_out, self.numel(), 0, 0, alpha.to<float>());
    return out;
}

at::Tensor npu_mul(const at::Tensor& self, const at::Tensor& other) {
    auto dev = self.device();
    NPUDeviceManager::current_device_index() = dev.index();
    auto out = at::empty_like(self);

    auto [d1, off1] = decode_handle(self.data_ptr());
    auto [d2, off2] = decode_handle(other.data_ptr());
    auto [d3, off_out] = decode_handle(out.data_ptr());

    auto ctx = NPUDeviceManager::get_instance().get_device(d1);
    submit_command(ctx, OP_COMPUTE_MUL, off1, off2, off_out, self.numel(), 0, 0);
    return out;
}

at::Tensor npu_mm(const at::Tensor& self, const at::Tensor& other) {
    int64_t M = self.size(0);
    int64_t K = self.size(1);
    int64_t N = other.size(1);

    auto dev = self.device();
    NPUDeviceManager::current_device_index() = dev.index();
    auto out = at::empty({M, N}, self.options());

    auto [d1, off1] = decode_handle(self.data_ptr());
    auto [d2, off2] = decode_handle(other.data_ptr());
    auto [d3, off_out] = decode_handle(out.data_ptr());

    auto ctx = NPUDeviceManager::get_instance().get_device(d1);
    submit_command(ctx, OP_COMPUTE_MATMUL, off1, off2, off_out, M, K, N);
    return out;
}

// ==========================================
// 5. Legacy
// ==========================================
void npu_ioctl(const std::string& op_code, uintptr_t src1, uintptr_t src2, uintptr_t dst, uint32_t size) {
    // Legacy defaults to Device 0
    auto ctx = NPUDeviceManager::get_instance().get_device(0);
    if (!ctx) {
        NPUDeviceManager::get_instance().init_device(0);
    }

    // Re-use logic: Pass handles directly (npu_compute decodes them)
    npu_compute(0, op_code, src1, src2, dst, size, 0);
}

void npu_wait() {
    // No-op or check device 0
}

void h2d_copy(torch::Tensor src, uint64_t offset) {
    npu_h2d(0, src, encode_handle(0, offset));
}

void d2h_copy(torch::Tensor dst, uint64_t offset) {
    npu_d2h(0, dst, encode_handle(0, offset));
}

void npu_add_legacy(uint64_t src, uint64_t dst, uint32_t size, float scalar) {
     npu_compute(0, "OP_ADD", encode_handle(0, src), 0, encode_handle(0, dst), size, scalar);
}

void npu_exit() {
     auto ctx = NPUDeviceManager::get_instance().get_device(0);
     if(ctx) submit_command(ctx, OP_EXIT, 0, 0, 0, 0, 0, 0);
}

// ==========================================
// 6. Registration
// ==========================================

void npu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    // Fallback logic remains similar, but need to ensure copies use correct device
    // Simplify: Just print warning or delegate to generic copy
    // For now, keep as is but aware of resolve_host_ptr
    // Implementation of fallback relies on npu_copy_from which handles the magic

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
    // Copy back logic omitted for brevity in fallback, usually not needed for simple ops
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
