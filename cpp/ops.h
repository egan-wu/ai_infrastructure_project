#pragma once
#include <torch/extension.h>
#include <string>

// Initialize the extension (allocator, backend registration)
void init_x_tpu_extension();

// Manual Driver API
void npu_ioctl(const std::string& op_code, uintptr_t src1, uintptr_t src2, uintptr_t dst, uint32_t size);
void npu_wait();

// Legacy / Debugging Functions
void h2d_copy(torch::Tensor src, uint64_t offset);
void d2h_copy(torch::Tensor dst, uint64_t offset);
void npu_add_legacy(uint64_t src_offset, uint64_t dst_offset, uint32_t size, float scalar);
void npu_exit();
