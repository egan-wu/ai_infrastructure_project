#pragma once
#include <torch/extension.h>

// Initialize the extension (allocator, backend registration)
void init_x_tpu_extension();

// Legacy / Debugging Functions
void h2d_copy(torch::Tensor src, uint64_t offset);
void d2h_copy(torch::Tensor dst, uint64_t offset);
void npu_add_legacy(uint64_t src_offset, uint64_t dst_offset, uint32_t size, float scalar);
void npu_exit();
