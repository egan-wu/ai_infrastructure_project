#pragma once
#include <torch/extension.h>

torch::Tensor weighted_sum(torch::Tensor a, torch::Tensor b, float alpha, float beta);

#ifdef WITH_CUDA
torch::Tensor weighted_sum_cuda(torch::Tensor a, torch::Tensor b, float alpha, float beta);
#endif

// NPU Driver Simulation Functions
void h2d_copy(torch::Tensor src, uint64_t offset);
void d2h_copy(torch::Tensor dst, uint64_t offset);
void npu_add(uint64_t src_offset, uint64_t dst_offset, uint32_t size, float scalar);
void npu_exit(); // Helper to send exit command
