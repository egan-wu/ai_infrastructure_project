#pragma once
#include <torch/extension.h>

// Forward declarations
torch::Tensor weighted_sum_cpu(torch::Tensor a, torch::Tensor b, float alpha, float beta);
torch::Tensor weighted_sum_cuda(torch::Tensor a, torch::Tensor b, float alpha, float beta);

// Main entry point that dispatches to CPU or CUDA
torch::Tensor weighted_sum(torch::Tensor a, torch::Tensor b, float alpha, float beta);

// NPU IPC Call
torch::Tensor call_npu(torch::Tensor input, float scalar, uint64_t offset);
