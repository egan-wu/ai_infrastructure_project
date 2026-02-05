#pragma once
#include <torch/extension.h>
#include <string>

// Initialize the extension (allocator, backend registration)
void init_x_tpu_extension();

// Multi-Device Management
void init_device(int device_id);

// Memory Management
uint64_t npu_malloc(int device_id, size_t size);
void npu_free(int device_id, uint64_t addr);

// Data Movement
void npu_h2d(int device_id, torch::Tensor src, uint64_t dst_addr);
void npu_d2h(int device_id, torch::Tensor dst, uint64_t src_addr);
void npu_d2d(int src_dev, uint64_t src_addr, int dst_dev, uint64_t dst_addr, size_t size);

// Execution
void npu_compute(int device_id, const std::string& op_code, uint64_t src1, uint64_t src2, uint64_t dst, uint32_t size, float scalar);

// Legacy / Backward Compatibility (Defaults to Device 0)
void npu_ioctl(const std::string& op_code, uintptr_t src1, uintptr_t src2, uintptr_t dst, uint32_t size);
void npu_wait();
void h2d_copy(torch::Tensor src, uint64_t offset);
void d2h_copy(torch::Tensor dst, uint64_t offset);
void npu_add_legacy(uint64_t src_offset, uint64_t dst_offset, uint32_t size, float scalar);
void npu_exit();
