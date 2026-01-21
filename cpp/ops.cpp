#include "ops.h"
#include <iostream>

// CPU Implementation using TensorAccessor
// The task requires demonstrating receiving a torch::Tensor and accessing its data efficiently using torch::TensorAccessor.
torch::Tensor weighted_sum_cpu(torch::Tensor a, torch::Tensor b, float alpha, float beta) {
    // Check inputs
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensors must have the same size");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "Tensors must have the same dtype");

    // Create output tensor
    auto result = torch::empty_like(a);

    // For this example, we support float32 and 2D tensors to demonstrate TensorAccessor<float, 2>
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "Only float32 supported for this example");
    TORCH_CHECK(a.dim() == 2, "This example supports 2D tensors");

    // Get accessors
    // TensorAccessor gives efficient element access avoiding overhead of full tensor operations in a loop
    auto a_acc = a.accessor<float, 2>();
    auto b_acc = b.accessor<float, 2>();
    auto res_acc = result.accessor<float, 2>();

    // Iterate and compute
    for(int i = 0; i < a.size(0); i++) {
        for(int j = 0; j < a.size(1); j++) {
            res_acc[i][j] = alpha * a_acc[i][j] + beta * b_acc[i][j];
        }
    }

    return result;
}

// Dispatcher
torch::Tensor weighted_sum(torch::Tensor a, torch::Tensor b, float alpha, float beta) {
    TORCH_CHECK(a.device() == b.device(), "Tensors must be on the same device");
    if (a.device().is_cuda()) {
        #ifdef WITH_CUDA
        return weighted_sum_cuda(a, b, alpha, beta);
        #else
        TORCH_CHECK(false, "CUDA support was not compiled in.");
        #endif
    } else {
        return weighted_sum_cpu(a, b, alpha, beta);
    }
}
