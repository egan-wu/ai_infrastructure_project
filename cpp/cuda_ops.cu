#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Simple kernel for weighted sum
__global__ void weighted_sum_kernel(float* out, const float* a, const float* b, float alpha, float beta, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = alpha * a[idx] + beta * b[idx];
    }
}

torch::Tensor weighted_sum_cuda(torch::Tensor a, torch::Tensor b, float alpha, float beta) {
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "Only float32 supported for this example");
    auto result = torch::empty_like(a);

    int size = a.numel();
    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;

    weighted_sum_kernel<<<blocks, threads>>>(
        result.data_ptr<float>(),
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        alpha,
        beta,
        size
    );

    return result;
}
