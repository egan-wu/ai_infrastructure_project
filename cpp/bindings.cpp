#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &init_x_tpu_extension, "Initialize X_NPU backend");

    // Explicit Memory API
    m.def("npu_malloc", &npu_malloc, "Allocate memory on NPU");
    m.def("npu_free", &npu_free, "Free memory on NPU");
    m.def("npu_h2d", &npu_h2d, "Host to Device Copy");
    m.def("npu_d2h", &npu_d2h, "Device to Host Copy");
    m.def("npu_compute", &npu_compute, "Dispatch Compute Op");
}
