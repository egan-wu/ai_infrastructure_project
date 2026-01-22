#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("weighted_sum", &weighted_sum, "Weighted element-wise sum");
    m.def("h2d_copy", &h2d_copy, "Host to Device Copy");
    m.def("d2h_copy", &d2h_copy, "Device to Host Copy");
    m.def("npu_add", &npu_add, "NPU Add Operation");
    m.def("npu_exit", &npu_exit, "Send Exit signal to NPU");
}
