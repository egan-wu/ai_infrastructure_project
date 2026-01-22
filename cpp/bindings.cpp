#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &init_x_tpu_extension, "Initialize X_TPU backend");
    m.def("h2d_copy", &h2d_copy, "Host to Device Copy");
    m.def("d2h_copy", &d2h_copy, "Device to Host Copy");
    // Bind legacy function for backward compatibility or explicit low-level testing
    m.def("npu_add", &npu_add_legacy, "NPU Add Operation (Legacy)");
    m.def("npu_exit", &npu_exit, "Send Exit signal to NPU");
}
