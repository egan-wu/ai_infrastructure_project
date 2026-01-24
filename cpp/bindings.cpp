#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &init_x_tpu_extension, "Initialize X_TPU backend");

    // Multi-Device Management
    m.def("init_device", &init_device, "Initialize NPU Device");

    // Memory
    m.def("npu_malloc", &npu_malloc, "Allocate memory on NPU");
    m.def("npu_free", &npu_free, "Free memory on NPU");

    // Data Movement
    m.def("npu_h2d", &npu_h2d, "Host to Device Copy");
    m.def("npu_d2h", &npu_d2h, "Device to Host Copy");
    m.def("npu_d2d", &npu_d2d, "Device to Device Copy");

    // Execution
    m.def("npu_compute", &npu_compute, "Execute command on NPU");

    // Legacy
    m.def("npu_ioctl", &npu_ioctl, "Send command to NPU Driver");
    m.def("npu_wait", &npu_wait, "Wait for NPU idle");
    m.def("h2d_copy", &h2d_copy, "Host to Device Copy (Legacy)");
    m.def("d2h_copy", &d2h_copy, "Device to Host Copy (Legacy)");
    m.def("npu_add", &npu_add_legacy, "NPU Add Operation (Legacy)");
    m.def("npu_exit", &npu_exit, "Send Exit signal to NPU");
}
