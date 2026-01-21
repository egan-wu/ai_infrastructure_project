#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("weighted_sum", &weighted_sum, "Weighted element-wise sum");
}
