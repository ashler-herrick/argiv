#pragma once

#include <memory>

#include <arrow/api.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace argiv {

// Import an Arrow table from a Python object supporting the PyCapsule
// (__arrow_c_stream__) protocol.
std::shared_ptr<arrow::Table> import_table(py::object obj);

// Export an Arrow table back to Python as a pyarrow.Table via PyCapsule.
py::object export_table(const std::shared_ptr<arrow::Table>& table);

}  // namespace argiv
