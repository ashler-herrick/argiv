#include "argiv/arrow_interop.hpp"

#include <cstring>

#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>

namespace argiv {

std::shared_ptr<arrow::Table> import_table(py::object obj) {
    // Call __arrow_c_stream__ on the Python object to get a PyCapsule
    py::object capsule = obj.attr("__arrow_c_stream__")();
    auto* c_stream =
        reinterpret_cast<ArrowArrayStream*>(PyCapsule_GetPointer(
            capsule.ptr(), "arrow_array_stream"));
    if (!c_stream) {
        throw std::runtime_error("Failed to extract ArrowArrayStream from PyCapsule");
    }

    // Import into C++ RecordBatchReader, then read all into a Table
    auto reader_result = arrow::ImportRecordBatchReader(c_stream);
    if (!reader_result.ok()) {
        throw std::runtime_error("ImportRecordBatchReader failed: " +
                                 reader_result.status().ToString());
    }
    auto reader = reader_result.MoveValueUnsafe();

    auto table_result = reader->ToTable();
    if (!table_result.ok()) {
        throw std::runtime_error("ToTable failed: " +
                                 table_result.status().ToString());
    }
    return table_result.MoveValueUnsafe();
}

py::object export_table(const std::shared_ptr<arrow::Table>& table) {
    // Create a RecordBatchReader from the table
    auto reader = std::make_shared<arrow::TableBatchReader>(table);

    // Allocate an ArrowArrayStream on the heap and export into it
    auto* c_stream = new ArrowArrayStream;
    std::memset(c_stream, 0, sizeof(ArrowArrayStream));
    auto status = arrow::ExportRecordBatchReader(reader, c_stream);
    if (!status.ok()) {
        delete c_stream;
        throw std::runtime_error("ExportRecordBatchReader failed: " +
                                 status.ToString());
    }

    // Use pyarrow's _import_from_c with the raw address.
    // PyArrow takes ownership of the stream and will call release().
    py::module_ pa = py::module_::import("pyarrow");
    auto addr = reinterpret_cast<uintptr_t>(c_stream);
    py::object rb_reader =
        pa.attr("RecordBatchReader").attr("_import_from_c")(addr);
    py::object result = rb_reader.attr("read_all")();

    // Stream has been consumed by pyarrow; just free the struct memory.
    delete c_stream;

    return result;
}

}  // namespace argiv
