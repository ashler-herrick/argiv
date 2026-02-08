#include <pybind11/pybind11.h>

#include "argiv/arrow_interop.hpp"
#include "argiv/compute.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "argiv: high-performance options Greeks via QuantLib + Arrow";

    m.def(
        "compute_greeks",
        [](py::object input_table) -> py::object {
            // Import pyarrow table -> C++ Arrow table (needs GIL for PyArrow)
            auto table = argiv::import_table(input_table);
            const int64_t n = table->num_rows();
            if (n==0){
                // Handle empty table: return immediately with output columns appended.
                auto empty_arr = std::make_shared<arrow::DoubleArray>(0, nullptr);
                auto empty_chunked = std::make_shared<arrow::ChunkedArray>(empty_arr);
                auto result = table;
                for (const auto& name : {"iv", "delta", "gamma", "vega", "theta", "rho"}) {
                    result = *result->AddColumn(result->num_columns(),
                                                arrow::field(name, arrow::float64()),
                                                empty_chunked);
                }
                return argiv::export_table(result);
            }
            // Release GIL for the CPU-bound computation so OpenMP threads can run
            std::shared_ptr<arrow::Table> result;
            {
                py::gil_scoped_release release;
                result = argiv::compute_greeks_table(table);
            }

            // Export back to pyarrow table (needs GIL for PyArrow)
            return argiv::export_table(result);
        },
        py::arg("table"),
        R"(Compute implied volatility and Greeks for a table of options.

        Parameters
        ----------
        table : pyarrow.Table
            Must contain columns: option_type (int32, 1=call/-1=put),
            spot, strike, expiry, rate, dividend_yield, market_price (all float64).

        Returns
        -------
        pyarrow.Table
            Input columns plus: iv, delta, gamma, vega, theta, rho.
        )");
}
