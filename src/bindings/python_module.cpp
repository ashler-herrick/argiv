#include <pybind11/pybind11.h>

#include "argiv/arrow_interop.hpp"
#include "argiv/compute.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "argiv: high-performance options Greeks via QuantLib + Arrow";

    m.def(
        "compute_greeks",
        [](py::object input_table) -> py::object {
            // Import pyarrow table -> C++ Arrow table
            auto table = argiv::import_table(input_table);
            // Compute IV + Greeks
            auto result = argiv::compute_greeks_table(table);
            // Export back to pyarrow table
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
