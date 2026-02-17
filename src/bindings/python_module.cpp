#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "argiv/arrow_interop.hpp"
#include "argiv/compute.hpp"
#include "argiv/surface.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "argiv: high-performance options Greeks via QuantLib + Arrow";

    m.def(
        "compute_greeks",
        [](py::object input_table) -> py::object {
            auto table = argiv::import_table(input_table);
            std::shared_ptr<arrow::Table> result;
            {
                py::gil_scoped_release release;
                result = argiv::compute_greeks_table(table);
            }
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

    m.def(
        "fit_vol_surface",
        [](py::object input_table, py::object delta_pillars_obj) -> py::object {
            auto table = argiv::import_table(input_table);
            argiv::SurfaceConfig config;
            if (!delta_pillars_obj.is_none()) {
                config.delta_pillars.clear();
                for (auto item : delta_pillars_obj)
                    config.delta_pillars.push_back(item.cast<double>());
            }
            std::shared_ptr<arrow::Table> result;
            {
                py::gil_scoped_release release;
                result = argiv::fit_vol_surface_table(table, config);
            }
            return argiv::export_table(result);
        },
        py::arg("table"), py::arg("delta_pillars") = py::none(),
        R"(Fit a vol surface via delta-space cubic spline interpolation.

        Parameters
        ----------
        table : pyarrow.Table
            Must contain columns: iv (float64), delta (float64),
            timestamp (timestamp), expiration (date32).
        delta_pillars : list of float, optional
            Absolute delta percentages for pillar points (default: [5,10,...,50]).

        Returns
        -------
        pyarrow.Table
            One row per (timestamp, expiration) group with iv_pNN and iv_cNN columns.
        )");

    m.def(
        "compute_fit_vol_surface",
        [](py::object input_table, py::object delta_pillars_obj) -> py::object {
            auto table = argiv::import_table(input_table);
            argiv::SurfaceConfig config;
            if (!delta_pillars_obj.is_none()) {
                config.delta_pillars.clear();
                for (auto item : delta_pillars_obj)
                    config.delta_pillars.push_back(item.cast<double>());
            }
            std::shared_ptr<arrow::Table> result;
            {
                py::gil_scoped_release release;
                auto enriched = argiv::compute_greeks_table(table);
                result = argiv::fit_vol_surface_table(enriched, config);
            }
            return argiv::export_table(result);
        },
        py::arg("table"), py::arg("delta_pillars") = py::none(),
        R"(Compute Greeks and fit a vol surface in one step.

        Parameters
        ----------
        table : pyarrow.Table
            Must contain columns: option_type (int32, 1=call/-1=put),
            spot, strike, expiry, rate, dividend_yield, market_price (all float64),
            timestamp (timestamp), expiration (date32).
        delta_pillars : list of float, optional
            Absolute delta percentages for pillar points (default: [5,10,...,50]).

        Returns
        -------
        pyarrow.Table
            One row per (timestamp, expiration) group with iv_pNN and iv_cNN columns.
        )");
}
