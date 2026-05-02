#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

#include "argiv/arrow_interop.hpp"
#include "argiv/compute.hpp"
#include "argiv/core.hpp"
#include "argiv/greeks_from_iv.hpp"
#include "argiv/surface.hpp"

namespace py = pybind11;

namespace {
argiv::IVSolver parse_solver(const std::string& s) {
    if (s == "numerical") return argiv::IVSolver::Numerical;
    if (s == "schadner") return argiv::IVSolver::Schadner;
    if (s == "lookup") return argiv::IVSolver::Lookup;
    throw std::invalid_argument(
        "iv_solver must be 'numerical', 'schadner', or 'lookup', got '" + s + "'");
}
}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "argiv: high-performance options Greeks via QuantLib + Arrow";

    m.def(
        "compute_greeks",
        [](py::object input_table, std::string iv_solver) -> py::object {
            auto solver = parse_solver(iv_solver);
            auto table = argiv::import_table(input_table);
            std::shared_ptr<arrow::Table> result;
            {
                py::gil_scoped_release release;
                result = argiv::compute_greeks_table(table, solver);
            }
            return argiv::export_table(result);
        },
        py::arg("table"), py::arg("iv_solver") = std::string("numerical"),
        R"(Compute implied volatility and Greeks for a table of options.

        Parameters
        ----------
        table : pyarrow.Table
            Must contain columns: option_type (int32, 1=call/-1=put),
            spot, strike, expiry, rate, dividend_yield, market_price (all float64).
            Optional: bid_price, ask_price (float64) for bid/ask IV bounds.

        Returns
        -------
        pyarrow.Table
            Input columns plus: iv, delta, gamma, vega, theta, rho.
            If bid_price and ask_price are present: also iv_bid, iv_ask.
        )");

    m.def(
        "compute_greeks_from_iv",
        [](py::object input_table) -> py::object {
            auto table = argiv::import_table(input_table);
            std::shared_ptr<arrow::Table> result;
            {
                py::gil_scoped_release release;
                result = argiv::compute_greeks_from_iv_table(table);
            }
            return argiv::export_table(result);
        },
        py::arg("table"),
        R"(Compute Greeks from pre-computed implied volatility.

        Parameters
        ----------
        table : pyarrow.Table
            Must contain columns: option_type (int32, 1=call/-1=put),
            spot, strike, expiry, rate, dividend_yield, iv (all float64).

        Returns
        -------
        pyarrow.Table
            Input columns plus: delta, gamma, vega, theta, rho.
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
        R"(Fit a vol surface using OTM options with a shared ATM anchor.

        Parameters
        ----------
        table : pyarrow.Table
            Must contain columns: iv (float64), delta (float64),
            option_type (int32, 1=call/-1=put),
            timestamp (timestamp), expiration (date32).
            Optional: spot (float64), strike (float64) for log_moneyness.
            Optional: iv_bid, iv_ask (float64) for bid/ask IV surface bounds.
        delta_pillars : list of float, optional
            Wing delta percentages, must be < 50 (default: [5,10,...,45]).
            ATM (delta=0.50) is always computed automatically.

        Returns
        -------
        pyarrow.Table
            One row per (timestamp, expiration, delta) with columns:
            timestamp, expiration, delta (signed), iv, log_moneyness.
            If iv_bid and iv_ask are present: also iv_bid, iv_ask.
        )");

    m.def(
        "compute_fit_vol_surface",
        [](py::object input_table, py::object delta_pillars_obj,
           std::string iv_solver) -> py::object {
            auto solver = parse_solver(iv_solver);
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
                auto enriched = argiv::compute_greeks_table(table, solver);
                result = argiv::fit_vol_surface_table(enriched, config);
            }
            return argiv::export_table(result);
        },
        py::arg("table"), py::arg("delta_pillars") = py::none(),
        py::arg("iv_solver") = std::string("numerical"),
        R"(Compute Greeks and fit a vol surface in one step.

        Parameters
        ----------
        table : pyarrow.Table
            Must contain columns: option_type (int32, 1=call/-1=put),
            spot, strike, expiry, rate, dividend_yield, market_price (all float64),
            timestamp (timestamp), expiration (date32).
            Optional: bid_price, ask_price (float64) for bid/ask IV bounds.
        delta_pillars : list of float, optional
            Wing delta percentages, must be < 50 (default: [5,10,...,45]).
            ATM (delta=0.50) is always computed automatically.

        Returns
        -------
        pyarrow.Table
            One row per (timestamp, expiration, delta) with columns:
            timestamp, expiration, delta (signed), iv, log_moneyness.
            If bid_price and ask_price are present: also iv_bid, iv_ask.
        )");
}
