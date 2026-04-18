#pragma once

#include <memory>

#include <arrow/api.h>

namespace argiv {

// Compute Greeks from pre-computed IV (no Brent solve).
//
// Input table must contain: option_type (int32, 1=call/-1=put),
// spot, strike, expiry, rate, dividend_yield, iv (all float64).
//
// Returns the input table with additional columns:
// delta, gamma, vega, theta, rho (all float64).
std::shared_ptr<arrow::Table> compute_greeks_from_iv_table(
    const std::shared_ptr<arrow::Table>& input);

}  // namespace argiv
