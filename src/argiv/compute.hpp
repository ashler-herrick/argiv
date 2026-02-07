#pragma once

#include <memory>
#include <arrow/api.h>

namespace argiv {

// Takes a table with columns: option_type (int: 1=call, -1=put),
// spot, strike, expiry (years), rate, dividend_yield, market_price (all double).
// Returns a new table with the input columns plus: iv, delta, gamma, vega, theta, rho.
std::shared_ptr<arrow::Table> compute_greeks_table(
    const std::shared_ptr<arrow::Table>& input);

}  // namespace argiv
