#pragma once

#include <memory>
#include <vector>

#include <arrow/api.h>

namespace argiv {

struct SurfaceConfig {
    // Wing delta percentages (must be < 50). ATM (50-delta) is always computed
    // as the average of the nearest OTM put and call IVs and output as iv_50.
    std::vector<double> delta_pillars = {5, 10, 15, 20, 25, 30, 35, 40, 45};
};

// Fit a vol surface using OTM options only with a shared ATM anchor.
//
// Input table must contain: iv (float64), delta (float64),
// option_type (int32, 1=call/-1=put), timestamp (timestamp), expiration (date32).
// Optional: spot (float64), strike (float64) — used to compute log_moneyness.
//
// Returns an unpivoted table with one row per (timestamp, expiration, delta)
// combination. Columns: timestamp, expiration, delta (float64, signed:
// negative for puts, positive for calls, 0.50 for ATM), iv (float64),
// log_moneyness (float64, log(K/S) — null if spot/strike not in input).
std::shared_ptr<arrow::Table> fit_vol_surface_table(
    const std::shared_ptr<arrow::Table>& input,
    const SurfaceConfig& config = SurfaceConfig{});

}  // namespace argiv
