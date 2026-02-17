#pragma once

#include <memory>
#include <vector>

#include <arrow/api.h>

namespace argiv {

struct SurfaceConfig {
    // Absolute delta percentages for pillar points (e.g., 5 means 5-delta).
    // Generates both put (negative delta) and call (positive delta) columns.
    std::vector<double> delta_pillars = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50};
};

// Fit a vol surface via delta-space cubic spline interpolation.
//
// Input table must contain: iv (float64), delta (float64),
// timestamp (timestamp), expiration (date32).
//
// Returns a table with one row per (timestamp, expiration) group, containing
// the grouping columns plus iv_pNN and iv_cNN columns for each delta pillar.
std::shared_ptr<arrow::Table> fit_vol_surface_table(
    const std::shared_ptr<arrow::Table>& input,
    const SurfaceConfig& config = SurfaceConfig{});

}  // namespace argiv
