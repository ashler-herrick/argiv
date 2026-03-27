#include "argiv/surface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include <arrow/api.h>
#include <arrow/table.h>
#include <ql/math/interpolations/cubicinterpolation.hpp>
#include <ql/math/interpolations/linearinterpolation.hpp>

namespace argiv {

namespace {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

const double* get_double_col(const std::shared_ptr<arrow::Table>& table,
                             const std::string& name) {
    auto col = table->GetColumnByName(name);
    if (!col)
        throw std::runtime_error("Missing column: " + name);
    auto arr = std::static_pointer_cast<arrow::DoubleArray>(col->chunk(0));
    return arr->raw_values();
}

const int32_t* get_int32_col(const std::shared_ptr<arrow::Table>& table,
                             const std::string& name) {
    auto col = table->GetColumnByName(name);
    if (!col)
        throw std::runtime_error("Missing column: " + name);
    auto arr = std::static_pointer_cast<arrow::Int32Array>(col->chunk(0));
    return arr->raw_values();
}

const int64_t* get_int64_col(const std::shared_ptr<arrow::Table>& table,
                             const std::string& name) {
    auto col = table->GetColumnByName(name);
    if (!col)
        throw std::runtime_error("Missing column: " + name);
    auto arr = std::static_pointer_cast<arrow::Int64Array>(col->chunk(0));
    return arr->raw_values();
}

// Build output column names: iv_p45, iv_p40, ..., iv_p5, iv_50, iv_c5, ..., iv_c45
std::vector<std::string> build_column_names(
    const std::vector<double>& pillars) {
    std::vector<std::string> names;
    names.reserve(pillars.size() * 2 + 1);
    // Put side: descending magnitude
    for (auto it = pillars.rbegin(); it != pillars.rend(); ++it) {
        int p = static_cast<int>(*it);
        names.push_back("iv_p" + std::to_string(p));
    }
    // ATM
    names.push_back("iv_50");
    // Call side: ascending magnitude
    for (double p : pillars) {
        int pi = static_cast<int>(p);
        names.push_back("iv_c" + std::to_string(pi));
    }
    return names;
}

// Build wing pillar delta values in absolute delta space (e.g., 0.05, 0.10, ..., 0.45)
std::vector<double> build_wing_deltas(const std::vector<double>& pillars) {
    std::vector<double> deltas;
    deltas.reserve(pillars.size());
    for (double p : pillars) {
        deltas.push_back(p / 100.0);
    }
    return deltas;
}

// Deduplicate (x, y) pairs sorted by x: average y values for identical x.
void dedup_sorted(std::vector<double>& x, std::vector<double>& y,
                  const std::vector<std::pair<double, double>>& points) {
    x.clear();
    y.clear();
    x.reserve(points.size());
    y.reserve(points.size());
    size_t i = 0;
    while (i < points.size()) {
        double d = points[i].first;
        double sum_iv = 0.0;
        size_t count = 0;
        while (i < points.size() && points[i].first == d) {
            sum_iv += points[i].second;
            ++count;
            ++i;
        }
        x.push_back(d);
        y.push_back(sum_iv / count);
    }
}

// Evaluate a spline (cubic or linear depending on point count) at target deltas.
// Only evaluates within the spline's domain [x.front(), x.back()].
void evaluate_spline(const std::vector<double>& x,
                     const std::vector<double>& y,
                     const std::vector<double>& targets,
                     std::vector<double>& out) {
    if (x.size() < 2) {
        // Not enough points — leave as NaN
        return;
    }

    if (x.size() >= 4) {
        QuantLib::CubicNaturalSpline spline(x.begin(), x.end(), y.begin());
        for (size_t i = 0; i < targets.size(); ++i) {
            double t = targets[i];
            if (t >= x.front() && t <= x.back()) {
                double val = spline(t);
                out[i] = val > 0.0 ? val : NaN;
            }
        }
    } else {
        QuantLib::LinearInterpolation interp(x.begin(), x.end(), y.begin());
        for (size_t i = 0; i < targets.size(); ++i) {
            double t = targets[i];
            if (t >= x.front() && t <= x.back()) {
                double val = interp(t);
                out[i] = val > 0.0 ? val : NaN;
            }
        }
    }
}

}  // namespace

std::shared_ptr<arrow::Table> fit_vol_surface_table(
    const std::shared_ptr<arrow::Table>& input,
    const SurfaceConfig& config) {

    // Validate pillars are all < 50
    for (double p : config.delta_pillars) {
        if (p >= 50.0)
            throw std::runtime_error(
                "delta_pillars must be < 50 (ATM is computed automatically)");
    }

    // Combine chunks for raw pointer access
    auto combined_result = input->CombineChunks();
    if (!combined_result.ok())
        throw std::runtime_error("Failed to combine chunks: " +
                                 combined_result.status().ToString());
    auto table = combined_result.MoveValueUnsafe();
    const int64_t n = table->num_rows();

    // Determine pillar layout
    auto sorted_pillars = config.delta_pillars;
    std::sort(sorted_pillars.begin(), sorted_pillars.end());
    auto col_names = build_column_names(sorted_pillars);
    auto wing_deltas = build_wing_deltas(sorted_pillars);
    const size_t num_wing = wing_deltas.size();
    // Total output IV columns: put wings + iv_50 + call wings
    const size_t num_iv_cols = num_wing * 2 + 1;

    // Preserve timestamp type from input schema
    auto ts_field = table->schema()->GetFieldByName("timestamp");
    if (!ts_field)
        throw std::runtime_error("Missing column: timestamp");
    auto ts_type = ts_field->type();

    // Handle empty input
    if (n == 0) {
        arrow::FieldVector fields;
        fields.push_back(arrow::field("timestamp", ts_type));
        fields.push_back(arrow::field("expiration", arrow::date32()));
        for (const auto& name : col_names)
            fields.push_back(arrow::field(name, arrow::float64()));
        auto schema = arrow::schema(fields);
        return arrow::Table::Make(schema,
            std::vector<std::shared_ptr<arrow::Array>>(
                fields.size(), std::make_shared<arrow::NullArray>(0)),
            0);
    }

    // --- Phase 1: Read columns ---
    const double* iv = get_double_col(table, "iv");
    const double* delta = get_double_col(table, "delta");
    const int32_t* option_type = get_int32_col(table, "option_type");
    const int64_t* timestamp = get_int64_col(table, "timestamp");
    const int32_t* expiration = get_int32_col(table, "expiration");

    // --- Phase 2: Group by (timestamp, expiration) ---
    using GroupKey = std::pair<int64_t, int32_t>;
    std::map<GroupKey, std::vector<size_t>> groups;
    for (int64_t i = 0; i < n; ++i) {
        if (std::isnan(iv[i]) || std::isnan(delta[i]))
            continue;
        groups[{timestamp[i], expiration[i]}].push_back(
            static_cast<size_t>(i));
    }

    // Convert to vector for indexed/parallel access
    std::vector<std::pair<GroupKey, std::vector<size_t>>> group_vec(
        groups.begin(), groups.end());
    const size_t num_groups = group_vec.size();

    // --- Phase 3-4: OTM filter, ATM anchor, fit splines per group ---
    // Output storage: [group_idx][col_idx]
    // Column order: put wings (descending) | iv_50 | call wings (ascending)
    std::vector<std::vector<double>> out_ivs(
        num_groups, std::vector<double>(num_iv_cols, NaN));
    std::vector<int64_t> out_timestamps(num_groups);
    std::vector<int32_t> out_expirations(num_groups);

    #pragma omp parallel for schedule(dynamic)
    for (size_t g = 0; g < num_groups; ++g) {
        const auto& [key, indices] = group_vec[g];
        out_timestamps[g] = key.first;
        out_expirations[g] = key.second;

        // Split into OTM puts and OTM calls
        // OTM puts:  option_type == -1 && |delta| <= 0.50
        // OTM calls: option_type ==  1 &&  delta  <= 0.50
        std::vector<std::pair<double, double>> put_points;  // (abs_delta, iv)
        std::vector<std::pair<double, double>> call_points; // (delta, iv)

        double best_put_delta_dist = 1.0, best_put_iv = NaN;
        double best_call_delta_dist = 1.0, best_call_iv = NaN;

        for (size_t idx : indices) {
            int32_t ot = option_type[idx];
            double d = delta[idx];
            double v = iv[idx];

            if (ot == -1 && d < 0.0 && std::abs(d) <= 0.50) {
                double abs_d = std::abs(d);
                put_points.emplace_back(abs_d, v);
                // Track closest to 0.50 for ATM anchor
                double dist = std::abs(abs_d - 0.50);
                if (dist < best_put_delta_dist) {
                    best_put_delta_dist = dist;
                    best_put_iv = v;
                }
            } else if (ot == 1 && d > 0.0 && d <= 0.50) {
                call_points.emplace_back(d, v);
                double dist = std::abs(d - 0.50);
                if (dist < best_call_delta_dist) {
                    best_call_delta_dist = dist;
                    best_call_iv = v;
                }
            }
        }

        // Compute shared ATM vol
        double iv_atm = NaN;
        if (!std::isnan(best_put_iv) && !std::isnan(best_call_iv)) {
            iv_atm = (best_put_iv + best_call_iv) / 2.0;
        } else if (!std::isnan(best_put_iv)) {
            iv_atm = best_put_iv;
        } else if (!std::isnan(best_call_iv)) {
            iv_atm = best_call_iv;
        } else {
            continue;  // No data at all
        }

        // Store iv_50 (ATM) — it's at index num_wing (middle of output)
        out_ivs[g][num_wing] = iv_atm;

        // --- Put-side spline ---
        // Sort by abs_delta ascending, deduplicate, anchor at 0.50
        std::sort(put_points.begin(), put_points.end());
        std::vector<double> px, py;
        dedup_sorted(px, py, put_points);
        // Replace or insert the 0.50 anchor
        if (!px.empty() && px.back() == 0.50) {
            py.back() = iv_atm;
        } else {
            px.push_back(0.50);
            py.push_back(iv_atm);
        }

        // Evaluate put wing pillars (output indices 0..num_wing-1, reversed)
        // Column order is descending: iv_p45, iv_p40, ..., iv_p5
        // wing_deltas is ascending: 0.05, 0.10, ..., 0.45
        std::vector<double> put_results(num_wing, NaN);
        evaluate_spline(px, py, wing_deltas, put_results);
        for (size_t i = 0; i < num_wing; ++i) {
            // Output index for iv_pNN: reversed (iv_p45 first, iv_p5 last)
            out_ivs[g][num_wing - 1 - i] = put_results[i];
        }

        // --- Call-side spline ---
        std::sort(call_points.begin(), call_points.end());
        std::vector<double> cx, cy;
        dedup_sorted(cx, cy, call_points);
        // Replace or insert the 0.50 anchor
        if (!cx.empty() && cx.back() == 0.50) {
            cy.back() = iv_atm;
        } else {
            cx.push_back(0.50);
            cy.push_back(iv_atm);
        }

        // Evaluate call wing pillars (output indices num_wing+1 .. num_iv_cols-1)
        std::vector<double> call_results(num_wing, NaN);
        evaluate_spline(cx, cy, wing_deltas, call_results);
        for (size_t i = 0; i < num_wing; ++i) {
            out_ivs[g][num_wing + 1 + i] = call_results[i];
        }
    }

    // --- Phase 5: Build output Arrow table ---
    // Timestamp column (preserve input type)
    auto ts_builder_result = arrow::MakeBuilder(ts_type);
    if (!ts_builder_result.ok())
        throw std::runtime_error("Failed to create timestamp builder");
    auto ts_builder = std::move(*ts_builder_result);
    for (size_t g = 0; g < num_groups; ++g) {
        auto status = ts_builder->AppendScalar(
            *arrow::MakeScalar(ts_type, out_timestamps[g]).MoveValueUnsafe());
        if (!status.ok())
            throw std::runtime_error("Timestamp append failed");
    }
    std::shared_ptr<arrow::Array> ts_array;
    if (!ts_builder->Finish(&ts_array).ok())
        throw std::runtime_error("Timestamp finish failed");

    // Expiration column (date32)
    arrow::Date32Builder exp_builder;
    for (size_t g = 0; g < num_groups; ++g) {
        auto status = exp_builder.Append(out_expirations[g]);
        if (!status.ok())
            throw std::runtime_error("Expiration append failed");
    }
    std::shared_ptr<arrow::Array> exp_array;
    if (!exp_builder.Finish(&exp_array).ok())
        throw std::runtime_error("Expiration finish failed");

    // IV columns
    std::vector<std::shared_ptr<arrow::Array>> iv_arrays(num_iv_cols);
    for (size_t c = 0; c < num_iv_cols; ++c) {
        arrow::DoubleBuilder builder;
        for (size_t g = 0; g < num_groups; ++g) {
            double val = out_ivs[g][c];
            if (std::isnan(val)) {
                auto status = builder.AppendNull();
                if (!status.ok())
                    throw std::runtime_error("AppendNull failed");
            } else {
                auto status = builder.Append(val);
                if (!status.ok())
                    throw std::runtime_error("Append failed");
            }
        }
        if (!builder.Finish(&iv_arrays[c]).ok())
            throw std::runtime_error("Finish failed for IV column");
    }

    // Assemble schema and arrays
    arrow::FieldVector fields;
    fields.push_back(arrow::field("timestamp", ts_type));
    fields.push_back(arrow::field("expiration", arrow::date32()));
    for (const auto& name : col_names)
        fields.push_back(arrow::field(name, arrow::float64()));

    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.push_back(ts_array);
    arrays.push_back(exp_array);
    for (auto& arr : iv_arrays)
        arrays.push_back(arr);

    auto schema = arrow::schema(fields);
    return arrow::Table::Make(schema, arrays,
                              static_cast<int64_t>(num_groups));
}

}  // namespace argiv
