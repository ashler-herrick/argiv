#include "argiv/surface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
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

// Build pillar column names: iv_p50, iv_p45, ..., iv_p5, iv_c5, ..., iv_c50
std::vector<std::string> build_pillar_names(
    const std::vector<double>& pillars) {
    std::vector<std::string> names;
    names.reserve(pillars.size() * 2);
    // Put side: descending magnitude
    for (auto it = pillars.rbegin(); it != pillars.rend(); ++it) {
        int p = static_cast<int>(*it);
        names.push_back("iv_p" + std::to_string(p));
    }
    // Call side: ascending magnitude
    for (double p : pillars) {
        int pi = static_cast<int>(p);
        names.push_back("iv_c" + std::to_string(pi));
    }
    return names;
}

// Build pillar delta values (put negative, call positive) in column order
std::vector<double> build_pillar_deltas(const std::vector<double>& pillars) {
    std::vector<double> deltas;
    deltas.reserve(pillars.size() * 2);
    // Put side: descending (e.g., -0.50, -0.45, ..., -0.05)
    for (auto it = pillars.rbegin(); it != pillars.rend(); ++it) {
        deltas.push_back(-(*it) / 100.0);
    }
    // Call side: ascending (e.g., +0.05, +0.10, ..., +0.50)
    for (double p : pillars) {
        deltas.push_back(p / 100.0);
    }
    return deltas;
}

}  // namespace

std::shared_ptr<arrow::Table> fit_vol_surface_table(
    const std::shared_ptr<arrow::Table>& input,
    const SurfaceConfig& config) {

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
    auto pillar_names = build_pillar_names(sorted_pillars);
    auto pillar_deltas = build_pillar_deltas(sorted_pillars);
    const size_t num_pillars = pillar_deltas.size();

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
        for (const auto& name : pillar_names)
            fields.push_back(arrow::field(name, arrow::float64()));
        auto schema = arrow::schema(fields);
        return arrow::Table::Make(schema,
            std::vector<std::shared_ptr<arrow::Array>>(
                fields.size(), std::make_shared<arrow::NullArray>(0)),
            0);
    }

    // --- Phase 1: Read precomputed IVs and deltas ---
    const double* iv = get_double_col(table, "iv");
    const double* delta = get_double_col(table, "delta");
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

    // --- Phase 3-4: Fit spline and evaluate per group ---
    // Output storage: [group_idx][pillar_idx]
    std::vector<std::vector<double>> pillar_ivs(
        num_groups, std::vector<double>(num_pillars, NaN));
    std::vector<int64_t> out_timestamps(num_groups);
    std::vector<int32_t> out_expirations(num_groups);

    #pragma omp parallel for schedule(dynamic)
    for (size_t g = 0; g < num_groups; ++g) {
        const auto& [key, indices] = group_vec[g];
        out_timestamps[g] = key.first;
        out_expirations[g] = key.second;

        // Collect (delta, iv) pairs, sorted by delta
        std::vector<std::pair<double, double>> points;
        points.reserve(indices.size());
        for (size_t idx : indices) {
            points.emplace_back(delta[idx], iv[idx]);
        }
        std::sort(points.begin(), points.end());

        // Deduplicate: average IVs for identical delta values
        std::vector<double> x, y;
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

        if (x.size() < 2) {
            // All NaN (already initialized)
            continue;
        }

        // Evaluate at each pillar delta
        if (x.size() >= 4) {
            // Cubic natural spline
            QuantLib::CubicNaturalSpline spline(
                x.begin(), x.end(), y.begin());
            for (size_t p = 0; p < num_pillars; ++p) {
                double target = pillar_deltas[p];
                if (target >= x.front() && target <= x.back()) {
                    double val = spline(target);
                    pillar_ivs[g][p] = val > 0.0 ? val : NaN;
                }
            }
        } else {
            // 2-3 points: linear interpolation
            QuantLib::LinearInterpolation interp(
                x.begin(), x.end(), y.begin());
            for (size_t p = 0; p < num_pillars; ++p) {
                double target = pillar_deltas[p];
                if (target >= x.front() && target <= x.back()) {
                    double val = interp(target);
                    pillar_ivs[g][p] = val > 0.0 ? val : NaN;
                }
            }
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

    // Pillar IV columns
    std::vector<std::shared_ptr<arrow::Array>> pillar_arrays(num_pillars);
    for (size_t p = 0; p < num_pillars; ++p) {
        arrow::DoubleBuilder builder;
        for (size_t g = 0; g < num_groups; ++g) {
            double val = pillar_ivs[g][p];
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
        if (!builder.Finish(&pillar_arrays[p]).ok())
            throw std::runtime_error("Finish failed for pillar column");
    }

    // Assemble schema and arrays
    arrow::FieldVector fields;
    fields.push_back(arrow::field("timestamp", ts_type));
    fields.push_back(arrow::field("expiration", arrow::date32()));
    for (const auto& name : pillar_names)
        fields.push_back(arrow::field(name, arrow::float64()));

    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.push_back(ts_array);
    arrays.push_back(exp_array);
    for (auto& arr : pillar_arrays)
        arrays.push_back(arr);

    auto schema = arrow::schema(fields);
    return arrow::Table::Make(schema, arrays,
                              static_cast<int64_t>(num_groups));
}

}  // namespace argiv
