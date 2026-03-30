#include "argiv/surface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
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

const double* try_get_double_col(const std::shared_ptr<arrow::Table>& table,
                                 const std::string& name) {
    auto col = table->GetColumnByName(name);
    if (!col)
        return nullptr;
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

// Deduplicate (x, y1, y2) triples sorted by x: average y1 and y2 for identical x.
void dedup_sorted2(std::vector<double>& x,
                   std::vector<double>& y1,
                   std::vector<double>& y2,
                   const std::vector<std::tuple<double, double, double>>& points) {
    x.clear();
    y1.clear();
    y2.clear();
    x.reserve(points.size());
    y1.reserve(points.size());
    y2.reserve(points.size());
    size_t i = 0;
    while (i < points.size()) {
        double d = std::get<0>(points[i]);
        double sum1 = 0.0, sum2 = 0.0;
        size_t count = 0;
        while (i < points.size() && std::get<0>(points[i]) == d) {
            sum1 += std::get<1>(points[i]);
            sum2 += std::get<2>(points[i]);
            ++count;
            ++i;
        }
        x.push_back(d);
        y1.push_back(sum1 / count);
        y2.push_back(sum2 / count);
    }
}

// Evaluate a spline (cubic or linear depending on point count) at target deltas.
// Only evaluates within the spline's domain [x.front(), x.back()].
// If clamp_positive is true, values <= 0 are replaced with NaN (for IV).
void evaluate_spline(const std::vector<double>& x,
                     const std::vector<double>& y,
                     const std::vector<double>& targets,
                     std::vector<double>& out,
                     bool clamp_positive = true) {
    if (x.size() < 2) {
        return;
    }

    if (x.size() >= 4) {
        QuantLib::CubicNaturalSpline spline(x.begin(), x.end(), y.begin());
        for (size_t i = 0; i < targets.size(); ++i) {
            double t = targets[i];
            if (t >= x.front() && t <= x.back()) {
                double val = spline(t);
                if (clamp_positive)
                    out[i] = val > 0.0 ? val : NaN;
                else
                    out[i] = val;
            }
        }
    } else {
        QuantLib::LinearInterpolation interp(x.begin(), x.end(), y.begin());
        for (size_t i = 0; i < targets.size(); ++i) {
            double t = targets[i];
            if (t >= x.front() && t <= x.back()) {
                double val = interp(t);
                if (clamp_positive)
                    out[i] = val > 0.0 ? val : NaN;
                else
                    out[i] = val;
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
    auto wing_deltas = build_wing_deltas(sorted_pillars);
    const size_t num_wing = wing_deltas.size();
    const size_t num_pillars = num_wing * 2 + 1;  // put wings + ATM + call wings

    // Preserve timestamp type from input schema
    auto ts_field = table->schema()->GetFieldByName("timestamp");
    if (!ts_field)
        throw std::runtime_error("Missing column: timestamp");
    auto ts_type = ts_field->type();

    // Detect optional bid/ask IV columns
    const double* iv_bid_col = try_get_double_col(table, "iv_bid");
    const double* iv_ask_col = try_get_double_col(table, "iv_ask");
    const bool has_bid_ask = (iv_bid_col != nullptr && iv_ask_col != nullptr);

    // Handle empty input
    if (n == 0) {
        arrow::FieldVector fields;
        fields.push_back(arrow::field("timestamp", ts_type));
        fields.push_back(arrow::field("expiration", arrow::date32()));
        fields.push_back(arrow::field("delta", arrow::float64()));
        fields.push_back(arrow::field("iv", arrow::float64()));
        if (has_bid_ask) {
            fields.push_back(arrow::field("iv_bid", arrow::float64()));
            fields.push_back(arrow::field("iv_ask", arrow::float64()));
        }
        fields.push_back(arrow::field("log_moneyness", arrow::float64()));
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

    // Optional columns for log moneyness
    const double* spot = try_get_double_col(table, "spot");
    const double* strike = try_get_double_col(table, "strike");
    const bool has_moneyness = (spot != nullptr && strike != nullptr);

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
    const size_t total_rows = num_groups * num_pillars;
    std::vector<int64_t>  out_timestamps(total_rows);
    std::vector<int32_t>  out_expirations(total_rows);
    std::vector<double>   out_deltas(total_rows, NaN);
    std::vector<double>   out_ivs(total_rows, NaN);
    std::vector<double>   out_iv_bid(has_bid_ask ? total_rows : 0, NaN);
    std::vector<double>   out_iv_ask(has_bid_ask ? total_rows : 0, NaN);
    std::vector<double>   out_lm(total_rows, NaN);

    // Pre-fill delta values for all groups (these are always populated)
    for (size_t g = 0; g < num_groups; ++g) {
        const size_t base = g * num_pillars;
        // Put side: descending magnitude (-0.45, -0.40, ..., -0.05)
        for (size_t i = 0; i < num_wing; ++i) {
            out_deltas[base + i] = -wing_deltas[num_wing - 1 - i];
        }
        // ATM
        out_deltas[base + num_wing] = 0.50;
        // Call side: ascending magnitude (0.05, 0.10, ..., 0.45)
        for (size_t i = 0; i < num_wing; ++i) {
            out_deltas[base + num_wing + 1 + i] = wing_deltas[i];
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t g = 0; g < num_groups; ++g) {
        const auto& [key, indices] = group_vec[g];
        const size_t base = g * num_pillars;

        // Fill timestamp/expiration for all rows in this group
        for (size_t i = 0; i < num_pillars; ++i) {
            out_timestamps[base + i] = key.first;
            out_expirations[base + i] = key.second;
        }

        // Split into OTM puts and OTM calls
        std::vector<std::pair<double, double>> put_points;   // (abs_delta, iv)
        std::vector<std::pair<double, double>> call_points;  // (delta, iv)
        // For log moneyness spline (parallel vectors)
        std::vector<std::tuple<double, double, double>> put_points_lm;   // (abs_delta, iv, lm)
        std::vector<std::tuple<double, double, double>> call_points_lm;  // (delta, iv, lm)

        // Bid/ask point vectors (only used if has_bid_ask)
        std::vector<std::pair<double, double>> put_points_bid, call_points_bid;
        std::vector<std::pair<double, double>> put_points_ask, call_points_ask;

        double best_put_delta_dist = 1.0, best_put_iv = NaN, best_put_lm = NaN;
        double best_call_delta_dist = 1.0, best_call_iv = NaN, best_call_lm = NaN;
        double best_put_iv_bid = NaN, best_put_iv_ask = NaN;
        double best_call_iv_bid = NaN, best_call_iv_ask = NaN;

        for (size_t idx : indices) {
            int32_t ot = option_type[idx];
            double d = delta[idx];
            double v = iv[idx];

            double lm = NaN;
            if (has_moneyness && spot[idx] > 0.0 && !std::isnan(spot[idx]) &&
                strike[idx] > 0.0 && !std::isnan(strike[idx])) {
                lm = std::log(strike[idx] / spot[idx]);
            }

            if (ot == -1 && d < 0.0 && std::abs(d) <= 0.50) {
                double abs_d = std::abs(d);
                put_points.emplace_back(abs_d, v);
                if (!std::isnan(lm))
                    put_points_lm.emplace_back(abs_d, v, lm);

                if (has_bid_ask) {
                    double vb = iv_bid_col[idx], va = iv_ask_col[idx];
                    if (!std::isnan(vb)) put_points_bid.emplace_back(abs_d, vb);
                    if (!std::isnan(va)) put_points_ask.emplace_back(abs_d, va);
                }

                double dist = std::abs(abs_d - 0.50);
                if (dist < best_put_delta_dist) {
                    best_put_delta_dist = dist;
                    best_put_iv = v;
                    best_put_lm = lm;
                    if (has_bid_ask) {
                        best_put_iv_bid = iv_bid_col[idx];
                        best_put_iv_ask = iv_ask_col[idx];
                    }
                }
            } else if (ot == 1 && d > 0.0 && d <= 0.50) {
                call_points.emplace_back(d, v);
                if (!std::isnan(lm))
                    call_points_lm.emplace_back(d, v, lm);

                if (has_bid_ask) {
                    double vb = iv_bid_col[idx], va = iv_ask_col[idx];
                    if (!std::isnan(vb)) call_points_bid.emplace_back(d, vb);
                    if (!std::isnan(va)) call_points_ask.emplace_back(d, va);
                }

                double dist = std::abs(d - 0.50);
                if (dist < best_call_delta_dist) {
                    best_call_delta_dist = dist;
                    best_call_iv = v;
                    best_call_lm = lm;
                    if (has_bid_ask) {
                        best_call_iv_bid = iv_bid_col[idx];
                        best_call_iv_ask = iv_ask_col[idx];
                    }
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

        // Compute shared ATM log moneyness
        double lm_atm = NaN;
        if (!std::isnan(best_put_lm) && !std::isnan(best_call_lm)) {
            lm_atm = (best_put_lm + best_call_lm) / 2.0;
        } else if (!std::isnan(best_put_lm)) {
            lm_atm = best_put_lm;
        } else if (!std::isnan(best_call_lm)) {
            lm_atm = best_call_lm;
        }

        // Store ATM row
        out_ivs[base + num_wing] = iv_atm;
        out_lm[base + num_wing] = lm_atm;

        // --- Put-side IV spline ---
        std::sort(put_points.begin(), put_points.end());
        std::vector<double> px, py;
        dedup_sorted(px, py, put_points);
        if (!px.empty() && px.back() == 0.50) {
            py.back() = iv_atm;
        } else {
            px.push_back(0.50);
            py.push_back(iv_atm);
        }

        std::vector<double> put_iv_results(num_wing, NaN);
        evaluate_spline(px, py, wing_deltas, put_iv_results, true);

        // Put-side log moneyness spline
        std::vector<double> put_lm_results(num_wing, NaN);
        if (has_moneyness && !put_points_lm.empty()) {
            // Sort by abs_delta for spline
            std::sort(put_points_lm.begin(), put_points_lm.end(),
                      [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
            std::vector<double> plx, ply_iv, ply_lm;
            dedup_sorted2(plx, ply_iv, ply_lm, put_points_lm);
            // Anchor at 0.50
            if (!plx.empty() && plx.back() == 0.50) {
                ply_lm.back() = lm_atm;
            } else if (!std::isnan(lm_atm)) {
                plx.push_back(0.50);
                ply_lm.push_back(lm_atm);
            }
            evaluate_spline(plx, ply_lm, wing_deltas, put_lm_results, false);
        }

        // Write put wing rows (descending magnitude: -0.45, -0.40, ..., -0.05)
        for (size_t i = 0; i < num_wing; ++i) {
            // wing_deltas[i] is ascending (0.05, 0.10, ...)
            // Output index for descending: num_wing - 1 - i
            size_t out_idx = base + num_wing - 1 - i;
            out_ivs[out_idx] = put_iv_results[i];
            out_lm[out_idx] = put_lm_results[i];
        }

        // --- Call-side IV spline ---
        std::sort(call_points.begin(), call_points.end());
        std::vector<double> cx, cy;
        dedup_sorted(cx, cy, call_points);
        if (!cx.empty() && cx.back() == 0.50) {
            cy.back() = iv_atm;
        } else {
            cx.push_back(0.50);
            cy.push_back(iv_atm);
        }

        std::vector<double> call_iv_results(num_wing, NaN);
        evaluate_spline(cx, cy, wing_deltas, call_iv_results, true);

        // Call-side log moneyness spline
        std::vector<double> call_lm_results(num_wing, NaN);
        if (has_moneyness && !call_points_lm.empty()) {
            std::sort(call_points_lm.begin(), call_points_lm.end(),
                      [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
            std::vector<double> clx, cly_iv, cly_lm;
            dedup_sorted2(clx, cly_iv, cly_lm, call_points_lm);
            if (!clx.empty() && clx.back() == 0.50) {
                cly_lm.back() = lm_atm;
            } else if (!std::isnan(lm_atm)) {
                clx.push_back(0.50);
                cly_lm.push_back(lm_atm);
            }
            evaluate_spline(clx, cly_lm, wing_deltas, call_lm_results, false);
        }

        // Write call wing rows (ascending: 0.05, 0.10, ..., 0.45)
        for (size_t i = 0; i < num_wing; ++i) {
            size_t out_idx = base + num_wing + 1 + i;
            out_ivs[out_idx] = call_iv_results[i];
            out_lm[out_idx] = call_lm_results[i];
        }

        // --- Bid/Ask splines ---
        if (has_bid_ask) {
            // Helper to compute ATM anchor from best put/call values
            auto compute_atm = [](double best_put, double best_call) -> double {
                if (!std::isnan(best_put) && !std::isnan(best_call))
                    return (best_put + best_call) / 2.0;
                if (!std::isnan(best_put)) return best_put;
                if (!std::isnan(best_call)) return best_call;
                return NaN;
            };

            // Helper to fit one wing and write results
            auto fit_wing = [&](std::vector<std::pair<double, double>>& points,
                                double atm_val,
                                std::vector<double>& results) {
                std::sort(points.begin(), points.end());
                std::vector<double> fx, fy;
                dedup_sorted(fx, fy, points);
                if (!std::isnan(atm_val)) {
                    if (!fx.empty() && fx.back() == 0.50)
                        fy.back() = atm_val;
                    else {
                        fx.push_back(0.50);
                        fy.push_back(atm_val);
                    }
                }
                evaluate_spline(fx, fy, wing_deltas, results, true);
            };

            double iv_atm_bid = compute_atm(best_put_iv_bid, best_call_iv_bid);
            double iv_atm_ask = compute_atm(best_put_iv_ask, best_call_iv_ask);

            // Store ATM bid/ask
            out_iv_bid[base + num_wing] = iv_atm_bid;
            out_iv_ask[base + num_wing] = iv_atm_ask;

            // Bid splines
            std::vector<double> put_bid_results(num_wing, NaN);
            std::vector<double> call_bid_results(num_wing, NaN);
            fit_wing(put_points_bid, iv_atm_bid, put_bid_results);
            fit_wing(call_points_bid, iv_atm_bid, call_bid_results);

            // Ask splines
            std::vector<double> put_ask_results(num_wing, NaN);
            std::vector<double> call_ask_results(num_wing, NaN);
            fit_wing(put_points_ask, iv_atm_ask, put_ask_results);
            fit_wing(call_points_ask, iv_atm_ask, call_ask_results);

            // Write put wing (descending magnitude)
            for (size_t i = 0; i < num_wing; ++i) {
                size_t out_idx = base + num_wing - 1 - i;
                out_iv_bid[out_idx] = put_bid_results[i];
                out_iv_ask[out_idx] = put_ask_results[i];
            }
            // Write call wing (ascending)
            for (size_t i = 0; i < num_wing; ++i) {
                size_t out_idx = base + num_wing + 1 + i;
                out_iv_bid[out_idx] = call_bid_results[i];
                out_iv_ask[out_idx] = call_ask_results[i];
            }
        }
    }

    // --- Phase 5: Build output Arrow table ---
    // Timestamp column (preserve input type)
    auto ts_builder_result = arrow::MakeBuilder(ts_type);
    if (!ts_builder_result.ok())
        throw std::runtime_error("Failed to create timestamp builder");
    auto ts_builder = std::move(*ts_builder_result);
    for (size_t i = 0; i < total_rows; ++i) {
        auto status = ts_builder->AppendScalar(
            *arrow::MakeScalar(ts_type, out_timestamps[i]).MoveValueUnsafe());
        if (!status.ok())
            throw std::runtime_error("Timestamp append failed");
    }
    std::shared_ptr<arrow::Array> ts_array;
    if (!ts_builder->Finish(&ts_array).ok())
        throw std::runtime_error("Timestamp finish failed");

    // Expiration column (date32)
    arrow::Date32Builder exp_builder;
    for (size_t i = 0; i < total_rows; ++i) {
        auto status = exp_builder.Append(out_expirations[i]);
        if (!status.ok())
            throw std::runtime_error("Expiration append failed");
    }
    std::shared_ptr<arrow::Array> exp_array;
    if (!exp_builder.Finish(&exp_array).ok())
        throw std::runtime_error("Expiration finish failed");

    // Delta column
    arrow::DoubleBuilder delta_builder;
    for (size_t i = 0; i < total_rows; ++i) {
        auto status = delta_builder.Append(out_deltas[i]);
        if (!status.ok())
            throw std::runtime_error("Delta append failed");
    }
    std::shared_ptr<arrow::Array> delta_array;
    if (!delta_builder.Finish(&delta_array).ok())
        throw std::runtime_error("Delta finish failed");

    // IV column (NaN -> null)
    arrow::DoubleBuilder iv_builder;
    for (size_t i = 0; i < total_rows; ++i) {
        double val = out_ivs[i];
        if (std::isnan(val)) {
            auto status = iv_builder.AppendNull();
            if (!status.ok())
                throw std::runtime_error("IV AppendNull failed");
        } else {
            auto status = iv_builder.Append(val);
            if (!status.ok())
                throw std::runtime_error("IV Append failed");
        }
    }
    std::shared_ptr<arrow::Array> iv_array;
    if (!iv_builder.Finish(&iv_array).ok())
        throw std::runtime_error("IV finish failed");

    // Log moneyness column (NaN -> null)
    arrow::DoubleBuilder lm_builder;
    for (size_t i = 0; i < total_rows; ++i) {
        double val = out_lm[i];
        if (std::isnan(val)) {
            auto status = lm_builder.AppendNull();
            if (!status.ok())
                throw std::runtime_error("Log moneyness AppendNull failed");
        } else {
            auto status = lm_builder.Append(val);
            if (!status.ok())
                throw std::runtime_error("Log moneyness Append failed");
        }
    }
    std::shared_ptr<arrow::Array> lm_array;
    if (!lm_builder.Finish(&lm_array).ok())
        throw std::runtime_error("Log moneyness finish failed");

    // Bid/Ask IV columns (NaN -> null), built conditionally
    std::shared_ptr<arrow::Array> iv_bid_array, iv_ask_array;
    if (has_bid_ask) {
        auto build_nullable_double = [&](const std::vector<double>& data,
                                         const char* label) {
            arrow::DoubleBuilder builder;
            for (size_t i = 0; i < total_rows; ++i) {
                double val = data[i];
                if (std::isnan(val)) {
                    if (!builder.AppendNull().ok())
                        throw std::runtime_error(
                            std::string(label) + " AppendNull failed");
                } else {
                    if (!builder.Append(val).ok())
                        throw std::runtime_error(
                            std::string(label) + " Append failed");
                }
            }
            std::shared_ptr<arrow::Array> arr;
            if (!builder.Finish(&arr).ok())
                throw std::runtime_error(
                    std::string(label) + " finish failed");
            return arr;
        };
        iv_bid_array = build_nullable_double(out_iv_bid, "iv_bid");
        iv_ask_array = build_nullable_double(out_iv_ask, "iv_ask");
    }

    // Assemble schema and arrays
    arrow::FieldVector fields = {
        arrow::field("timestamp", ts_type),
        arrow::field("expiration", arrow::date32()),
        arrow::field("delta", arrow::float64()),
        arrow::field("iv", arrow::float64()),
    };
    std::vector<std::shared_ptr<arrow::Array>> arrays = {
        ts_array, exp_array, delta_array, iv_array,
    };

    if (has_bid_ask) {
        fields.push_back(arrow::field("iv_bid", arrow::float64()));
        fields.push_back(arrow::field("iv_ask", arrow::float64()));
        arrays.push_back(iv_bid_array);
        arrays.push_back(iv_ask_array);
    }

    fields.push_back(arrow::field("log_moneyness", arrow::float64()));
    arrays.push_back(lm_array);

    auto schema = arrow::schema(fields);
    return arrow::Table::Make(schema, arrays, static_cast<int64_t>(total_rows));
}

}  // namespace argiv
