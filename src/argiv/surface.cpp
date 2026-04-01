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
#include <ql/experimental/volatility/sviinterpolation.hpp>
#include <ql/math/optimization/endcriteria.hpp>
#include <ql/math/optimization/levenbergmarquardt.hpp>
#include <ql/math/solvers1d/brent.hpp>
#include <ql/pricingengines/blackcalculator.hpp>

namespace argiv {

namespace {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

const double* get_double_col(const std::shared_ptr<arrow::Table>& table,
                             const std::string& name) {
    auto col = table->GetColumnByName(name);
    if (!col)
        throw std::runtime_error("Missing column: " + name);
    auto chunk = col->chunk(0);
    if (chunk->type_id() != arrow::Type::DOUBLE) {
        throw std::runtime_error(
            "Column '" + name + "' has type " + chunk->type()->ToString() +
            ", expected float64. Cast the column before passing to argiv.");
    }
    if (chunk->null_count() > 0) {
        throw std::runtime_error(
            "Column '" + name + "' contains " +
            std::to_string(chunk->null_count()) + " null values of " +
            std::to_string(chunk->length()) + " total. "
            "Fill or drop nulls before passing to argiv.");
    }
    return std::static_pointer_cast<arrow::DoubleArray>(chunk)->raw_values();
}

const double* try_get_double_col(const std::shared_ptr<arrow::Table>& table,
                                 const std::string& name) {
    auto col = table->GetColumnByName(name);
    if (!col)
        return nullptr;
    auto chunk = col->chunk(0);
    if (chunk->type_id() != arrow::Type::DOUBLE) {
        throw std::runtime_error(
            "Column '" + name + "' has type " + chunk->type()->ToString() +
            ", expected float64. Cast the column before passing to argiv.");
    }
    if (chunk->null_count() > 0) {
        throw std::runtime_error(
            "Column '" + name + "' contains " +
            std::to_string(chunk->null_count()) + " null values of " +
            std::to_string(chunk->length()) + " total. "
            "Fill or drop nulls before passing to argiv.");
    }
    return std::static_pointer_cast<arrow::DoubleArray>(chunk)->raw_values();
}

// Accepts int32 or date32 (both use int32 physical storage).
const int32_t* get_int32_col(const std::shared_ptr<arrow::Table>& table,
                             const std::string& name) {
    auto col = table->GetColumnByName(name);
    if (!col)
        throw std::runtime_error("Missing column: " + name);
    auto chunk = col->chunk(0);
    bool ok = chunk->type_id() == arrow::Type::INT32 ||
              chunk->type_id() == arrow::Type::DATE32;
    if (!ok) {
        throw std::runtime_error(
            "Column '" + name + "' has type " + chunk->type()->ToString() +
            ", expected int32 or date32. Cast the column before passing to argiv.");
    }
    if (chunk->null_count() > 0) {
        throw std::runtime_error(
            "Column '" + name + "' contains " +
            std::to_string(chunk->null_count()) + " null values of " +
            std::to_string(chunk->length()) + " total. "
            "Fill or drop nulls before passing to argiv.");
    }
    return std::static_pointer_cast<arrow::Int32Array>(chunk)->raw_values();
}

// Accepts int64 or timestamp (both use int64 physical storage).
const int64_t* get_int64_col(const std::shared_ptr<arrow::Table>& table,
                             const std::string& name) {
    auto col = table->GetColumnByName(name);
    if (!col)
        throw std::runtime_error("Missing column: " + name);
    auto chunk = col->chunk(0);
    bool ok = chunk->type_id() == arrow::Type::INT64 ||
              chunk->type_id() == arrow::Type::TIMESTAMP;
    if (!ok) {
        throw std::runtime_error(
            "Column '" + name + "' has type " + chunk->type()->ToString() +
            ", expected int64 or timestamp. Cast the column before passing to argiv.");
    }
    if (chunk->null_count() > 0) {
        throw std::runtime_error(
            "Column '" + name + "' contains " +
            std::to_string(chunk->null_count()) + " null values of " +
            std::to_string(chunk->length()) + " total. "
            "Fill or drop nulls before passing to argiv.");
    }
    return std::static_pointer_cast<arrow::Int64Array>(chunk)->raw_values();
}

std::vector<double> build_wing_deltas(const std::vector<double>& pillars) {
    std::vector<double> deltas;
    deltas.reserve(pillars.size());
    for (double p : pillars) {
        deltas.push_back(p / 100.0);
    }
    return deltas;
}

// Deduplicate (x, y) pairs sorted by x: average y for identical x.
void dedup_sorted_pairs(std::vector<double>& x, std::vector<double>& y,
                        const std::vector<std::pair<double, double>>& points) {
    x.clear();
    y.clear();
    if (points.empty()) return;
    x.reserve(points.size());
    y.reserve(points.size());
    size_t i = 0;
    while (i < points.size()) {
        double k = points[i].first;
        double sum = 0.0;
        size_t count = 0;
        while (i < points.size() && points[i].first == k) {
            sum += points[i].second;
            ++count;
            ++i;
        }
        x.push_back(k);
        y.push_back(sum / count);
    }
}

// Find the strike K such that BS delta(K, sigma(K)) == target_delta.
// vol_at_strike is a callable returning IV at a given strike.
// min_data_strike / max_data_strike bound the search to where the SVI model
// is reliable and BS delta remains monotonic in strike.
template <typename VolFunc>
double strike_at_delta(double target_delta,
                       QuantLib::Option::Type option_type,
                       double spot, double forward, double T, double r,
                       double min_data_strike, double max_data_strike,
                       const VolFunc& vol_at_strike) {
    try {
        auto objective = [&](double K) -> double {
            double sigma = vol_at_strike(K);
            if (sigma <= 0.0 || std::isnan(sigma))
                return target_delta;
            double stdDev = sigma * std::sqrt(T);
            double discount = std::exp(-r * T);
            QuantLib::BlackCalculator calc(option_type, K, forward, stdDev,
                                           discount);
            return calc.delta(spot) - target_delta;
        };

        QuantLib::Brent solver;
        solver.setMaxEvaluations(200);

        double lo, hi;
        if (option_type == QuantLib::Option::Put) {
            lo = min_data_strike * 0.5;
            hi = forward;
        } else {
            lo = forward;
            hi = max_data_strike * 2.0;
        }

        // Verify the bracket contains a sign change
        double f_lo = objective(lo);
        double f_hi = objective(hi);
        if (f_lo * f_hi > 0.0)
            return NaN;  // no root in bracket

        return solver.solve(objective, 1e-6, 0.5 * (lo + hi), lo, hi);
    } catch (...) {
        return NaN;
    }
}

// Try to fit SVI to (strikes, vols) data. Returns true on success.
// On success, the SviInterpolation is ready for evaluation.
bool try_fit_svi(const std::vector<double>& strikes,
                 const std::vector<double>& vols,
                 double T, double forward,
                 QuantLib::SviInterpolation& svi) {
    try {
        svi.update();
        double rms = svi.rmsError();
        if (std::isnan(rms) || rms > 0.5)
            return false;
        // Quick sanity: ATM vol should be positive
        double atm_vol = svi(forward, true);
        return atm_vol > 0.0 && !std::isnan(atm_vol) && atm_vol < 10.0;
    } catch (...) {
        return false;
    }
}

}  // namespace

std::shared_ptr<arrow::Table> fit_vol_surface_table(
    const std::shared_ptr<arrow::Table>& input,
    const SurfaceConfig& config) {

    for (double p : config.delta_pillars) {
        if (p >= 50.0)
            throw std::runtime_error(
                "delta_pillars must be < 50 (ATM is computed automatically)");
    }

    auto combined_result = input->CombineChunks();
    if (!combined_result.ok())
        throw std::runtime_error("Failed to combine chunks: " +
                                 combined_result.status().ToString());
    auto table = combined_result.MoveValueUnsafe();
    const int64_t n = table->num_rows();

    auto sorted_pillars = config.delta_pillars;
    std::sort(sorted_pillars.begin(), sorted_pillars.end());
    auto wing_deltas = build_wing_deltas(sorted_pillars);
    const size_t num_wing = wing_deltas.size();
    const size_t num_pillars = num_wing * 2 + 1;

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
    const double* iv_col = get_double_col(table, "iv");
    const int32_t* option_type = get_int32_col(table, "option_type");
    const int64_t* ts_col = get_int64_col(table, "timestamp");
    const int32_t* expiration_col = get_int32_col(table, "expiration");
    const double* spot_col = get_double_col(table, "spot");
    const double* strike_col = get_double_col(table, "strike");
    const double* expiry_col = get_double_col(table, "expiry");

    // Optional: rate and dividend_yield (default to 0 if absent)
    const double* rate_col = try_get_double_col(table, "rate");
    const double* div_col = try_get_double_col(table, "dividend_yield");

    // --- Phase 2: Group by (timestamp, expiration) ---
    using GroupKey = std::pair<int64_t, int32_t>;
    std::map<GroupKey, std::vector<size_t>> groups;
    for (int64_t i = 0; i < n; ++i) {
        if (std::isnan(iv_col[i]) || iv_col[i] <= 0.0)
            continue;
        groups[{ts_col[i], expiration_col[i]}].push_back(
            static_cast<size_t>(i));
    }

    std::vector<std::pair<GroupKey, std::vector<size_t>>> group_vec(
        groups.begin(), groups.end());
    const size_t num_groups = group_vec.size();

    // --- Allocate outputs ---
    const size_t total_rows = num_groups * num_pillars;
    std::vector<int64_t>  out_timestamps(total_rows);
    std::vector<int32_t>  out_expirations(total_rows);
    std::vector<double>   out_deltas(total_rows, NaN);
    std::vector<double>   out_ivs(total_rows, NaN);
    std::vector<double>   out_iv_bid(has_bid_ask ? total_rows : 0, NaN);
    std::vector<double>   out_iv_ask(has_bid_ask ? total_rows : 0, NaN);
    std::vector<double>   out_lm(total_rows, NaN);

    // Pre-fill delta values (always populated regardless of SVI success)
    for (size_t g = 0; g < num_groups; ++g) {
        const size_t base = g * num_pillars;
        for (size_t i = 0; i < num_wing; ++i) {
            out_deltas[base + i] = -wing_deltas[num_wing - 1 - i];
        }
        out_deltas[base + num_wing] = 0.50;
        for (size_t i = 0; i < num_wing; ++i) {
            out_deltas[base + num_wing + 1 + i] = wing_deltas[i];
        }
    }

    // --- Phase 3-4: SVI fitting per group ---
    #pragma omp parallel for schedule(dynamic)
    for (size_t g = 0; g < num_groups; ++g) {
        const auto& [key, indices] = group_vec[g];
        const size_t base = g * num_pillars;

        for (size_t i = 0; i < num_pillars; ++i) {
            out_timestamps[base + i] = key.first;
            out_expirations[base + i] = key.second;
        }

        // Group-level parameters from first valid row
        size_t first_idx = indices[0];
        double group_spot = spot_col[first_idx];
        double group_T = expiry_col[first_idx];
        double group_r = rate_col ? rate_col[first_idx] : 0.0;
        double group_q = div_col ? div_col[first_idx] : 0.0;

        if (group_spot <= 0.0 || group_T <= 0.0)
            continue;

        double forward = group_spot * std::exp((group_r - group_q) * group_T);
        if (forward <= 0.0)
            continue;

        // Collect OTM (strike, iv) pairs
        std::vector<std::pair<double, double>> otm_points;
        std::vector<std::pair<double, double>> otm_bid, otm_ask;

        for (size_t idx : indices) {
            double K = strike_col[idx];
            double v = iv_col[idx];
            int32_t ot = option_type[idx];

            if (K <= 0.0 || std::isnan(K) || v <= 0.0 || std::isnan(v))
                continue;

            bool is_otm = (ot == -1 && K < forward) || (ot == 1 && K > forward);
            if (!is_otm) continue;

            otm_points.emplace_back(K, v);

            if (has_bid_ask) {
                double vb = iv_bid_col[idx], va = iv_ask_col[idx];
                if (!std::isnan(vb) && vb > 0.0)
                    otm_bid.emplace_back(K, vb);
                if (!std::isnan(va) && va > 0.0)
                    otm_ask.emplace_back(K, va);
            }
        }

        if (otm_points.size() < 5) continue;

        // Sort and deduplicate by strike
        std::sort(otm_points.begin(), otm_points.end());
        std::vector<double> strikes, ivs;
        dedup_sorted_pairs(strikes, ivs, otm_points);

        if (strikes.size() < 5) continue;

        // --- Fit SVI ---
        try {
            // ATM IV guess from nearest-to-forward option
            double atm_iv_guess = 0.2;
            double best_dist = 1e18;
            for (size_t i = 0; i < strikes.size(); ++i) {
                double dist = std::abs(strikes[i] - forward);
                if (dist < best_dist) {
                    best_dist = dist;
                    atm_iv_guess = ivs[i];
                }
            }

            double a0 = atm_iv_guess * atm_iv_guess * group_T;
            double b0 = 0.1;
            double sigma0 = 0.1;
            double rho0 = -0.4;
            double m0 = 0.0;

            auto endCrit = QuantLib::ext::make_shared<QuantLib::EndCriteria>(
                200, 40, 1e-6, 1e-6, 1e-6);
            auto optMethod =
                QuantLib::ext::make_shared<QuantLib::LevenbergMarquardt>();

            QuantLib::SviInterpolation svi(
                strikes.begin(), strikes.end(), ivs.begin(),
                group_T, forward,
                a0, b0, sigma0, rho0, m0,
                false, false, false, false, false,
                true, endCrit, optMethod,
                0.0020, false, 5);

            if (!try_fit_svi(strikes, ivs, group_T, forward, svi))
                continue;

            auto vol_fn = [&](double K) -> double {
                return svi(K, true);
            };

            double min_K = strikes.front();
            double max_K = strikes.back();

            // ATM
            double iv_atm = svi(forward, true);
            if (iv_atm <= 0.0 || std::isnan(iv_atm) || iv_atm > 10.0)
                continue;
            out_ivs[base + num_wing] = iv_atm;
            out_lm[base + num_wing] = std::log(forward / group_spot);

            // Put wing pillars (descending magnitude in output)
            for (size_t i = 0; i < num_wing; ++i) {
                double target = -wing_deltas[i];
                double K = strike_at_delta(target, QuantLib::Option::Put,
                                           group_spot, forward, group_T,
                                           group_r, min_K, max_K, vol_fn);
                if (std::isnan(K) || K <= 0.0) continue;
                double v = svi(K, true);
                if (v <= 0.0 || std::isnan(v) || v > 10.0) continue;
                size_t out_idx = base + num_wing - 1 - i;
                out_ivs[out_idx] = v;
                out_lm[out_idx] = std::log(K / group_spot);
            }

            // Call wing pillars (ascending in output)
            for (size_t i = 0; i < num_wing; ++i) {
                double target = wing_deltas[i];
                double K = strike_at_delta(target, QuantLib::Option::Call,
                                           group_spot, forward, group_T,
                                           group_r, min_K, max_K, vol_fn);
                if (std::isnan(K) || K <= 0.0) continue;
                double v = svi(K, true);
                if (v <= 0.0 || std::isnan(v) || v > 10.0) continue;
                size_t out_idx = base + num_wing + 1 + i;
                out_ivs[out_idx] = v;
                out_lm[out_idx] = std::log(K / group_spot);
            }

            // --- Bid/Ask SVI fits ---
            if (has_bid_ask) {
                // Helper: fit SVI for bid or ask, write results at same
                // put/call strike layout already computed above.
                // Use fitted mid params as warm start for bid/ask
                double mid_a = svi.a(), mid_b = svi.b();
                double mid_sig = svi.sigma(), mid_rho = svi.rho();
                double mid_m = svi.m();

                auto fit_bid_ask_svi = [&](
                    std::vector<std::pair<double, double>>& raw_points,
                    std::vector<double>& out_vec) {

                    if (raw_points.size() < 5) return;
                    std::sort(raw_points.begin(), raw_points.end());
                    std::vector<double> ba_strikes, ba_ivs;
                    dedup_sorted_pairs(ba_strikes, ba_ivs, raw_points);
                    if (ba_strikes.size() < 5) return;

                    QuantLib::SviInterpolation ba_svi(
                        ba_strikes.begin(), ba_strikes.end(), ba_ivs.begin(),
                        group_T, forward,
                        mid_a, mid_b, mid_sig, mid_rho, mid_m,
                        false, false, false, false, false,
                        true, endCrit, optMethod,
                        0.0020, false, 3);
                    if (!try_fit_svi(ba_strikes, ba_ivs, group_T, forward,
                                     ba_svi))
                        return;

                    // ATM
                    double ba_atm = ba_svi(forward, true);
                    if (ba_atm > 0.0 && !std::isnan(ba_atm) && ba_atm < 10.0)
                        out_vec[base + num_wing] = ba_atm;

                    for (size_t i = 0; i < num_wing; ++i) {
                        double target_put = -wing_deltas[i];
                        double Kp = strike_at_delta(
                            target_put, QuantLib::Option::Put,
                            group_spot, forward, group_T, group_r,
                            min_K, max_K,
                            [&](double K) { return ba_svi(K, true); });
                        if (!std::isnan(Kp) && Kp > 0.0) {
                            double v = ba_svi(Kp, true);
                            if (v > 0.0 && !std::isnan(v) && v < 10.0) {
                                size_t out_idx = base + num_wing - 1 - i;
                                out_vec[out_idx] = v;
                            }
                        }

                        double target_call = wing_deltas[i];
                        double Kc = strike_at_delta(
                            target_call, QuantLib::Option::Call,
                            group_spot, forward, group_T, group_r,
                            min_K, max_K,
                            [&](double K) { return ba_svi(K, true); });
                        if (!std::isnan(Kc) && Kc > 0.0) {
                            double v = ba_svi(Kc, true);
                            if (v > 0.0 && !std::isnan(v) && v < 10.0) {
                                size_t out_idx = base + num_wing + 1 + i;
                                out_vec[out_idx] = v;
                            }
                        }
                    }
                };

                fit_bid_ask_svi(otm_bid, out_iv_bid);
                fit_bid_ask_svi(otm_ask, out_iv_ask);
            }

        } catch (...) {
            continue;  // SVI fit failed entirely; leave all NaN
        }
    }

    // --- Phase 5: Build output Arrow table ---
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

    arrow::Date32Builder exp_builder;
    for (size_t i = 0; i < total_rows; ++i) {
        auto status = exp_builder.Append(out_expirations[i]);
        if (!status.ok())
            throw std::runtime_error("Expiration append failed");
    }
    std::shared_ptr<arrow::Array> exp_array;
    if (!exp_builder.Finish(&exp_array).ok())
        throw std::runtime_error("Expiration finish failed");

    arrow::DoubleBuilder delta_builder;
    for (size_t i = 0; i < total_rows; ++i) {
        auto status = delta_builder.Append(out_deltas[i]);
        if (!status.ok())
            throw std::runtime_error("Delta append failed");
    }
    std::shared_ptr<arrow::Array> delta_array;
    if (!delta_builder.Finish(&delta_array).ok())
        throw std::runtime_error("Delta finish failed");

    arrow::DoubleBuilder iv_builder;
    for (size_t i = 0; i < total_rows; ++i) {
        double val = out_ivs[i];
        if (std::isnan(val)) {
            if (!iv_builder.AppendNull().ok())
                throw std::runtime_error("IV AppendNull failed");
        } else {
            if (!iv_builder.Append(val).ok())
                throw std::runtime_error("IV Append failed");
        }
    }
    std::shared_ptr<arrow::Array> iv_array;
    if (!iv_builder.Finish(&iv_array).ok())
        throw std::runtime_error("IV finish failed");

    arrow::DoubleBuilder lm_builder;
    for (size_t i = 0; i < total_rows; ++i) {
        double val = out_lm[i];
        if (std::isnan(val)) {
            if (!lm_builder.AppendNull().ok())
                throw std::runtime_error("Log moneyness AppendNull failed");
        } else {
            if (!lm_builder.Append(val).ok())
                throw std::runtime_error("Log moneyness Append failed");
        }
    }
    std::shared_ptr<arrow::Array> lm_array;
    if (!lm_builder.Finish(&lm_array).ok())
        throw std::runtime_error("Log moneyness finish failed");

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
