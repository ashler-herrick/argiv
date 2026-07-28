#include "argiv/arrow_helpers.hpp"
#include "argiv/core.hpp"
#include "argiv/greeks_from_iv.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include <arrow/api.h>
#include <arrow/table.h>
#include <ql/pricingengines/blackcalculator.hpp>

namespace argiv {

namespace {
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
}  // namespace

std::shared_ptr<arrow::Table> compute_greeks_from_iv_table(
    const std::shared_ptr<arrow::Table>& input, bool higher_order) {

    auto table = combine_and_validate(input);
    const int64_t n = table->num_rows();

    // Extract input columns
    const int32_t* option_type = get_int_col(table, "option_type");
    const double* spot = get_double_col(table, "spot");
    const double* strike = get_double_col(table, "strike");
    const double* expiry = get_double_col(table, "expiry");
    const double* rate = get_double_col(table, "rate");
    const double* dividend_yield = get_double_col(table, "dividend_yield");
    const double* iv = get_double_col(table, "iv");

    // Pre-allocate output vectors
    std::vector<double> delta(n), gamma(n), vega(n), theta(n), rho(n);
    const int64_t hn = higher_order ? n : 0;
    std::vector<double> vanna(hn), volga(hn), charm(hn), speed(hn), zomma(hn),
        color(hn);

    #pragma omp parallel for schedule(dynamic, 256)
    for (int64_t i = 0; i < n; ++i) {
        double T = expiry[i];
        double S = spot[i];
        double K = strike[i];
        double sigma = iv[i];
        double r = rate[i];
        double q = dividend_yield[i];

        double discount = std::exp(-r * T);
        double forward = S * std::exp((r - q) * T);
        double stdDev = sigma * std::sqrt(T);

        // forward/discount/stdDev cover non-finite r and q (neither is
        // sign-constrained, so they are only checked through the quantities
        // QuantLib consumes).
        if (!pos_finite(T) || !pos_finite(S) || !pos_finite(K) ||
            !pos_finite(sigma) || !pos_finite(forward) ||
            !pos_finite(discount) || !pos_finite(stdDev)) {
            delta[i] = NaN;
            gamma[i] = NaN;
            vega[i] = NaN;
            theta[i] = NaN;
            rho[i] = NaN;
            if (higher_order) {
                vanna[i] = NaN;
                volga[i] = NaN;
                charm[i] = NaN;
                speed[i] = NaN;
                zomma[i] = NaN;
                color[i] = NaN;
            }
            continue;
        }

        auto ql_type = (option_type[i] == 1) ? QuantLib::Option::Call
                                              : QuantLib::Option::Put;

        QuantLib::BlackCalculator calc(ql_type, K, forward, stdDev, discount);

        delta[i] = calc.delta(S);
        gamma[i] = calc.gamma(S);
        vega[i] = calc.vega(T);
        theta[i] = calc.theta(S, T);
        rho[i] = calc.rho(T);

        if (higher_order) {
            auto h = higher_order_greeks(S, K, forward, T, r, q, sigma,
                                         delta[i], gamma[i], vega[i]);
            vanna[i] = h.vanna;
            volga[i] = h.volga;
            charm[i] = h.charm;
            speed[i] = h.speed;
            zomma[i] = h.zomma;
            color[i] = h.color;
        }
    }

    // Build output columns
    auto build_col = [&](const std::string& name,
                         const std::vector<double>& data)
        -> std::shared_ptr<arrow::ChunkedArray> {
        arrow::DoubleBuilder builder;
        auto status = builder.AppendValues(data);
        if (!status.ok())
            throw std::runtime_error("AppendValues failed for " + name);
        std::shared_ptr<arrow::Array> arr;
        status = builder.Finish(&arr);
        if (!status.ok())
            throw std::runtime_error("Finish failed for " + name);
        return std::make_shared<arrow::ChunkedArray>(arr);
    };

    auto result = table;
    result = *result->AddColumn(result->num_columns(),
                                arrow::field("delta", arrow::float64()),
                                build_col("delta", delta));
    result = *result->AddColumn(result->num_columns(),
                                arrow::field("gamma", arrow::float64()),
                                build_col("gamma", gamma));
    result = *result->AddColumn(result->num_columns(),
                                arrow::field("vega", arrow::float64()),
                                build_col("vega", vega));
    result = *result->AddColumn(result->num_columns(),
                                arrow::field("theta", arrow::float64()),
                                build_col("theta", theta));
    result = *result->AddColumn(result->num_columns(),
                                arrow::field("rho", arrow::float64()),
                                build_col("rho", rho));

    if (higher_order) {
        auto add_col = [&](const char* name, const std::vector<double>& data) {
            result = *result->AddColumn(result->num_columns(),
                                        arrow::field(name, arrow::float64()),
                                        build_col(name, data));
        };
        add_col("vanna", vanna);
        add_col("volga", volga);
        add_col("charm", charm);
        add_col("speed", speed);
        add_col("zomma", zomma);
        add_col("color", color);
    }

    return result;
}

}  // namespace argiv
