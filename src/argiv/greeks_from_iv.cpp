#include "argiv/arrow_helpers.hpp"
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
    const std::shared_ptr<arrow::Table>& input) {

    auto table = combine_and_validate(input);
    const int64_t n = table->num_rows();

    // Handle empty table
    if (n == 0) {
        auto result = table;
        for (const auto& name : {"delta", "gamma", "vega", "theta", "rho"}) {
            auto empty_arr = std::make_shared<arrow::DoubleArray>(0, nullptr);
            auto empty_chunked = std::make_shared<arrow::ChunkedArray>(empty_arr);
            result = *result->AddColumn(result->num_columns(),
                                        arrow::field(name, arrow::float64()),
                                        empty_chunked);
        }
        return result;
    }

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

    #pragma omp parallel for schedule(dynamic, 256)
    for (int64_t i = 0; i < n; ++i) {
        double T = expiry[i];
        double S = spot[i];
        double K = strike[i];
        double sigma = iv[i];

        if (T <= 0.0 || S <= 0.0 || K <= 0.0 || sigma <= 0.0) {
            delta[i] = NaN;
            gamma[i] = NaN;
            vega[i] = NaN;
            theta[i] = NaN;
            rho[i] = NaN;
            continue;
        }

        double r = rate[i];
        double q = dividend_yield[i];
        auto ql_type = (option_type[i] == 1) ? QuantLib::Option::Call
                                              : QuantLib::Option::Put;

        double discount = std::exp(-r * T);
        double forward = S * std::exp((r - q) * T);
        double stdDev = sigma * std::sqrt(T);

        QuantLib::BlackCalculator calc(ql_type, K, forward, stdDev, discount);

        delta[i] = calc.delta(S);
        gamma[i] = calc.gamma(S);
        vega[i] = calc.vega(T);
        theta[i] = calc.theta(S, T);
        rho[i] = calc.rho(T);
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

    return result;
}

}  // namespace argiv
