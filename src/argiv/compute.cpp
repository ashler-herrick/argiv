#include "argiv/compute.hpp"
#include "argiv/core.hpp"

#include <vector>

#include <arrow/api.h>
#include <arrow/table.h>

namespace argiv {

namespace {

// Helper to get a double column's raw data (requires single-chunk table).
const double* get_double_col(const std::shared_ptr<arrow::Table>& table,
                             const std::string& name) {
    auto col = table->GetColumnByName(name);
    if (!col) {
        throw std::runtime_error("Missing column: " + name);
    }
    auto arr = std::static_pointer_cast<arrow::DoubleArray>(col->chunk(0));
    return arr->raw_values();
}

// Helper to get int32 column's raw data (requires single-chunk table).
const int32_t* get_int_col(const std::shared_ptr<arrow::Table>& table,
                           const std::string& name) {
    auto col = table->GetColumnByName(name);
    if (!col) {
        throw std::runtime_error("Missing column: " + name);
    }
    auto arr = std::static_pointer_cast<arrow::Int32Array>(col->chunk(0));
    return arr->raw_values();
}

}  // namespace

std::shared_ptr<arrow::Table> compute_greeks_table(
    const std::shared_ptr<arrow::Table>& input) {

    // Combine chunks upfront so raw pointers remain valid.
    auto combined_result = input->CombineChunks();
    if (!combined_result.ok()) {
        throw std::runtime_error("Failed to combine chunks: " +
                                 combined_result.status().ToString());
    }
    auto table = combined_result.MoveValueUnsafe();
    const int64_t n = table->num_rows();

    // Handle empty table
    if (n == 0) {
        auto result = table;
        for (const auto& name : {"iv", "delta", "gamma", "vega", "theta", "rho"}) {
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
    const double* market_price = get_double_col(table, "market_price");

    // Pre-allocate output vectors
    std::vector<double> iv(n), delta(n), gamma(n), vega(n), theta(n), rho(n);

    // Parallel computation
    #pragma omp parallel for schedule(dynamic, 256)
    for (int64_t i = 0; i < n; ++i) {
        auto res = compute_single(option_type[i], spot[i], strike[i],
                                  expiry[i], rate[i], dividend_yield[i],
                                  market_price[i]);
        iv[i] = res.iv;
        delta[i] = res.delta;
        gamma[i] = res.gamma;
        vega[i] = res.vega;
        theta[i] = res.theta;
        rho[i] = res.rho;
    }

    // Build output columns
    auto build_col = [&](const std::string& name,
                         const std::vector<double>& data)
        -> std::shared_ptr<arrow::ChunkedArray> {
        arrow::DoubleBuilder builder;
        auto status = builder.AppendValues(data);
        if (!status.ok()) {
            throw std::runtime_error("AppendValues failed for " + name);
        }
        std::shared_ptr<arrow::Array> arr;
        status = builder.Finish(&arr);
        if (!status.ok()) {
            throw std::runtime_error("Finish failed for " + name);
        }
        return std::make_shared<arrow::ChunkedArray>(arr);
    };

    // Start with input columns, append output columns
    auto result = table;
    result = *result->AddColumn(result->num_columns(),
                                arrow::field("iv", arrow::float64()),
                                build_col("iv", iv));
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
