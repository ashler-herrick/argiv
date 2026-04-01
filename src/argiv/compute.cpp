#include "argiv/compute.hpp"
#include "argiv/core.hpp"

#include <vector>

#include <arrow/api.h>
#include <arrow/table.h>

namespace argiv {

namespace {

// Helper to get a double column's raw data (requires single-chunk table).
// Validates that the column is actually float64 and contains no nulls.
const double* get_double_col(const std::shared_ptr<arrow::Table>& table,
                             const std::string& name) {
    auto col = table->GetColumnByName(name);
    if (!col) {
        throw std::runtime_error("Missing column: " + name);
    }
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

// Helper to get int32 column's raw data (requires single-chunk table).
// Validates that the column is actually int32 and contains no nulls.
const int32_t* get_int_col(const std::shared_ptr<arrow::Table>& table,
                           const std::string& name) {
    auto col = table->GetColumnByName(name);
    if (!col) {
        throw std::runtime_error("Missing column: " + name);
    }
    auto chunk = col->chunk(0);
    if (chunk->type_id() != arrow::Type::INT32) {
        throw std::runtime_error(
            "Column '" + name + "' has type " + chunk->type()->ToString() +
            ", expected int32. Cast the column before passing to argiv.");
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

// Optional column accessor (returns nullptr if column not present).
// Validates type and nulls if column exists.
const double* try_get_double_col(const std::shared_ptr<arrow::Table>& table,
                                 const std::string& name) {
    auto col = table->GetColumnByName(name);
    if (!col) return nullptr;
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

    // Sanity check: CombineChunks must produce single-chunk columns.
    if (n > 0) {
        for (int c = 0; c < table->num_columns(); ++c) {
            auto col = table->column(c);
            if (col->num_chunks() != 1) {
                throw std::runtime_error(
                    "Internal error: column '" +
                    table->schema()->field(c)->name() + "' has " +
                    std::to_string(col->num_chunks()) +
                    " chunks after CombineChunks (expected 1)");
            }
        }
    }

    // Detect optional bid/ask columns
    const double* bid_price_col = try_get_double_col(table, "bid_price");
    const double* ask_price_col = try_get_double_col(table, "ask_price");
    const bool has_bid_ask = (bid_price_col != nullptr && ask_price_col != nullptr);

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
        if (has_bid_ask) {
            for (const auto& name : {"iv_bid", "iv_ask"}) {
                auto empty_arr = std::make_shared<arrow::DoubleArray>(0, nullptr);
                auto empty_chunked = std::make_shared<arrow::ChunkedArray>(empty_arr);
                result = *result->AddColumn(result->num_columns(),
                                            arrow::field(name, arrow::float64()),
                                            empty_chunked);
            }
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
    std::vector<double> iv_bid, iv_ask;
    if (has_bid_ask) {
        iv_bid.resize(n);
        iv_ask.resize(n);
    }

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

        if (has_bid_ask) {
            iv_bid[i] = compute_single(option_type[i], spot[i], strike[i],
                                       expiry[i], rate[i], dividend_yield[i],
                                       bid_price_col[i]).iv;
            iv_ask[i] = compute_single(option_type[i], spot[i], strike[i],
                                       expiry[i], rate[i], dividend_yield[i],
                                       ask_price_col[i]).iv;
        }
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

    if (has_bid_ask) {
        result = *result->AddColumn(result->num_columns(),
                                    arrow::field("iv_bid", arrow::float64()),
                                    build_col("iv_bid", iv_bid));
        result = *result->AddColumn(result->num_columns(),
                                    arrow::field("iv_ask", arrow::float64()),
                                    build_col("iv_ask", iv_ask));
    }

    return result;
}

}  // namespace argiv
