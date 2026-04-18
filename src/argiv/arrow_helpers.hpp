#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <arrow/api.h>
#include <arrow/table.h>

namespace argiv {

// Combine all chunks in a table and validate single-chunk invariant.
inline std::shared_ptr<arrow::Table> combine_and_validate(
    const std::shared_ptr<arrow::Table>& input) {
    auto combined_result = input->CombineChunks();
    if (!combined_result.ok()) {
        throw std::runtime_error("Failed to combine chunks: " +
                                 combined_result.status().ToString());
    }
    auto table = combined_result.MoveValueUnsafe();
    if (table->num_rows() > 0) {
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
    return table;
}

// Get a required float64 column's raw data (requires single-chunk table).
inline const double* get_double_col(
    const std::shared_ptr<arrow::Table>& table,
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

// Get an optional float64 column (returns nullptr if not present).
inline const double* try_get_double_col(
    const std::shared_ptr<arrow::Table>& table,
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

// Get a required int32 column's raw data.
inline const int32_t* get_int_col(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& name) {
    auto col = table->GetColumnByName(name);
    if (!col)
        throw std::runtime_error("Missing column: " + name);
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

// Get a required int32 or date32 column (both use int32 physical storage).
inline const int32_t* get_int32_col(
    const std::shared_ptr<arrow::Table>& table,
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

// Get a required int64 or timestamp column (both use int64 physical storage).
inline const int64_t* get_int64_col(
    const std::shared_ptr<arrow::Table>& table,
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

}  // namespace argiv
