"""Regression tests for type-mismatch and null-value bugs.

The compute_greeks C++ code uses static_pointer_cast to access Arrow column
buffers.  If the column type doesn't match (e.g. int8 instead of int32 for
option_type), the cast silently reinterprets the buffer with the wrong element
stride, causing out-of-bounds reads and NaN corruption that scales with table
size.  Similarly, Arrow null values are read as uninitialized buffer content.

These tests verify that:
  1. The Python wrapper auto-casts mismatched types transparently.
  2. Null values are rejected with a clear error.
  3. Missing columns are rejected with a clear error.
  4. The C++ safety net catches type mismatches if the wrapper is bypassed.
  5. Auto-cast results are identical to correctly-typed results.
  6. Large multi-chunk tables with wrong types don't produce silent corruption.
"""

import math

import numpy as np
import pyarrow as pa
import pytest

import argiv
from argiv._core import compute_greeks as _raw_compute_greeks
from argiv.helpers import _bs_call_price, _bs_put_price, generate_dataset

GREEK_COLS = ["iv", "delta", "gamma", "vega", "theta", "rho"]


def _make_correct_table(n=200):
    """Build a correctly-typed table with known-good data."""
    return generate_dataset(n)


def _replace_column(table, name, new_type):
    """Replace a column with the same data cast to a different Arrow type."""
    idx = table.schema.get_field_index(name)
    return table.set_column(
        idx,
        pa.field(name, new_type),
        table.column(idx).cast(new_type),
    )


def _assert_greeks_match(result_a, result_b, label=""):
    """Assert all Greek columns are identical (NaN == NaN)."""
    for col in GREEK_COLS:
        a = np.array(result_a.column(col).to_pylist(), dtype=float)
        b = np.array(result_b.column(col).to_pylist(), dtype=float)
        match = (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=1e-12)
        assert np.all(match), (
            f"{label}: column '{col}' differs — "
            f"{(~match).sum()} / {len(a)} mismatches"
        )


# ---------------------------------------------------------------------------
# Auto-casting: wrong types should be transparently fixed
# ---------------------------------------------------------------------------


class TestAutoCastOptionType:
    """option_type column: int8/int16/int64/uint8 should all auto-cast to int32."""

    @pytest.fixture(scope="class")
    def baseline(self):
        table = _make_correct_table()
        return table, argiv.compute_greeks(table)

    @pytest.mark.parametrize("dtype", [pa.int8(), pa.int16(), pa.int64()])
    def test_int_type_auto_cast(self, baseline, dtype):
        table, expected = baseline
        wrong = _replace_column(table, "option_type", dtype)
        assert wrong.schema.field("option_type").type == dtype
        result = argiv.compute_greeks(wrong)
        _assert_greeks_match(expected, result, f"option_type as {dtype}")


class TestAutoCastDoubleColumns:
    """Float columns: float32 should auto-cast to float64."""

    DOUBLE_COLS = ["spot", "strike", "expiry", "rate", "dividend_yield", "market_price"]

    @pytest.fixture(scope="class")
    def baseline(self):
        table = _make_correct_table()
        return table, argiv.compute_greeks(table)

    @pytest.mark.parametrize("col_name", DOUBLE_COLS)
    def test_float32_auto_cast(self, baseline, col_name):
        table, expected = baseline
        wrong = _replace_column(table, col_name, pa.float32())
        result = argiv.compute_greeks(wrong)
        # float32 truncation means results won't be bit-identical, but should
        # be close (the important thing is no silent corruption / all-NaN).
        for greek in GREEK_COLS:
            a = np.array(expected.column(greek).to_pylist(), dtype=float)
            b = np.array(result.column(greek).to_pylist(), dtype=float)
            both_nan = np.isnan(a) & np.isnan(b)
            both_valid = ~np.isnan(a) & ~np.isnan(b)
            # NaN pattern should be very similar
            assert np.sum(np.isnan(a)) == pytest.approx(
                np.sum(np.isnan(b)), abs=5
            ), f"{col_name} as float32: NaN count mismatch in {greek}"
            # Valid values should be close
            if both_valid.any():
                assert np.allclose(
                    a[both_valid], b[both_valid], rtol=1e-4, atol=1e-8
                ), f"{col_name} as float32: values diverge in {greek}"


class TestAutoCastBidAsk:
    """Optional bid_price/ask_price columns: float32 should auto-cast."""

    def test_bid_ask_float32(self):
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20
        mid = _bs_call_price(S, K, T, r, q, sigma)
        bid = _bs_call_price(S, K, T, r, q, sigma - 0.02)
        ask = _bs_call_price(S, K, T, r, q, sigma + 0.02)

        table = pa.table({
            "option_type": pa.array([1], type=pa.int32()),
            "spot": pa.array([S], type=pa.float64()),
            "strike": pa.array([K], type=pa.float64()),
            "expiry": pa.array([T], type=pa.float64()),
            "rate": pa.array([r], type=pa.float64()),
            "dividend_yield": pa.array([q], type=pa.float64()),
            "market_price": pa.array([mid], type=pa.float64()),
            "bid_price": pa.array([bid], type=pa.float32()),
            "ask_price": pa.array([ask], type=pa.float32()),
        })
        result = argiv.compute_greeks(table)
        assert "iv_bid" in result.column_names
        assert "iv_ask" in result.column_names
        iv_bid = result.column("iv_bid")[0].as_py()
        iv_ask = result.column("iv_ask")[0].as_py()
        assert not math.isnan(iv_bid)
        assert not math.isnan(iv_ask)
        assert iv_bid < iv_ask


# ---------------------------------------------------------------------------
# Null rejection
# ---------------------------------------------------------------------------


class TestNullRejection:
    """Null values in any required column should raise ValueError."""

    REQUIRED_COLS = [
        "option_type", "spot", "strike", "expiry",
        "rate", "dividend_yield", "market_price",
    ]

    def _table_with_null(self, col_name):
        """Create a 10-row table with a null in the specified column."""
        table = _make_correct_table(10)
        idx = table.schema.get_field_index(col_name)
        col = table.column(idx)
        arr = col.chunk(0)
        # Replace first value with null
        values = arr.to_pylist()
        values[0] = None
        new_arr = pa.array(values, type=arr.type)
        return table.set_column(idx, table.schema.field(idx), new_arr)

    @pytest.mark.parametrize("col_name", REQUIRED_COLS)
    def test_null_in_required_column_raises(self, col_name):
        table = self._table_with_null(col_name)
        with pytest.raises(ValueError, match="null"):
            argiv.compute_greeks(table)


# ---------------------------------------------------------------------------
# Missing column rejection
# ---------------------------------------------------------------------------


class TestMissingColumns:
    """Missing required columns should raise ValueError."""

    REQUIRED_COLS = [
        "option_type", "spot", "strike", "expiry",
        "rate", "dividend_yield", "market_price",
    ]

    @pytest.mark.parametrize("col_name", REQUIRED_COLS)
    def test_missing_required_column_raises(self, col_name):
        table = _make_correct_table(10)
        idx = table.schema.get_field_index(col_name)
        table = table.remove_column(idx)
        with pytest.raises(ValueError, match="Missing"):
            argiv.compute_greeks(table)


# ---------------------------------------------------------------------------
# C++ safety net: bypassing the Python wrapper
# ---------------------------------------------------------------------------


class TestCppSafetyNet:
    """Direct calls to _core with wrong types should raise RuntimeError."""

    def test_int8_option_type_caught(self):
        table = _make_correct_table(10)
        wrong = _replace_column(table, "option_type", pa.int8())
        with pytest.raises(RuntimeError, match="expected int32"):
            _raw_compute_greeks(wrong)

    def test_float32_column_caught(self):
        table = _make_correct_table(10)
        wrong = _replace_column(table, "spot", pa.float32())
        with pytest.raises(RuntimeError, match="expected float64"):
            _raw_compute_greeks(wrong)

    def test_null_column_caught(self):
        table = _make_correct_table(10)
        idx = table.schema.get_field_index("spot")
        values = table.column(idx).chunk(0).to_pylist()
        values[0] = None
        new_arr = pa.array(values, type=pa.float64())
        table = table.set_column(idx, table.schema.field(idx), new_arr)
        with pytest.raises(RuntimeError, match="null"):
            _raw_compute_greeks(table)


# ---------------------------------------------------------------------------
# Regression: large multi-chunk tables with wrong types
#
# This is the specific scenario from the original bug report. With int8
# option_type on a large table, static_pointer_cast<Int32Array> reads with
# 4-byte stride from a 1-byte buffer, causing OOB reads and NaN corruption.
# ---------------------------------------------------------------------------


class TestLargeTableTypeSafety:
    """Verify no silent corruption on large tables with various type widths."""

    @pytest.fixture(scope="class")
    def baseline_large(self):
        table = generate_dataset(50_000)
        return table, argiv.compute_greeks(table)

    def test_int8_option_type_large(self, baseline_large):
        """int8 option_type on 50K rows must match int32 baseline."""
        table, expected = baseline_large
        wrong = _replace_column(table, "option_type", pa.int8())
        result = argiv.compute_greeks(wrong)
        _assert_greeks_match(expected, result, "50K rows int8 option_type")

    def test_int64_option_type_large(self, baseline_large):
        """int64 option_type on 50K rows must match int32 baseline."""
        table, expected = baseline_large
        wrong = _replace_column(table, "option_type", pa.int64())
        result = argiv.compute_greeks(wrong)
        _assert_greeks_match(expected, result, "50K rows int64 option_type")

    def test_float32_spot_large(self, baseline_large):
        """float32 spot on 50K rows — NaN count should be similar to baseline."""
        table, expected = baseline_large
        wrong = _replace_column(table, "spot", pa.float32())
        result = argiv.compute_greeks(wrong)
        for greek in GREEK_COLS:
            a = np.array(expected.column(greek).to_pylist(), dtype=float)
            b = np.array(result.column(greek).to_pylist(), dtype=float)
            nan_diff = abs(int(np.isnan(a).sum()) - int(np.isnan(b).sum()))
            assert nan_diff < 50, (
                f"float32 spot: NaN count diff {nan_diff} in {greek}"
            )

    def test_multichunk_wrong_type(self, baseline_large):
        """Multi-chunk table with wrong type should still auto-cast correctly."""
        table, expected = baseline_large
        wrong = _replace_column(table, "option_type", pa.int8())
        # Split into 50 chunks of 1000
        chunks = []
        for start in range(0, wrong.num_rows, 1000):
            end = min(start + 1000, wrong.num_rows)
            chunks.append(wrong.slice(start, end - start))
        multichunk = pa.concat_tables(chunks)
        assert multichunk.column("option_type").num_chunks > 1

        result = argiv.compute_greeks(multichunk)
        _assert_greeks_match(expected, result, "multi-chunk int8 option_type")


# ---------------------------------------------------------------------------
# Regression: NaN rate should not depend on batch size
# ---------------------------------------------------------------------------


class TestBatchSizeInvariance:
    """The NaN rate for identical data must not change with batch size.

    This is the core regression test for the original bug: processing the
    same data as one large table vs many small batches must produce the
    same NaN pattern.
    """

    def test_nan_rate_matches_across_batch_sizes(self):
        table = generate_dataset(10_000)
        # Process as one large batch
        full_result = argiv.compute_greeks(table)
        full_iv = np.array(full_result.column("iv").to_pylist(), dtype=float)
        full_nan = np.isnan(full_iv)

        # Process in small batches of 500
        batch_ivs = []
        for start in range(0, table.num_rows, 500):
            end = min(start + 500, table.num_rows)
            batch = table.slice(start, end - start)
            result = argiv.compute_greeks(batch)
            batch_ivs.extend(result.column("iv").to_pylist())
        batch_iv = np.array(batch_ivs, dtype=float)
        batch_nan = np.isnan(batch_iv)

        # NaN patterns must be identical
        assert np.array_equal(full_nan, batch_nan), (
            f"NaN pattern differs: full has {full_nan.sum()}, "
            f"batched has {batch_nan.sum()}"
        )

        # Non-NaN values must be identical
        both_valid = ~full_nan & ~batch_nan
        if both_valid.any():
            assert np.allclose(
                full_iv[both_valid], batch_iv[both_valid], rtol=1e-12
            ), "Valid IV values differ between full and batched processing"
