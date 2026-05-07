import math

import numpy as np
import pyarrow as pa
import pytest

import argiv


class TestComputeGreeksFromIV:
    """Test the iv-input path of compute_greeks (no Brent solve)."""

    def test_greeks_match_price_path(self):
        """Greeks from pre-computed IV should match those from Brent solve."""
        table = pa.table({
            "option_type": pa.array([1, -1, 1, -1], type=pa.int32()),
            "spot": [100.0, 100.0, 100.0, 100.0],
            "strike": [105.0, 95.0, 110.0, 90.0],
            "expiry": [0.5, 0.5, 1.0, 1.0],
            "rate": [0.05, 0.05, 0.03, 0.03],
            "dividend_yield": [0.02, 0.02, 0.01, 0.01],
            "market_price": [5.0, 3.5, 6.0, 4.0],
        })
        enriched = argiv.compute_greeks(table)

        from_iv_input = pa.table({
            "option_type": enriched.column("option_type"),
            "spot": enriched.column("spot"),
            "strike": enriched.column("strike"),
            "expiry": enriched.column("expiry"),
            "rate": enriched.column("rate"),
            "dividend_yield": enriched.column("dividend_yield"),
            "iv": enriched.column("iv"),
        })
        from_iv = argiv.compute_greeks(from_iv_input)

        for col in ["delta", "gamma", "vega", "theta", "rho"]:
            a = enriched.column(col).to_pylist()
            b = from_iv.column(col).to_pylist()
            assert np.allclose(a, b, rtol=1e-10), f"{col} mismatch"

    def test_known_bs_values(self):
        """Verify against hand-computed Black-Scholes Greeks."""
        from scipy.stats import norm

        S, K, T, r, q, sigma = 100.0, 100.0, 0.25, 0.05, 0.02, 0.20
        table = pa.table({
            "option_type": pa.array([1], type=pa.int32()),
            "spot": [S], "strike": [K], "expiry": [T],
            "rate": [r], "dividend_yield": [q], "iv": [sigma],
        })
        result = argiv.compute_greeks(table)

        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        expected_delta = math.exp(-q * T) * norm.cdf(d1)
        assert abs(result.column("delta")[0].as_py() - expected_delta) < 1e-6

        assert result.column("gamma")[0].as_py() > 0
        assert result.column("vega")[0].as_py() > 0

    def test_empty_table(self):
        """Empty input should return empty output with correct schema."""
        table = pa.table({
            "option_type": pa.array([], type=pa.int32()),
            "spot": pa.array([], type=pa.float64()),
            "strike": pa.array([], type=pa.float64()),
            "expiry": pa.array([], type=pa.float64()),
            "rate": pa.array([], type=pa.float64()),
            "dividend_yield": pa.array([], type=pa.float64()),
            "iv": pa.array([], type=pa.float64()),
        })
        result = argiv.compute_greeks(table)
        assert result.num_rows == 0
        assert "delta" in result.column_names
        assert "gamma" in result.column_names

    def test_put_call_symmetry(self):
        """ATM call and put gamma/vega should be identical."""
        table = pa.table({
            "option_type": pa.array([1, -1], type=pa.int32()),
            "spot": [100.0, 100.0],
            "strike": [100.0, 100.0],
            "expiry": [0.5, 0.5],
            "rate": [0.05, 0.05],
            "dividend_yield": [0.05, 0.05],  # r=q so forward=spot
            "iv": [0.25, 0.25],
        })
        result = argiv.compute_greeks(table)
        gamma = result.column("gamma").to_pylist()
        vega = result.column("vega").to_pylist()

        assert abs(gamma[0] - gamma[1]) < 1e-10
        assert abs(vega[0] - vega[1]) < 1e-10

    def test_missing_iv_and_price_raises(self):
        """Without either iv or market_price, compute_greeks should error."""
        table = pa.table({
            "option_type": pa.array([1], type=pa.int32()),
            "spot": [100.0],
            "strike": [100.0],
            "expiry": [1.0],
            "rate": [0.05],
            "dividend_yield": [0.0],
        })
        with pytest.raises(ValueError, match="iv.*market_price|market_price.*iv"):
            argiv.compute_greeks(table)
