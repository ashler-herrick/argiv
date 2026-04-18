import datetime
import math

import numpy as np
import pyarrow as pa
import pytest

import argiv


def _bs_price(option_type, S, K, T, r, q, sigma):
    from scipy.stats import norm

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 1:
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


class TestComputeGreeksFromIV:
    """Test that compute_greeks_from_iv matches compute_greeks output."""

    def test_greeks_match_compute_greeks(self):
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
        from_iv = argiv.compute_greeks_from_iv(from_iv_input)

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
        result = argiv.compute_greeks_from_iv(table)

        # Delta should be close to N(d1) * exp(-q*T) for a call
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        expected_delta = math.exp(-q * T) * norm.cdf(d1)
        assert abs(result.column("delta")[0].as_py() - expected_delta) < 1e-6

        # Gamma must be positive
        assert result.column("gamma")[0].as_py() > 0
        # Vega must be positive
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
        result = argiv.compute_greeks_from_iv(table)
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
        result = argiv.compute_greeks_from_iv(table)
        gamma = result.column("gamma").to_pylist()
        vega = result.column("vega").to_pylist()

        assert abs(gamma[0] - gamma[1]) < 1e-10
        assert abs(vega[0] - vega[1]) < 1e-10

    def test_missing_column_raises(self):
        """Missing required column should raise ValueError."""
        table = pa.table({
            "option_type": pa.array([1], type=pa.int32()),
            "spot": [100.0],
            "strike": [100.0],
            # missing: expiry, rate, dividend_yield, iv
        })
        with pytest.raises(ValueError, match="Missing required column"):
            argiv.compute_greeks_from_iv(table)


class TestGreeksAtPillars:
    """Test the greeks_at_pillars helper."""

    def _make_enriched_and_surface(self):
        S, T, r, q = 100.0, 0.5, 0.05, 0.01
        base_sigma, skew = 0.20, 0.00005
        strikes = np.linspace(80, 120, 30)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp_date = datetime.date(2024, 7, 15)

        options = []
        for K in strikes:
            sigma = base_sigma + skew * (K - S) ** 2
            ot = -1 if K < S else 1
            options.append({
                "option_type": ot, "spot": S, "strike": K, "expiry": T,
                "rate": r, "dividend_yield": q,
                "market_price": _bs_price(ot, S, K, T, r, q, sigma),
            })

        n = len(options)
        raw = pa.table({
            "option_type": pa.array([o["option_type"] for o in options], type=pa.int32()),
            "spot": [o["spot"] for o in options],
            "strike": [o["strike"] for o in options],
            "expiry": [o["expiry"] for o in options],
            "rate": [o["rate"] for o in options],
            "dividend_yield": [o["dividend_yield"] for o in options],
            "market_price": [o["market_price"] for o in options],
            "timestamp": pa.array([ts] * n, type=pa.timestamp("us")),
            "expiration": pa.array([exp_date] * n, type=pa.date32()),
        })

        enriched = argiv.compute_greeks(raw)
        surface = argiv.fit_vol_surface(enriched)
        return enriched, surface

    def test_output_columns(self):
        enriched, surface = self._make_enriched_and_surface()
        result = argiv.greeks_at_pillars(enriched, surface)

        for col in ["strike", "gamma", "vega", "theta", "rho"]:
            assert col in result.column_names

        # Should also retain surface columns
        for col in ["timestamp", "expiration", "delta", "iv"]:
            assert col in result.column_names

    def test_greeks_reasonable(self):
        enriched, surface = self._make_enriched_and_surface()
        result = argiv.greeks_at_pillars(enriched, surface)
        df = result.to_pandas()

        assert (df["gamma"] > 0).all(), "gamma should be positive"
        assert (df["vega"] > 0).all(), "vega should be positive"
        assert (df["theta"] < 0).all(), "theta should be negative"

    def test_strike_from_log_moneyness(self):
        """Strike should equal spot * exp(log_moneyness)."""
        enriched, surface = self._make_enriched_and_surface()
        result = argiv.greeks_at_pillars(enriched, surface)
        df = result.to_pandas()

        # Get spot from enriched
        spot = enriched.column("spot")[0].as_py()
        expected_strikes = spot * np.exp(df["log_moneyness"].values)
        np.testing.assert_allclose(df["strike"].values, expected_strikes, rtol=1e-10)

    def test_missing_enriched_column_raises(self):
        _, surface = self._make_enriched_and_surface()
        bad_enriched = pa.table({"timestamp": pa.array([], type=pa.timestamp("us"))})
        with pytest.raises(ValueError, match="Missing required column"):
            argiv.greeks_at_pillars(bad_enriched, surface)
