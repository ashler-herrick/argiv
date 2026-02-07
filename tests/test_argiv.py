import math

import pyarrow as pa
import pytest

import argiv


def _bs_call_price(S, K, T, r, q, sigma):
    """Reference Black-Scholes call price for test verification."""
    from math import exp, log, sqrt

    from scipy.stats import norm

    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def _bs_put_price(S, K, T, r, q, sigma):
    """Reference Black-Scholes put price via put-call parity."""
    from math import exp

    call = _bs_call_price(S, K, T, r, q, sigma)
    return call - S * exp(-q * T) + K * exp(-r * T)


def _make_table(option_type, spot, strike, expiry, rate, div_yield, price):
    """Build a single-row pyarrow table."""
    return pa.table(
        {
            "option_type": pa.array([option_type], type=pa.int32()),
            "spot": pa.array([spot], type=pa.float64()),
            "strike": pa.array([strike], type=pa.float64()),
            "expiry": pa.array([expiry], type=pa.float64()),
            "rate": pa.array([rate], type=pa.float64()),
            "dividend_yield": pa.array([div_yield], type=pa.float64()),
            "market_price": pa.array([price], type=pa.float64()),
        }
    )


class TestIVRecovery:
    """Test that implied volatility is correctly recovered from known BS prices."""

    def test_call_iv_recovery(self):
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
        price = _bs_call_price(S, K, T, r, q, sigma)
        table = _make_table(1, S, K, T, r, q, price)
        result = argiv.compute_greeks(table)
        iv = result.column("iv")[0].as_py()
        assert abs(iv - sigma) < 1e-4, f"Expected IV ~{sigma}, got {iv}"

    def test_put_iv_recovery(self):
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
        price = _bs_put_price(S, K, T, r, q, sigma)
        table = _make_table(-1, S, K, T, r, q, price)
        result = argiv.compute_greeks(table)
        iv = result.column("iv")[0].as_py()
        assert abs(iv - sigma) < 1e-4, f"Expected IV ~{sigma}, got {iv}"

    def test_otm_call_iv(self):
        S, K, T, r, q, sigma = 100.0, 120.0, 0.5, 0.03, 0.01, 0.3
        price = _bs_call_price(S, K, T, r, q, sigma)
        table = _make_table(1, S, K, T, r, q, price)
        result = argiv.compute_greeks(table)
        iv = result.column("iv")[0].as_py()
        assert abs(iv - sigma) < 1e-4

    def test_itm_put_iv(self):
        S, K, T, r, q, sigma = 100.0, 80.0, 0.25, 0.02, 0.0, 0.25
        price = _bs_put_price(S, K, T, r, q, sigma)
        table = _make_table(-1, S, K, T, r, q, price)
        result = argiv.compute_greeks(table)
        iv = result.column("iv")[0].as_py()
        assert abs(iv - sigma) < 1e-4


class TestGreeksSigns:
    """Test that Greeks have correct signs."""

    def test_call_delta_positive(self):
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
        price = _bs_call_price(S, K, T, r, q, sigma)
        table = _make_table(1, S, K, T, r, q, price)
        result = argiv.compute_greeks(table)
        delta = result.column("delta")[0].as_py()
        assert 0 < delta < 1, f"Call delta should be in (0,1), got {delta}"

    def test_put_delta_negative(self):
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
        price = _bs_put_price(S, K, T, r, q, sigma)
        table = _make_table(-1, S, K, T, r, q, price)
        result = argiv.compute_greeks(table)
        delta = result.column("delta")[0].as_py()
        assert -1 < delta < 0, f"Put delta should be in (-1,0), got {delta}"

    def test_gamma_positive(self):
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
        price = _bs_call_price(S, K, T, r, q, sigma)
        table = _make_table(1, S, K, T, r, q, price)
        result = argiv.compute_greeks(table)
        gamma = result.column("gamma")[0].as_py()
        assert gamma > 0, f"Gamma should be positive, got {gamma}"

    def test_vega_positive(self):
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
        price = _bs_call_price(S, K, T, r, q, sigma)
        table = _make_table(1, S, K, T, r, q, price)
        result = argiv.compute_greeks(table)
        vega = result.column("vega")[0].as_py()
        assert vega > 0, f"Vega should be positive, got {vega}"

    def test_call_rho_positive(self):
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
        price = _bs_call_price(S, K, T, r, q, sigma)
        table = _make_table(1, S, K, T, r, q, price)
        result = argiv.compute_greeks(table)
        rho = result.column("rho")[0].as_py()
        assert rho > 0, f"Call rho should be positive, got {rho}"

    def test_put_rho_negative(self):
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
        price = _bs_put_price(S, K, T, r, q, sigma)
        table = _make_table(-1, S, K, T, r, q, price)
        result = argiv.compute_greeks(table)
        rho = result.column("rho")[0].as_py()
        assert rho < 0, f"Put rho should be negative, got {rho}"


class TestMultiRow:
    """Test batch computation with multiple rows."""

    def test_batch(self):
        n = 100
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
        call_price = _bs_call_price(S, K, T, r, q, sigma)
        table = pa.table(
            {
                "option_type": pa.array([1] * n, type=pa.int32()),
                "spot": pa.array([S] * n, type=pa.float64()),
                "strike": pa.array([K] * n, type=pa.float64()),
                "expiry": pa.array([T] * n, type=pa.float64()),
                "rate": pa.array([r] * n, type=pa.float64()),
                "dividend_yield": pa.array([q] * n, type=pa.float64()),
                "market_price": pa.array([call_price] * n, type=pa.float64()),
            }
        )
        result = argiv.compute_greeks(table)
        assert result.num_rows == n
        assert result.num_columns == 7 + 6  # input + output columns
        ivs = result.column("iv").to_pylist()
        for iv in ivs:
            assert abs(iv - sigma) < 1e-4
