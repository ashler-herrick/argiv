"""Tests for the Schadner closed-form IV solver (arXiv:2604.24480)."""

import numpy as np
import pyarrow as pa
import pytest
from scipy.stats import norm

from argiv import compute_greeks


def _bs(S, K, T, r, q, sigma, is_call):
    F = S * np.exp((r - q) * T)
    D = np.exp(-r * T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


_CASES = [
    # (option_type, S, K, T, r, q, sigma_true)
    (1, 100.0, 100.0, 1.0, 0.05, 0.0, 0.20),    # ATM call
    (1, 100.0, 110.0, 1.0, 0.05, 0.0, 0.25),    # OTM call
    (1, 100.0, 90.0, 1.0, 0.05, 0.0, 0.30),     # ITM call
    (1, 100.0, 80.0, 0.5, 0.03, 0.01, 0.40),
    (1, 100.0, 120.0, 2.0, 0.04, 0.02, 0.15),
    (1, 100.0, 130.0, 0.05, 0.01, 0.0, 0.60),   # short-dated OTM
    (-1, 100.0, 100.0, 1.0, 0.05, 0.0, 0.20),   # ATM put
    (-1, 100.0, 110.0, 1.0, 0.05, 0.0, 0.25),   # ITM put
    (-1, 100.0, 90.0, 1.0, 0.05, 0.0, 0.30),    # OTM put
    (-1, 100.0, 80.0, 0.5, 0.03, 0.01, 0.40),
    (-1, 100.0, 120.0, 2.0, 0.04, 0.02, 0.15),
    (-1, 100.0, 70.0, 0.05, 0.01, 0.0, 0.60),   # short-dated OTM put
]


def _build_table(cases):
    prices = [
        _bs(S, K, T, r, q, sigma, ot == 1)
        for ot, S, K, T, r, q, sigma in cases
    ]
    return pa.table({
        "option_type": pa.array([c[0] for c in cases], type=pa.int32()),
        "spot": [c[1] for c in cases],
        "strike": [c[2] for c in cases],
        "expiry": [c[3] for c in cases],
        "rate": [c[4] for c in cases],
        "dividend_yield": [c[5] for c in cases],
        "market_price": prices,
    })


def test_schadner_recovers_true_iv():
    """Schadner closed-form should recover true IV to ~1e-12 for calls and puts."""
    tbl = _build_table(_CASES)
    out = compute_greeks(tbl, iv_solver="schadner")
    ivs = np.array(out["iv"].to_pylist())
    truth = np.array([c[6] for c in _CASES])
    assert np.allclose(ivs, truth, atol=1e-12)


def test_schadner_matches_numerical():
    """Schadner and numerical solvers should agree to within Brent's tolerance."""
    tbl = _build_table(_CASES)
    out_n = compute_greeks(tbl, iv_solver="numerical")
    out_s = compute_greeks(tbl, iv_solver="schadner")
    iv_n = np.array(out_n["iv"].to_pylist())
    iv_s = np.array(out_s["iv"].to_pylist())
    assert np.allclose(iv_n, iv_s, atol=1e-10)
    # Greeks consistent at the resolved IV
    for col in ("delta", "gamma", "vega", "theta", "rho"):
        a = np.array(out_n[col].to_pylist())
        b = np.array(out_s[col].to_pylist())
        assert np.allclose(a, b, atol=1e-7)


def test_schadner_default_is_numerical():
    """Default solver should match explicit numerical."""
    tbl = _build_table(_CASES[:3])
    out_default = compute_greeks(tbl)
    out_num = compute_greeks(tbl, iv_solver="numerical")
    assert out_default["iv"].to_pylist() == out_num["iv"].to_pylist()


def test_schadner_invalid_solver_raises():
    tbl = _build_table(_CASES[:1])
    with pytest.raises(Exception, match="iv_solver"):
        compute_greeks(tbl, iv_solver="bogus")


def test_lookup_solver_4_decimal_accuracy():
    """LUT-based solver should hit 4-decimal accuracy on σ over a wide grid."""
    rng = np.random.default_rng(0)
    n = 2000
    spot = np.full(n, 100.0)
    strike = spot * np.exp(rng.uniform(-0.8, 0.8, n))
    expiry = rng.uniform(0.05, 2.0, n)
    rate = rng.uniform(0.0, 0.05, n)
    div = rng.uniform(0.0, 0.03, n)
    sigma = rng.uniform(0.05, 1.0, n)
    opt = rng.choice([1, -1], size=n).astype(np.int8)
    prices = np.array([
        _bs(spot[i], strike[i], expiry[i], rate[i], div[i], sigma[i],
            opt[i] == 1)
        for i in range(n)
    ])
    tbl = pa.table({
        "option_type": pa.array(opt.astype(np.int32)),
        "spot": spot, "strike": strike, "expiry": expiry,
        "rate": rate, "dividend_yield": div, "market_price": prices,
    })
    out_n = compute_greeks(tbl, iv_solver="numerical")
    out_l = compute_greeks(tbl, iv_solver="lookup")
    iv_n = np.array(out_n["iv"].to_pylist())
    iv_l = np.array(out_l["iv"].to_pylist())
    mask = ~np.isnan(iv_n) & ~np.isnan(iv_l)
    err = np.abs(iv_l[mask] - iv_n[mask])
    # 4 decimals on σ (allow small margin for grid-edge cases)
    assert err.max() < 2e-4
    # Median must be well under that
    assert np.median(err) < 1e-6


def test_lookup_solver_smoke():
    """Lookup solver should at least match numerical on the basic _CASES."""
    tbl = _build_table(_CASES)
    out = compute_greeks(tbl, iv_solver="lookup")
    iv = np.array(out["iv"].to_pylist())
    truth = np.array([c[6] for c in _CASES])
    assert np.allclose(iv, truth, atol=1e-4)


def test_schadner_bad_inputs_return_nan():
    """Out-of-domain rows should still produce NaN, not raise."""
    tbl = pa.table({
        "option_type": pa.array([1, 1], type=pa.int32()),
        "spot": [100.0, 100.0],
        "strike": [100.0, 100.0],
        "expiry": [-1.0, 1.0],       # negative T
        "rate": [0.05, 0.05],
        "dividend_yield": [0.0, 0.0],
        "market_price": [10.0, -1.0],  # negative price
    })
    out = compute_greeks(tbl, iv_solver="schadner")
    ivs = out["iv"].to_pylist()
    assert all(np.isnan(v) for v in ivs)
