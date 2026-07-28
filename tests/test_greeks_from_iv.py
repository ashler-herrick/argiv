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


HIGHER_ORDER_COLS = ["vanna", "volga", "charm", "speed", "zomma", "color"]


def _grid():
    """(option_type, S, K, T, r, q, sigma) across moneyness/tenor/vol/carry."""
    return [
        (ot, 100.0, K, T, r, q, sigma)
        for ot in (1, -1)
        for K in (80.0, 95.0, 100.0, 105.0, 130.0)
        for T in (0.08, 0.5, 2.0)
        for sigma in (0.12, 0.30, 0.75)
        for r, q in ((0.04, 0.03), (0.0, 0.0), (0.02, 0.06))
    ]


def _greeks(rows, *, dS=0.0, dsig=0.0, dT=0.0, higher_order=False):
    table = pa.table({
        "option_type": pa.array([r[0] for r in rows], type=pa.int32()),
        "spot": pa.array([r[1] + dS for r in rows], type=pa.float64()),
        "strike": pa.array([r[2] for r in rows], type=pa.float64()),
        "expiry": pa.array([r[3] + dT for r in rows], type=pa.float64()),
        "rate": pa.array([r[4] for r in rows], type=pa.float64()),
        "dividend_yield": pa.array([r[5] for r in rows], type=pa.float64()),
        "iv": pa.array([r[6] + dsig for r in rows], type=pa.float64()),
    })
    return argiv.compute_greeks(table, higher_order=higher_order)


class TestHigherOrderGreeks:
    def test_default_off_is_unchanged(self):
        """higher_order=False must yield exactly today's table."""
        rows = _grid()
        base = _greeks(rows)
        assert base.column_names == [
            "option_type", "spot", "strike", "expiry", "rate",
            "dividend_yield", "iv", "delta", "gamma", "vega", "theta", "rho",
        ]
        assert not any(c in base.column_names for c in HIGHER_ORDER_COLS)

        # Enabling the flag appends, it never perturbs or reorders.
        ho = _greeks(rows, higher_order=True)
        assert ho.column_names == base.column_names + HIGHER_ORDER_COLS
        assert ho.select(base.column_names).equals(base)

    def test_price_path_default_off_is_unchanged(self):
        table = pa.table({
            "option_type": pa.array([1, -1], type=pa.int32()),
            "spot": [100.0, 100.0],
            "strike": [95.0, 105.0],
            "expiry": [0.5, 1.0],
            "rate": [0.04, 0.04],
            "dividend_yield": [0.02, 0.02],
            "market_price": [9.0, 8.0],
            "bid_price": [8.9, 7.9],
            "ask_price": [9.1, 8.1],
        })
        base = argiv.compute_greeks(table)
        ho = argiv.compute_greeks(table, higher_order=True)
        assert base.column_names[-2:] == ["iv_bid", "iv_ask"]
        assert set(HIGHER_ORDER_COLS).issubset(ho.column_names)
        assert ho.select(base.column_names).equals(base)

    def test_dtypes_are_float64(self):
        ho = _greeks(_grid()[:8], higher_order=True)
        for col in HIGHER_ORDER_COLS:
            assert ho.schema.field(col).type == pa.float64()

    @pytest.mark.parametrize(
        "greek,source,bump,sign",
        [
            # vanna = d(vega)/dS, volga = d(vega)/dsigma, speed = d(gamma)/dS,
            # zomma = d(gamma)/dsigma. charm/color differentiate w.r.t. calendar
            # time, i.e. minus the derivative w.r.t. time-to-expiry.
            ("vanna", "vega", "dS", +1),
            ("volga", "vega", "dsig", +1),
            ("charm", "delta", "dT", -1),
            ("speed", "gamma", "dS", +1),
            ("zomma", "gamma", "dsig", +1),
            ("color", "gamma", "dT", -1),
        ],
    )
    def test_matches_central_finite_difference(self, greek, source, bump, sign):
        """Central difference of QuantLib's own first-order Greeks."""
        h = {"dS": 1e-3, "dsig": 1e-5, "dT": 1e-5}[bump]
        rows = _grid()
        analytic = np.array(
            _greeks(rows, higher_order=True).column(greek).to_pylist()
        )
        up = np.array(_greeks(rows, **{bump: +h}).column(source).to_pylist())
        dn = np.array(_greeks(rows, **{bump: -h}).column(source).to_pylist())
        fd = sign * (up - dn) / (2.0 * h)

        err = np.abs(analytic - fd)
        tol = 1e-7 + 1e-6 * np.abs(fd)
        assert np.all(err < tol), (
            f"{greek}: max abs err {err.max():.3e} at row {int(err.argmax())}"
        )

    def test_atm_reference_values(self):
        """Independent closed-form check at a single ATM point."""
        from scipy.stats import norm

        S, K, T, r, q, sigma = 100.0, 100.0, 0.25, 0.05, 0.02, 0.20
        res = _greeks([(1, S, K, T, r, q, sigma)], higher_order=True)

        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        pdf = norm.pdf(d1)
        vega = S * math.exp(-q * T) * pdf * math.sqrt(T)

        assert res.column("vanna")[0].as_py() == pytest.approx(
            -math.exp(-q * T) * pdf * d2 / sigma, rel=1e-12
        )
        assert res.column("volga")[0].as_py() == pytest.approx(
            vega * d1 * d2 / sigma, rel=1e-12
        )

    def test_nan_propagation(self):
        """Rows that yield NaN first-order Greeks yield NaN higher-order too."""
        rows = [
            (1, 100.0, 100.0, 0.0, 0.05, 0.0, 0.20),    # T = 0
            (1, 100.0, 100.0, 0.5, 0.05, 0.0, 0.0),     # sigma = 0
            (1, 0.0, 100.0, 0.5, 0.05, 0.0, 0.20),      # spot = 0
            (1, 100.0, 0.0, 0.5, 0.05, 0.0, 0.20),      # strike = 0
            (1, 100.0, 100.0, 0.5, 0.05, 0.0, 0.20),    # valid control
        ]
        res = _greeks(rows, higher_order=True)
        for col in ["delta", "gamma", "vega"] + HIGHER_ORDER_COLS:
            vals = np.array(res.column(col).to_pylist(), dtype=float)
            assert np.all(np.isnan(vals[:4])), f"{col} should be NaN"
            assert np.isfinite(vals[4]), f"{col} should be finite"

    def test_nan_propagation_price_path(self):
        table = pa.table({
            "option_type": pa.array([1, 1], type=pa.int32()),
            "spot": [100.0, 100.0],
            "strike": [100.0, 100.0],
            "expiry": [0.0, 0.5],
            "rate": [0.05, 0.05],
            "dividend_yield": [0.0, 0.0],
            "market_price": [5.0, 5.0],
        })
        res = argiv.compute_greeks(table, higher_order=True)
        for col in HIGHER_ORDER_COLS:
            vals = np.array(res.column(col).to_pylist(), dtype=float)
            assert np.isnan(vals[0])
            assert np.isfinite(vals[1])

    def test_empty_table_with_higher_order(self):
        empty = pa.table({
            "option_type": pa.array([], type=pa.int32()),
            "spot": pa.array([], type=pa.float64()),
            "strike": pa.array([], type=pa.float64()),
            "expiry": pa.array([], type=pa.float64()),
            "rate": pa.array([], type=pa.float64()),
            "dividend_yield": pa.array([], type=pa.float64()),
            "iv": pa.array([], type=pa.float64()),
        })
        res = argiv.compute_greeks(empty, higher_order=True)
        assert res.num_rows == 0
        assert set(HIGHER_ORDER_COLS).issubset(res.column_names)
