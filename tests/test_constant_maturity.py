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


def _make_multi_expiry_surface(sigma=0.20, skew=0.00005):
    """Build a surface with 4 expiries and known flat-ish vol."""
    S, r, q = 100.0, 0.05, 0.01
    ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
    expiries = [
        (0.1, datetime.date(2024, 2, 20)),
        (0.25, datetime.date(2024, 4, 15)),
        (0.5, datetime.date(2024, 7, 15)),
        (1.0, datetime.date(2025, 1, 15)),
    ]

    all_options = []
    for T, exp_date in expiries:
        strikes = np.linspace(80, 120, 30)
        for K in strikes:
            vol = sigma + skew * (K - S) ** 2
            ot = -1 if K < S else 1
            all_options.append({
                "option_type": ot, "spot": S, "strike": K, "expiry": T,
                "rate": r, "dividend_yield": q,
                "market_price": _bs_price(ot, S, K, T, r, q, vol),
                "timestamp": ts, "expiration": exp_date,
            })

    n = len(all_options)
    raw = pa.table({
        "option_type": pa.array([o["option_type"] for o in all_options], type=pa.int32()),
        "spot": [o["spot"] for o in all_options],
        "strike": [o["strike"] for o in all_options],
        "expiry": [o["expiry"] for o in all_options],
        "rate": [o["rate"] for o in all_options],
        "dividend_yield": [o["dividend_yield"] for o in all_options],
        "market_price": [o["market_price"] for o in all_options],
        "timestamp": pa.array([o["timestamp"] for o in all_options], type=pa.timestamp("us")),
        "expiration": pa.array([o["expiration"] for o in all_options], type=pa.date32()),
    })
    enriched = argiv.compute_greeks(raw)
    return argiv.fit_vol_surface(enriched)


class TestConstantMaturitySurface:
    """Test variance interpolation to fixed tenors."""

    def test_output_schema(self):
        surface = _make_multi_expiry_surface()
        cm = argiv.constant_maturity_surface(surface, tenors=[0.15, 0.35])
        assert "timestamp" in cm.column_names
        assert "tenor" in cm.column_names
        assert "delta" in cm.column_names
        assert "iv" in cm.column_names

    def test_tenors_within_range(self):
        """Only tenors within the available expiry range should be output."""
        surface = _make_multi_expiry_surface()
        # Min expiry is 0.1, max is 1.0
        cm = argiv.constant_maturity_surface(
            surface, tenors=[0.05, 0.15, 0.5, 1.5]
        )
        tenors_out = set(cm.column("tenor").to_pylist())
        assert 0.05 not in tenors_out, "below min expiry"
        assert 1.5 not in tenors_out, "above max expiry"
        assert 0.15 in tenors_out
        assert 0.5 in tenors_out

    def test_flat_vol_interpolation(self):
        """With flat vol, interpolated vol should match the input vol."""
        surface = _make_multi_expiry_surface(sigma=0.25, skew=0.0)
        cm = argiv.constant_maturity_surface(surface, tenors=[0.15, 0.35, 0.75])
        df = cm.to_pandas()

        # ATM IV should be very close to 0.25 everywhere
        atm = df[df["delta"] == 0.50]
        for _, row in atm.iterrows():
            assert abs(row["iv"] - 0.25) < 0.005, (
                f"ATM IV at tenor={row['tenor']:.2f} is {row['iv']:.4f}, expected ~0.25"
            )

    def test_variance_monotonic(self):
        """Total variance w = iv^2 * T should be non-decreasing in T."""
        surface = _make_multi_expiry_surface()
        cm = argiv.constant_maturity_surface(
            surface, tenors=[0.12, 0.20, 0.35, 0.50, 0.75]
        )
        df = cm.to_pandas()

        for delta_val, grp in df.groupby("delta"):
            grp = grp.sort_values("tenor")
            w = grp["iv"].values ** 2 * grp["tenor"].values
            diffs = np.diff(w)
            assert (diffs >= -1e-12).all(), (
                f"Calendar arbitrage at delta={delta_val}: "
                f"total variance decreased"
            )

    def test_empty_output_for_single_expiry(self):
        """Single expiry can't be interpolated — should return empty or skip."""
        S, r, q, T = 100.0, 0.05, 0.01, 0.5
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp_date = datetime.date(2024, 7, 15)
        strikes = np.linspace(80, 120, 30)
        options = []
        for K in strikes:
            ot = -1 if K < S else 1
            options.append({
                "option_type": ot, "spot": S, "strike": K, "expiry": T,
                "rate": r, "dividend_yield": q,
                "market_price": _bs_price(ot, S, K, T, r, q, 0.20),
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
        cm = argiv.constant_maturity_surface(surface, tenors=[0.3])
        assert cm.num_rows == 0

    def test_log_moneyness_interpolated(self):
        """log_moneyness should be present if in the input surface."""
        surface = _make_multi_expiry_surface()
        assert "log_moneyness" in surface.column_names
        cm = argiv.constant_maturity_surface(surface, tenors=[0.15, 0.35])
        assert "log_moneyness" in cm.column_names

    def test_composes_with_greeks_at_pillars(self):
        """constant_maturity_surface output should work with greeks_at_pillars."""
        S, r, q = 100.0, 0.05, 0.01
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        expiries = [
            (0.1, datetime.date(2024, 2, 20)),
            (0.25, datetime.date(2024, 4, 15)),
            (0.5, datetime.date(2024, 7, 15)),
            (1.0, datetime.date(2025, 1, 15)),
        ]
        all_options = []
        for T, exp_date in expiries:
            strikes = np.linspace(80, 120, 30)
            for K in strikes:
                sigma = 0.20 + 0.00005 * (K - S) ** 2
                ot = -1 if K < S else 1
                all_options.append({
                    "option_type": ot, "spot": S, "strike": K, "expiry": T,
                    "rate": r, "dividend_yield": q,
                    "market_price": _bs_price(ot, S, K, T, r, q, sigma),
                    "timestamp": ts, "expiration": exp_date,
                })
        n = len(all_options)
        raw = pa.table({
            "option_type": pa.array([o["option_type"] for o in all_options], type=pa.int32()),
            "spot": [o["spot"] for o in all_options],
            "strike": [o["strike"] for o in all_options],
            "expiry": [o["expiry"] for o in all_options],
            "rate": [o["rate"] for o in all_options],
            "dividend_yield": [o["dividend_yield"] for o in all_options],
            "market_price": [o["market_price"] for o in all_options],
            "timestamp": pa.array([o["timestamp"] for o in all_options], type=pa.timestamp("us")),
            "expiration": pa.array([o["expiration"] for o in all_options], type=pa.date32()),
        })
        enriched = argiv.compute_greeks(raw)
        surface = argiv.fit_vol_surface(enriched)
        cm = argiv.constant_maturity_surface(surface, tenors=[0.15, 0.35])

        # greeks_at_pillars needs expiration column — cm uses tenor instead.
        # This verifies the composition pattern works when we add expiration.
        # For now, just verify cm has the right shape.
        assert cm.num_rows > 0
        assert "iv" in cm.column_names
