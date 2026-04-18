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


def _make_surface_with_event(sigma_d=0.20, event_var=0.01, event_date=None):
    """Build a multi-expiry surface with a known event variance injection."""
    S, r, q = 100.0, 0.05, 0.01
    ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
    ts_date = datetime.date(2024, 1, 15)
    if event_date is None:
        event_date = datetime.date(2024, 2, 10)

    days_to_event = (event_date - ts_date).days
    event_T = days_to_event / 365.0

    expiries = [
        (0.04, datetime.date(2024, 1, 30)),
        (0.055, datetime.date(2024, 2, 5)),
        (0.10, datetime.date(2024, 2, 20)),
        (0.25, datetime.date(2024, 4, 15)),
        (0.50, datetime.date(2024, 7, 15)),
    ]

    all_options = []
    for T, exp_date in expiries:
        w = sigma_d**2 * T
        if T >= event_T:
            w += event_var
        total_sigma = math.sqrt(w / T)

        strikes = np.linspace(80, 120, 30)
        for K in strikes:
            ot = -1 if K < S else 1
            all_options.append({
                "option_type": ot, "spot": S, "strike": K, "expiry": T,
                "rate": r, "dividend_yield": q,
                "market_price": _bs_price(ot, S, K, T, r, q, total_sigma),
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
    return surface, event_T


class TestDecomposeVariance:
    """Test NNLS variance decomposition."""

    def test_single_event_recovery(self):
        """Recover known diffusive vol and event variance."""
        sigma_d = 0.20
        event_var = 0.01
        surface, event_T = _make_surface_with_event(sigma_d=sigma_d, event_var=event_var)

        events = pa.table({
            "event_date": pa.array([datetime.date(2024, 2, 10)], type=pa.date32()),
            "event_type": ["earnings"],
        })

        decomp = argiv.decompose_variance(surface, events)
        df = decomp.to_pandas()

        # Check ATM decomposition
        atm = df[df["delta"] == 0.50]
        assert len(atm) == 1
        assert abs(atm["sigma_d_sq"].iloc[0] - sigma_d**2) < 0.002
        assert abs(atm["event_var"].iloc[0] - event_var) < 0.002
        assert atm["event_type"].iloc[0] == "earnings"

    def test_no_events(self):
        """With no events, sigma_d_sq should equal total variance rate."""
        sigma_d = 0.25
        surface, _ = _make_surface_with_event(sigma_d=sigma_d, event_var=0.0)

        # Provide events table with an event far in the future (irrelevant)
        events = pa.table({
            "event_date": pa.array([datetime.date(2025, 12, 1)], type=pa.date32()),
            "event_type": ["earnings"],
        })

        decomp = argiv.decompose_variance(surface, events)
        df = decomp.to_pandas()

        # No events should be within the surface range
        atm = df[df["delta"] == 0.50]
        assert len(atm) == 1
        assert atm["event_expiry"].iloc[0] is None or np.isnan(atm["event_expiry"].iloc[0])
        assert abs(atm["sigma_d_sq"].iloc[0] - sigma_d**2) < 0.002

    def test_event_type_preserved(self):
        """Event type labels should pass through to output."""
        surface, _ = _make_surface_with_event()
        events = pa.table({
            "event_date": pa.array([datetime.date(2024, 2, 10)], type=pa.date32()),
            "event_type": ["fomc"],
        })
        decomp = argiv.decompose_variance(surface, events)
        df = decomp.to_pandas()
        assert "fomc" in df["event_type"].values

    def test_output_schema(self):
        surface, _ = _make_surface_with_event()
        events = pa.table({
            "event_date": pa.array([datetime.date(2024, 2, 10)], type=pa.date32()),
        })
        decomp = argiv.decompose_variance(surface, events)
        assert "timestamp" in decomp.column_names
        assert "delta" in decomp.column_names
        assert "sigma_d_sq" in decomp.column_names
        assert "event_expiry" in decomp.column_names
        assert "event_var" in decomp.column_names

    def test_all_variances_non_negative(self):
        """NNLS should ensure all variance components are >= 0."""
        surface, _ = _make_surface_with_event(sigma_d=0.20, event_var=0.015)
        events = pa.table({
            "event_date": pa.array([datetime.date(2024, 2, 10)], type=pa.date32()),
            "event_type": ["earnings"],
        })
        decomp = argiv.decompose_variance(surface, events)
        df = decomp.to_pandas()

        assert (df["sigma_d_sq"] >= 0).all()
        assert (df["event_var"] >= 0).all()

    def test_composes_with_constant_maturity(self):
        """Decomposition should feed into constant_maturity_surface."""
        sigma_d = 0.20
        event_var = 0.01
        surface, _ = _make_surface_with_event(sigma_d=sigma_d, event_var=event_var)

        events = pa.table({
            "event_date": pa.array([datetime.date(2024, 2, 10)], type=pa.date32()),
            "event_type": ["earnings"],
        })
        decomp = argiv.decompose_variance(surface, events)

        # This should work without error
        cm = argiv.constant_maturity_surface(
            surface,
            tenors=[0.15, 0.35],
            decomposition=decomp,
        )
        assert cm.num_rows > 0
        assert "iv" in cm.column_names

        # IV should be reasonable
        df = cm.to_pandas()
        assert (df["iv"] > 0).all()
        assert (df["iv"] < 1.0).all()
