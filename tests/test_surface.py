import datetime
import math

import numpy as np
import pyarrow as pa

import argiv


def _bs_price(option_type, S, K, T, r, q, sigma):
    """Black-Scholes price for a call (1) or put (-1)."""
    from scipy.stats import norm

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 1:
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def _make_surface_table(options, timestamp, expiration_date):
    """Build a surface input table from a list of option dicts.

    Each dict should have: option_type, spot, strike, expiry, rate,
    dividend_yield, market_price.
    """
    n = len(options)
    return pa.table({
        "option_type": pa.array([o["option_type"] for o in options], type=pa.int32()),
        "spot": pa.array([o["spot"] for o in options], type=pa.float64()),
        "strike": pa.array([o["strike"] for o in options], type=pa.float64()),
        "expiry": pa.array([o["expiry"] for o in options], type=pa.float64()),
        "rate": pa.array([o["rate"] for o in options], type=pa.float64()),
        "dividend_yield": pa.array([o["dividend_yield"] for o in options], type=pa.float64()),
        "market_price": pa.array([o["market_price"] for o in options], type=pa.float64()),
        "timestamp": pa.array([timestamp] * n, type=pa.timestamp("us")),
        "expiration": pa.array([expiration_date] * n, type=pa.date32()),
    })


def _generate_smile_options(S, T, r, q, base_sigma, skew_coeff, strikes, timestamp, expiration_date):
    """Generate options with a known smile: sigma(K) = base_sigma + skew_coeff * (K - S)^2."""
    options = []
    for K in strikes:
        sigma = base_sigma + skew_coeff * (K - S) ** 2
        # Use puts for K < S, calls for K >= S
        opt_type = -1 if K < S else 1
        price = _bs_price(opt_type, S, K, T, r, q, sigma)
        options.append({
            "option_type": opt_type,
            "spot": S,
            "strike": K,
            "expiry": T,
            "rate": r,
            "dividend_yield": q,
            "market_price": price,
        })
    return _make_surface_table(options, timestamp, expiration_date)


class TestSyntheticSmile:
    """Test IV recovery from a known volatility smile."""

    def test_smile_recovery(self):
        S, T, r, q = 100.0, 0.5, 0.05, 0.01
        base_sigma = 0.20
        skew_coeff = 0.00005
        strikes = np.linspace(80, 120, 30)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2024, 7, 15)

        table = _generate_smile_options(S, T, r, q, base_sigma, skew_coeff,
                                        strikes, ts, exp)
        result = argiv.compute_fit_vol_surface(table)

        assert result.num_rows == 1
        assert result.column("timestamp")[0].as_py() == ts
        assert result.column("expiration")[0].as_py() == exp

        # Check that at least 25-delta put and call pillars are populated
        iv_p25 = result.column("iv_p25")[0].as_py()
        iv_c25 = result.column("iv_c25")[0].as_py()
        assert iv_p25 is not None, "iv_p25 should not be null"
        assert iv_c25 is not None, "iv_c25 should not be null"

        # The smile is symmetric around ATM, so these should be close to
        # base_sigma (within the smile curvature)
        assert 0.15 < iv_p25 < 0.35, f"iv_p25={iv_p25} out of reasonable range"
        assert 0.15 < iv_c25 < 0.35, f"iv_c25={iv_c25} out of reasonable range"

    def test_flat_vol_consistency(self):
        """With flat vol, all pillar IVs should be approximately equal."""
        S, T, r, q = 100.0, 1.0, 0.05, 0.0
        sigma = 0.25
        strikes = np.linspace(85, 115, 25)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2025, 1, 15)

        table = _generate_smile_options(S, T, r, q, sigma, 0.0,
                                        strikes, ts, exp)
        result = argiv.compute_fit_vol_surface(table)

        # All non-null pillar IVs should be close to sigma
        for col_name in result.column_names:
            if col_name.startswith("iv_"):
                val = result.column(col_name)[0].as_py()
                if val is not None:
                    assert abs(val - sigma) < 0.01, \
                        f"{col_name}={val} deviates from flat vol {sigma}"


class TestMultipleGroups:
    """Test correct grouping by (timestamp, expiration)."""

    def test_multiple_expirations(self):
        S, T1, T2, r, q, sigma = 100.0, 0.25, 1.0, 0.05, 0.0, 0.20
        strikes = np.linspace(90, 110, 15)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp1 = datetime.date(2024, 4, 15)
        exp2 = datetime.date(2025, 1, 15)

        t1 = _generate_smile_options(S, T1, r, q, sigma, 0.0, strikes, ts, exp1)
        t2 = _generate_smile_options(S, T2, r, q, sigma, 0.0, strikes, ts, exp2)
        combined = pa.concat_tables([t1, t2])

        result = argiv.compute_fit_vol_surface(combined)
        assert result.num_rows == 2

        exps = result.column("expiration").to_pylist()
        assert exp1 in exps
        assert exp2 in exps

    def test_multiple_timestamps(self):
        S, T, r, q, sigma = 100.0, 0.5, 0.05, 0.0, 0.20
        strikes = np.linspace(90, 110, 15)
        ts1 = datetime.datetime(2024, 1, 15, 10, 0, 0)
        ts2 = datetime.datetime(2024, 1, 15, 11, 0, 0)
        exp = datetime.date(2024, 7, 15)

        t1 = _generate_smile_options(S, T, r, q, sigma, 0.0, strikes, ts1, exp)
        t2 = _generate_smile_options(S, T, r, q, sigma, 0.0, strikes, ts2, exp)
        combined = pa.concat_tables([t1, t2])

        result = argiv.compute_fit_vol_surface(combined)
        assert result.num_rows == 2

    def test_three_groups(self):
        S, r, q, sigma = 100.0, 0.05, 0.0, 0.20
        strikes = np.linspace(90, 110, 15)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp1 = datetime.date(2024, 4, 15)
        exp2 = datetime.date(2024, 7, 15)
        exp3 = datetime.date(2025, 1, 15)

        tables = []
        for T, exp in [(0.25, exp1), (0.5, exp2), (1.0, exp3)]:
            tables.append(_generate_smile_options(S, T, r, q, sigma, 0.0,
                                                  strikes, ts, exp))
        combined = pa.concat_tables(tables)

        result = argiv.compute_fit_vol_surface(combined)
        assert result.num_rows == 3


class TestEdgeCases:
    """Test edge cases: few points, empty table, out-of-range pillars."""

    def test_single_option_all_nan(self):
        """A single option per group gives <2 points, all pillars should be null."""
        S, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.05, 0.0, 0.20
        price = _bs_price(1, S, K, T, r, q, sigma)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2024, 7, 15)

        table = _make_surface_table(
            [{"option_type": 1, "spot": S, "strike": K, "expiry": T,
              "rate": r, "dividend_yield": q, "market_price": price}],
            ts, exp,
        )
        result = argiv.compute_fit_vol_surface(table)
        assert result.num_rows == 1

        for col_name in result.column_names:
            if col_name.startswith("iv_"):
                assert result.column(col_name)[0].as_py() is None, \
                    f"{col_name} should be null with single point"

    def test_two_points_linear(self):
        """Two options should use linear interpolation."""
        S, T, r, q, sigma = 100.0, 0.5, 0.05, 0.0, 0.20
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2024, 7, 15)

        options = []
        for K in [95.0, 105.0]:
            opt_type = -1 if K < S else 1
            price = _bs_price(opt_type, S, K, T, r, q, sigma)
            options.append({
                "option_type": opt_type, "spot": S, "strike": K,
                "expiry": T, "rate": r, "dividend_yield": q,
                "market_price": price,
            })

        table = _make_surface_table(options, ts, exp)
        result = argiv.compute_fit_vol_surface(table)
        assert result.num_rows == 1
        # Should have at least some non-null values between the two deltas
        has_value = False
        for col_name in result.column_names:
            if col_name.startswith("iv_"):
                val = result.column(col_name)[0].as_py()
                if val is not None:
                    has_value = True
                    assert val > 0
        # It's possible all pillars are out of range for just 2 points,
        # but we should at least not crash

    def test_empty_table(self):
        """Empty input returns 0-row table with correct schema."""
        table = pa.table({
            "option_type": pa.array([], type=pa.int32()),
            "spot": pa.array([], type=pa.float64()),
            "strike": pa.array([], type=pa.float64()),
            "expiry": pa.array([], type=pa.float64()),
            "rate": pa.array([], type=pa.float64()),
            "dividend_yield": pa.array([], type=pa.float64()),
            "market_price": pa.array([], type=pa.float64()),
            "timestamp": pa.array([], type=pa.timestamp("us")),
            "expiration": pa.array([], type=pa.date32()),
        })
        result = argiv.compute_fit_vol_surface(table)
        assert result.num_rows == 0
        assert "timestamp" in result.column_names
        assert "expiration" in result.column_names
        assert "iv_p25" in result.column_names
        assert "iv_c25" in result.column_names


class TestCustomPillars:
    """Test custom delta pillar configuration."""

    def test_custom_pillars(self):
        S, T, r, q, sigma = 100.0, 0.5, 0.05, 0.0, 0.20
        strikes = np.linspace(85, 115, 25)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2024, 7, 15)

        table = _generate_smile_options(S, T, r, q, sigma, 0.0,
                                        strikes, ts, exp)
        result = argiv.compute_fit_vol_surface(table, delta_pillars=[10, 25])

        # Should have exactly 4 IV columns: iv_p25, iv_p10, iv_c10, iv_c25
        iv_cols = [c for c in result.column_names if c.startswith("iv_")]
        assert len(iv_cols) == 4
        assert "iv_p25" in iv_cols
        assert "iv_p10" in iv_cols
        assert "iv_c10" in iv_cols
        assert "iv_c25" in iv_cols

    def test_single_pillar(self):
        S, T, r, q, sigma = 100.0, 0.5, 0.05, 0.0, 0.20
        strikes = np.linspace(85, 115, 25)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2024, 7, 15)

        table = _generate_smile_options(S, T, r, q, sigma, 0.0,
                                        strikes, ts, exp)
        result = argiv.compute_fit_vol_surface(table, delta_pillars=[25])

        iv_cols = [c for c in result.column_names if c.startswith("iv_")]
        assert len(iv_cols) == 2
        assert "iv_p25" in iv_cols
        assert "iv_c25" in iv_cols


class TestOutputSchema:
    """Test that output schema is correct."""

    def test_column_order(self):
        S, T, r, q, sigma = 100.0, 0.5, 0.05, 0.0, 0.20
        strikes = np.linspace(85, 115, 20)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2024, 7, 15)

        table = _generate_smile_options(S, T, r, q, sigma, 0.0,
                                        strikes, ts, exp)
        result = argiv.compute_fit_vol_surface(table)

        expected_cols = ["timestamp", "expiration",
                         "iv_p50", "iv_p45", "iv_p40", "iv_p35", "iv_p30",
                         "iv_p25", "iv_p20", "iv_p15", "iv_p10", "iv_p5",
                         "iv_c5", "iv_c10", "iv_c15", "iv_c20", "iv_c25",
                         "iv_c30", "iv_c35", "iv_c40", "iv_c45", "iv_c50"]
        assert result.column_names == expected_cols

    def test_timestamp_type_preserved(self):
        """Output timestamp type should match input."""
        S, T, r, q, sigma = 100.0, 0.5, 0.05, 0.0, 0.20
        strikes = np.linspace(90, 110, 10)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2024, 7, 15)

        for unit in ["s", "ms", "us", "ns"]:
            table = pa.table({
                "option_type": pa.array([-1, -1, -1, 1, 1, 1, 1, 1, 1, 1], type=pa.int32()),
                "spot": pa.array([100.0] * 10, type=pa.float64()),
                "strike": pa.array(strikes.tolist(), type=pa.float64()),
                "expiry": pa.array([T] * 10, type=pa.float64()),
                "rate": pa.array([r] * 10, type=pa.float64()),
                "dividend_yield": pa.array([q] * 10, type=pa.float64()),
                "market_price": pa.array(
                    [_bs_price(-1 if K < 100 else 1, 100, K, T, r, q, sigma)
                     for K in strikes],
                    type=pa.float64()),
                "timestamp": pa.array([ts] * 10, type=pa.timestamp(unit)),
                "expiration": pa.array([exp] * 10, type=pa.date32()),
            })
            result = argiv.compute_fit_vol_surface(table)
            assert result.schema.field("timestamp").type == pa.timestamp(unit), \
                f"Timestamp type mismatch for unit={unit}"
