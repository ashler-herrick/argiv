import datetime
import math

import numpy as np
import pyarrow as pa
import pytest

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
    """Build a surface input table from a list of option dicts."""
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


def _get_iv_at_delta(result, target_delta, tol=1e-9):
    """Get the IV value from the result table at the given signed delta."""
    deltas = result.column("delta").to_pylist()
    ivs = result.column("iv").to_pylist()
    for d, v in zip(deltas, ivs):
        if abs(d - target_delta) < tol:
            return v
    return None


def _get_row_at_delta(result, target_delta, tol=1e-9):
    """Get all column values from the result table at the given signed delta."""
    deltas = result.column("delta").to_pylist()
    for i, d in enumerate(deltas):
        if abs(d - target_delta) < tol:
            return {col: result.column(col)[i].as_py() for col in result.column_names}
    return None


# Default config: 9 wing pillars -> 9 put + ATM + 9 call = 19 rows per group
DEFAULT_PILLARS_PER_GROUP = 19


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

        assert result.num_rows == DEFAULT_PILLARS_PER_GROUP
        assert result.column_names == ["timestamp", "expiration", "delta", "iv", "log_moneyness"]

        # Check 25-delta put and call
        iv_p25 = _get_iv_at_delta(result, -0.25)
        iv_c25 = _get_iv_at_delta(result, 0.25)
        assert iv_p25 is not None, "iv at delta=-0.25 should not be null"
        assert iv_c25 is not None, "iv at delta=0.25 should not be null"
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

        ivs = result.column("iv").to_pylist()
        for i, val in enumerate(ivs):
            if val is not None:
                assert abs(val - sigma) < 0.01, \
                    f"Row {i}: iv={val} deviates from flat vol {sigma}"


class TestATMVol:
    """Test the ATM (delta=0.50) anchor computation."""

    def test_atm_present(self):
        """A row with delta=0.50 should always be in the output."""
        S, T, r, q, sigma = 100.0, 0.5, 0.05, 0.0, 0.20
        strikes = np.linspace(85, 115, 25)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2024, 7, 15)

        table = _generate_smile_options(S, T, r, q, sigma, 0.0,
                                        strikes, ts, exp)
        result = argiv.compute_fit_vol_surface(table)

        deltas = result.column("delta").to_pylist()
        assert 0.50 in deltas

    def test_atm_convergence_flat_vol(self):
        """With flat vol, ATM IV should match wing pillars."""
        S, T, r, q = 100.0, 1.0, 0.05, 0.0
        sigma = 0.25
        strikes = np.linspace(85, 115, 25)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2025, 1, 15)

        table = _generate_smile_options(S, T, r, q, sigma, 0.0,
                                        strikes, ts, exp)
        result = argiv.compute_fit_vol_surface(table)

        iv_atm = _get_iv_at_delta(result, 0.50)
        assert iv_atm is not None, "ATM iv should not be null"
        assert abs(iv_atm - sigma) < 0.01, f"iv_atm={iv_atm} deviates from {sigma}"

    def test_atm_is_average(self):
        """With a symmetric smile, ATM IV should be close to the base level."""
        S, T, r, q = 100.0, 0.5, 0.05, 0.0
        base_sigma = 0.25
        skew_coeff = 0.0001
        strikes = np.linspace(80, 120, 30)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2024, 7, 15)

        table = _generate_smile_options(S, T, r, q, base_sigma, skew_coeff,
                                        strikes, ts, exp)
        result = argiv.compute_fit_vol_surface(table)

        iv_atm = _get_iv_at_delta(result, 0.50)
        assert iv_atm is not None
        assert abs(iv_atm - base_sigma) < 0.02, \
            f"iv_atm={iv_atm} too far from base_sigma={base_sigma}"

    def test_no_negative_50_delta(self):
        """No row should have delta=-0.50 (ATM is 0.50, not a put wing)."""
        S, T, r, q, sigma = 100.0, 0.5, 0.05, 0.0, 0.20
        strikes = np.linspace(85, 115, 25)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2024, 7, 15)

        table = _generate_smile_options(S, T, r, q, sigma, 0.0,
                                        strikes, ts, exp)
        result = argiv.compute_fit_vol_surface(table)

        deltas = result.column("delta").to_pylist()
        assert -0.50 not in deltas


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
        assert result.num_rows == 2 * DEFAULT_PILLARS_PER_GROUP

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
        assert result.num_rows == 2 * DEFAULT_PILLARS_PER_GROUP

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
        assert result.num_rows == 3 * DEFAULT_PILLARS_PER_GROUP


class TestEdgeCases:
    """Test edge cases: few points, empty table, out-of-range pillars."""

    def test_single_option_all_nan(self):
        """A single option per group gives <2 points, all wing IVs should be null."""
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
        assert result.num_rows == DEFAULT_PILLARS_PER_GROUP

        ivs = result.column("iv").to_pylist()
        deltas = result.column("delta").to_pylist()
        for d, v in zip(deltas, ivs):
            if abs(d - 0.50) > 1e-9:
                assert v is None, f"iv at delta={d} should be null with single point"

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
        assert result.num_rows == DEFAULT_PILLARS_PER_GROUP

        iv_atm = _get_iv_at_delta(result, 0.50)
        if iv_atm is not None:
            assert iv_atm > 0

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
        assert result.column_names == ["timestamp", "expiration", "delta", "iv", "log_moneyness"]


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

        # 2 put + ATM + 2 call = 5 rows
        assert result.num_rows == 5
        deltas = sorted(result.column("delta").to_pylist())
        assert deltas == pytest.approx([-0.25, -0.10, 0.10, 0.25, 0.50])

    def test_single_pillar(self):
        S, T, r, q, sigma = 100.0, 0.5, 0.05, 0.0, 0.20
        strikes = np.linspace(85, 115, 25)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2024, 7, 15)

        table = _generate_smile_options(S, T, r, q, sigma, 0.0,
                                        strikes, ts, exp)
        result = argiv.compute_fit_vol_surface(table, delta_pillars=[25])

        assert result.num_rows == 3
        deltas = sorted(result.column("delta").to_pylist())
        assert deltas == pytest.approx([-0.25, 0.25, 0.50])

    def test_pillars_must_be_below_50(self):
        """Passing delta_pillars containing 50 should raise."""
        S, T, r, q, sigma = 100.0, 0.5, 0.05, 0.0, 0.20
        strikes = np.linspace(85, 115, 25)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2024, 7, 15)

        table = _generate_smile_options(S, T, r, q, sigma, 0.0,
                                        strikes, ts, exp)
        with pytest.raises(RuntimeError, match="must be < 50"):
            argiv.compute_fit_vol_surface(table, delta_pillars=[25, 50])


class TestOptionTypeRequired:
    """Test that fit_vol_surface requires option_type column."""

    def test_missing_option_type_raises(self):
        """fit_vol_surface should raise if option_type is missing."""
        table = pa.table({
            "iv": pa.array([0.2, 0.25], type=pa.float64()),
            "delta": pa.array([-0.3, 0.3], type=pa.float64()),
            "timestamp": pa.array(
                [datetime.datetime(2024, 1, 15, 10, 0, 0)] * 2,
                type=pa.timestamp("us")),
            "expiration": pa.array(
                [datetime.date(2024, 7, 15)] * 2,
                type=pa.date32()),
        })
        with pytest.raises((ValueError, RuntimeError), match="[Mm]issing"):
            argiv.fit_vol_surface(table)


class TestOutputSchema:
    """Test that output schema is correct."""

    def test_column_names(self):
        S, T, r, q, sigma = 100.0, 0.5, 0.05, 0.0, 0.20
        strikes = np.linspace(85, 115, 20)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2024, 7, 15)

        table = _generate_smile_options(S, T, r, q, sigma, 0.0,
                                        strikes, ts, exp)
        result = argiv.compute_fit_vol_surface(table)

        assert result.column_names == ["timestamp", "expiration", "delta", "iv", "log_moneyness"]

    def test_delta_ordering(self):
        """Within a group, deltas should be ordered: puts descending, ATM, calls ascending."""
        S, T, r, q, sigma = 100.0, 0.5, 0.05, 0.0, 0.20
        strikes = np.linspace(85, 115, 20)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2024, 7, 15)

        table = _generate_smile_options(S, T, r, q, sigma, 0.0,
                                        strikes, ts, exp)
        result = argiv.compute_fit_vol_surface(table)

        deltas = result.column("delta").to_pylist()
        expected = [-0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05,
                    0.50,
                    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
        assert deltas == pytest.approx(expected)

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


class TestLogMoneyness:
    """Test log moneyness computation."""

    def test_log_moneyness_populated(self):
        """compute_fit_vol_surface should populate log_moneyness."""
        S, T, r, q = 100.0, 0.5, 0.05, 0.0
        sigma = 0.25
        strikes = np.linspace(85, 115, 25)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2024, 7, 15)

        table = _generate_smile_options(S, T, r, q, sigma, 0.0,
                                        strikes, ts, exp)
        result = argiv.compute_fit_vol_surface(table)

        # ATM log_moneyness should be near 0
        atm_row = _get_row_at_delta(result, 0.50)
        assert atm_row is not None
        lm_atm = atm_row["log_moneyness"]
        assert lm_atm is not None, "ATM log_moneyness should not be null"
        assert abs(lm_atm) < 0.1, f"ATM log_moneyness={lm_atm} should be near 0"

        # Put side should have negative log_moneyness (K < S for OTM puts)
        lm_p25 = _get_row_at_delta(result, -0.25)
        if lm_p25 and lm_p25["log_moneyness"] is not None:
            assert lm_p25["log_moneyness"] < 0, \
                f"Put log_moneyness={lm_p25['log_moneyness']} should be negative"

        # Call side should have positive log_moneyness (K > S for OTM calls)
        lm_c25 = _get_row_at_delta(result, 0.25)
        if lm_c25 and lm_c25["log_moneyness"] is not None:
            assert lm_c25["log_moneyness"] > 0, \
                f"Call log_moneyness={lm_c25['log_moneyness']} should be positive"

    def test_missing_spot_strike_raises(self):
        """fit_vol_surface without spot/strike/expiry should raise ValueError."""
        table = pa.table({
            "iv": pa.array([0.20, 0.22, 0.25, 0.22, 0.20], type=pa.float64()),
            "option_type": pa.array([-1, -1, -1, 1, 1], type=pa.int32()),
            "timestamp": pa.array(
                [datetime.datetime(2024, 1, 15, 10, 0, 0)] * 5,
                type=pa.timestamp("us")),
            "expiration": pa.array(
                [datetime.date(2024, 7, 15)] * 5,
                type=pa.date32()),
        })
        with pytest.raises((ValueError, RuntimeError)):
            argiv.fit_vol_surface(table)


class TestBidAskSurface:
    """Test bid/ask IV bounds in the vol surface."""

    def _generate_bid_ask_options(self, S, T, r, q, sigma, strikes, ts, exp,
                                  spread_sigma=0.02):
        """Generate options with bid/ask prices (mid ± spread in vol space)."""
        options = {
            "option_type": [], "spot": [], "strike": [], "expiry": [],
            "rate": [], "dividend_yield": [],
            "market_price": [], "bid_price": [], "ask_price": [],
        }
        for K in strikes:
            opt_type = -1 if K < S else 1
            mid = _bs_price(opt_type, S, K, T, r, q, sigma)
            bid = _bs_price(opt_type, S, K, T, r, q, sigma - spread_sigma)
            ask = _bs_price(opt_type, S, K, T, r, q, sigma + spread_sigma)
            options["option_type"].append(opt_type)
            options["spot"].append(S)
            options["strike"].append(K)
            options["expiry"].append(T)
            options["rate"].append(r)
            options["dividend_yield"].append(q)
            options["market_price"].append(mid)
            options["bid_price"].append(max(bid, 1e-10))
            options["ask_price"].append(ask)
        n = len(strikes)
        return pa.table({
            "option_type": pa.array(options["option_type"], type=pa.int32()),
            "spot": pa.array(options["spot"], type=pa.float64()),
            "strike": pa.array(options["strike"], type=pa.float64()),
            "expiry": pa.array(options["expiry"], type=pa.float64()),
            "rate": pa.array(options["rate"], type=pa.float64()),
            "dividend_yield": pa.array(options["dividend_yield"], type=pa.float64()),
            "market_price": pa.array(options["market_price"], type=pa.float64()),
            "bid_price": pa.array(options["bid_price"], type=pa.float64()),
            "ask_price": pa.array(options["ask_price"], type=pa.float64()),
            "timestamp": pa.array([ts] * n, type=pa.timestamp("us")),
            "expiration": pa.array([exp] * n, type=pa.date32()),
        })

    def test_surface_bid_ask_schema(self):
        """Output should have iv_bid and iv_ask columns when bid/ask provided."""
        S, T, r, q, sigma = 100.0, 0.5, 0.05, 0.0, 0.20
        strikes = np.linspace(85, 115, 25)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2024, 7, 15)

        table = self._generate_bid_ask_options(S, T, r, q, sigma, strikes, ts, exp)
        result = argiv.compute_fit_vol_surface(table)

        assert "iv_bid" in result.column_names
        assert "iv_ask" in result.column_names

    def test_surface_bid_ask_bands(self):
        """iv_bid <= iv <= iv_ask at each pillar delta."""
        S, T, r, q, sigma = 100.0, 0.5, 0.05, 0.0, 0.25
        strikes = np.linspace(85, 115, 25)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2024, 7, 15)

        table = self._generate_bid_ask_options(S, T, r, q, sigma, strikes, ts, exp)
        result = argiv.compute_fit_vol_surface(table)

        ivs = result.column("iv").to_pylist()
        iv_bids = result.column("iv_bid").to_pylist()
        iv_asks = result.column("iv_ask").to_pylist()

        for i in range(result.num_rows):
            if ivs[i] is None or iv_bids[i] is None or iv_asks[i] is None:
                continue
            assert iv_bids[i] <= ivs[i] + 1e-6, \
                f"Row {i}: iv_bid={iv_bids[i]} > iv={ivs[i]}"
            assert ivs[i] <= iv_asks[i] + 1e-6, \
                f"Row {i}: iv={ivs[i]} > iv_ask={iv_asks[i]}"

    def test_surface_no_bid_ask_unchanged(self):
        """Without bid/ask, output should not have iv_bid/iv_ask."""
        S, T, r, q, sigma = 100.0, 0.5, 0.05, 0.0, 0.20
        strikes = np.linspace(85, 115, 25)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2024, 7, 15)

        table = _generate_smile_options(S, T, r, q, sigma, 0.0, strikes, ts, exp)
        result = argiv.compute_fit_vol_surface(table)

        assert "iv_bid" not in result.column_names
        assert "iv_ask" not in result.column_names


# ---------------------------------------------------------------------------
# SVI-specific regression tests
# ---------------------------------------------------------------------------


class TestSviFitting:
    """Tests specific to the SVI model replacing cubic splines."""

    def test_short_dated_no_blowup(self):
        """0-DTE options with narrow delta range must not produce 100+ IVs.

        This is the core regression test for the cubic spline blowup bug.
        """
        S, T, r, q = 100.0, 0.005, 0.05, 0.0  # ~2 trading hours
        sigma = 0.30
        # Narrow strikes near ATM — deep OTM deltas will be small
        strikes = np.linspace(97, 103, 15)
        ts = datetime.datetime(2024, 1, 15, 15, 0, 0)
        exp = datetime.date(2024, 1, 16)

        table = _generate_smile_options(S, T, r, q, sigma, 0.0,
                                        strikes, ts, exp)
        result = argiv.compute_fit_vol_surface(table)

        ivs = result.column("iv").to_pylist()
        for i, val in enumerate(ivs):
            if val is not None:
                assert 0 < val < 2.0, (
                    f"Row {i}: iv={val} is unreasonable for short-dated"
                )

    def test_flat_vol_recovery_svi(self):
        """Flat vol at 0.25, wide strikes — all pillar IVs should be ~0.25."""
        S, T, r, q = 100.0, 1.0, 0.05, 0.0
        sigma = 0.25
        strikes = np.linspace(70, 130, 40)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2025, 1, 15)

        table = _generate_smile_options(S, T, r, q, sigma, 0.0,
                                        strikes, ts, exp)
        result = argiv.compute_fit_vol_surface(table)

        ivs = result.column("iv").to_pylist()
        non_null = [v for v in ivs if v is not None]
        assert len(non_null) >= 10, f"Only {len(non_null)} non-null IVs"
        for v in non_null:
            assert abs(v - sigma) < 0.03, f"iv={v} deviates from flat {sigma}"

    def test_too_few_points_produces_nan(self):
        """Group with < 5 OTM options → all IVs should be null."""
        S, T, r, q, sigma = 100.0, 0.5, 0.05, 0.0, 0.20
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2024, 7, 15)

        # Only 3 OTM options — not enough for 5-param SVI
        options = []
        for K in [95.0, 105.0, 110.0]:
            opt_type = -1 if K < S else 1
            price = _bs_price(opt_type, S, K, T, r, q, sigma)
            options.append({
                "option_type": opt_type, "spot": S, "strike": K,
                "expiry": T, "rate": r, "dividend_yield": q,
                "market_price": price,
            })

        table = _make_surface_table(options, ts, exp)
        result = argiv.compute_fit_vol_surface(table)

        ivs = result.column("iv").to_pylist()
        # With < 5 OTM points, SVI can't fit — all should be null
        assert all(v is None for v in ivs), (
            f"Expected all null IVs with < 5 points, got {ivs}"
        )

    def test_wing_ivs_bounded(self):
        """All non-null output IVs must be in (0, 5.0) for any reasonable input."""
        S, T, r, q = 100.0, 0.5, 0.05, 0.0
        sigma = 0.25
        skew = 0.0001
        strikes = np.linspace(80, 120, 30)
        ts = datetime.datetime(2024, 1, 15, 10, 0, 0)
        exp = datetime.date(2024, 7, 15)

        table = _generate_smile_options(S, T, r, q, sigma, skew,
                                        strikes, ts, exp)
        result = argiv.compute_fit_vol_surface(table)

        ivs = result.column("iv").to_pylist()
        non_null = [v for v in ivs if v is not None]
        assert len(non_null) > 0
        for v in non_null:
            assert 0 < v < 5.0, f"iv={v} out of (0, 5.0)"


# ---------------------------------------------------------------------------
# AAPL regression test using real market data
# ---------------------------------------------------------------------------


class TestAaplRegression:
    """Regression tests using real AAPL data from test_data/."""

    @pytest.fixture(scope="class")
    def aapl_enriched(self):
        """Load AAPL enriched data."""
        import polars as pl
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "test_data",
                            "aapl_enriched.parquet")
        if not os.path.exists(path):
            pytest.skip("test_data/aapl_enriched.parquet not found")
        return pl.read_parquet(path)

    def test_aapl_2018_07_26_no_blowup(self, aapl_enriched):
        """2018-07-26 had 1-DTE exp (2018-07-27) that produced 139x IVs.

        With SVI, all surface IVs must be in (0, 2.0).
        """
        import polars as pl

        day_data = aapl_enriched.filter(
            pl.col("timestamp").cast(pl.Date) == pl.lit("2018-07-26").cast(pl.Date)
        )
        if day_data.shape[0] == 0:
            pytest.skip("2018-07-26 not in test data")

        table = day_data.to_arrow()
        result = argiv.compute_fit_vol_surface(table)

        ivs = result.column("iv").to_pylist()
        non_null = [v for v in ivs if v is not None]
        assert len(non_null) > 0, "No IVs computed for 2018-07-26"
        for v in non_null:
            assert 0 < v < 2.0, (
                f"2018-07-26 surface IV={v} is unreasonable (blowup?)"
            )

    def test_aapl_2018_07_26_atm_reasonable(self, aapl_enriched):
        """ATM IV for the 1-DTE expiration should be in [0.1, 1.0]."""
        import polars as pl

        day_data = aapl_enriched.filter(
            pl.col("timestamp").cast(pl.Date) == pl.lit("2018-07-26").cast(pl.Date)
        )
        if day_data.shape[0] == 0:
            pytest.skip("2018-07-26 not in test data")

        table = day_data.to_arrow()
        result_pl = pl.from_arrow(argiv.compute_fit_vol_surface(table))

        # Filter for shortest-dated expiration
        short_dated = result_pl.filter(
            pl.col("expiration") == pl.lit("2018-07-27").cast(pl.Date)
        )
        if short_dated.shape[0] == 0:
            pytest.skip("2018-07-27 expiration not in surface output")

        atm = short_dated.filter(pl.col("delta") == 0.5)
        if atm.shape[0] > 0:
            iv_atm = atm["iv"][0]
            if iv_atm is not None:
                assert 0.1 < iv_atm < 1.0, f"ATM IV={iv_atm} unreasonable"
