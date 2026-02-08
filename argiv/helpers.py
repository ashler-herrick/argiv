from math import exp, log, sqrt

from scipy.stats import norm
import numpy as np
import pyarrow as pa
import QuantLib as ql


# Cache for the base 1000 contracts
_BASE_TABLE = None


def _generate_base_contracts():
    """Generate 1000 base contracts using QuantLib for fast pricing."""
    rng = np.random.default_rng(42)
    n = 1000

    # Generate random parameters
    option_type = rng.choice([1, -1], size=n).astype(np.int32)
    spot = np.full(n, 100.0)
    strike = rng.uniform(80.0, 120.0, size=n)
    expiry = rng.uniform(0.1, 2.0, size=n)
    rate = np.full(n, 0.05)
    dividend_yield = np.full(n, 0.01)
    true_sigma = rng.uniform(0.1, 0.5, size=n)

    # Setup QuantLib
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    # Compute market prices using QuantLib (much faster than scipy)
    market_price = np.empty(n)
    for i in range(n):
        S, K, T, r, q, sigma = spot[i], strike[i], expiry[i], rate[i], dividend_yield[i], true_sigma[i]
        opt_type = ql.Option.Call if option_type[i] == 1 else ql.Option.Put

        # Maturity date
        maturity_days = max(int(round(T * 365)), 1)
        maturity = today + ql.Period(maturity_days, ql.Days)

        # Setup option
        payoff = ql.PlainVanillaPayoff(opt_type, K)
        exercise = ql.EuropeanExercise(maturity)
        option = ql.VanillaOption(payoff, exercise)

        # Setup market data
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
        rate_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
        div_curve = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count))
        vol_handle = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, calendar, sigma, day_count)
        )

        # Price with analytic engine
        process = ql.BlackScholesMertonProcess(spot_handle, div_curve, rate_curve, vol_handle)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        market_price[i] = option.NPV()

    return pa.table({
        "option_type": pa.array(option_type, type=pa.int32()),
        "spot": pa.array(spot, type=pa.float64()),
        "strike": pa.array(strike, type=pa.float64()),
        "expiry": pa.array(expiry, type=pa.float64()),
        "rate": pa.array(rate, type=pa.float64()),
        "dividend_yield": pa.array(dividend_yield, type=pa.float64()),
        "market_price": pa.array(market_price, type=pa.float64()),
    })


def generate_dataset(n):
    """Generate dataset by repeating 1000 base contracts.

    Much faster than generating random data - simply tiles the base contracts
    to reach the desired size.
    """
    global _BASE_TABLE

    if _BASE_TABLE is None:
        _BASE_TABLE = _generate_base_contracts()

    # If n <= 1000, just slice
    if n <= 1000:
        return _BASE_TABLE.slice(0, n)

    # Otherwise, tile to reach n rows
    repeats = (n + 999) // 1000  # Ceiling division
    tables = [_BASE_TABLE] * repeats
    combined = pa.concat_tables(tables, promote_options="none").combine_chunks()

    # Slice to exact size
    return combined.slice(0, n)


def _bs_call_price(S, K, T, r, q, sigma):
    """Reference Black-Scholes call price for test verification."""


    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def _bs_put_price(S, K, T, r, q, sigma):
    """Reference Black-Scholes put price via put-call parity."""

    call = _bs_call_price(S, K, T, r, q, sigma)
    return call - S * exp(-q * T) + K * exp(-r * T)