"""Benchmark argiv (C++/OpenMP) vs pure-Python Black-Scholes Greeks computation."""

import math
import statistics
import time
from dataclasses import dataclass, field

import pyarrow as pa
from scipy.optimize import brentq
from scipy.stats import norm
import QuantLib as ql

from argiv.helpers import generate_dataset

# ---------------------------------------------------------------------------
# BenchmarkResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    library: str
    size: int
    times: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.mean(self.times)

    @property
    def std(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.times)

    @property
    def min(self) -> float:
        return min(self.times)

    @property
    def max(self) -> float:
        return max(self.times)


# ---------------------------------------------------------------------------
# Pure-Python Black-Scholes baseline
# ---------------------------------------------------------------------------


def _bs_call_price(S, K, T, r, q, sigma):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def _bs_put_price(S, K, T, r, q, sigma):
    call = _bs_call_price(S, K, T, r, q, sigma)
    return call - S * math.exp(-q * T) + K * math.exp(-r * T)


def _bs_greeks(option_type, S, K, T, r, q, sigma):
    """Compute all Greeks for a single option at the given implied vol."""
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    discount = math.exp(-r * T)
    growth = math.exp(-q * T)

    if option_type == 1:  # call
        delta = growth * norm.cdf(d1)
        rho = K * T * discount * norm.cdf(d2)
    else:  # put
        delta = -growth * norm.cdf(-d1)
        rho = -K * T * discount * norm.cdf(-d2)

    gamma = growth * norm.pdf(d1) / (S * sigma * sqrtT)
    vega = S * growth * norm.pdf(d1) * sqrtT
    theta = (
        -S * growth * norm.pdf(d1) * sigma / (2 * sqrtT)
        - option_type * r * K * discount * norm.cdf(option_type * d2)
        + option_type * q * S * growth * norm.cdf(option_type * d1)
    )

    return delta, gamma, vega, theta, rho


def compute_greeks_python(table):
    """Pure-Python compute_greeks — same interface as argiv.compute_greeks."""
    n = table.num_rows
    opt = table.column("option_type").to_pylist()
    spot = table.column("spot").to_pylist()
    strike = table.column("strike").to_pylist()
    expiry = table.column("expiry").to_pylist()
    rate = table.column("rate").to_pylist()
    div = table.column("dividend_yield").to_pylist()
    mkt = table.column("market_price").to_pylist()

    iv_out = []
    delta_out = []
    gamma_out = []
    vega_out = []
    theta_out = []
    rho_out = []

    for i in range(n):
        ot, S, K, T, r, q, price = opt[i], spot[i], strike[i], expiry[i], rate[i], div[i], mkt[i]
        pricer = _bs_call_price if ot == 1 else _bs_put_price

        def objective(sigma, _S=S, _K=K, _T=T, _r=r, _q=q, _price=price, _pricer=pricer):
            return _pricer(_S, _K, _T, _r, _q, sigma) - _price

        sigma = brentq(objective, 1e-6, 5.0, xtol=1e-6, maxiter=100)
        d, g, v, th, rh = _bs_greeks(ot, S, K, T, r, q, sigma)

        iv_out.append(sigma)
        delta_out.append(d)
        gamma_out.append(g)
        vega_out.append(v)
        theta_out.append(th)
        rho_out.append(rh)

    columns = {name: table.column(name) for name in table.column_names}
    columns["iv"] = pa.array(iv_out, type=pa.float64())
    columns["delta"] = pa.array(delta_out, type=pa.float64())
    columns["gamma"] = pa.array(gamma_out, type=pa.float64())
    columns["vega"] = pa.array(vega_out, type=pa.float64())
    columns["theta"] = pa.array(theta_out, type=pa.float64())
    columns["rho"] = pa.array(rho_out, type=pa.float64())

    return pa.table(columns)


# ---------------------------------------------------------------------------
# QuantLib-Python baseline
# ---------------------------------------------------------------------------


def compute_greeks_quantlib(table):
    """Row-by-row QuantLib BSM Greeks computation."""
    n = table.num_rows
    opt = table.column("option_type").to_pylist()
    spot = table.column("spot").to_pylist()
    strike = table.column("strike").to_pylist()
    expiry = table.column("expiry").to_pylist()
    rate = table.column("rate").to_pylist()
    div = table.column("dividend_yield").to_pylist()
    mkt = table.column("market_price").to_pylist()

    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    for i in range(n):
        S, K, T, r, q, price = spot[i], strike[i], expiry[i], rate[i], div[i], mkt[i]
        option_type = ql.Option.Call if opt[i] == 1 else ql.Option.Put

        # Maturity date from T in years
        maturity_days = max(int(round(T * 365)), 1)
        maturity = today + ql.Period(maturity_days, ql.Days)

        payoff = ql.PlainVanillaPayoff(option_type, K)
        exercise = ql.EuropeanExercise(maturity)
        option = ql.VanillaOption(payoff, exercise)

        spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
        rate_curve = ql.YieldTermStructureHandle(
            ql.FlatForward(today, r, day_count)
        )
        div_curve = ql.YieldTermStructureHandle(
            ql.FlatForward(today, q, day_count)
        )
        vol_handle = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, calendar, 0.20, day_count)
        )

        process = ql.BlackScholesMertonProcess(
            spot_handle, div_curve, rate_curve, vol_handle
        )

        # Solve for IV
        try:
            sigma = option.impliedVolatility(price, process)
        except RuntimeError:
            continue

        # Re-price with solved vol to extract Greeks
        solved_vol = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, calendar, sigma, day_count)
        )
        solved_process = ql.BlackScholesMertonProcess(
            spot_handle, div_curve, rate_curve, solved_vol
        )
        option.setPricingEngine(ql.AnalyticEuropeanEngine(solved_process))

        option.delta()
        option.gamma()
        option.vega()
        option.theta()
        option.rho()


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def benchmark_fn(fn, *args, warmup=2, trials=10):
    """Run warmup discarded invocations, then return list of trial times."""
    for _ in range(warmup):
        fn(*args)

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return times


def print_comparison_table(size, results, argiv_result):
    """Print a markdown table comparing libraries for a given dataset size."""
    print(f"\n### {size:,} rows\n")
    print("| Library        |   Mean (s) |    Std (s) | Median (s) |    Min (s) |    Max (s) | vs argiv |")
    print("|----------------|------------|------------|------------|------------|------------|----------|")

    for r in results:
        if r.library == "argiv":
            speedup_str = "       -"
        else:
            speedup_str = f"{r.median / argiv_result.median:>6.0f}x "
        print(
            f"| {r.library:<14s} "
            f"| {r.mean:>10.4f} "
            f"| {r.std:>10.4f} "
            f"| {r.median:>10.4f} "
            f"| {r.min:>10.4f} "
            f"| {r.max:>10.4f} "
            f"| {speedup_str} |"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import argiv

    # -- Header ----------------------------------------------------------------
    print("# argiv benchmark\n")
    print(f"  argiv:          installed")
    print(f"  Python (scipy): installed")
    print(f"  QuantLib:       installed")
    print()

    # -- Size scaling ----------------------------------------------------------
    # Size strategy:
    #   1,000:   Python, QuantLib, argiv
    #   10,000:  Python, QuantLib, argiv
    #   100,000: argiv only
    sizes = [1_000, 10_000, 100_000]

    print("## Size Scaling (all threads)\n")

    for n in sizes:
        table = generate_dataset(n)
        results = []

        # argiv (always run)
        times = benchmark_fn(argiv.compute_greeks, table)
        argiv_result = BenchmarkResult(library="argiv", size=n, times=times)
        results.append(argiv_result)

        # Python/scipy baseline (skip at 100k)
        if n <= 10_000:
            times = benchmark_fn(compute_greeks_python, table, warmup=1, trials=3)
            results.append(BenchmarkResult(library="Python (scipy)", size=n, times=times))

        # QuantLib (skip at 100k)
        if n <= 10_000:
            times = benchmark_fn(compute_greeks_quantlib, table, warmup=1, trials=3)
            results.append(BenchmarkResult(library="QuantLib", size=n, times=times))

        print_comparison_table(n, results, argiv_result)


if __name__ == "__main__":
    main()
