"""Benchmark argiv (C++/OpenMP) vs pure-Python Black-Scholes Greeks computation."""

import math
import os
import statistics
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
from scipy.optimize import brentq
from scipy.stats import norm


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
# Data generation
# ---------------------------------------------------------------------------

def generate_dataset(n):
    """Generate a realistic mixed options dataset with known-valid IV solutions."""
    rng = np.random.default_rng(42)

    option_type = rng.choice([1, -1], size=n).astype(np.int32)
    spot = np.full(n, 100.0)
    strike = rng.uniform(80.0, 120.0, size=n)
    expiry = rng.uniform(0.1, 2.0, size=n)
    rate = np.full(n, 0.05)
    dividend_yield = np.full(n, 0.01)
    true_sigma = rng.uniform(0.1, 0.5, size=n)

    # Compute market prices from known sigmas via BS formula
    market_price = np.empty(n)
    for i in range(n):
        if option_type[i] == 1:
            market_price[i] = _bs_call_price(
                spot[i], strike[i], expiry[i], rate[i], dividend_yield[i], true_sigma[i]
            )
        else:
            market_price[i] = _bs_put_price(
                spot[i], strike[i], expiry[i], rate[i], dividend_yield[i], true_sigma[i]
            )

    return pa.table({
        "option_type": pa.array(option_type, type=pa.int32()),
        "spot": pa.array(spot, type=pa.float64()),
        "strike": pa.array(strike, type=pa.float64()),
        "expiry": pa.array(expiry, type=pa.float64()),
        "rate": pa.array(rate, type=pa.float64()),
        "dividend_yield": pa.array(dividend_yield, type=pa.float64()),
        "market_price": pa.array(market_price, type=pa.float64()),
    })


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def time_fn(fn, *args, repeats=3):
    """Return median wall-clock time over `repeats` runs."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


# ---------------------------------------------------------------------------
# Thread-scaling sub-process runner
# ---------------------------------------------------------------------------

_THREAD_BENCH_SCRIPT = """\
import os, sys, time, statistics
os.environ["OMP_NUM_THREADS"] = sys.argv[1]

import pyarrow.ipc as ipc
import argiv

reader = ipc.open_file(sys.argv[2])
table = reader.read_all()

times = []
for _ in range(3):
    t0 = time.perf_counter()
    argiv.compute_greeks(table)
    times.append(time.perf_counter() - t0)

print(statistics.median(times))
"""


def bench_threads(table, threads_list):
    """Run argiv in a subprocess for each thread count, return {threads: median_seconds}."""
    import tempfile

    # Write table to a temp IPC file for subprocess to read
    tmp = tempfile.NamedTemporaryFile(suffix=".arrow", delete=False)
    writer = pa.ipc.new_file(tmp, table.schema)
    writer.write_table(table)
    writer.close()
    tmp.close()

    results = {}
    for t in threads_list:
        proc = subprocess.run(
            [sys.executable, "-c", _THREAD_BENCH_SCRIPT, str(t), tmp.name],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"  Thread={t} FAILED: {proc.stderr.strip()}", file=sys.stderr)
            results[t] = float("nan")
        else:
            results[t] = float(proc.stdout.strip())
    os.unlink(tmp.name)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argiv

    sizes = [1_000, 10_000]

    # -- Size scaling -----------------------------------------------------------
    print("## Size Scaling (all threads)\n")
    print("| Rows      | Python (s) | C++ (s)  | Speedup |")
    print("|-----------|------------|----------|---------|")

    for n in sizes:
        table = generate_dataset(n)

        cpp_t = time_fn(argiv.compute_greeks, table)

        py_t = time_fn(compute_greeks_python, table)
        speedup = py_t / cpp_t
        print(f"| {n:>9,} | {py_t:10.4f} | {cpp_t:8.4f} | {speedup:>5.0f}x  |")


    # -- Thread scaling ---------------------------------------------------------
    print()
    print("## Thread Scaling (100k rows)\n")
    print("| Threads | C++ (s)  | vs 1-thread |")
    print("|---------|----------|-------------|")

    table_100k = generate_dataset(1_000_000)
    threads = [1, 2, 4, 8, 16]
    results = bench_threads(table_100k, threads)

    base = results.get(1, float("nan"))
    for t in threads:
        sec = results[t]
        if math.isnan(sec) or math.isnan(base):
            print(f"| {t:>7} | {sec:8.4f} |         n/a |")
        else:
            scaling = base / sec
            print(f"| {t:>7} | {sec:8.4f} | {scaling:>9.2f}x |")


if __name__ == "__main__":
    main()
