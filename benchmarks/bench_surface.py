"""Benchmark argiv vol surface fitting on real AAPL enriched data."""

import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import argiv


# ---------------------------------------------------------------------------
# BenchmarkResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    label: str
    rows_in: int
    groups: int
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
# Helpers
# ---------------------------------------------------------------------------


def benchmark_fn(fn, *args, warmup=1, trials=5):
    """Run warmup discarded invocations, then return list of trial times."""
    for _ in range(warmup):
        fn(*args)

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return times


def count_groups(table: pa.Table) -> int:
    """Count unique (timestamp, expiration) pairs."""
    combo = pc.binary_join_element_wise(
        table.column("timestamp").cast(pa.utf8()),
        table.column("expiration").cast(pa.utf8()),
        "|",
    )
    return len(pc.unique(combo))


def read_n_row_groups(path, n):
    """Read the first n row groups from a parquet file."""
    pf = pq.ParquetFile(path)
    n = min(n, pf.metadata.num_row_groups)
    tables = [pf.read_row_group(i) for i in range(n)]
    return pa.concat_tables(tables), n


def print_result_table(results: list[BenchmarkResult]):
    """Print a markdown table of benchmark results."""
    print()
    print("| Label                          |    Rows In |   Groups |   Mean (s) |    Std (s) | Median (s) |    Min (s) |    Max (s) |     rows/s |")
    print("|--------------------------------|------------|----------|------------|------------|------------|------------|------------|------------|")
    for r in results:
        rows_per_sec = r.rows_in / r.median if r.median > 0 else float("inf")
        print(
            f"| {r.label:<30s} "
            f"| {r.rows_in:>10,} "
            f"| {r.groups:>8,} "
            f"| {r.mean:>10.4f} "
            f"| {r.std:>10.4f} "
            f"| {r.median:>10.4f} "
            f"| {r.min:>10.4f} "
            f"| {r.max:>10.4f} "
            f"| {rows_per_sec:>10,.0f} |"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).resolve().parent.parent / "test_data" / "aapl_enriched.parquet"

# Row-group slices to benchmark (each ~250K rows)
SLICES = [1, 2, 5, 10]


def main():
    print("# argiv surface fitting benchmark\n")
    print(f"  CPU cores:  {os.cpu_count()}")
    print(f"  Data:       {DATA_PATH.name}")

    meta = pq.read_metadata(DATA_PATH)
    print(f"  Total rows: {meta.num_rows:,}")
    print(f"  Row groups: {meta.num_row_groups} (~{meta.row_group(0).num_rows:,} rows each)")
    print()

    trials = 3
    warmup = 1

    # ------------------------------------------------------------------
    # Phase 1: compute_fit_vol_surface (greeks + SVI in one shot)
    # ------------------------------------------------------------------
    print("## compute_fit_vol_surface (greeks + SVI fit)\n")
    results_combo = []
    for n_rg in SLICES:
        table, actual_rg = read_n_row_groups(DATA_PATH, n_rg)
        nrows = len(table)
        n_unique = count_groups(table)
        label = f"{actual_rg} rg ({nrows:,} rows)"

        print(f"  {label} — {n_unique} groups ...", end=" ", flush=True)
        times = benchmark_fn(argiv.compute_fit_vol_surface, table, warmup=warmup, trials=trials)
        print(f"median {statistics.median(times):.3f}s")

        results_combo.append(BenchmarkResult(label=label, rows_in=nrows, groups=n_unique, times=times))

    print_result_table(results_combo)

    # ------------------------------------------------------------------
    # Phase 2: fit_vol_surface only (pre-computed IV)
    # ------------------------------------------------------------------
    print("## fit_vol_surface only (pre-computed IV)\n")
    results_fit = []
    for n_rg in SLICES:
        table, actual_rg = read_n_row_groups(DATA_PATH, n_rg)
        enriched = argiv.compute_greeks(table)
        nrows = len(enriched)
        n_unique = count_groups(enriched)
        label = f"{actual_rg} rg ({nrows:,} rows)"

        print(f"  {label} — {n_unique} groups ...", end=" ", flush=True)
        times = benchmark_fn(argiv.fit_vol_surface, enriched, warmup=warmup, trials=trials)
        print(f"median {statistics.median(times):.3f}s")

        results_fit.append(BenchmarkResult(label=label, rows_in=nrows, groups=n_unique, times=times))

    print_result_table(results_fit)


if __name__ == "__main__":
    main()
