"""Memory diagnostic: track RSS and Arrow pool bytes across repeated compute_greeks calls."""

import gc
import os

import pyarrow as pa

import argiv
from argiv.helpers import generate_dataset


def rss_mb():
    """Current process RSS in MB (Linux)."""
    with open(f"/proc/{os.getpid()}/statm") as f:
        pages = int(f.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)


def main():
    n = 100_000
    iterations = 50

    print(f"Dataset size: {n:,} rows")
    print(f"Iterations:   {iterations}")
    print()

    table = generate_dataset(n)
    pool = pa.default_memory_pool()

    # Warm up
    result = argiv.compute_greeks(table)
    del result
    gc.collect()

    print(f"{'Iter':>4s}  {'RSS (MB)':>10s}  {'Arrow alloc (MB)':>18s}  {'Arrow peak (MB)':>18s}")
    print("-" * 60)

    rss_start = rss_mb()
    for i in range(1, iterations + 1):
        result = argiv.compute_greeks(table)
        del result
        # Force Python GC to release any ref-counted Arrow buffers
        gc.collect()

        if i % 5 == 0 or i == 1:
            rss_now = rss_mb()
            alloc = pool.bytes_allocated() / (1024 * 1024)
            peak = pool.max_memory() / (1024 * 1024)
            print(f"{i:4d}  {rss_now:10.1f}  {alloc:18.1f}  {peak:18.1f}")

    rss_end = rss_mb()
    alloc_end = pool.bytes_allocated() / (1024 * 1024)
    print()
    print(f"RSS delta:          {rss_end - rss_start:+.1f} MB")
    print(f"Arrow pool final:   {alloc_end:.1f} MB")
    print(f"Arrow pool peak:    {pool.max_memory() / (1024 * 1024):.1f} MB")

    if alloc_end > 50:
        print("\n** Arrow pool has significant outstanding allocations — possible leak in C++ side")
    elif rss_end - rss_start > 100:
        print("\n** RSS grew but Arrow pool is flat — likely jemalloc retention, not a real leak")
    else:
        print("\n** Looks clean.")


if __name__ == "__main__":
    main()
