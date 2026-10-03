"""Noise-summary benchmark: python bench/noise.py FILE  (e.g. from bench/gen_noise.py).

Sums the "total" noise contribution of every device per frequency, several ways.
"""
import sys
import time

import numpy as np
import polars as pl

import polars_psf as pp

path = sys.argv[1]


def t(label, fn, reps=3):
    best = 1e9
    for _ in range(reps):
        a = time.perf_counter()
        r = fn()
        best = min(best, time.perf_counter() - a)
    print(f"| {label} | {best * 1e3:,.1f} ms |")
    return r


f = pp.open(path).file()  # low-level reader for A-C
devs = f.names_with_field("total")
print(f"{len(devs)} devices x {f.header['PSF sweep points']} frequencies\n")
print("| approach (open + read + sum per frequency) | time |\n|---|---|")


def loop(n):
    g = pp.open(path).file()
    acc = 0
    for c in devs[:n]:
        acc = acc + g.read_signal(c)[c].struct.field("total").to_numpy()
    return acc


n_loop = 1000
per = t(f"A. Python loop, one read per device ({n_loop} devices; x{len(devs) // n_loop} for all)", lambda: loop(n_loop), reps=1)


def wide():
    df = pp.open(path).scan().collect()
    return df.select("freq", pl.sum_horizontal([pl.col(c).struct.field("total") for c in devs]).alias("total"))


w = t("B. wide table + sum_horizontal", wide)


def long_eager():
    return pp.open(path).file().to_polars_long(field="total").group_by("freq").agg(pl.col("value").sum())


t("C. to_polars_long(field='total') + group_by", long_eager)


def long_lazy():
    return pp.open(path).scan_long(field="total").group_by("freq").agg(pl.col("value").sum()).collect()


lz = t("D. scan_long(field='total') + group_by (lazy)", long_lazy)


def long_block():
    return (
        pp.open(path).scan_long(field="total")
        .filter(pl.col("signal").cast(pl.String).str.starts_with("x3."))
        .group_by("freq")
        .agg(pl.col("value").sum())
        .collect()
    )


t("E. scan_long, one block of 1000 devices (name filter pushed down)", long_block)
ref = w.sort("freq")["total"].to_numpy()
assert np.allclose(lz.sort("freq")["value"].to_numpy(), ref, rtol=1e-12)
print("\nresults agree")
