"""Example Polars queries on noise contributions (pnoise): python python/examples/noise_queries.py [FILE]"""
import sys

import polars as pl

import polars_psf as pp

pl.Config.set_tbl_rows(8)
pl.Config.set_tbl_hide_dataframe_shape(True)

path = sys.argv[1] if len(sys.argv) > 1 else "testdata/pycircuit/psf/pnoise0.pnoise"

# every device's total contribution as a lazy long table: freq | signal | value
d = pp.open(path)
noise = d.scan_long(field="total")

print("1. total noise per frequency, checked against the simulator's output noise")
out = d.scan().select("freq", "out")
print(noise.group_by("freq").agg(pl.col("value").sum().alias("sum_contrib"))
      .join(out, on="freq")
      .with_columns(ratio=pl.col("sum_contrib") / pl.col("out") ** 2)
      .sort("freq").collect().head(3))

print("2. top contributors at 10 kHz, with share of the total")
print(noise.filter(pl.col("freq").is_between(9e3, 11e3))
      .with_columns(share=pl.col("value") / pl.col("value").sum())
      .sort("value", descending=True).head(5).collect())

print("3. share per hierarchy block at 10 kHz (block = first 2 path levels, computed once per name)")
blocks = pl.LazyFrame({"signal": d.names(field="total")},
                      schema={"signal": pl.Categorical}).with_columns(
    block=pl.col("signal").cast(pl.String).str.extract(r"^([^.]+\.[^.]+)"))
print(noise.filter(pl.col("freq").is_between(9e3, 11e3)).join(blocks, on="signal")
      .group_by("block").agg(pl.col("value").sum().alias("V2/Hz"))
      .with_columns(share=pl.col("V2/Hz") / pl.col("V2/Hz").sum())
      .sort("V2/Hz", descending=True).collect().head(5))

print("4. only resistors (filter pushed down: only those signals are decoded)")
print(noise.filter(pl.col("signal").cast(pl.String).str.contains(r"\.r\d+$"))
      .group_by("freq").agg(pl.col("value").sum().alias("resistor noise"))
      .sort("freq").collect().head(3))

print("5. one mechanism: flicker noise (fn), integrated over frequency, per device that has it")
fn = d.scan_long(field="fn")      # resistors and bjts have 'fn'; others are left out
print(fn.sort("signal", "freq")
      .with_columns(df=pl.col("freq").diff().over("signal"),
                    avg=(pl.col("value") + pl.col("value").shift().over("signal")) / 2)
      .group_by("signal").agg((pl.col("df") * pl.col("avg")).sum().alias("integrated V2"))
      .sort("integrated V2", descending=True).collect().head(3))
