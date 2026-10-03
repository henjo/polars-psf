"""Corner report: measurements over all corners as Polars tables (uses the files in testdata/).

    python python/examples/corner_report.py

Every measurement on a family (one curve per corner) returns a table with one row per corner, so
joining, filtering, grouping and pivoting turn simulation results into a spec report.
"""

from pathlib import Path

import polars as pl

import polars_psf as pp

pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True)

TD = Path(__file__).resolve().parents[2] / "testdata"
r = pp.open(TD / "spectre25/sweep_psfbin")  # 10-stage RC ladder, temp x rval sweep, ac1 + tran1
ac, tran = r.ac1, r.tran1
corners = ["temp", "rval"]

# --- one measurement per corner: a table with one row per corner ---------------------------------
bw = ac.v("n10").bandwidth()  # temp | rval | bandwidth
gain = ac.v("n10").db20().value(1e6)  # temp | rval | value
delay = tran.v("in").delay(tran.v("n5"), 0.3, type="rising")  # temp | rval | delay
peak = tran.v("n5").ymax()  # temp | rval | ymax

# --- line them up per corner with join, then add readable columns -------------------------------
report = bw.join(gain, on=corners).join(delay, on=corners).join(peak, on=corners)
report = report.with_columns(
    bw_MHz=pl.col("bandwidth") / 1e6,
    gain_1MHz_dB=pl.col("value"),
    delay_ns=pl.col("delay") * 1e9,
    peak_n5_V=pl.col("ymax"),
)
report = report.select(*corners, "bw_MHz", "gain_1MHz_dB", "delay_ns", "peak_n5_V").with_columns(
    pl.selectors.float().exclude(corners).round(3)
)

# --- the spec: a True/False column ----------------------------------------------------------------
report = report.with_columns(ok=(pl.col("bw_MHz") > 2) & (pl.col("delay_ns") < 30))

print("all corners:")
print(report)

print("\nfailing corners:")
print(report.filter(~pl.col("ok")))

print("\nworst case per rval (temperature does not matter for this ladder):")
print(report.group_by("rval").agg(pl.col("bw_MHz").min(), pl.col("delay_ns").max()).sort("rval"))

# --- every node at every rval, as one pivot table ----------------------------------------------
nodes = [f"n{k}" for k in range(1, 11)]
bw = pl.concat(ac.v(n, temp=27).bandwidth().with_columns(node=pl.lit(n)) for n in nodes)
print("\n-3 dB bandwidth [MHz] per node and rval (27 °C):")
print(bw.with_columns((pl.col("bandwidth") / 1e6).round(1)).pivot("rval", index="node", values="bandwidth"))

# --- export: any Polars writer (csv, parquet, excel with xlsxwriter installed) -----------------
out = Path("corner_report.csv")
report.write_csv(out)
print(f"\nwrote {out} ({report.height} rows)")
out.unlink()  # keep the example side-effect free
