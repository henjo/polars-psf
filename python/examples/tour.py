# /// script
# requires-python = ">=3.11"
# dependencies = ["polars-psf", "marimo", "altair"]
# ///
"""Tour of polars-psf: marimo edit python/examples/tour.py (uses the files in testdata/)."""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    import polars_psf as pp

    alt.data_transformers.disable_max_rows()
    TD = (mo.notebook_dir() / "../../testdata").resolve()
    return TD, alt, mo, pl, pp


@app.cell
def _(mo):
    mo.md("""
    # polars-psf tour

    Spectre PSF results (psfbin, psfascii, PSFXL) as Polars tables. Everything is lazy: opening a
    file reads only its declarations, and filters on sweep parameters and signal names are pushed
    into the reader, so only the data a query needs is decoded. All queries run in Polars' Rust
    engine.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. One file
    """)
    return


@app.cell
def _(TD, pp):
    tran = pp.open(TD / "spectre25/psfxl_ac/tran1.tran.tran")  # PSFXL: stub + .psfxl picked up
    tran, tran.layout, tran.is_complete
    return (tran,)


@app.cell
def _(mo, tran):
    header = {k: tran.header[k] for k in ("simulator", "version", "analysis type", "stop")}
    picked = mo.ui.multiselect(tran.names, value=["in", "n1", "n5", "n10"], label="signals")
    mo.vstack([mo.md(f"Header: `{header}`"), picked])
    return (picked,)


@app.cell
def _(alt, picked, tran):
    # read only the selected signals; unpivot to long form for plotting
    wave = tran.to_polars(picked.value).unpivot(index="time", variable_name="signal")
    alt.Chart(wave).mark_line().encode(
        x=alt.X("time", title="time [s]"), y=alt.Y("value", title="V"), color="signal"
    ).properties(height=260, title=f"{len(picked.value)} signals x {wave.height // max(1, len(picked.value)):,} points")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. A result directory: sweeps as columns

    `logFile` / `runObjFile` trees (nested sweeps, Monte Carlo, ADE parametric runs) become one
    table with a column per outer parameter. Values come from the `.sweep` files at full precision.
    """)
    return


@app.cell
def _(TD, pp):
    res = pp.results(TD / "spectre25/sweep_psfbin")
    res.analyses
    return (res,)


@app.cell
def _(res):
    res.leaves("tran1").drop("path")
    return


@app.cell
def _(mo, res):
    temps = sorted(res.leaves("tran1")["temp"].unique().to_list())
    temp = mo.ui.dropdown({f"{t:g} °C": t for t in temps}, value=f"{temps[1]:g} °C", label="temp")
    temp
    return (temp,)


@app.cell
def _(alt, pl, res, temp):
    # the filter on `temp` is pushed down: leaf files of other temperatures are never opened
    one_temp = (
        res.scan("tran1")
        .filter(pl.col("temp") == temp.value)
        .select("rval", "time", "n10")
        .collect()
    )
    alt.Chart(one_temp).mark_line().encode(
        x="time", y=alt.Y("n10", title="n10 [V]"), color="rval:N"
    ).properties(height=240, title=f"transient at {temp.value:g} °C, one line per rval")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Complex AC data

    Complex signals are `Struct{re, im}`; the `.cx` namespace gives `db20`, `phase`, `abs`, ...
    """)
    return


@app.cell
def _(alt, pl, res, temp):
    bode = (
        res.scan("ac1")
        .filter(pl.col("temp") == temp.value)
        .select("rval", "freq", db=pl.col("n10").cx.db20(), phase=pl.col("n10").cx.phase())
        .collect()
    )
    _base = alt.Chart(bode).encode(x=alt.X("freq", scale=alt.Scale(type="log")), color="rval:N")
    (_base.mark_line().encode(y=alt.Y("db", title="|n10| [dB]")).properties(height=180)
     & _base.mark_line().encode(y=alt.Y("phase", title="phase [deg]")).properties(height=180))
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Long format: aggregate across corners

    `scan_long` gives `[params..., sweep, signal, value]`. Filters on parameters and signal
    names are both pushed down; the aggregation runs in Polars.
    """)
    return


@app.cell
def _(pl, res):
    (
        res.scan_long("tran1")
        .filter(pl.col("signal").is_in(["n1", "n5", "n10"]))
        .group_by("temp", "rval", "signal")
        .agg(pl.col("value").max().alias("peak"))
        .collect()
        .pivot("signal", index=["temp", "rval"], values="peak")
        .sort("temp", "rval")
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Monte Carlo

    Monte Carlo iterations are just another parameter column.
    """)
    return


@app.cell
def _(TD, alt, pl, pp):
    mc = pp.results(TD / "psf-parser/binary").scan("mymonte_tran1").collect()
    _stats = mc.group_by("iteration").agg(pl.col("out").max().alias("max out")).sort("iteration")
    _chart = alt.Chart(mc).mark_line().encode(x="time", y="out", color="iteration:N")
    _chart.properties(height=220, title="out per Monte Carlo iteration") | alt.Chart(_stats).mark_bar().encode(
        x="iteration:N", y="max out"
    ).properties(height=220, width=120)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Noise contributions

    Noise analyses hold one struct per device. `scan_long(field="total")` turns every device's
    total contribution into rows, so a noise summary is a group-by.
    """)
    return


@app.cell
def _(TD, mo, pp):
    pnoise = TD / "pycircuit/psf/pnoise0.pnoise"
    freqs = pp.open(pnoise).to_polars(["out"])["freq"].to_list()
    at = mo.ui.slider(steps=freqs, value=freqs[len(freqs) // 2], label="frequency [Hz]", show_value=True)
    at
    return at, pnoise


@app.cell
def _(alt, at, pl, pnoise, pp):
    top = (
        pp.scan_long(pnoise, field="total")
        .filter(pl.col("freq") == at.value)
        .with_columns(share=pl.col("value") / pl.col("value").sum())
        .sort("value", descending=True)
        .head(10)
        .collect()
    )
    alt.Chart(top).mark_bar().encode(
        x=alt.X("share", axis=alt.Axis(format="%")), y=alt.Y("signal:N", sort="-x")
    ).properties(height=240, title=f"top 10 noise contributors at {at.value:.3g} Hz")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Operating point and device info
    """)
    return


@app.cell
def _(TD, pl, pp):
    op = pp.open(TD / "pycircuit/psf/dcOpInfo.info")
    # non-swept values: one struct per device; keep the bipolar transistors
    (
        pl.DataFrame(
            [{"device": k, **v} for k, v in op.values().items() if isinstance(v, dict) and "betadc" in v]
        )
        .select("device", "ic", "vbe", "gm", "betadc", "ft")
        .sort("ic")
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. libpsf-compatible numpy API

    Scripts written against libpsf run unchanged with `import polars_psf.compat as libpsf`.
    """)
    return


@app.cell
def _(TD):
    import polars_psf.compat as libpsf

    ds = libpsf.PSFDataSet(str(TD / "spectre25/psfxl_ac/tran1.tran.tran"))
    t, v = ds.get_sweep_values(), ds.get_signal("n10")
    type(v), v.dtype, t[:3], v[:3]
    return


if __name__ == "__main__":
    app.run()
