# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "polars-psf",
#     "marimo",
#     "altair",
#     "numpy",
#     "polars==1.44.2",
# ]
#
# [tool.uv.sources]
# polars-psf = { path = ".." }
# ///
"""Waveform tour (uses the files in testdata/).

    uvx marimo edit --sandbox python/examples/waveform.py

``--sandbox`` builds the environment from the header above; ``polars-psf`` is taken from this
checkout (``python/``) and built with maturin, so a Rust toolchain is needed. With ``polars-psf``
already installed, plain ``marimo edit python/examples/waveform.py`` works too.

`polars_psf.post.Waveform` is a value column over one or more index columns, backed by a Polars
`LazyFrame` (materialized on demand): numpy-style post-processing (transfer functions, metrics,
differentiation, spectra), Altair plotting, and multi-dimensional sweeps (Monte Carlo, corners)
without leaving Polars.
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    import polars_psf as pp

    alt.data_transformers.disable_max_rows()
    TD = (mo.notebook_dir() / "../../testdata").resolve()
    return TD, alt, mo, np, pl, pp


@app.cell
def _(mo):
    mo.md("""
    # `Waveform`: signals with numpy-style post-processing

    A `Waveform` is **a value column over one or more index columns**, backed by a `pl.LazyFrame`;
    expression operations build a query plan and the frame is materialized only when you touch a
    value (`w.x`/`w.y`, `w.to_polars()`, reductions, plots).
    Every operation stays in Polars; no data is copied out until you ask for it.

    - `w.x` / `w.y` are `pl.Series`, `w.to_polars()` is the frame, `w.index` lists the index columns.
    - The last index column is the **sampling axis** (`time`, `freq`, …); any other index columns
      (Monte Carlo iterations, corner parameters) are carried through every operation.
    - Complex signals are `Struct{re, im}`; `+ - * / **`, `abs`, `phase`, `real`, `imag`, `conj`
      are built on those fields.
    - `w.plot()` draws with matplotlib; `w.plot(backend="altair")` gives an interactive Altair chart, used here.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. One signal

    `pp.wave(file, name)` reads a single trace with **its own sweep axis** (so PSFXL files, whose
    signals may have different time grids, work too).
    """)
    return


@app.cell
def _(TD, pp):
    tran = pp.open(TD / "pycircuit/psf/tran.tran")
    return (tran,)


@app.cell
def _(mo, tran):
    signal = mo.ui.dropdown(tran.names(), value="I1:d", label="signal")
    signal
    return (signal,)


@app.cell
def _(pp, signal, tran):
    w = pp.wave(tran, signal.value)
    w
    return (w,)


@app.cell
def _(mo, w):
    mo.hstack(
        [
            mo.stat(f"{w.ymax():.4g}", label=f"max [{w.yunit}]", caption=f"at {w.argmax():.3g} s"),
            mo.stat(f"{w.ymin():.4g}", label=f"min [{w.yunit}]"),
            mo.stat(f"{w.rms():.4g}", label=f"rms [{w.yunit}]"),
            mo.stat(f"{w.average():.4g}", label=f"mean [{w.yunit}]"),
        ],
        justify="start",
    )
    return


@app.cell
def _(w):
    w.plot(backend="altair").properties(height=240, title=f"{w.yname} over {w.xname}")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Resample, differentiate

    `value(x)` interpolates **linearly between samples** (clamped outside the range). `deriv()` is a
    forward difference, one point shorter — and it keeps the rest of the index columns.
    """)
    return


@app.cell
def _(mo, w):
    tq = mo.ui.slider(
        start=0.0,
        stop=float(w.x.max()),
        value=float(w.x.max()) * 0.5,
        label="query time [s]",
        show_value=True,
    )
    tq
    return (tq,)


@app.cell
def _(mo, tq, w):
    mo.md(f"""
    `w.value({tq.value:.4g})` = **{w.value(tq.value):.5g} {w.yunit}**
    """)
    return


@app.cell
def _(w):
    d = w.deriv()
    d
    return (d,)


@app.cell
def _(alt, d, w):
    _x = alt.X(field="time", type="quantitative", title="time [s]")
    (
        alt.Chart(w.to_polars())
        .mark_line()
        .encode(x=_x, y=alt.Y(field=w.yname, type="quantitative", title=f"{w.yname} [{w.yunit}]"))
        .properties(height=170)
        & alt.Chart(d.to_polars())
        .mark_line(color="#e45756")
        .encode(x=_x, y=alt.Y(field=d.yname, type="quantitative", title="derivative"))
        .properties(height=170)
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Complex AC: a transfer function

    Dividing two complex waveforms gives the transfer function, still as `Struct{re, im}`. Parameters
    of a sweep are picked with `dataset.wave(signal, result, temp=…, rval=…)` — the leaf is selected
    before anything is read.
    """)
    return


@app.cell
def _(TD, pp):
    res = pp.open(TD / "spectre25/sweep_psfbin")
    return (res,)


@app.cell
def _(mo, res):
    _leaves = res.leaves("ac1")
    _temps = sorted(_leaves["temp"].unique().to_list())
    _rvals = sorted(_leaves["rval"].unique().to_list())
    temp = mo.ui.dropdown({f"{t:g} °C": t for t in _temps}, value=f"{_temps[1]:g} °C", label="temp")
    rval = mo.ui.dropdown({f"{r:g} Ω": r for r in _rvals}, value=f"{_rvals[0]:g} Ω", label="rval")
    mo.hstack([temp, rval])
    return rval, temp


@app.cell
def _(res, rval, temp):
    bode = res.wave("n10", "ac1", temp=temp.value, rval=rval.value)
    return (bode,)


@app.cell
def _(bode, mo):
    mo.hstack(
        [
            mo.stat(f"{bode.bandwidth():.5g}", label="bandwidth (-3 dB) [Hz]"),
            mo.stat(f"{abs(bode).value(bode.x.min()):.5g}", label="|H| at the lowest freq"),
            mo.stat(f"{len(bode):,}", label="points"),
        ],
        justify="start",
    )
    return


@app.cell
def _(alt, bode, pl):
    _x = alt.X(field="freq", type="quantitative", scale=alt.Scale(type="log"), title="freq [Hz]")
    _rule = (
        alt.Chart(pl.DataFrame({"f": [bode.bandwidth()]}))
        .mark_rule(color="#e45756", strokeDash=[4, 4])
        .encode(x=alt.X(field="f", type="quantitative"))
    )
    _db = bode.db20().to_polars()
    _ph = bode.phase().to_polars()
    (
        (
            alt.Chart(_db)
            .mark_line()
            .encode(x=_x, y=alt.Y(field=_db.columns[1], type="quantitative", title="|n10| [dB]"))
            .properties(height=180)
            + _rule
        )
        & alt.Chart(_ph)
        .mark_line()
        .encode(x=_x, y=alt.Y(field=_ph.columns[1], type="quantitative", title="phase [deg]"))
        .properties(height=180)
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Loop-gain metrics

    `unity_gain_frequency()`, `phase_margin()` and `bandwidth()` work on any complex waveform — here a
    synthetic two-pole loop gain `H = 1000 / ((1 + jf/1e4)(1 + jf/2.2e7))`, built with `from_series`
    (≈ 9.2 MHz unity gain, ≈ 67° phase margin).

    The sampling axis is designated by `index[-1]`, and `w.value()` / `cross()` interpolate on it.
    """)
    return


@app.cell
def _(np, pl, pp):
    _f = np.logspace(1, 9, 601)
    _h = 1e3 / ((1 + 1j * _f / 1e4) * (1 + 1j * _f / 2.2e7))
    # Polars has no complex dtype: complex signals are Struct{re, im} (as the readers produce)
    loop = pp.Waveform.from_series(
        pl.Series("freq", _f),
        pl.DataFrame({"re": _h.real, "im": _h.imag}).to_struct("loopGain"),
    )
    loop
    return (loop,)


@app.cell
def _(loop, mo, pp):
    mo.hstack(
        [
            mo.stat(f"{loop.unity_gain_frequency():.5g}", label="unity-gain freq [Hz]"),
            mo.stat(f"{pp.phase_margin(loop):.4g}", label="phase margin [deg]"),
            mo.stat(f"{loop.bandwidth():.5g}", label="bandwidth (-3 dB) [Hz]"),
        ],
        justify="start",
    )
    return


@app.cell
def _(alt, loop):
    _x = alt.X(field="freq", type="quantitative", scale=alt.Scale(type="log"), title="freq [Hz]")
    _db = loop.db20().to_polars()
    _ph = loop.phase().to_polars()
    (
        alt.Chart(_db)
        .mark_line()
        .encode(x=_x, y=alt.Y(field=_db.columns[1], type="quantitative", title="|H| [dB]"))
        .properties(height=180)
        & alt.Chart(_ph)
        .mark_line()
        .encode(x=_x, y=alt.Y(field=_ph.columns[1], type="quantitative", title="phase [deg]"))
        .properties(height=180)
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. More than one index column

    A Monte Carlo run is one waveform with `index=["iteration", "time"]`: the sampling axis stays
    `time`, and `iteration` rides along. Filtering is a Polars expression on the frame — no separate
    objects per iteration.
    """)
    return


@app.cell
def _(TD, pp):
    mc = pp.Waveform(
        pp.open(TD / "psf-parser/binary")
        .scan("mymonte_tran1")
        .select("iteration", "time", "out"),  # lazy: decoded on first value access
        "out",
        index=["iteration", "time"],
    )
    mc
    return (mc,)


@app.cell
def _(mc, mo):
    _items = {f"{i:g}": i for i in sorted(mc.to_polars()["iteration"].unique().to_list())}
    iteration = mo.ui.dropdown(_items, value=next(iter(_items)), label="iteration")
    iteration
    return (iteration,)


@app.cell
def _(iteration, mc, pl):
    sel = mc.filter(pl.col("iteration") == iteration.value)
    return (sel,)


@app.cell
def _(alt, mc, sel):
    _x = alt.X(field="time", type="quantitative", title="time [s]")
    _y = alt.Y(field="out", type="quantitative", title="out [V]")
    (
        alt.Chart(mc.to_polars())
        .mark_line(opacity=0.3)
        .encode(x=_x, y=_y, detail=alt.Detail(field="iteration", type="nominal"))
        .properties(height=200, title="all iterations")
        & alt.Chart(sel.to_polars())
        .mark_line(color="#e45756")
        .encode(x=_x, y=_y)
        .properties(height=200, title=f"selected (peak {sel.ymax():.4g} V)")
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Spectrum

    `dft()` is a single-sided amplitude spectrum (numpy's FFT, imported only inside the method); the
    result is itself a `Waveform` over `freq`.
    """)
    return


@app.cell
def _(np, pl, pp):
    _t = np.arange(0.0, 1.0, 1 / 512)
    tone = pp.Waveform.from_series(
        pl.Series("t", _t),
        pl.Series("v", np.sin(2 * np.pi * 13 * _t) + 0.3 * np.sin(2 * np.pi * 40 * _t)),
    )
    tone
    return (tone,)


@app.cell
def _(tone):
    tone.dft().plot(backend="altair").properties(height=200, title="single-sided amplitude spectrum")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Escape hatch

    `to_polars()` hands the frame straight to Polars, so expressions and the `.cx` namespace are one
    call away.
    """)
    return


@app.cell
def _(bode, pl):
    bode.to_polars().select(
        pl.col("freq"),
        pl.col(bode.yname).cx.db20().alias("|n10| [dB]"),
        pl.col(bode.yname).cx.phase().alias("phase [deg]"),
    ).head(5)
    return


if __name__ == "__main__":
    app.run()
