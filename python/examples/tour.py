# /// script
# requires-python = ">=3.11"
# dependencies = ["polars-psf", "marimo", "altair", "numpy"]
# ///
"""polars-psf tour (uses the files in testdata/). From python/:

    uv run --group examples marimo edit examples/tour.py     # or: marimo run examples/tour.py
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="polars-psf tour")


@app.cell
def _():
    import math
    import time

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    import polars_psf as pp

    alt.data_transformers.disable_max_rows()
    TD = (mo.notebook_dir() / "../../testdata").resolve()
    ROOT = TD.parent

    def timed(fn):
        """(result, milliseconds) of ``fn()``."""
        t0 = time.perf_counter()
        out = fn()
        return out, (time.perf_counter() - t0) * 1e3

    def ladder_theory(n, r, c, f):
        """Node voltages of an n-stage RC ladder driven by 1 V at f (nodal analysis)."""
        w = 2 * math.pi * f
        y = np.zeros((n, n), complex)
        for k in range(n):
            y[k, k] += 1 / r + 1j * w * c
            if k + 1 < n:
                y[k, k] += 1 / r
                y[k, k + 1] = y[k + 1, k] = -1 / r
        b = np.zeros(n, complex)
        b[0] = 1 / r
        return np.linalg.solve(y, b)

    return ROOT, TD, alt, ladder_theory, math, mo, np, pl, pp, timed


@app.cell
def _(mo):
    mo.md(r"""
    # polars-psf

    **Spectre simulation results as lazy Polars queries.** One call, `pp.open(path)`, opens a
    single PSF file (psfbin, psfascii, PSFXL) or a whole result directory with nested sweeps and
    Monte Carlo runs. Opening reads only declarations. Each query hands its column selection and
    filters down to the Rust reader, so leaf files outside the filter are never opened and signals
    you do not ask for are never decoded.

    This tour checks real circuit physics against the simulator output, and the checks themselves
    are a handful of Polars expressions.
    """)
    return


@app.cell
def _(TD, mo, pp, timed):
    corners, t_open = timed(lambda: pp.open(TD / "spectre25/sweep_psfbin"))
    corner_results, t_list = timed(lambda: corners.results)
    _files = list((TD / "spectre25/sweep_psfbin").iterdir())
    _mb = sum(f.stat().st_size for f in _files) / 1e6
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{t_open + t_list:.1f} ms", label="open + list results", caption="logFile only"),
                    mo.stat(f"{len(_files)}", label="files on disk", caption=f"{_mb:.1f} MB"),
                    mo.stat(f"{corner_results['leaves'][0]}", label="corners", caption="temp × rval"),
                    mo.stat("0", label="values decoded so far", caption="nothing is read eagerly"),
                ],
                justify="space-around",
            ),
            corner_results,
        ]
    )
    return (corners,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Corner explorer

    A 10-stage RC ladder (`samplegen/sweep_tran.scs`) simulated in Spectre 25 inside two nested
    sweeps (`temp` × `rval`), each with a `tran1` and an `ac1` analysis. A result is named after its
    analysis (`tran1`), and its outer sweep parameters become columns. Pick corners and a node:
    the parameter filter skips the leaf files of the other corners, and only the chosen node is
    decoded from the files that remain.
    """)
    return


@app.cell
def _(corners, mo):
    _leaves = corners.leaves("tran1")
    temps = mo.ui.multiselect(
        {f"{t:g} °C": t for t in sorted(_leaves["temp"].unique())}, value=["27 °C"], label="temp"
    )
    rvals = mo.ui.multiselect(
        {f"{r:g} Ω": r for r in sorted(_leaves["rval"].unique())},
        value=["500 Ω", "1000 Ω", "2000 Ω"],
        label="rval",
    )
    node = mo.ui.dropdown([f"n{k}" for k in range(1, 11)], value="n5", label="node")
    mo.hstack([temps, rvals, node], justify="start", gap=2)
    return node, rvals, temps


@app.cell
def _(alt, corners, mo, node, pl, rvals, temps, timed):
    _pick = pl.col("temp").is_in(temps.value) & pl.col("rval").is_in(rvals.value)
    _q = corners.scan("tran1").filter(_pick).select("temp", "rval", "time", "in", node.value)
    tran_sel, _ms = timed(_q.collect)
    _n = corners.leaves("tran1").filter(_pick).height
    _x = alt.X("time", title="time [s]", axis=alt.Axis(format="~s"))
    _src = (
        alt.Chart(tran_sel.unique("time", keep="first").sort("time"))
        .mark_line(color="lightgray", interpolate="step-after")
        .encode(x=_x, y="in")
    )
    _out = (
        alt.Chart(tran_sel)
        .mark_line(strokeWidth=1.8)
        .encode(
            x=_x,
            y=alt.Y(node.value, title=f"V({node.value}), input in gray"),
            color=alt.Color("rval:N", title="rval [Ω]"),
            strokeDash=alt.StrokeDash("temp:N", title="temp [°C]"),
            tooltip=["temp", "rval", "time", node.value],
        )
    )
    _chart = (_src + _out).properties(width="container", height=260).interactive(bind_y=False)
    _signals = len(corners.names("tran1"))
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{_n} / 9", label="leaf files opened"),
                    mo.stat(f"2 / {_signals}", label="signals decoded per file"),
                    mo.stat(f"{tran_sel.height:,}", label="rows"),
                    mo.stat(f"{_ms:.1f} ms", label="query"),
                ],
                justify="space-around",
            ),
            _chart,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Every corner and every node in one query: measured bandwidth vs. Elmore

    For each corner and node, the −3 dB bandwidth comes from a single lazy query on the long
    table `[temp, rval, freq, signal, value]`: magnitude in dB with `.cx.db20()`, the first point
    below DC − 3 dB, then log-frequency interpolation, all inside `group_by().agg()`.

    The **Elmore delay** of node $k$ in an $N$-stage ladder,
    $\tau_k = RC\sum_{i=1}^{k}(N-i+1)$, predicts a dominant pole near $1/(2\pi\tau_k)$. The
    90 points should hug the diagonal. **Click a point** to open its Bode plot below. The three
    temperatures land on top of each other because the resistors have no temperature coefficient.
    """)
    return


@app.cell
def _(corners, math, pl, timed):
    _N, _C = 10, 1e-12
    _f, _db, _i = pl.col("f"), pl.col("db"), pl.col("i")
    _q = (
        corners.scan_long("ac1")
        .filter(pl.col("signal").cast(pl.String).str.contains(r"^n\d+$"))
        .with_columns(db=pl.col("value").cx.db20())
        .sort("temp", "rval", "signal", "freq")
        .group_by("temp", "rval", "signal")
        .agg(
            i=(pl.col("db") < pl.col("db").first() - 3).arg_max(),
            f=pl.col("freq"),
            db=pl.col("db"),
        )
        .with_columns(
            f0=_f.list.get(_i - 1).log10(),
            f1=_f.list.get(_i).log10(),
            d0=_db.list.get(_i - 1),
            d1=_db.list.get(_i),
            dc=_db.list.first(),
            k=pl.col("signal").cast(pl.String).str.slice(1).cast(pl.Int32),
        )
        .select(
            "temp",
            "rval",
            pl.col("signal").cast(pl.String),
            "k",
            f3db=10 ** (pl.col("f0") + (pl.col("dc") - 3 - pl.col("d0")) / (pl.col("d1") - pl.col("d0")) * (pl.col("f1") - pl.col("f0"))),
            f_elmore=1 / (2 * math.pi * pl.col("rval") * _C * (pl.col("k") * (_N + 1) - pl.col("k") * (pl.col("k") + 1) / 2)),
        )
    )
    bw, bw_ms = timed(_q.collect)
    return bw, bw_ms


@app.cell
def _(alt, bw, bw_ms, mo, pl):
    _lo, _hi = bw["f3db"].min() / 1.5, bw["f3db"].max() * 1.5
    _diag = alt.Chart(pl.DataFrame({"x": [_lo, _hi]})).mark_line(color="gray", strokeDash=[4, 4]).encode(x="x", y="x")
    _sel = alt.selection_point(fields=["rval", "signal"], on="click", name="corner")
    _pts = (
        alt.Chart(bw)
        .mark_circle(size=80)
        .encode(
            x=alt.X("f_elmore", title="Elmore estimate 1/(2πτ) [Hz]", scale=alt.Scale(type="log"), axis=alt.Axis(format="~s")),
            y=alt.Y("f3db", title="measured −3 dB [Hz]", scale=alt.Scale(type="log"), axis=alt.Axis(format="~s")),
            color=alt.Color("rval:N", title="rval [Ω]"),
            opacity=alt.condition(_sel, alt.value(0.95), alt.value(0.25)),
            tooltip=["signal", "temp", "rval", alt.Tooltip("f3db", format=".4s"), alt.Tooltip("f_elmore", format=".4s")],
        )
        .add_params(_sel)
    )
    bw_chart = mo.ui.altair_chart((_diag + _pts).properties(width="container", height=320))
    _ratio = bw["f3db"] / bw["f_elmore"]
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{bw.height}", label="corner × node bandwidths"),
                    mo.stat(f"{bw_ms:.0f} ms", label="one lazy query", caption="9 files, complex AC"),
                    mo.stat(f"{_ratio.median():.2f}", label="median measured / Elmore"),
                ],
                justify="space-around",
            ),
            bw_chart,
        ]
    )
    return (bw_chart,)


@app.cell
def _(alt, bw, bw_chart, corners, mo, np, pl):
    _picked = bw_chart.apply_selection(bw)  # layered chart: filter the data by the selection
    _chosen = 0 < _picked.height < bw.height
    _row = _picked.row(0, named=True) if _chosen else {"signal": "n10", "rval": 1000.0}
    bode = corners.ac1.v(_row["signal"], temp=27, rval=_row["rval"])  # lazy Waveform
    _f3 = bode.bandwidth()
    _bode = pl.DataFrame(
        {
            "freq": bode.x,
            "mag_db": bode.db20().y,
            "phase_deg": np.degrees(np.unwrap(np.radians(bode.phase().y.to_numpy()))),
        }
    ).filter(pl.col("mag_db") > -160)  # below that the ladder output is numerical noise
    _x = alt.X("freq", scale=alt.Scale(type="log"), axis=alt.Axis(format="~s"), title="freq [Hz]")
    _rule = alt.Chart(pl.DataFrame({"freq": [_f3]})).mark_rule(color="crimson", strokeDash=[4, 3]).encode(x=_x)
    _chart = alt.vconcat(
        *[
            (alt.Chart(_bode).mark_line().encode(x=_x, y=alt.Y(_col, title=_title)) + _rule).properties(width=950, height=140)
            for _col, _title in [("mag_db", "|H| [dB]"), ("phase_deg", "phase [deg]")]
        ]
    )
    mo.vstack(
        [
            mo.md(
                f"**{bode.yname}** at rval = {_row['rval']:g} Ω, 27 °C: `wave().bandwidth()` = "
                f"**{_f3 / 1e6:.3f} MHz**, |H| at the lowest frequency = {abs(bode).value(bode.x.min()):.4f}"
            ),
            _chart,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. A sine travelling down a 20-stage ladder (PSFXL)

    A 1 MHz sine drives a 20-stage RC ladder for 10 µs. Spectre stored the result as PSFXL, a
    stub plus a `.psfxl` data file where each signal has its own time axis. The long table puts
    all signals on one axis column, and the heatmap is a `group_by` over time buckets.
    """)
    return


@app.cell
def _(TD, mo, pp):
    ladder = pp.open(TD / "spectre25/psfxl_ac/tran1.tran.tran")
    window = mo.ui.range_slider(start=0, stop=10, step=0.1, value=[0, 3], label="time window [µs]", show_value=True)
    window
    return ladder, window


@app.cell
def _(alt, ladder, mo, pl, timed, window):
    _t0, _t1 = window.value[0] * 1e-6, window.value[1] * 1e-6
    _dt = max((_t1 - _t0) / 300, 1e-9)
    _q = (
        ladder.scan_long()
        .filter(pl.col("signal").cast(pl.String).str.contains(r"^n\d+$") & pl.col("time").is_between(_t0, _t1))
        .with_columns(
            k=pl.col("signal").cast(pl.String).str.slice(1).cast(pl.Int32),
            t=(pl.col("time") / _dt).floor() * _dt,
        )
        .group_by("k", "t")
        .agg(pl.col("value").mean())
        .with_columns(t2=pl.col("t") + _dt)
    )
    heat, _ms = timed(_q.collect)
    _chart = (
        alt.Chart(heat)
        .mark_rect()
        .encode(
            x=alt.X("t:Q", title="time [s]", axis=alt.Axis(format="~s")),
            x2="t2:Q",
            y=alt.Y("k:O", title="node", sort="descending"),
            color=alt.Color("value", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True), title="V"),
            tooltip=["k", alt.Tooltip("t", format=".3s"), alt.Tooltip("value", format=".3f")],
        )
        .properties(width="container", height=300)
    )
    mo.vstack([mo.md(f"{heat.height:,} cells from 20 PSFXL signals in {_ms:.0f} ms"), _chart])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Measured phasors vs. nodal analysis

    Over the last 5 µs (steady state), each node's 1 MHz phasor is a Fourier sum written as Polars
    expressions: $\frac{2}{T}\sum v\,e^{-j\omega t}\,\Delta t$, for all 20 nodes in one `group_by`.
    The theory solves the ladder's 20 × 20 admittance matrix with numpy.
    """)
    return


@app.cell
def _(ladder, ladder_theory, math, np, pl):
    _f0, _T = 1e6, 5e-6
    _w = 2 * math.pi * _f0
    _ph = (
        ladder.scan_long()
        .filter(pl.col("time") >= 10e-6 - _T)
        .sort("signal", "time")
        .with_columns(dt=pl.col("time").diff().over("signal").fill_null(0.0))
        .group_by("signal")
        .agg(
            re=(pl.col("value") * (_w * pl.col("time")).cos() * pl.col("dt")).sum() * 2 / _T,
            im=(-pl.col("value") * (_w * pl.col("time")).sin() * pl.col("dt")).sum() * 2 / _T,
        )
        .collect()
    )
    _src = _ph.filter(pl.col("signal") == "in").row(0, named=True)
    _ref = complex(_src["re"], _src["im"])
    _meas = (
        _ph.filter(pl.col("signal").cast(pl.String).str.contains(r"^n\d+$"))
        .with_columns(k=pl.col("signal").cast(pl.String).str.slice(1).cast(pl.Int32))
        .sort("k")
    )
    _h = (_meas["re"].to_numpy() + 1j * _meas["im"].to_numpy()) / _ref

    _v = ladder_theory(20, 1e3, 1e-12, _f0)  # R = 1 kΩ, C = 1 pF, source through R1
    phasors = pl.DataFrame(
        {
            "node": np.arange(1, len(_v) + 1),
            "|H| measured": np.abs(_h),
            "|H| theory": np.abs(_v),
            "phase measured": np.degrees(np.angle(_h)),
            "phase theory": np.degrees(np.angle(_v)),
        }
    )
    return (phasors,)


@app.cell
def _(alt, mo, phasors, pl):
    def _panel(what, title):
        _d = phasors.select(
            "node", measured=pl.col(f"{what} measured"), theory=pl.col(f"{what} theory")
        ).unpivot(index="node", variable_name="source")
        _base = alt.Chart(_d).encode(x=alt.X("node:Q", title="node"), y=alt.Y("value", title=title, scale=alt.Scale(zero=False)))
        return (
            _base.transform_filter("datum.source == 'theory'").mark_line(color="gray")
            + _base.transform_filter("datum.source == 'measured'").mark_point(filled=True, size=60, color="crimson")
        ).properties(width=300, height=220)

    _err_mag = (phasors["|H| measured"] / phasors["|H| theory"] - 1).abs().max()
    _err_ph = (phasors["phase measured"] - phasors["phase theory"]).abs().max()
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{_err_mag * 100:.3f} %", label="max |H| error", caption="measured vs. theory"),
                    mo.stat(f"{_err_ph:.3f}°", label="max phase error"),
                    mo.stat(f"{phasors['phase theory'][-1]:.1f}°", label="lag at n20"),
                ],
                justify="space-around",
            ),
            mo.hstack([_panel("|H|", "|V(n)/V(in)|"), _panel("phase", "phase [deg]")], justify="center"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Monte Carlo: a lazy waveform, and an iteration that blew up

    Monte Carlo iterations are another parameter column. A `Waveform` can have several index
    columns. Here it holds `iteration × time`, and per-sample operations such as `deriv()` keep
    iterations apart. The waveform is built from an uncollected scan, so nothing is decoded until
    the statistics need it. One query per statistic is enough to flag the run that diverged.
    """)
    return


@app.cell
def _(TD, alt, mo, pl, pp):
    mc = pp.open(TD / "psf-parser/binary").mymonte_tran1.v("out")  # one curve per iteration
    _d = mc.deriv()
    _slew = _d.to_polars().group_by("iteration").agg(pl.col(_d.yname).abs().max().alias("max |dV/dt| [V/s]"))
    _stats = (
        mc.to_polars()
        .group_by("iteration")
        .agg(pl.col("out").abs().max().alias("max |out| [V]"), pl.len().alias("time steps"))
        .join(_slew, on="iteration")
        .with_columns(
            pl.selectors.float().round_sig_figs(4),
            status=pl.when(pl.col("max |out| [V]") > 10).then(pl.lit("⚠ diverged")).otherwise(pl.lit("ok")),
        )
        .sort("iteration")
    )
    _chart = (
        alt.Chart(mc.to_polars())
        .mark_line()
        .encode(x=alt.X("time", axis=alt.Axis(format="~s", tickCount=4)), y=alt.Y("out", axis=alt.Axis(format="~s")), color="iteration:N")
        .properties(width=250, height=160)
        .facet(column=alt.Column("iteration:N"))
        .resolve_scale(y="independent")
    )
    mo.vstack([mo.md(f"`{mc!r}`"), _stats, _chart])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Noise contributions with pushed-down name filters

    A pnoise result holds one struct per device (`rn`, `fn`, `total`, ...).
    `scan_long(field="total")` turns all of them into rows. A regex on `signal` is evaluated on the
    list of names before reading, so only the matching devices are decoded.
    """)
    return


@app.cell
def _(mo):
    pattern = mo.ui.text(value=r"\.r\w*$", label="device regex (try . for all)", full_width=False)
    top_n = mo.ui.slider(3, 12, value=6, label="top N", show_value=True)
    mo.hstack([pattern, top_n], justify="start", gap=2)
    return pattern, top_n


@app.cell
def _(TD, alt, mo, pattern, pl, pp, timed, top_n):
    _noise = pp.open(TD / "pycircuit/psf/pnoise0.pnoise")
    _all = _noise.names(field="total")
    _q = _noise.scan_long(field="total").filter(pl.col("signal").cast(pl.String).str.contains(pattern.value))
    _rows, _ms = timed(_q.collect)
    _rank = _rows.group_by("signal").agg(pl.col("value").mean()).sort("value", descending=True)
    _top = _rank["signal"].head(top_n.value).cast(pl.String).to_list()
    _stack = (
        _rows.with_columns(device=pl.when(pl.col("signal").cast(pl.String).is_in(_top)).then(pl.col("signal").cast(pl.String)).otherwise(pl.lit("other")))
        .group_by("freq", "device")
        .agg(pl.col("value").sum())
    )
    _share = _rows.group_by("freq").agg(pl.col("value").sum()).sort("freq")
    _chart = (
        alt.Chart(_stack)
        .mark_area()
        .encode(
            x=alt.X("freq", scale=alt.Scale(type="log"), axis=alt.Axis(format="~s"), title="offset freq [Hz]"),
            y=alt.Y("value", stack="normalize", title="share among matching devices", axis=alt.Axis(format="%")),
            color=alt.Color("device:N", sort=[*_top, "other"]),
            tooltip=["device", alt.Tooltip("freq", format=".3s"), alt.Tooltip("value", format=".3e")],
        )
        .properties(width="container", height=280)
    )
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{_rank.height} / {len(_all)}", label="devices decoded"),
                    mo.stat(f"{_ms:.1f} ms", label="query"),
                    mo.stat(f"{_share['value'][0]:.3g}", label="their V²/Hz at lowest freq"),
                ],
                justify="space-around",
            ),
            _chart,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Operating point: gm/Id of 705 MOSFETs

    A non-swept file scans to a single row with one struct per device. Unnesting the MOS structs
    gives a device table, and the gm/Id design chart follows. It sits on the strong-inversion
    square law $2/V_{ov}$ and flattens toward the weak-inversion limit $1/(nV_T)$.
    **Drag a box** over the scatter to list devices.
    """)
    return


@app.cell
def _(TD, pl, pp):
    _op = pp.open(TD / "pycircuit/psf/dcOpInfo.info").scan().collect()
    _mos = [c for c, t in _op.schema.items() if isinstance(t, pl.Struct) and "vt1" in [f.name for f in t.fields]]
    mos = (
        pl.concat([_op.select(pl.lit(c).alias("device"), pl.col(c).struct.unnest()) for c in _mos])
        .filter(pl.col("ids").abs() > 1e-9, pl.col("gm") > 0)
        .select(
            "device",
            kind=pl.when(pl.col("vgs") < 0).then(pl.lit("pmos")).otherwise(pl.lit("nmos")),
            ids_uA=pl.col("ids").abs() * 1e6,
            vov=pl.col("vgs").abs() - pl.col("vt1").abs(),
            gm_id=pl.col("gm") / pl.col("ids").abs(),
            fug_MHz=pl.col("fug") / 1e6,
        )
        .with_columns(pl.selectors.float().round_sig_figs(4))
    )
    return (mos,)


@app.cell
def _(alt, mo, mos, np, pl):
    _vt = 0.02585
    _v = np.linspace(-0.3, 1.2, 200)
    _theory = pl.concat(
        [
            pl.DataFrame({"vov": _v[_v > 0.08], "gm_id": 2 / _v[_v > 0.08], "model": "square law 2/Vov"}),
            pl.DataFrame({"vov": _v, "gm_id": np.full_like(_v, 1 / (1.3 * _vt)), "model": "weak inversion 1/(n·VT), n = 1.3"}),
        ]
    )
    _brush = alt.selection_interval(name="box")
    _pts = (
        alt.Chart(mos)
        .mark_circle(size=40, opacity=0.7)
        .encode(
            x=alt.X("vov", title="Vov = |Vgs| − |Vth| [V]"),
            y=alt.Y("gm_id", title="gm/Id [1/V]", scale=alt.Scale(type="log")),
            color=alt.Color("kind:N"),
            tooltip=["device", "kind", alt.Tooltip("ids_uA", format=".3g"), alt.Tooltip("gm_id", format=".3g")],
        )
        .add_params(_brush)
    )
    _lines = alt.Chart(_theory).mark_line(strokeDash=[5, 4], color="gray").encode(x="vov", y="gm_id", detail="model")
    gm_chart = mo.ui.altair_chart((_lines + _pts).properties(width="container", height=320))
    gm_chart
    return (gm_chart,)


@app.cell
def _(gm_chart, mo, mos):
    _picked = gm_chart.apply_selection(mos)  # layered chart: filter the data by the selection
    _chosen = 0 < _picked.height < mos.height
    _sel = _picked if _chosen else mos.sort("gm_id", descending=True).head(5)
    _note = "" if _chosen else " (top gm/Id; drag a box to choose)"
    mo.vstack([mo.md(f"{_sel.height} devices{_note}"), _sel.sort("gm_id", descending=True)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Scale: a 2.4 GB transient and eight decades of attenuation

    `samplegen/run.sh` writes a 1 M-point Spectre 25 psfbin file (not in git): a 1 MHz sine
    driving a 300-stage RC ladder. Opening it reads only the header. The amplitude profile below
    decodes 16 of the 302 signals and reduces 16 M samples to 16 numbers in Polars. The zoom uses
    `head()`, so the reader stops after the first chunk of the file.
    """)
    return


@app.cell
def _(ROOT, mo, pp, timed):
    _big_path = ROOT / "samplegen/out/psfbin_chunked/tran1.tran.tran"
    mo.stop(
        not _big_path.exists(),
        mo.callout(mo.md("Run `samplegen/run.sh` (needs Spectre) to generate the 2.4 GB file."), kind="info"),
    )
    big, big_open_ms = timed(lambda: pp.open(_big_path))
    big_gb = _big_path.stat().st_size / 1e9
    return big, big_gb, big_open_ms


@app.cell
def _(alt, big, big_gb, big_open_ms, ladder_theory, mo, np, pl, timed):
    _ks = [1, 2, 5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300]
    _nodes = [f"n{k}" for k in _ks]
    _q = big.scan().filter(pl.col("time") > 0.9e-3).select(pl.col(_nodes).abs().max())  # steady state
    _peak, _ms = timed(_q.collect)
    _theory = np.abs(ladder_theory(300, 1e3, 1e-12, 1e6))
    _prof = pl.DataFrame({"node": _ks, "measured": _peak.row(0)})
    _line = alt.Chart(pl.DataFrame({"node": np.arange(1, 301), "theory": _theory})).mark_line(color="gray").encode(
        x=alt.X("node", title="node"), y=alt.Y("theory", scale=alt.Scale(type="log"), title="|V| [V], steady state", axis=alt.Axis(format=".0e"))
    )
    _pts = alt.Chart(_prof).mark_point(filled=True, size=70, color="crimson").encode(
        x="node", y="measured", tooltip=["node", alt.Tooltip("measured", format=".3e")]
    )
    _err = float(np.max(np.abs(_prof["measured"].to_numpy() / _theory[np.array(_ks) - 1] - 1)))
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{big_gb:.1f} GB", label="file", caption="1 M points × 302 signals"),
                    mo.stat(f"{big_open_ms:.1f} ms", label="open"),
                    mo.stat(f"{_ms:.0f} ms", label="16 M samples → profile"),
                    mo.stat(f"{_err * 100:.2f} %", label="max deviation from theory", caption="over 8 decades"),
                ],
                justify="space-around",
            ),
            (_line + _pts).properties(width="container", height=260),
        ]
    )
    return


@app.cell
def _(big, mo):
    big_signal = mo.ui.dropdown(big.names()[1:], value="n20", label="zoom on signal", searchable=True)
    big_signal
    return (big_signal,)


@app.cell
def _(alt, big, big_signal, mo, timed):
    _zoom, _ms = timed(big.scan().select("time", big_signal.value).head(5000).collect)
    _chart = (
        alt.Chart(_zoom)
        .mark_line()
        .encode(x=alt.X("time", title="time [s]", axis=alt.Axis(format="~s")), y=alt.Y(big_signal.value, title="V"))
        .properties(width="container", height=180)
    )
    mo.vstack([mo.md(f"first {_zoom.height:,} points of `{big_signal.value}` in **{_ms:.0f} ms**: `head()` stops the reader early"), _chart])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. libpsf-compatible numpy API

    Scripts written against libpsf run unchanged with `import polars_psf.compat as libpsf`, and they
    read more formats (PSFXL, Spectre 25 psfbin, psfascii).
    """)
    return


@app.cell
def _(TD):
    import polars_psf.compat as libpsf

    _ds = libpsf.PSFDataSet(str(TD / "spectre25/psfxl_ac/tran1.tran.tran"))
    _t, _v = _ds.get_sweep_values(), _ds.get_signal("n10")
    type(_v), _v.dtype, _t[:3], _v[:3]
    return


if __name__ == "__main__":
    app.run()
