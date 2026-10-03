# polars-psf

Fast reader for Cadence® Spectre® PSF simulation results (psfbin, psfascii, PSFXL) into
[Polars](https://pola.rs), with Rust underneath. Result directories (nested sweeps, Monte Carlo, ADE
parametric runs) become one lazy table; filters on sweep parameters and signal names are pushed down, so only
the data you query is read. A libpsf-compatible numpy API is included for existing scripts.

polars-psf is an independent community project, not affiliated with or endorsed by the Polars project or by
Cadence Design Systems, Inc. Cadence® and Spectre® are registered trademarks of Cadence Design Systems, Inc.

Prebuilt wheels (Linux x86_64/aarch64, macOS, Windows; Python >= 3.11) are attached to
[GitHub Releases](https://github.com/henjo/polars-psf/releases):

```sh
pip install polars-psf --find-links https://github.com/henjo/polars-psf/releases/expanded_assets/v0.1.0
# or: uv pip install polars-psf --find-links ...
```

From source (needs a Rust toolchain): `pip install "git+https://github.com/henjo/polars-psf#subdirectory=python"`.

## Python

```python
import polars as pl
import polars_psf as pp

# one entry point, lazy throughout: opening reads metadata, values are decoded on collect()
f = pp.open("ac.ac")                      # PSFXL stubs pick up <stub>.psfxl automatically
lf = f.scan()                             # LazyFrame: freq | signals (complex as Struct{re, im})
lf.select("freq", pl.col("out").cx.db20(), pl.col("out").cx.phase()).collect()   # decodes only "out"
f.names(), f.file().units("out"), f.file().header, pp.open("dcOp.dc").values()

# many small signals (e.g. noise contributions): long table built in Rust, filters pushed down
lf = pp.open("noise.pnoise").scan_long(field="total")      # freq | signal | value
lf.filter(pl.col("signal").cast(pl.String).str.starts_with("xa1.")).group_by("freq").agg(pl.col("value").sum())

r = pp.open("sim.raw")                    # directory with logFile or runObjFile: same API
r.results                                 # one row per analysis result: name, type, params, leaves
lf = r.scan("tran1")                      # wide: <params...> | time | signals
lf.filter(pl.col("temp") == 27).select("rval", "time", "out").collect()   # skips other files
# long format across corners / Monte Carlo: filters on parameters and signal names both push down
(r.scan_long("tran1").filter((pl.col("temp") == 27) & (pl.col("signal") == "out"))
  .group_by("rval").agg(pl.col("value").abs().max()).collect())
```

A single file is a dataset with one result and one leaf, so `result` can be left out. The native
reader is still reachable for one leaf via `d.file(result, **params)` (header, props, Arrow batches).
Queries run in Polars' Rust engine; Python only builds the plan. More: `python/examples/noise_queries.py`, and an
interactive tour of the features as a [marimo](https://marimo.io) notebook: from `python/`,
`uv run --group examples marimo edit examples/tour.py`.

### Post-processing convenience layer

`polars_psf.post` holds a `Waveform` — a value column over one or more index columns, backed by a
`pl.LazyFrame` (materialized on demand) — and offers the imperative, numpy-style post-processing of
[pycircuit](https://github.com/henjo/pycircuit)'s `post` module, still backed by Polars (`w.x`/`w.y`
are `pl.Series`, `w.to_polars()` is the frame, and every operation builds Polars expressions):

```python
import polars_psf as pp

ac = pp.open("ac.ac")                   # a single file holds one result: ac.v("out")
h = ac.v("out") / ac.v("in")            # complex arithmetic on Struct{re, im}
h.db20(), h.phase()                     # also +, -, *, /, **, abs, real, imag, conj
h.bandwidth(), h.unity_gain_frequency(), h.phase_margin(), h.gain_margin()
h.value(1e6), h.cross(0.0), h.deriv(), h.rms()          # resample/query/reduce
pp.iip3(out, inn, f1, f2), pp.compression_point(gain_db)

r = pp.open("sim.raw")
r.ac1.v("n10").bandwidth()              # result objects: one curve per corner -> temp | rval | bandwidth
w = r.tran1.v("out", temp=27)           # parameters narrow the leaves
r.tran1.waves(["in", "out"], temp=27)   # several signals, decoded together on first access
w.to_polars()                           # index column(s) then the value: rval | time | out
w.rise_time(), w.slew_rate(), w.overshoot(), w.settling_time(), w.frequency()
w.leaf(rval=1e3), w.integ(), w.delay(r.tran1.v("in", temp=27), 0.5)
mc = r.tran2.v("out")                   # Monte Carlo: groups = ["iteration"]
mc.cross(0.5), mc.ymax()                 # one row per iteration (a single curve gives a scalar)
mc - mc.mean(), mc / mc.ymax()           # per-curve results broadcast, joined on the group columns
mc.deriv(), mc.clip(1e-9, 5e-9)          # per iteration, iteration column kept
```

Measurements follow OCEAN semantics: `xmax()` is the x at the largest y (the x range is `w.x.min()`
/ `w.x.max()`), and `cross(threshold, edge=1, type="either")` counts edges from 1 (negative from
the end). The OCEAN names are aliases of the Python names, for Cadence users: `pp.dB20`,
`pp.unityGainFreq`, `pp.phaseMargin`, `pp.riseTime`, `pp.leafValue`, `pp.openResults`, ...

A waveform wraps a `Dataset.scan` plan projected to its signal, so nothing is decoded until a value
is needed, and the convenience layer keeps the pushdown behaviour of the queries. Complex traces are `Struct{re, im}`; `abs`/`phase`/`real`/`imag`/`conj` and mixed real/complex
arithmetic are built on those fields. `w.plot()` returns an Altair chart via Polars' `DataFrame.plot`
(needs `polars[plot]`, i.e. `altair>=5.4`). Walkthrough:
`python/examples/post_processing.py`; interactive demo:
`uvx marimo edit --sandbox python/examples/waveform.py`.

### libpsf-compatible numpy API

```python
import polars_psf.compat as libpsf        # drop-in for scripts written against libpsf
d = libpsf.PSFDataSet("ac.ac")
f, v = d.get_sweep_values(), d.get_signal("out")   # numpy float64 / complex128
```

Walkthrough (transient, AC, noise structs, operating point, PSFXL): `python/examples/libpsf_compat.py`.

## Rust

The Python package is built on three Polars-independent crates that produce Arrow:

| crate | |
|---|---|
| `psfkit` (`crates/psfkit`) | psfbin (non-swept, row records incl. groups, windowed transient, truncated files), psfascii, PSFXL (stub + `.psfxl`, Blosc; feature `psfxl`) |
| `psfkit-arrow` | columns -> Arrow arrays without copying (complex = `Struct{re, im}`, long format, PSF props as field metadata) |
| `psfkit-results` | logFile / runObjFile trees: Spectre nested sweeps, Monte Carlo, ADE parametric runs -> leaves with outer parameters (full precision from `.sweep` files) |

```rust
let f = psfkit::PsfFile::open("ac.ac")?;
let d = f.read(&["out"])?;                     // sweep + requested traces, one pass
let (re, im) = d.traces[0].1.as_complex_f64().unwrap();
```

Examples: `cargo run --example psfdump -- FILE [SIGNAL...]`, `cargo run --release --example bench -- FILE`.
MSRV: `psfkit`, `psfkit-results` 1.85; `psfkit-arrow` and the Python extension 1.88 (arrow).

## Design

Everything is lazy: opening a file parses only declarations; values are decoded on request and only for the
requested signals (non-swept values are indexed on first lookup and decoded one by one); result directories
read parent sweep files and check data files only for the result you ask for; selective reads of transient
files prefetch exactly the needed blocks. Performance and a comparison with libpsf: `docs/benchmarks.md`.
File formats (psfbin, psfascii, PSFXL, result directories): `docs/format.md`.

## Development

- Dev shell: `nix develop` (Rust, Python, uv, maturin; sets `LD_LIBRARY_PATH` for manylinux wheels).
- Rust tests: `cargo test` (golden comparisons of every psfbin file against its psfascii twin, Spectre 25.1
  samples, truncation robustness). Test data and licenses: `testdata/README.md`.
- Python: `cd python && maturin build --release -o dist`, then from the repository root `pytest python/tests`
  with the wheel installed.

## License

MIT.
