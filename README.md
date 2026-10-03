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

### Quick start

```python
import polars_psf as pp

r   = pp.open("sim.raw")             # a result directory or a single PSF file
ac  = r.ac1                          # the ac1 analysis
out = ac.v("out")                    # all corners at once, nothing read yet

out.bandwidth()                      # -3 dB bandwidth, one row per corner
out.db20().value(1e6)                # gain at 1 MHz in dB
out.leaf(temp=27, rval=1e3)          # one corner
ac.v("out", temp=27)                 # or narrow the corners up front
```

A swept result is a *family*: one curve per corner (or Monte Carlo iteration). Every measurement
works per curve, and the answer is a table with one row per corner:

```
>>> r.ac1.v("n10").bandwidth()
┌───────┬────────┬───────────┐
│ temp  ┆ rval   ┆ bandwidth │
╞═══════╪════════╪═══════════╡
│ -40.0 ┆ 500.0  ┆ 6.9899e6  │
│ -40.0 ┆ 1000.0 ┆ 3.4951e6  │
│ …     ┆ …      ┆ …         │
│ 125.0 ┆ 2000.0 ┆ 1.7476e6  │
└───────┴────────┴───────────┘
```

A single curve gives a plain number: `r.ac1.v("n10", temp=27, rval=1e3).bandwidth()` → `3495082.18`.

### Measurements

| OCEAN | polars-psf | |
|---|---|---|
| `openResults("sim.raw")` | `r = pp.open("sim.raw")` | lazy: reads the logFile only |
| `selectResult('tran1)` | `t = r.tran1` | `r.result("tran1")` for any name |
| `v("out")`, `i("V1:p")` | `t.v("out")`, `t.i("V1:p")` | |
| `dB20(w)`, `phase(w)`, `mag(w)` | `w.db20()`, `w.phase()`, `w.mag()` | also `+ - * / **` on complex data |
| `value(w 1e-9)` | `w.value(1e-9)` | |
| `ymax(w)`, `xmax(w)` | `w.ymax()`, `w.xmax()` | `xmax` = x at the largest y |
| `cross(w 0.5 1 "rising")` | `w.cross(0.5, 1, "rising")` | edges count from 1, `-1` = last |
| `bandwidth(w 3 "low")` | `w.bandwidth(3, "low")` | `"high"`, `"band"` too |
| `unityGainFreq(w)`, `phaseMargin(w)`, `gainMargin(w)` | `w.unity_gain_frequency()`, `w.phase_margin()`, `w.gain_margin()` | |
| `riseTime(w ...)`, `slewRate(w ...)` | `w.rise_time()`, `w.slew_rate()` | 10–90 % by default |
| `overshoot(w ...)`, `settlingTime(w ...)` | `w.overshoot()`, `w.settling_time(tolerance=1)` | |
| `delay(?wf1 a ?wf2 b ...)` | `a.delay(b, 0.5)` | |
| `frequency(w)`, `average(w)`, `rms(w)` | `w.frequency()`, `w.average()`, `w.rms()` | |
| `integ(w)`, `deriv(w)`, `clip(w 1n 5n)` | `w.integ()`, `w.deriv()`, `w.clip(1e-9, 5e-9)` | |
| `leafValue(w "temp" 27)` | `w.leaf(temp=27)` | |
| `ocnPrint(...)` | `print(df)`, `df.write_csv(...)` | results are Polars DataFrames |

The OCEAN names also work as functions, so `pp.bandwidth(w, 3, "low")`, `pp.dB20(w)` and
`pp.riseTime(w)` work too.

### Results are tables

A measurement over corners comes back as a table (a [Polars](https://pola.rs) DataFrame), ready
to sort, filter and save:

```python
import polars as pl                   # the table library underneath

bw = r.ac1.v("n10").bandwidth()       # temp | rval | bandwidth, one row per corner

bw.sort("bandwidth")                  # slowest corners first
bw.filter(pl.col("bandwidth") < 2e6)  # only the corners below 2 MHz
bw.write_csv("bandwidth.csv")         # or write_excel(...), for the spreadsheet
```

The filter gives:

```
┌───────┬────────┬───────────┐
│ temp  ┆ rval   ┆ bandwidth │
╞═══════╪════════╪═══════════╡
│ -40.0 ┆ 2000.0 ┆ 1.7476e6  │
│ 27.0  ┆ 2000.0 ┆ 1.7476e6  │
│ 125.0 ┆ 2000.0 ┆ 1.7476e6  │
└───────┴────────┴───────────┘
```

Two measurements of the same corners line up with `join`, here an AC bandwidth and a transient
delay:

```python
delay = r.tran1.v("in").delay(r.tran1.v("n5"), 0.3)     # temp | rval | delay
table = bw.join(delay, on=["temp", "rval"])            # temp | rval | bandwidth | delay

table.filter(pl.col("temp") == 27)
```
```
┌──────┬────────┬───────────┬───────────┐
│ temp ┆ rval   ┆ bandwidth ┆ delay     │
╞══════╪════════╪═══════════╪═══════════╡
│ 27.0 ┆ 500.0  ┆ 6.9899e6  ┆ 5.8106e-9 │
│ 27.0 ┆ 1000.0 ┆ 3.4951e6  ┆ 2.6562e-8 │
│ 27.0 ┆ 2000.0 ┆ 1.7476e6  ┆ 5.1510e-8 │
└──────┴────────┴───────────┴───────────┘
```

And a worst case over temperature, for each rval:

```python
table.group_by("rval").agg(pl.col("delay").max())
```

**When you want more.** The same few verbs scale up. Bandwidth of every node at every rval as one
pivot table:

```python
nodes = [f"n{k}" for k in range(1, 11)]
bw = pl.concat(r.ac1.v(n, temp=27).bandwidth().with_columns(node=pl.lit(n)) for n in nodes)
bw.pivot("rval", index="node", values="bandwidth")
```

The top noise contributors of a pnoise analysis, out of thousands of devices:

```python
noise = pp.open("noise.pnoise")
(noise.scan_long(field="total")                     # freq | signal | value, per device and freq
      .group_by("signal").agg(pl.col("value").mean())
      .sort("value", descending=True)
      .head(5)
      .collect())
```

**Monte Carlo** works the same way: `mc = r.tran2.v("out")` holds one curve per iteration, so
`mc.cross(0.5)` is a table, `mc.ymax().describe()` gives mean/std/min/max over the runs, and
`mc - mc.mean()` subtracts each curve's own mean.

More: `python/examples/corner_report.py` (a corner spec report as a script), `python/examples/post_processing.py`, and the interactive tour, a [marimo](https://marimo.io)
notebook: from `python/`, `uv run --group examples marimo edit examples/tour.py`.

### Lazy queries

Under the waveforms are lazy Polars queries you can use directly. Opening reads only metadata;
filters on sweep parameters skip whole files, and only the selected signals are decoded:

```python
import polars as pl
import polars_psf as pp

r = pp.open("sim.raw")
lf = r.tran1.scan()                       # LazyFrame: temp | rval | time | signals...
lf.filter(pl.col("temp") == 27).select("rval", "time", "out").collect()   # reads 3 of 9 files, 1 signal

# long format [params..., sweep, signal, value]: filters on parameters and signal names push down
(r.tran1.scan_long().filter((pl.col("temp") == 27) & (pl.col("signal") == "out"))
   .group_by("rval").agg(pl.col("value").abs().max()).collect())

f = pp.open("ac.ac")                      # a single file; PSFXL stubs pick up <stub>.psfxl automatically
f.scan().select("freq", pl.col("out").cx.db20(), pl.col("out").cx.phase()).collect()  # complex = Struct{re, im}
f.names(), f.file().units("out"), f.file().header, pp.open("dcOp.dc").values()
```

Queries run in Polars' Rust engine; Python only builds the plan. A `Waveform` wraps such a query
(`w.lazy`, `w.to_polars()`), and `w.plot()` returns an Altair chart via Polars' `DataFrame.plot`
(needs `polars[plot]`). The post-processing follows
[pycircuit](https://github.com/henjo/pycircuit)'s `post` module (its names are aliases too:
`unityGainFrequency`, `IIP3`, ...); interactive demo: `uvx marimo edit --sandbox
python/examples/waveform.py`.

### polars-waveform

`Waveform`, the measurements and `cx` live in their own pure-Python package,
[polars-waveform](https://git.johome.net/proj/polars-waveform) (`import polars_waveform as pw`), which polars-psf depends on
and re-exports. It works on any Polars data, for example lab measurements in Parquet, including
nested channels with their own sweep (`pw.from_nested`).

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
  with the wheel installed. polars-waveform is its own repository; to work on both, install a checkout
  of it in editable mode (`uv pip install -e ../polars-waveform`).

## License

MIT.
