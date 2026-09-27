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

f = pp.open("ac.ac")                      # PSFXL stubs pick up <stub>.psfxl automatically
df = f.to_polars(["out"])                 # freq + out (complex as Struct{re, im})
df.select("freq", pl.col("out").cx.db20(), pl.col("out").cx.phase())
f.units("out"), f.header, pp.open("dcOp.dc").values()

# many small signals (e.g. noise contributions): long table built in Rust, filters pushed down
lf = pp.scan_long("noise.pnoise", field="total")      # freq | signal | value
lf.filter(pl.col("signal").cast(pl.String).str.starts_with("xa1.")).group_by("freq").agg(pl.col("value").sum())

r = pp.results("sim.raw")                 # directory with logFile or runObjFile
r.analyses                                # name, type, params, leaves
lf = r.scan("tran1")                      # wide: <params...> | time | signals
lf.filter(pl.col("temp") == 27).select("rval", "time", "out").collect()   # skips other files
# long format across corners / Monte Carlo: filters on parameters and signal names both push down
(r.scan_long("tran1").filter((pl.col("temp") == 27) & (pl.col("signal") == "out"))
  .group_by("rval").agg(pl.col("value").abs().max()).collect())
```

Queries run in Polars' Rust engine; Python only builds the plan. More: `python/examples/noise_queries.py`, and an
interactive tour of the features as a [marimo](https://marimo.io) notebook: `pip install marimo altair`, then
`marimo edit python/examples/tour.py`.

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
read parent sweep files and check data files only for the analysis you ask for; selective reads of transient
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
