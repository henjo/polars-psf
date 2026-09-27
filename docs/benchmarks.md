# Benchmark: polars-psf vs libpsf

2026-09-27. polars-psf at the commit that adds this file. libpsf 0.1.4 from PyPI (bjmuld/libpsf-python, the
maintained Python 3 build of libpsf) and, for reference, the original C++ libpsf (`../libpsf`, git `cfe5c6f`).

## Summary

- **Compatibility is the main difference.** libpsf 0.1.4 cannot open any transient file written by Spectre 25.1
  (psfbin or PSFXL), nor psfbinf (float32), psfascii, or the old `dcOpInfo` info file. polars-psf reads all of them.
- **Where both work, polars-psf is 1.5-3x faster on small and medium files and 3-30x faster on large ones**:
  libpsf re-scans the value section for every signal and is single-threaded.
- **libpsf is faster on tiny files with many signals** (0.6 MB DC file with 1432 signals: 7.8 vs 10.5 ms;
  0.03 MB pnoise: 2.4 vs 3.4 ms). Building a Polars DataFrame costs ~3 us per column; libpsf returns bare numpy
  arrays.
- Wherever both read a file, the values are identical (max relative difference 0).

## Method

- Machine: 12-core x86_64 Linux, 15 GB RAM. Python 3.11, polars 1.x, numpy 1.26 (the libpsf wheels need
  numpy < 2).
- Every measurement is a fresh Python process (`bench/bench_one.py`), imports not timed, file in the page cache
  (warm), best of 3. Timed: open + list signal names + read the first N signals.
  - libpsf: `PSFDataSet(f)`, `get_signal_names()`, `get_sweep_values()`, then `get_signal(name)` for N names
    (one numpy array per signal).
  - polars-psf: `pp.open(f)`, `.names`, `.to_polars(names[:N])` (one Polars DataFrame: sweep + N columns).
    Non-swept files: `value(name)` for N names in both.
  - polars-psf with `RAYON_NUM_THREADS=1` and with all 12 threads; libpsf is single-threaded.
- Values compared with `bench/check_equal.py` (first 50 signals) wherever libpsf succeeds.
- Files: Spectre 25.1 output from `samplegen/` (not in git), `testdata/`, and two synthetic files from
  `bench/gen_big.py` (200 signals each: 643 MB row-record sweep, 804 MB windowed transient).

## Results (Python, warm cache)

| case | size | N | libpsf 0.1.4 | polars-psf, 1 thread | polars-psf, 12 threads | max rel diff |
|---|---|---|---|---|---|---|
| Spectre 25.1 tran psfbin, 302 sig | 2,430 MB | 1 | fails | 32.4 ms | 32.3 ms |  |
| Spectre 25.1 tran psfbin, 302 sig | 2,430 MB | 10 | fails | 112.8 ms | 37.9 ms |  |
| Spectre 25.1 tran psfbin, 302 sig | 2,430 MB | 302 | fails | 2,536.6 ms | 414.9 ms |  |
| Spectre 25.1 tran PSFXL, 102 sig | 412 MB | 1 | fails | 20.3 ms | 20.5 ms |  |
| Spectre 25.1 tran PSFXL, 102 sig | 412 MB | 10 | fails | 87.9 ms | 38.0 ms |  |
| Spectre 25.1 tran PSFXL, 102 sig | 412 MB | 102 | fails | 744.1 ms | 241.2 ms |  |
| Spectre 25.1 tran psfbin, 11 sig (swept leaf) | 0.3 MB | 11 | fails | 2.6 ms | 2.6 ms |  |
| Spectre 25.1 ac psfbin complex, 22 sig | 5.4 MB | 22 | 15.6 ms | 7.7 ms (2.0x) | 5.2 ms (3.0x) | 0 |
| Spectre 25.1 ac psfbinf (float32), 22 sig | 3.6 MB | 22 | fails | 7.1 ms | 4.7 ms |  |
| Spectre 25.1 ac psfascii, 22 sig | 12 MB | 22 | fails | 139.4 ms | 139.2 ms |  |
| Spectre 4.4 tran psfbin, 32 sig | 0.8 MB | 32 | 4.8 ms | 3.2 ms (1.5x) | 3.2 ms (1.5x) | 0 |
| Eldo tran psfbin (1 window), 144 sig | 0.6 MB | 144 | 3.5 ms | 3.7 ms (0.9x) | 3.4 ms (1.0x) | 0 |
| Spectre dc, 1432 sig | 0.6 MB | 1432 | 7.8 ms | 10.5 ms (0.7x) | 10.5 ms (0.7x) | 0 |
| Spectre pnoise (struct traces), 26 sig | 0.03 MB | 26 | 2.4 ms | 3.5 ms (0.7x) | 3.4 ms (0.7x) | 0 |
| Spectre op info, 3800 values | 0.7 MB | 1 | fails | 3.0 ms | 2.9 ms |  |
| Spectre op info, 3800 values | 0.7 MB | 3800 | fails | 17.8 ms | 17.8 ms |  |
| synthetic row-record sweep, 200 sig | 643 MB | 1 | fails | 27.6 ms | 27.7 ms |  |
| synthetic row-record sweep, 200 sig | 643 MB | 10 | fails | 62.9 ms | 31.7 ms |  |
| synthetic row-record sweep, 200 sig | 643 MB | 200 | fails | 813.7 ms | 180.5 ms |  |
| synthetic tran windowed, 200 sig | 804 MB | 1 | 42.1 ms | 13.1 ms (3.2x) | 12.6 ms (3.3x) | 0 |
| synthetic tran windowed, 200 sig | 804 MB | 10 | 254.8 ms | 46.5 ms (5.5x) | 15.7 ms (16.2x) | 0 |
| synthetic tran windowed, 200 sig | 804 MB | 200 | 4,040.2 ms | 736.2 ms (5.5x) | 132.1 ms (30.6x) | 0 |

"fails": libpsf raises `RuntimeError: std::exception` (psfbinf: `Unknown type 9`). Speed-ups are libpsf time /
polars-psf time. Raw data: `bench/results-2026-09-27.json`.

## Why libpsf fails

| input | cause |
|---|---|
| Spectre 25.1 transient psfbin (any size) | Spectre 25.1 writes transient psfbin **without the end table**: the file ends in window-buffer padding, with no `Clarissa` trailer. libpsf finds sections only through that table, sees no trace section and fails on open. polars-psf parses sections sequentially through their links and never needs the table. |
| PSFXL (Spectre's default for transient) | not supported (data is in `.psfxl`, Blosc-compressed) |
| psfbinf | float32 type codes 9/10 unknown |
| psfascii | not supported |
| `dcOpInfo.info` (Spectre info file, structs) | `get_signal` raises (the original C++ libpsf reads this file) |
| synthetic row-record file | open fails (the original C++ libpsf reads it; cause not investigated, synthetic only) |

The original C++ libpsf (`../libpsf`, `cfe5c6f`, rebuilt with g++ 15) is worse on transients: it throws on any
transient longer than one window, because it reads into the unused tail of Spectre's window buffer, and it
segfaults on files over 2 GB. The PyPI build reads multi-window transients that have an end table (Spectre 4.4
`tran.tran`, the synthetic 804 MB file), so it includes fixes the local checkout lacks.

## Why polars-psf is faster

- **One pass for many signals.** libpsf walks the value section again for every `get_signal`, so its cost is
  signals x file size (804 MB transient, 200 signals: 4.0 s vs 0.13 s).
- **Layout planned once**, then columns are gathered directly instead of decoding value by value through
  virtual calls.
- **Parallel decode** across signals (12 threads: up to 6x over 1 thread on large reads).
- **Lazy**: opening parses only declarations; non-swept values are decoded on request.

## Where polars-psf is slower

Building a Polars DataFrame through Arrow costs ~3 us per column, which dominates only for tiny files with many
signals (1432 signals x 21 points: +3 ms). Already applied: no field metadata on the Polars path (Polars drops
it anyway), parallel decode only above ~1 MB of work (thread-pool start-up), O(1) property lookup. A plain
numpy API could remove the rest; not added.

Row-record files (the dc/ac/noise layout) interleave all signals on every page, so even one signal maps the
whole file: the 643 MB synthetic file needs ~25 ms for the sweep column alone (about 5,800 page faults).
Transient files avoid this (one signal of the 2.4 GB file: 32 ms, 3.6% of the file read when cold).

## Direct C++ comparison (original libpsf, `cfe5c6f`, g++ -O3)

Rust example `crates/psfkit/examples/cmp_bench.rs` against a C++ program linking libpsf (same steps, same machine,
warm, best of 3). The row-record row used an earlier variant of the synthetic file with grouped traces, which
the C++ libpsf reads.

| file | N | libpsf C++ | psfkit, 1 thread | psfkit, 12 threads |
|---|---|---|---|---|
| synthetic row-record sweep 643 MB, 200 sig | 1 / 10 / 200 | 26 / 292 / 6,829 ms | 7 / 28 / 465 ms | 7 / 12 / 173 ms |
| ac 5.4 MB complex, 22 sig | 22 | 11.9 ms | 5.2 ms | 2.8 ms |
| dc 0.6 MB, 1432 sig | 1432 | 5.2 ms | 2.3 ms | 2.6 ms |
| op info, 3800 values | 1 / 3800 | 15.3 / 23.7 ms | 3.0 / 7.9 ms | 2.7 / 8.1 ms |
| any multi-window transient | | fails | see above | see above |

In Rust, psfkit is faster than the C++ library on every file both can read; the Python-level gap on tiny files
comes from the DataFrame construction described above.

## Queries: noise summary with 20,000 devices

Polars queries execute in Polars' Rust engine; Python only builds the plan. What matters is the table shape.
A pnoise-style file with 20,000 per-device noise structs (fields differ per device type) x 101 frequencies,
81 MB (`bench/gen_noise.py`), summing the `total` contribution per frequency (`bench/noise.py`, warm, best of 3,
open + read + query):

| approach | time |
|---|---|
| A. Python loop, one read per device (libpsf-style `noisesummary.py`) | ~70 s (3.5 s per 1000 devices) |
| B. wide table (20,000 struct columns) + `pl.sum_horizontal` | 874 ms |
| C. `to_polars_long(field="total")` + `group_by("freq").sum()` | 135 ms |
| D. `pp.scan_long(path, field="total")` + `group_by` (lazy) | 137 ms |
| E. `scan_long` filtered to one block of 1000 devices by name | 36 ms |

- A wide table costs per column (Arrow import, plan size), not per value: 20,000 columns cost more than the
  arithmetic. The long format (`freq | signal | value`, `signal` Categorical) is built in Rust as three flat
  columns, and Polars aggregates 2M rows in a few ms.
- `field="total"` decodes only that struct member (row-record files: its byte offset inside each record) and
  leaves out traces without it (the plain `out` trace of a noise file).
- `scan_long` pushes filters that use only `signal` down to the name list, so polars-psf decodes only the matching
  devices (E). Other filters, `head()` and projections still run in Polars; reads are chunked (~4M rows).
- Grouping by block: extract the block once per signal name, not per row
  (`pl.col("signal").to_physical() // k` or a join with a 20,000-row name table); a string regex on 2M
  Categorical rows costs ~270 ms.

## Reproduce

```sh
nix develop
samplegen/run.sh                                  # Spectre samples, needs a licensed Spectre
python bench/gen_big.py /tmp/big_simple.psf 200 200000 simple
python bench/gen_big.py /tmp/big_win.psf 200 500000 windowed
(cd python && maturin build --release -o dist)
SYNTH_DIR=/tmp uv run --no-project --python 3.11 --with libpsf==0.1.4 --with 'numpy<2' --with polars \
  --with python/dist/polars_psf-*.whl python bench/run.py results.json
python bench/report.py results.json               # Markdown table
```
