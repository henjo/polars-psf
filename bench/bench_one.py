"""One timed case: `python bench_one.py {libpsf|polars-psf} FILE N` -> JSON on stdout.

Both produce numeric data for the first N signals: libpsf numpy arrays via get_signal(),
polars-psf: a polars DataFrame (sweep + N signals) via to_polars(). Non-swept files: N values.
"""
import json
import sys
import time

lib, path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
# imports (polars, numpy) are not timed
if lib == "libpsf":
    import libpsf
else:
    import polars  # noqa: F401
    import polars_psf as pp
t0 = time.perf_counter()
if lib == "libpsf":
    d = libpsf.PSFDataSet(path)
    names = list(d.get_signal_names())
    t1 = time.perf_counter()
    if d.is_swept():
        d.get_sweep_values()
    out = [d.get_signal(s) for s in names[:n]]
else:
    f = pp.open(path)
    names = f.names
    t1 = time.perf_counter()
    if f.is_swept:
        out = f.to_polars(names[:n])
    else:
        out = [f.value(s) for s in names[:n]]
t2 = time.perf_counter()
print(json.dumps({"open_ms": (t1 - t0) * 1e3, "read_ms": (t2 - t1) * 1e3, "total_ms": (t2 - t0) * 1e3, "signals": len(names)}))
