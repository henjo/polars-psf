"""`python check_equal.py FILE N`: max relative difference libpsf vs polars-psf over the first N signals."""
import sys

import numpy as np

import libpsf
import polars_psf as pp

path, n = sys.argv[1], int(sys.argv[2])
d, f = libpsf.PSFDataSet(path), pp.open(path)
names = list(d.get_signal_names())[:n]
assert names == f.names[: len(names)], "signal names differ"
worst = 0.0
if f.is_swept:
    df = f.to_polars(names)
    pairs = [(np.asarray(d.get_sweep_values()), df[f.sweep_name].to_numpy())]
    for s in names:
        a = d.get_signal(s)
        col = df[s]
        if col.dtype == __import__("polars").Struct({"re": __import__("polars").Float64, "im": __import__("polars").Float64}):
            b = pp.to_numpy_complex(col)
        elif isinstance(a, dict):  # struct traces: dict of arrays (invertstruct) or array of dicts
            continue
        else:
            b = col.to_numpy()
        pairs.append((np.asarray(a), b))
    for a, b in pairs:
        if a.dtype == object:
            continue
        scale = max(np.abs(a).max(), np.abs(b).max(), 1e-300)
        worst = max(worst, float(np.abs(a - b).max() / scale))
else:
    for s in names:
        a, b = d.get_signal(s), f.value(s)
        if isinstance(a, dict):
            worst = max(worst, max(abs(a[k] - b[k]) / max(abs(a[k]), 1e-300) for k in a if isinstance(a[k], float)))
print(f"{worst:.2e}")
