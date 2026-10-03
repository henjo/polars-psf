"""Numpy-style post-processing with polars_psf.post (uses the files in testdata/).

    python python/examples/post_processing.py
"""

from pathlib import Path

import polars as pl

import polars_psf as pp

TD = Path(__file__).resolve().parents[2] / "testdata"

# --- transient: one signal, its own time axis --------------------------------------------------
tran = pp.open(TD / "pycircuit/psf/tran.tran")
i = tran.v("C0:1")
print("transient:", i)
print(f"  peak {i.ymax():.3e} {i.yunit} at t = {i.argmax():.3e} s, rms = {i.rms():.3e}")
print(f"  value at t=1 ns: {i.value(1e-9):.3e}, derivative points: {len(i.deriv())}")
print(f"  clipped to [0, 1 ns]: {len(i.clip(0.0, 1e-9))} points")
print("  batch read:", ", ".join(pp.waves(tran, ["C0:1", "I1:d"])))

# --- AC: complex transfer function and RF metrics ---------------------------------------------
ac = pp.open(TD / "psf-parser/binary/myac.ac")
h = ac.v("out") / ac.v("in")
f0 = h.x.min()
print("\ntransfer function out/in:", h)
print(f"  {h.db20().value(f0):+.4f} dB, {h.phase().value(f0):+.3f} deg at {f0:.0f} Hz")
print(f"  bandwidth (-3 dB):  {h.bandwidth():.4g} Hz")
print(f"  unity-gain freq:    {h.unity_gain_frequency():.4g} Hz")
print(f"  phase margin:       {pp.phase_margin(h):.4g} deg")

# --- result directory: select one corner -------------------------------------------------------
res = pp.open(TD / "spectre25/sweep_psfbin")
w = res.tran1.v("n10", temp=27, rval=1000.0)
print("\nsweep leaf:", w, f"| units {w.yunit}, stop {w.x.max():.3g} s")

# --- all corners at once: a family with one curve per corner ----------------------------------
fam = res.ac1.v("n10")
print("\nper-corner -3 dB bandwidth:", fam.groups)
print(fam.bandwidth().sort("bandwidth").head(3))
print("normalized to each corner's DC gain:", fam / fam.value(fam.x.min()))

# --- drop to Polars when a query is easier as an expression ------------------------------------
df = h.frame.select("freq", pl.col("freq").log10().alias("log_f"))
print("polars escape hatch:", df.head(2).to_dicts())
