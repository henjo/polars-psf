"""Using polars-psf through the libpsf-compatible numpy API.

Existing libpsf scripts only need a different import:

    import libpsf                    ->    import polars_psf.compat as libpsf

Run from the repository root: python python/examples/libpsf_compat.py
"""

from pathlib import Path

import numpy as np

import polars_psf.compat as libpsf

TD = Path(__file__).resolve().parents[2] / "testdata"

# --- transient: sweep values and a signal as numpy arrays --------------------------------------
tran = libpsf.PSFDataSet(TD / "pycircuit/psf/tran.tran")
print("tran swept:", tran.is_swept(), "| sweep:", tran.get_sweep_param_names(), "| points:", tran.get_sweep_npoints())
t = tran.get_sweep_values()  # numpy float64
v = tran.get_signal("C0:1")  # numpy float64, same length as t
print(f"  C0:1 peak {np.abs(v).max():.3e} A at t = {t[np.abs(v).argmax()]:.3e} s")
print("  properties:", tran.get_signal_properties("C0:1"))
print("  header:", {k: tran.get_header_properties()[k] for k in ("simulator", "analysis type", "stop")})

# --- AC: complex128 arrays -------------------------------------------------------------------
ac = libpsf.PSFDataSet(TD / "psf-parser/binary/myac.ac")
freq = ac.get_sweep_values()
print("\nac signals:", ac.get_signal_names())
h = ac.get_signal("out") / ac.get_signal("in")  # numpy complex128 transfer function
db, deg = 20 * np.log10(np.abs(h)), np.degrees(np.angle(h))
k = np.argmax(db < db[0] - 3)  # first point 3 dB below the pass band
print(f"  out/in: {h.dtype}, {abs(db[0]):.1f} dB at {freq[0]:.0f} Hz, -3 dB near {freq[k]:.3g} Hz ({deg[k]:.0f} deg)")

# --- noise: struct signals -------------------------------------------------------------------
noise = libpsf.PSFDataSet(TD / "pycircuit/psf/pnoise0.pnoise")
nfreq = noise.get_sweep_values()
r = noise.get_signal("xi1.r11")  # default: numpy object array, one dict per frequency
print(f"\nnoise xi1.r11 at {nfreq[0]:.0f} Hz:", r[0])
noise.invertstruct = True
r = noise.get_signal("xi1.r11")  # dict of numpy arrays, one per struct member
print("  invertstruct=True:", {k: (a.dtype.name, a.shape) for k, a in r.items()})

# classic noise summary: sum the 'total' contribution of every device (a loop over signals is
# fine: row-record files are read once and cached)
devices = [n for n in noise.get_signal_names() if n != "out"]
total = sum(noise.get_signal(n)["total"] for n in devices)
out = noise.get_signal("out")  # output noise in V/sqrt(Hz)
print(f"  {len(devices)} devices, sum of contributions / out^2 = {np.max(np.abs(total / out**2 - 1)):.1e} (max deviation from 1)")
top = sorted(devices, key=lambda n: -noise.get_signal(n)["total"][0])[:3]
print("  largest contributors at", f"{nfreq[0]:.0f} Hz:", top)

# --- operating point: plain Python values ----------------------------------------------------
op = libpsf.PSFDataSet(TD / "pycircuit/psf/dcOpInfo.info")
print("\nop swept:", op.is_swept(), "| values:", len(op.get_signal_names()))
dev = op.get_signal("IREG21U_0.MP5.b1")  # dict for struct values
print("  IREG21U_0.MP5.b1:", {k: dev[k] for k in list(dev)[:4]}, "...")

# --- formats libpsf cannot read ---------------------------------------------------------------
xl = libpsf.PSFDataSet(TD / "spectre25/psfxl_ac/tran1.tran.tran")  # PSFXL (Spectre default for tran)
print("\nPSFXL:", xl.get_sweep_npoints(), "points,", len(xl.get_signal_names()), "signals")
f32 = libpsf.PSFDataSet(TD / "spectre25/psfbinf_ac/tran1.tran")  # single-precision psfbin
print("psfbinf:", f32.get_signal(f32.get_signal_names()[0]).dtype)
