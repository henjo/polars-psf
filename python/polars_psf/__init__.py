"""Fast reader for Cadence Spectre PSF simulation results (psfbin, psfascii, PSFXL).

One lazy entry point for single PSF files and result directories::

>>> import polars_psf as pp
>>> d = pp.open("ac.ac")                       # metadata only; also a result dir / logFile
>>> d.scan().select("freq", pp.cx.db20("out")).collect()   # or pl.col("out").cx.db20()
>>> r = pp.open("sim.raw")
>>> r.scan("tran1").filter(pl.col("temp") == 27).select("time", "out").collect()

The :mod:`polars_psf.post` layer adds a numpy-style ``Waveform`` on top of the queries::

>>> w = pp.wave("ac.ac", "out")            # lazy: decoded on first value access
>>> w.db20(), w.phase(), w.bandwidth()     # complex arithmetic builds on Struct{re, im}

Results as objects, and calculator functions (OCEAN names such as ``dB20`` are aliases)::

>>> r = pp.open("sim.raw")
>>> r.ac1.v("n10").bandwidth()             # one row per corner
>>> pp.bandwidth(r.ac1.v("n10"), 3, "low"), pp.db20(r.ac1.v("n10"))
"""

from . import cx
from ._polars_psf import PsfError
from .dataset import Dataset, Result, open, openResults
from .post import *  # noqa: F403 - Waveform and the calculator functions
from .post import __all__ as _post_all

__all__ = ["Dataset", "PsfError", "Result", "cx", "open", "openResults", "to_numpy_complex", *_post_all]


def to_numpy_complex(series):
    """Convert a Struct{re, im} polars Series to a numpy complex128 array (one copy)."""
    import numpy as np

    re = series.struct.field("re").to_numpy()
    im = series.struct.field("im").to_numpy()
    out = np.empty(len(re), dtype=np.complex128)
    out.real = re
    out.imag = im
    return out
