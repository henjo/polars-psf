"""Fast reader for Cadence Spectre PSF simulation results (psfbin, psfascii, PSFXL).

>>> import polars_psf as pp
>>> f = pp.open("ac.ac")
>>> df = f.to_polars(["out"])                 # freq + out (Struct{re, im})
>>> df.select("freq", pp.cx.db20("out"))   # or pl.col("out").cx.db20()
"""

from os import PathLike

from ._polars_psf import PsfError, PsfFile
from . import cx
from .results import Results
from .lazy import scan_long

__all__ = ["PsfError", "PsfFile", "Results", "cx", "open", "read_polars", "results", "scan_long", "to_numpy_complex"]


def open(path: str | PathLike) -> PsfFile:  # noqa: A001 - mirrors builtins.open
    """Open a PSF file. For PSFXL stubs the sibling ``.psfxl`` data file is used automatically."""
    return PsfFile(path)


def results(path: str | PathLike) -> Results:
    """Open a Spectre/ADE result directory (logFile / runObjFile): nested sweeps, Monte Carlo."""
    return Results(path)


def read_polars(path: str | PathLike, names: list[str] | None = None):
    """Read sweep + traces of a swept PSF file into a polars DataFrame."""
    return PsfFile(path).to_polars(names)


def to_numpy_complex(series):
    """Convert a Struct{re, im} polars Series to a numpy complex128 array (one copy)."""
    import numpy as np

    re = series.struct.field("re").to_numpy()
    im = series.struct.field("im").to_numpy()
    out = np.empty(len(re), dtype=np.complex128)
    out.real = re
    out.imag = im
    return out
