"""libpsf-compatible numpy API on top of pp.

Drop-in for code written against libpsf (``import polars_psf.compat as libpsf``)::

    d = libpsf.PSFDataSet("tran.tran")
    t = d.get_sweep_values()          # numpy float64
    v = d.get_signal("out")           # numpy array (complex128 for AC, dicts for struct traces)

Arrays come from Polars (``Series.to_numpy``, zero-copy for real signals). Complex values are
combined into complex128 (one copy). Differences from libpsf 0.1.4, all deliberate:

- dict keys and strings are ``str`` (libpsf returns ``bytes`` on Python 3);
- ``get_signal_properties`` works on swept traces (libpsf crashes);
- reads more formats (Spectre 25.1 transient psfbin, PSFXL, psfbinf, psfascii).

Reading one signal from a row-record file (dc, ac, noise) touches the whole file, so for those, and
for any file under 64 MB, the first ``get_signal`` reads all signals once and caches them as numpy
arrays; code that loops over thousands of signals stays linear. Large transient and PSFXL files are
read per signal (selective I/O).
"""

import os
from os import PathLike

import numpy as np
import polars as pl

from ._polars_psf import PsfFile

__all__ = ["PSFDataSet"]

WHOLE_FILE_LIMIT = 64 << 20


def _to_numpy(s: pl.Series, invertstruct: bool):
    dt = s.dtype
    if isinstance(dt, pl.Struct):
        fields = [f.name for f in dt.fields]
        if fields == ["re", "im"]:
            out = np.empty(len(s), dtype=np.complex64 if dt.fields[0].dtype == pl.Float32 else np.complex128)
            out.real = s.struct.field("re").to_numpy()
            out.imag = s.struct.field("im").to_numpy()
            return out
        if invertstruct:
            return {f: _to_numpy(s.struct.field(f), invertstruct) for f in fields}
        return np.array(s.to_list(), dtype=object)
    return s.to_numpy()


class PSFDataSet:
    """libpsf ``PSFDataSet`` interface backed by :class:`pp.PsfFile`."""

    def __init__(self, filename: "str | PathLike"):
        self._f = PsfFile(filename)
        self.invertstruct = False
        self._all = None  # {name: Series} when the whole file is read at once
        self._np = {}  # (name, invertstruct) -> converted value
        self._sweep = None
        size = os.path.getsize(filename)
        if self._f.is_psfxl and os.path.exists(f"{filename}.psfxl"):
            size += os.path.getsize(f"{filename}.psfxl")
        self._whole = self._f.layout in ("rows", "ascii") or size < WHOLE_FILE_LIMIT

    def close(self):
        self._f = self._all = self._sweep = None
        self._np = {}

    # metadata -----------------------------------------------------------------------------
    def get_header_properties(self) -> dict:
        return self._f.header

    def get_signal_names(self) -> list:
        return self._f.names

    def is_swept(self) -> bool:
        return self._f.is_swept

    def get_nsweeps(self) -> int:
        return 1 if self._f.is_swept else 0

    def get_sweep_param_names(self) -> list:
        return [self._f.sweep_name] if self._f.is_swept else []

    def get_sweep_npoints(self) -> int:
        n = self._f.header.get("PSF sweep points") or 0
        if not n and self._f.is_swept:  # PSFXL stubs and psfascii omit it
            n = len(self.get_sweep_values())
        return n

    def get_signal_properties(self, name: str) -> dict:
        props = self._f.props(name)
        props.pop("type", None)
        return props

    # values -------------------------------------------------------------------------------
    def _load_all(self) -> dict:
        if self._all is None:
            df = self._f.to_polars()
            # positional access: no per-call name lookup in a wide frame
            self._all = dict(zip(df.columns, df.get_columns(), strict=True))
        return self._all

    def _series(self, name: str) -> pl.Series:
        if self._whole:
            return self._load_all()[name]
        return self._f.to_polars([name]).get_columns()[1]

    def get_sweep_values(self) -> np.ndarray:
        if not self._f.is_swept:
            raise ValueError("file is not swept")
        if self._sweep is None:
            if self._whole:
                s = self._load_all()[self._f.sweep_name]
            else:
                s = self._f.to_polars([]).get_columns()[0]
            self._sweep = s.to_numpy()
        return self._sweep

    def get_signal(self, name: str):
        """numpy array for swept files, Python value (dict for structs) for non-swept files."""
        if not self._f.is_swept:
            return self._f.value(name)
        key = (name, self.invertstruct)
        if key not in self._np:
            if name not in self._f:
                raise KeyError(name)
            self._np[key] = _to_numpy(self._series(name), self.invertstruct)
        return self._np[key]

    def __repr__(self) -> str:
        return f"PSFDataSet({self._f.path!r})"
