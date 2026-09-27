"""polars_psf.compat vs libpsf 0.1.4 from PyPI (needs Python <= 3.11 and numpy < 2; skipped otherwise)."""

from pathlib import Path

import numpy as np
import pytest

import polars_psf.compat as compat

P = Path(__file__).resolve().parents[2] / "testdata/pycircuit/psf"
libpsf = pytest.importorskip("libpsf", reason="libpsf 0.1.4 (PyPI) not installed")


def _norm(x):
    """libpsf returns bytes keys on Python 3."""
    if isinstance(x, dict):
        return {(k.decode() if isinstance(k, bytes) else k): _norm(v) for k, v in x.items()}
    if isinstance(x, bytes):
        return x.decode()
    return x


@pytest.mark.parametrize("name", ["srcSweep", "tran.tran", "frequencySweep", "dc.dc", "pnoise0.pnoise", "timeSweep", "opBegin"])
def test_same_as_libpsf(name):
    a, b = libpsf.PSFDataSet(str(P / name)), compat.PSFDataSet(P / name)
    names = list(a.get_signal_names())
    assert names == b.get_signal_names()
    assert a.is_swept() == b.is_swept()
    if a.is_swept():
        np.testing.assert_array_equal(a.get_sweep_values(), b.get_sweep_values())
        assert a.get_sweep_npoints() == b.get_sweep_npoints()
    for invert in (False, True):
        a.invertstruct = b.invertstruct = invert
        for n in names[:40]:
            x, y = _norm(a.get_signal(n)), b.get_signal(n)
            if isinstance(x, np.ndarray) and x.dtype == object:
                assert [_norm(e) for e in x] == list(y), n
            elif isinstance(x, dict) and isinstance(next(iter(x.values()), None), np.ndarray):
                assert x.keys() == y.keys() and all(np.array_equal(x[k], y[k]) for k in x), n
            elif isinstance(x, np.ndarray):
                assert x.dtype == y.dtype and np.array_equal(x, y), n
            else:
                assert x == y, n
