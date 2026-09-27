from pathlib import Path

import numpy as np
import pytest

import polars_psf.compat as compat

TD = Path(__file__).resolve().parents[2] / "testdata"
P = TD / "pycircuit/psf"


def test_swept_real_complex_struct():
    d = compat.PSFDataSet(P / "tran.tran")
    assert d.is_swept() and d.get_nsweeps() == 1 and d.get_sweep_param_names() == ["time"]
    t = d.get_sweep_values()
    assert t.dtype == np.float64 and len(t) == d.get_sweep_npoints() == 2757
    v = d.get_signal("C0:1")
    assert v.dtype == np.float64 and v.shape == t.shape
    assert not v.flags.writeable  # zero-copy view of the decoded buffer

    ac = compat.PSFDataSet(TD / "psf-parser/binary/myac.ac")
    z = ac.get_signal(ac.get_signal_names()[0])
    assert z.dtype == np.complex128

    n = compat.PSFDataSet(P / "pnoise0.pnoise")
    s = n.get_signal("xi1.r11")
    assert s.dtype == object and set(s[0]) == {"rn", "fn", "total"}  # str keys (libpsf: bytes)
    n.invertstruct = True
    s = n.get_signal("xi1.r11")
    assert isinstance(s, dict) and s["total"].dtype == np.float64 and s["total"].shape == (21,)
    assert n.get_signal_properties("out") == {}  # libpsf 0.1.4 segfaults here


def test_nonswept_and_properties():
    d = compat.PSFDataSet(P / "opBegin")
    assert not d.is_swept() and d.get_nsweeps() == 0
    name = d.get_signal_names()[0]
    assert isinstance(d.get_signal(name), dict)
    assert d.get_signal_properties("XIRXRFMIXTRIM0.XM1PDAC1.XMN.MAIN")["Region"] == "subthreshold"
    assert "PSFversion" in d.get_header_properties()
    with pytest.raises(ValueError):
        d.get_sweep_values()


def test_psfxl_npoints_and_row_cache():
    x = compat.PSFDataSet(TD / "psf-parser/psfxl/tran1.tran.tran")
    assert x.get_sweep_npoints() == 65  # header says 0 for PSFXL stubs
    d = compat.PSFDataSet(P / "dc.dc")
    names = d.get_signal_names()
    d.get_signal(names[0])
    cached = d._all
    total = sum(d.get_signal(n) for n in names)  # loop over 1432 signals, one read
    assert d._all is cached and total.shape == (21,)
    assert d.get_signal(names[0]) is d.get_signal(names[0])  # converted once
    with pytest.raises(KeyError):
        d.get_signal("nope")
