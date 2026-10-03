import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from conftest import recorded

import polars_psf as pp

TD = Path(__file__).resolve().parents[2] / "testdata"


def test_from_file_metadata_and_values():
    d = pp.open(TD / "pycircuit/psf/tran.tran")
    ref = d.scan().select("C0:1").collect()["C0:1"]
    w = pp.wave(d, "C0:1")
    assert (w.xname, w.yname) == ("time", "C0:1")
    assert (w.xunit, w.yunit) == ("s", "A")
    assert len(w) == len(ref)
    assert w.y.equals(ref)
    assert w.ymax() == w.y.max()
    assert w.value(0.0) == w.y[0]
    assert len(w.deriv()) == len(w) - 1

    many = pp.waves(d, ["C0:1", "I1:d"])
    assert set(many) == {"C0:1", "I1:d"}
    assert many["C0:1"].y.equals(w.y)


def test_wave_is_lazy_and_waves_share_one_read():
    d, rec = recorded(TD / "pycircuit/psf/tran.tran")
    w = d.wave("C0:1")
    d2 = w * 2 + 1  # elementwise ops stay lazy
    assert len(w) == 2757 and rec.calls == []  # length from the header
    assert d2.ymax() == pytest.approx(2 * w.ymax() + 1)
    assert len(rec.calls) == 2  # d2 and w each collect their own plan

    rec.clear()
    many = d.waves(["C0:1", "I1:d", "R0:1"])
    assert rec.calls == []
    assert [len(m.y) for m in many.values()] == [2757] * 3
    assert len(rec.calls) == 1


def test_psfxl_per_signal_axis():
    f = pp.open(TD / "spectre25/psfxl_ac/tran1.tran.tran")
    assert f.file().is_psfxl
    w = pp.wave(f, "n1")
    assert len(w) == len(pp.waves(f, ["n1"])["n1"]) and w.xname == "time"
    assert len(w) > 1
    assert len(pp.waves(f, ["n1", "n2"])) == 2


def test_complex_transfer_function_and_metrics():
    ac = pp.open(TD / "psf-parser/binary/myac.ac")
    out, inn = pp.wave(ac, "out"), pp.wave(ac, "in")
    assert out.is_complex

    h = out / inn
    freq = out.x.to_numpy()
    expected = np.array(
        [complex(v["re"], v["im"]) for v in out.y], dtype=np.complex128
    ) / np.array([complex(v["re"], v["im"]) for v in inn.y], dtype=np.complex128)
    assert np.allclose(h.to_numpy(), expected)

    np.testing.assert_allclose(h.db20().y.to_numpy(), 20 * np.log10(np.abs(expected)))
    np.testing.assert_allclose(h.phase().y.to_numpy(), np.degrees(np.angle(expected)))
    assert h.db20().yunit == "dB" and h.phase().yunit == "deg"
    assert abs(h).y.to_list() == pytest.approx(np.abs(expected))
    assert h.conj().to_numpy() == pytest.approx(np.conj(expected))

    # -3 dB point: independent numpy interpolation of |out/in|
    mag_np = np.abs(expected)
    target = mag_np[0] * 10 ** (-3 / 20)
    below = np.flatnonzero(mag_np < target)
    assert below.size
    i = int(below[0])
    t = (target - mag_np[i - 1]) / (mag_np[i] - mag_np[i - 1])
    assert h.bandwidth() == pytest.approx(freq[i - 1] + t * (freq[i] - freq[i - 1]))
    assert math.isnan(h.unity_gain_frequency())  # |out/in| stays just below 1

    with pytest.raises(TypeError):
        h.ymax()
    with pytest.raises(TypeError):
        h.cross()
    with pytest.raises(NotImplementedError):
        h**h  # waveform ** waveform is not supported


def test_results_wave_selects_leaf():
    r = pp.open(TD / "spectre25/sweep_psfbin")
    assert r.sweep_name("tran1") == "time"
    w = r.wave("n10", "tran1", temp=27, rval=1000.0)
    assert (w.xname, w.yname, w.xunit) == ("time", "n10", "s")
    assert w.y.equals(r.scan("tran1").filter(
        (pl.col("temp") == 27) & (pl.col("rval") == 1000.0)
    ).select("n10").collect()["n10"])
    assert pp.wave(r, "n10", "tran1", temp=27, rval=1000.0).y.equals(w.y)
    assert r.waves(["n10"], "tran1", temp=27, rval=1000.0)["n10"].y.equals(w.y)
    xl = pp.open(TD / "spectre25/sweep_psfxl").wave("n10", "tran1", temp=27, rval=1000.0)
    assert xl.y.equals(w.y)  # PSFXL leaves go through the long table

    all9 = r.wave("n10", "tran1")  # no parameters: one curve per corner
    assert all9.groups == ["temp", "rval"] and all9.index == ["temp", "rval", "time"]
    assert all9.ymax().height == 9
    hot = r.wave("n10", "tran1", temp=125)  # rval still varies
    assert hot.groups == ["rval"] and hot.ymax().height == 3
    with pytest.raises(ValueError):
        r.wave("n10")  # 2 results
    with pytest.raises(KeyError):
        r.wave("n10", "tran1", bogus=1)
    with pytest.raises(KeyError):
        r.wave("n10", "tran1", temp=999.0)


def test_errors():
    with pytest.raises(ValueError):
        pp.wave(TD / "pycircuit/psf/dcOpInfo.info", "some value")  # not swept


def test_binop_reads_only_index_until_values_are_needed():
    from conftest import recorded

    d, rec = recorded(TD / "psf-parser/binary/myac.ac")
    h = d.wave("out") / d.wave("in")
    assert [names for _, _, names in rec.calls] == [[], []]  # sweep checks only
    assert h.is_complex and len(h.y) == len(d.wave("out").y)


def test_corner_bandwidth_in_one_line():
    r = pp.open(TD / "spectre25/sweep_psfbin")
    bw = r.wave("n10", "ac1").bandwidth()
    assert bw.columns == ["temp", "rval", "bandwidth"] and bw.height == 9
    one = r.wave("n10", "ac1", temp=27, rval=1000.0).bandwidth()
    assert bw.filter((pl.col("temp") == 27) & (pl.col("rval") == 1000.0))["bandwidth"].item() == pytest.approx(one)


def test_result_objects_and_aliases():
    r = pp.open(TD / "spectre25/sweep_psfbin")
    ac = r.ac1
    assert "ac1" in dir(r) and ac.name == "ac1" and ac.type == "ac" and ac.params == ["temp", "rval"]
    assert "n10" in ac and ac.leaves.height == 9 and ac.sweep_name == "freq"
    assert ac.v("n10").bandwidth().equals(r.wave("n10", "ac1").bandwidth())
    assert r.result("ac1").v("n10", temp=27, rval=1e3).bandwidth() == pytest.approx(
        ac.v("n10").leaf(temp=27, rval=1e3).bandwidth()
    )
    assert ac.i("V1:p").yname == "V1:p"
    with pytest.raises(AttributeError):
        _ = r.nope
    f = pp.open(TD / "psf-parser/binary/myac.ac")  # a single file: one result
    assert f.myac.v("out").bandwidth() == pytest.approx(f.v("out").bandwidth())
    # OCEAN and pycircuit names are aliases of the Python names
    assert pp.dB20 is pp.db20 and pp.unityGainFreq is pp.unity_gain_frequency and pp.IIP3 is pp.iip3
    assert pp.riseTime is pp.rise_time and pp.leafValue is pp.leaf_value and pp.openResults is pp.open
