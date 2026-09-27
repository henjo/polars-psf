from pathlib import Path

import numpy as np
import polars as pl
import pytest

import polars_psf as pp

TD = Path(__file__).resolve().parents[2] / "testdata"


def test_ac_complex_struct():
    f = pp.open(TD / "psf-parser/binary/myac.ac")
    df = f.to_polars()
    assert df.columns[0] == f.sweep_name == "freq"
    assert df.schema["freq"] == pl.Float64
    cplx = [c for c, t in df.schema.items() if t == pl.Struct({"re": pl.Float64, "im": pl.Float64})]
    assert cplx
    name = cplx[0]
    z = pp.to_numpy_complex(df[name])
    got = df.select(pl.col(name).cx.db20()).to_series().to_numpy()
    np.testing.assert_allclose(got, 20 * np.log10(np.abs(z)))
    got = df.select(pp.cx.phase(name)).to_series().to_numpy()
    np.testing.assert_allclose(got, np.degrees(np.angle(z)))


def test_binary_equals_ascii_tran():
    b = pp.read_polars(TD / "pycircuit/psf/tran.tran")
    a = pp.read_polars(TD / "pycircuit/psfasc/tran.tran.asc")
    assert b.shape == a.shape == (2757, 33)
    assert b.columns == a.columns
    np.testing.assert_allclose(b.to_numpy(), a.to_numpy(), rtol=1e-5, atol=1e-30)


def test_projection_and_props():
    f = pp.open(TD / "pycircuit/psf/tran.tran")
    df = f.to_polars(["I1:d", "C0:1"])
    assert df.columns == ["time", "I1:d", "C0:1"]
    assert f.units("C0:1") == "A"
    assert f.props("C0:1")["type"] == "I"
    assert "C0:1" in f and len(f) == 32
    assert f.header["PSF sweep points"] == 2757
    with pytest.raises(KeyError):
        f.to_polars(["nope"])


def test_struct_traces():
    df = pp.open(TD / "pycircuit/psf/pnoise0.pnoise").to_polars(["xi1.qi71.ql", "xi1.r11"])
    assert df.schema["xi1.r11"] == pl.Struct({"rn": pl.Float64, "fn": pl.Float64, "total": pl.Float64})
    assert df["xi1.qi71.ql"].struct.field("total")[0] == pytest.approx(4.745056346150785e-10)


def test_nonswept_values():
    f = pp.open(TD / "pycircuit/psf/dcOpInfo.info")
    assert not f.is_swept and f.sweep_name is None
    v = f.value("IREG21U_0.MP5.b1")
    assert v["betadc"] == pytest.approx(4.7957014499434756)
    assert len(f.values()) == len(f)


def test_logfile():
    f = pp.open(TD / "pycircuit/pardcsweep.raw/logFile")
    v = f.value("sweep1-000_sweep2-001_dc1-dc")
    assert v["dataFile"] == "sweep1-000_sweep2-001_dc1.dc"
    assert v["sweepVariable"] == ["vdc1"]
    assert f.props("sweep1-000_sweep2-001_dc1-dc")["vdc2"] == pytest.approx(3.66667)


def test_psfxl():
    f = pp.open(TD / "psf-parser/psfxl/tran1.tran.tran")
    assert f.is_psfxl
    x = f.to_polars()
    a = pp.read_polars(TD / "psf-parser/ascii/tran1.tran.tran")
    assert x.columns == a.columns
    np.testing.assert_allclose(x.to_numpy(), a.to_numpy(), rtol=1e-14)
    assert f.psfxl_meta["cdnshsweepcount"] == 65
    s = f.read_signal("out")
    assert s.columns == ["time", "out"] and s.height == 65


def test_truncated():
    f = pp.open(TD / "pycircuit/resultdirs/parsweep/VDC1=0,VDC2=0/srcSweep")
    assert not f.is_complete
    assert f.to_polars().height == 0


def test_errors(tmp_path):
    bad = tmp_path / "bad"
    bad.write_bytes(b"not a psf file")
    with pytest.raises(pp.PsfError):
        pp.open(bad)
    with pytest.raises(OSError):
        pp.open(tmp_path / "missing")


def test_arrow_metadata_via_pyarrow():
    pa = pytest.importorskip("pyarrow")
    rb = pa.record_batch(pp.open(TD / "pycircuit/psf/tran.tran").read(["C0:1"]))
    assert rb.schema.field("C0:1").metadata[b"psf:units"] == b"A"


def test_scan_long_struct_field_and_pushdown():
    p = TD / "pycircuit/psf/pnoise0.pnoise"
    f = pp.open(p)
    lf = pp.scan_long(p, field="total")
    assert lf.collect_schema().names() == ["freq", "signal", "value"]
    full = lf.collect()
    devices = f.names_with_field("total")
    assert "out" in f.names and "out" not in devices  # plain output-noise trace is left out
    assert full.height == len(devices) * 21
    # eager long equals lazy long
    assert full.equals(f.to_polars_long(field="total"))
    # predicate only on signal: selected before reading
    sel = [n for n in devices if n.startswith("xi1.r")]
    part = lf.filter(pl.col("signal").is_in(sel)).collect()
    assert set(part["signal"].cast(pl.String).unique()) == set(sel)
    part2 = lf.filter(pl.col("signal").cast(pl.String).str.starts_with("xi1.r")).collect()
    assert part2.sort("signal", "freq").equals(part.sort("signal", "freq"))
    # total noise per frequency matches the wide table
    wide = f.to_polars()
    ref = np.sum([wide[c].struct.field("total").to_numpy() for c in devices], axis=0)
    got = lf.group_by("freq").agg(pl.col("value").sum()).sort("freq").collect()["value"].to_numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-12)
    # mixed predicate is applied after reading; head stops early
    assert lf.filter(pl.col("value") > 0).head(3).collect().height == 3


def test_long_plain_signals():
    df = pp.open(TD / "pycircuit/psf/tran.tran").to_polars_long()
    assert df.columns == ["time", "signal", "value"] and df.height == 32 * 2757
    with pytest.raises(pp.PsfError):
        pp.open(TD / "pycircuit/psf/pnoise0.pnoise").to_polars_long()  # structs need field=
