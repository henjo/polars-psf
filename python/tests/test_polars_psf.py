from pathlib import Path

import numpy as np
import polars as pl
import pytest
from conftest import recorded

import polars_psf as pp
from polars_psf._polars_psf import PsfFile

TD = Path(__file__).resolve().parents[2] / "testdata"


def test_ac_complex_struct():
    d = pp.open(TD / "psf-parser/binary/myac.ac")
    df = d.scan().collect()
    assert df.columns[0] == d.sweep_name() == "freq"
    assert df.schema["freq"] == pl.Float64
    cplx = [c for c, t in df.schema.items() if t == pl.Struct({"re": pl.Float64, "im": pl.Float64})]
    assert cplx
    name = cplx[0]
    z = pp.to_numpy_complex(df[name])
    got = df.select(pl.col(name).cx.db20()).to_series().to_numpy()
    np.testing.assert_allclose(got, 20 * np.log10(np.abs(z)))
    got = df.select(pp.cx.phase(name)).to_series().to_numpy()
    np.testing.assert_allclose(got, np.degrees(np.angle(z)))


def test_file_is_one_result():
    d = pp.open(TD / "psf-parser/binary/myac.ac")
    a = d.results
    assert a["name"].to_list() == ["myac"] and a["type"].to_list() == ["ac"]
    assert a["params"][0].to_list() == [] and a["leaves"].to_list() == [1]
    assert d.leaves().columns == ["leaf", "path"]
    assert d.scan("myac").collect().equals(d.scan("ac").collect())
    with pytest.raises(KeyError):
        d.scan("tran")
    assert d.warnings == []


def test_binary_equals_ascii_tran():
    b = pp.open(TD / "pycircuit/psf/tran.tran").scan().collect()
    a = pp.open(TD / "pycircuit/psfasc/tran.tran.asc").scan().collect()
    assert b.shape == a.shape == (2757, 33)
    assert b.columns == a.columns
    np.testing.assert_allclose(b.to_numpy(), a.to_numpy(), rtol=1e-5, atol=1e-30)


def test_scan_projection_pushdown_and_laziness():
    d, rec = recorded(TD / "pycircuit/psf/tran.tran")
    lf = d.scan()
    assert lf.collect_schema().names()[:1] == ["time"] and rec.calls == []  # nothing read yet
    df = lf.select("I1:d", "C0:1").collect()
    assert df.columns == ["I1:d", "C0:1"] and df.height == 2757
    assert [(k, i, sorted(n)) for k, i, n in rec.calls] == [("wide", [0], ["C0:1", "I1:d"])]
    assert d.names()[:2] == PsfFile(TD / "pycircuit/psf/tran.tran").names[:2]
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        d.scan().select("nope").collect()


def test_props_and_low_level_file():
    d = pp.open(TD / "pycircuit/psf/tran.tran")
    f = d.file()
    assert f.units("C0:1") == "A"
    assert f.props("C0:1")["type"] == "I"
    assert "C0:1" in f and len(f) == 32
    assert f.header["PSF sweep points"] == 2757
    with pytest.raises(KeyError):
        f.to_polars(["nope"])


def test_struct_traces():
    df = pp.open(TD / "pycircuit/psf/pnoise0.pnoise").scan().select("xi1.qi71.ql", "xi1.r11").collect()
    assert df.schema["xi1.r11"] == pl.Struct({"rn": pl.Float64, "fn": pl.Float64, "total": pl.Float64})
    assert df["xi1.qi71.ql"].struct.field("total")[0] == pytest.approx(4.745056346150785e-10)


def test_nonswept_values():
    d = pp.open(TD / "pycircuit/psf/dcOpInfo.info")
    assert not d.is_swept() and d.sweep_name() is None
    v = d.value("IREG21U_0.MP5.b1")
    assert v["betadc"] == pytest.approx(4.7957014499434756)
    assert len(d.values()) == len(d.names())
    one = d.scan().collect()
    assert one.height == 1
    assert one["IREG21U_0.MP5.b1"].struct.field("betadc")[0] == pytest.approx(4.7957014499434756)


def test_logfile_low_level():
    f = PsfFile(TD / "pycircuit/pardcsweep.raw/logFile")
    v = f.value("sweep1-000_sweep2-001_dc1-dc")
    assert v["dataFile"] == "sweep1-000_sweep2-001_dc1.dc"
    assert v["sweepVariable"] == ["vdc1"]
    assert f.props("sweep1-000_sweep2-001_dc1-dc")["vdc2"] == pytest.approx(3.66667)
    # pp.open on a logFile opens the result directory
    assert pp.open(TD / "pycircuit/pardcsweep.raw/logFile").results.height == 2


def test_psfxl():
    d = pp.open(TD / "psf-parser/psfxl/tran1.tran.tran")
    f = d.file()
    assert f.is_psfxl
    x = d.scan().collect()
    a = pp.open(TD / "psf-parser/ascii/tran1.tran.tran").scan().collect()
    assert x.columns == a.columns
    np.testing.assert_allclose(x.to_numpy(), a.to_numpy(), rtol=1e-14)
    assert f.psfxl_meta["cdnshsweepcount"] == 65
    s = f.read_signal("out")
    assert s.columns == ["time", "out"] and s.height == 65


def test_truncated():
    d = pp.open(TD / "pycircuit/resultdirs/parsweep/VDC1=0,VDC2=0/srcSweep")
    assert not d.file().is_complete
    assert d.scan().collect().height == 0


def test_errors(tmp_path):
    bad = tmp_path / "bad"
    bad.write_bytes(b"not a psf file")
    with pytest.raises(pp.PsfError):
        pp.open(bad)
    with pytest.raises(OSError):
        pp.open(tmp_path / "missing")


def test_arrow_metadata_via_pyarrow():
    pa = pytest.importorskip("pyarrow")
    rb = pa.record_batch(pp.open(TD / "pycircuit/psf/tran.tran").file().read(["C0:1"]))
    assert rb.schema.field("C0:1").metadata[b"psf:units"] == b"A"


def test_scan_long_struct_field_and_pushdown():
    p = TD / "pycircuit/psf/pnoise0.pnoise"
    d, rec = recorded(p)
    lf = d.scan_long(field="total")
    assert lf.collect_schema().names() == ["freq", "signal", "value"]
    full = lf.collect()
    devices = d.names(field="total")
    assert "out" in d.names() and "out" not in devices  # plain output-noise trace is left out
    assert full.height == len(devices) * 21
    assert full.equals(PsfFile(p).to_polars_long(field="total"))
    # predicate only on signal: selected before reading
    sel = [n for n in devices if n.startswith("xi1.r")]
    rec.clear()
    part = lf.filter(pl.col("signal").is_in(sel)).collect()
    assert rec.calls == [("long", [(0, sel)])]
    assert set(part["signal"].cast(pl.String).unique()) == set(sel)
    part2 = lf.filter(pl.col("signal").cast(pl.String).str.starts_with("xi1.r")).collect()
    assert part2.sort("signal", "freq").equals(part.sort("signal", "freq"))
    # total noise per frequency matches the wide table
    wide = d.scan().collect()
    ref = np.sum([wide[c].struct.field("total").to_numpy() for c in devices], axis=0)
    got = lf.group_by("freq").agg(pl.col("value").sum()).sort("freq").collect()["value"].to_numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-12)
    # mixed predicate is applied after reading; head stops early
    assert lf.filter(pl.col("value") > 0).head(3).collect().height == 3


def test_scan_long_chunks_signals():
    d, rec = recorded(TD / "pycircuit/psf/tran.tran")
    df = d.scan_long(chunk_rows=10 * 2757).collect()
    assert df.columns == ["time", "signal", "value"] and df.height == 32 * 2757
    assert [pl.DataFrame(b)["signal"].n_unique() for step in rec.steps for b in step] == [10, 10, 10, 2]
    rec.clear()
    assert d.scan_long(chunk_rows=10 * 2757).head(5).collect().height == 5
    assert len(rec.steps) == 1  # stops after the first chunk
    with pytest.raises(pl.exceptions.ComputeError, match="field="):
        pp.open(TD / "pycircuit/psf/pnoise0.pnoise").scan_long().collect()  # structs need field=
