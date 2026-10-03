from pathlib import Path

import polars as pl
import pytest
from conftest import recorded

import polars_psf as pp

TD = Path(__file__).resolve().parents[2] / "testdata"


def test_spectre_nested_sweep_read():
    r = pp.open(TD / "pycircuit/pardcsweep.raw")
    a = r.results
    assert a["name"].to_list() == ["dc1", "dc2"]  # short names; family names still resolve
    assert a["params"][0].to_list() == ["vdc3", "vdc2"]
    df = r.scan("dc1").collect()
    assert df.columns[:3] == ["vdc3", "vdc2", "vdc1"]
    assert df.height == 16 * 4
    assert 3.6666666666666665 in df["vdc2"].unique().to_list()
    assert r.warnings == []


def test_scan_pushdown_skips_files_and_signals():
    r, rec = recorded(TD / "pycircuit/pardcsweep.raw")
    trace = r.names("dc1")[0]
    out = r.scan("dc1").filter(pl.col("vdc3") == 10).select("vdc2", "vdc1", trace).collect()
    assert out.height == 4 * 4
    assert out.columns == ["vdc2", "vdc1", trace]
    read_idx = [i for _, idx, _ in rec.calls for i in idx]
    assert len(read_idx) == 4  # only leaves with vdc3 == 10
    assert all(names == [trace] for _, _, names in rec.calls)
    ref = r.scan("dc1").collect().filter(pl.col("vdc3") == 10)
    assert out["vdc2"].unique().sort().equals(ref["vdc2"].unique().sort())


def test_scan_matches_read_and_head():
    r = pp.open(TD / "pycircuit/pardcsweep.raw")
    full = r.scan("dc2").collect()
    assert r.scan("dc2").collect().equals(full)
    assert r.scan("dc2").head(5).collect().height == 5


def test_monte_carlo_tran():
    r = pp.open(TD / "psf-parser/binary/logFile")
    df = r.scan("mymonte_tran1").select("iteration", "time", "out").collect()
    assert df.columns == ["iteration", "time", "out"]
    assert df["iteration"].unique().sort().to_list() == [1.0, 2.0, 3.0]
    per = df.group_by("iteration").len().sort("iteration")["len"].to_list()
    assert len(set(per)) > 1  # adaptive time steps differ per iteration: long format, no grid


def test_nested_ac_complex():
    r = pp.open(TD / "psf-parser/binary/logFile")
    sig = r.names("ac2")[0]
    df = r.scan("ac2").select("R1:r", "C1:c", "freq", pl.col(sig).cx.db20().alias("db")).collect()
    assert df.select(pl.struct("R1:r", "C1:c").n_unique()).item() == 9


def test_ade_parametric_nonswept_op():
    r = pp.open(TD / "pycircuit/resultdirs/parsweep/psf")
    assert r.warnings == []  # nothing resolved yet
    df = r.scan("opBegin").collect()
    assert df.height == 9
    assert df.columns[:2] == ["VDC1", "VDC2"]
    assert r.leaves("srcSweep").height == 1
    assert sum("missing data file" in w for w in r.warnings) == 2


def test_unknown_result():
    r = pp.open(TD / "pycircuit/pardcsweep.raw")
    with pytest.raises(KeyError):
        r.scan("nope")


S25 = TD / "spectre25"


def jobs(rec):
    """(leaf, signals) jobs requested from the reader so far."""
    return [j for c in rec.calls for j in c[1]]


def test_scan_long_across_sweep_matches_formats():
    b = pp.open(S25 / "sweep_psfbin").scan_long("tran1").collect()
    x = pp.open(S25 / "sweep_psfxl").scan_long("tran1").collect()
    assert b.columns == ["temp", "rval", "time", "signal", "value"]
    assert b.select("temp", "rval").n_unique() == 9
    key = ["temp", "rval", "signal", "time"]
    assert b.sort(key).equals(x.sort(key))
    # same rows as reading every leaf wide
    wide = pp.open(S25 / "sweep_psfbin").scan("tran1").collect()
    signals = [c for c in wide.columns if c not in ("temp", "rval", "time")]
    assert b.height == wide.height * len(signals)


def test_scan_long_pushdown_params_signal_and_mixed():
    path = S25 / "sweep_psfbin"
    r, rec = recorded(path)
    lf = r.scan_long("tran1")
    # parameters only: only the 3 leaves at 27 degrees are read, all signals
    d = lf.filter(pl.col("temp") == 27).collect()
    assert len(jobs(rec)) == 3 and all(n is None for _, n in jobs(rec))
    assert d["temp"].unique().to_list() == [27.0]
    # signal only: every leaf, one signal
    rec.clear()
    d = lf.filter(pl.col("signal") == "n5").collect()
    assert len(jobs(rec)) == 9 and all(n == ["n5"] for _, n in jobs(rec))
    assert d["signal"].cast(pl.String).unique().to_list() == ["n5"]
    # mixed: per-leaf signal lists from the (leaves x names) table
    rec.clear()
    pred = ((pl.col("temp") == 27) & (pl.col("signal") == "n1")) | (
        (pl.col("rval") == 2000) & (pl.col("signal") == "n2")
    )
    d = lf.filter(pred).collect()
    assert 0 < len(jobs(rec)) <= 5 and all(len(n) <= 2 for _, n in jobs(rec))
    full = pp.open(path).scan_long("tran1").collect().filter(pred)
    key = ["temp", "rval", "signal", "time"]
    assert d.sort(key).equals(full.sort(key))


def test_scan_long_query_across_corners():
    # max |n10| per corner, and AC magnitude at the output via the cx namespace
    lf = pp.open(S25 / "sweep_psfbin").scan_long("tran1")
    peak = (lf.filter(pl.col("signal") == "n10")
              .group_by("temp", "rval").agg(pl.col("value").abs().max().alias("peak"))
              .sort("temp", "rval").collect())
    assert peak.height == 9 and peak["peak"].min() > 0
    ac = (pp.open(S25 / "sweep_psfxl").scan_long("ac1")
            .filter(pl.col("signal") == "n10")
            .with_columns(db=pl.col("value").cx.db20())
            .collect())
    assert ac.schema["value"] == pl.Struct({"re": pl.Float64, "im": pl.Float64})
    assert ac.select("temp", "rval").n_unique() == 9


def test_result_required_when_several():
    r = pp.open(TD / "pycircuit/pardcsweep.raw")
    with pytest.raises(ValueError, match="dc1"):
        r.scan()
    assert r.scan("sweep1_dc1").collect_schema() == r.scan("dc1").collect_schema()
    assert r.scan("dc1").collect_schema().names()[:3] == ["vdc3", "vdc2", "vdc1"]


def test_short_names_unless_shared():
    names = pp.open(TD / "psf-parser/binary/logFile").results["name"].to_list()
    assert "ac2" in names and "mysweep_ac2" not in names
    # plain tran1 and Monte Carlo mymonte_tran1 share the short name: family names are kept
    assert "tran1" in names and "mymonte_tran1" in names
    assert pp.open(TD / "spectre25/sweep_psfxl").results["name"].to_list() == ["tran1", "ac1"]


def test_params_match_rounded_logfile_values():
    r = pp.open(TD / "pycircuit/pardcsweep.raw")
    sig = r.names("dc1")[0]
    w = r.wave(sig, "dc1", vdc3=10, vdc2=3.66667)  # logFile shows 3.66667; sweep file 3.666...65
    ref = r.scan("dc1").filter((pl.col("vdc3") == 10) & (pl.col("vdc2") == 3.6666666666666665)).select(sig).collect()
    assert w.y.equals(ref[sig])
    with pytest.raises(KeyError):
        r.wave(sig, "dc1", vdc3=10, vdc2=3.7)


def test_contains_and_repr():
    r = pp.open(TD / "spectre25/sweep_psfbin")
    assert "tran1" in r and "swp_t_tran1" in r and "nope" not in r
    assert repr(r).startswith("Dataset(") and "dir" in repr(r)
    assert "file" in repr(pp.open(TD / "psf-parser/binary/myac.ac"))
