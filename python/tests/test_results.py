from pathlib import Path

import polars as pl
import pytest

import polars_psf as pp

TD = Path(__file__).resolve().parents[2] / "testdata"


def test_spectre_nested_sweep_read():
    r = pp.results(TD / "pycircuit/pardcsweep.raw")
    a = r.analyses
    assert a["name"].to_list() == ["sweep1_dc1", "sweep1_dc2"]
    assert a["params"][0].to_list() == ["vdc3", "vdc2"]
    df = r.read("dc1")
    assert df.columns[:3] == ["vdc3", "vdc2", "vdc1"]
    assert df.height == 16 * 4
    assert 3.6666666666666665 in df["vdc2"].unique().to_list()
    assert r.warnings == []


class Recorder:
    """Proxy around the native reader that records which leaves are read."""

    def __init__(self, inner):
        self.inner, self.read = inner, []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def read_leaves(self, analysis, idx, names=None):
        self.read.append((list(idx), names))
        return self.inner.read_leaves(analysis, idx, names)


def test_scan_pushdown_skips_files_and_signals():
    r = pp.results(TD / "pycircuit/pardcsweep.raw")
    rec = Recorder(r._r)
    r._r = rec
    trace = pp.open(r.leaves("dc1")["path"][0]).names[0]
    out = r.scan("dc1").filter(pl.col("vdc3") == 10).select("vdc2", "vdc1", trace).collect()
    assert out.height == 4 * 4
    assert out.columns == ["vdc2", "vdc1", trace]
    read_idx = [i for idx, _ in rec.read for i in idx]
    assert len(read_idx) == 4  # only leaves with vdc3 == 10
    assert all(names == [trace] for _, names in rec.read)
    assert (out["vdc2"].unique().sort() == r.read("dc1").filter(pl.col("vdc3") == 10)["vdc2"].unique().sort()).all()


def test_scan_matches_read_and_head():
    r = pp.results(TD / "pycircuit/pardcsweep.raw")
    full = r.read("dc2")
    assert r.scan("dc2").collect().equals(full)
    assert r.scan("dc2").head(5).collect().height == 5


def test_monte_carlo_tran():
    r = pp.results(TD / "psf-parser/binary/logFile")
    df = r.read("mymonte_tran1", ["out"])
    assert df.columns == ["iteration", "time", "out"]
    assert df["iteration"].unique().sort().to_list() == [1.0, 2.0, 3.0]
    per = df.group_by("iteration").len().sort("iteration")["len"].to_list()
    assert len(set(per)) > 1  # adaptive time steps differ per iteration: long format, no grid


def test_nested_ac_complex():
    r = pp.results(TD / "psf-parser/binary/logFile")
    sig = pp.open(r.leaves("ac2")["path"][0]).names[0]
    df = r.scan("ac2").select("R1:r", "C1:c", "freq", pl.col(sig).cx.db20().alias("db")).collect()
    assert df.select(pl.struct("R1:r", "C1:c").n_unique()).item() == 9


def test_ade_parametric_nonswept_op():
    r = pp.results(TD / "pycircuit/resultdirs/parsweep/psf")
    assert r.warnings == []  # nothing resolved yet
    df = r.read("opBegin")
    assert df.height == 9
    assert df.columns[:2] == ["VDC1", "VDC2"]
    assert r.leaves("srcSweep").height == 1
    assert sum("missing data file" in w for w in r.warnings) == 2


def test_unknown_analysis():
    r = pp.results(TD / "pycircuit/pardcsweep.raw")
    with pytest.raises(KeyError):
        r.read("nope")


S25 = TD / "spectre25"


class JobRecorder:
    """Proxy around the native reader recording (leaf, signals) jobs of read_leaves_long."""

    def __init__(self, inner):
        self.inner, self.jobs = inner, []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def read_leaves_long(self, analysis, jobs, field=None):
        self.jobs += jobs
        return self.inner.read_leaves_long(analysis, jobs, field)


def recorded(path):
    r = pp.results(path)
    r._r = JobRecorder(r._r)
    return r, r._r


def test_scan_long_across_sweep_matches_formats():
    b = pp.results(S25 / "sweep_psfbin").read_long("tran1")
    x = pp.results(S25 / "sweep_psfxl").read_long("tran1")
    assert b.columns == ["temp", "rval", "time", "signal", "value"]
    assert b.select("temp", "rval").n_unique() == 9
    key = ["temp", "rval", "signal", "time"]
    assert b.sort(key).equals(x.sort(key))
    # same rows as reading every leaf wide
    wide = pp.results(S25 / "sweep_psfbin").read("tran1")
    signals = [c for c in wide.columns if c not in ("temp", "rval", "time")]
    assert b.height == wide.height * len(signals)


def test_scan_long_pushdown_params_signal_and_mixed():
    path = S25 / "sweep_psfbin"
    r, rec = recorded(path)
    lf = r.scan_long("tran1")
    # parameters only: only the 3 leaves at 27 degrees are read, all signals
    d = lf.filter(pl.col("temp") == 27).collect()
    assert len(rec.jobs) == 3 and all(n is None for _, n in rec.jobs)
    assert d["temp"].unique().to_list() == [27.0]
    # signal only: every leaf, one signal
    rec.jobs.clear()
    d = lf.filter(pl.col("signal") == "n5").collect()
    assert len(rec.jobs) == 9 and all(n == ["n5"] for _, n in rec.jobs)
    assert d["signal"].cast(pl.String).unique().to_list() == ["n5"]
    # mixed: per-leaf signal lists from the (leaves x names) table
    rec.jobs.clear()
    pred = ((pl.col("temp") == 27) & (pl.col("signal") == "n1")) | ((pl.col("rval") == 2000) & (pl.col("signal") == "n2"))
    d = lf.filter(pred).collect()
    assert 0 < len(rec.jobs) <= 5 and all(len(n) <= 2 for _, n in rec.jobs)
    full = pp.results(path).read_long("tran1").filter(pred)
    key = ["temp", "rval", "signal", "time"]
    assert d.sort(key).equals(full.sort(key))


def test_scan_long_query_across_corners():
    # max |n10| per corner, and AC magnitude at the output via the cx namespace
    lf = pp.results(S25 / "sweep_psfbin").scan_long("tran1")
    peak = (lf.filter(pl.col("signal") == "n10")
              .group_by("temp", "rval").agg(pl.col("value").abs().max().alias("peak"))
              .sort("temp", "rval").collect())
    assert peak.height == 9 and peak["peak"].min() > 0
    ac = (pp.results(S25 / "sweep_psfxl").scan_long("ac1")
            .filter(pl.col("signal") == "n10")
            .with_columns(db=pl.col("value").cx.db20())
            .collect())
    assert ac.schema["value"] == pl.Struct({"re": pl.Float64, "im": pl.Float64})
    assert ac.select("temp", "rval").n_unique() == 9
