"""Checks on Spectre samples from samplegen/run.sh (large, not in git). Skipped if absent."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import polars_psf as pp

OUT = Path(__file__).resolve().parents[2] / "samplegen" / "out"
pytestmark = pytest.mark.skipif(not (OUT / "sweep_psfascii").is_dir(), reason="run samplegen/run.sh first")


def num(df):
    cols = []
    for c, t in df.schema.items():
        if isinstance(t, pl.Struct):
            cols += [df[c].struct.field(f).cast(pl.Float64).to_numpy() for f in ("re", "im")]
        else:
            cols.append(df[c].cast(pl.Float64).to_numpy())
    return np.column_stack(cols)


def assert_close(a, b, rtol):
    assert a.columns == b.columns
    x, y = num(a), num(b)
    assert x.shape == y.shape
    scale = np.maximum(np.abs(x).max(axis=0), 1e-300)
    assert (np.abs(x - y) / scale).max() <= rtol


def test_interrupted_psfxl_is_prefix_of_long():
    k = pp.open(OUT / "psfxl_killed/tran1.tran.tran")
    long = pp.open(OUT / "psfxl_long/tran1.tran.tran")
    assert not k.is_complete and long.is_complete
    dk = k.to_polars()
    assert dk.equals(long.to_polars().head(dk.height))
    assert long.psfxl_meta["cdnshnumflushes"] == 7


@pytest.mark.parametrize("ext,rtol", [("psfxl_ac", 1e-12), ("psfbinf_ac", 1e-6)])
def test_formats_agree(ext, rtol):
    for name, asc in [("ac1.ac", "ac1.ac"), ("tran1.tran", "tran1.tran.tran")]:
        f = OUT / ext / name
        if not f.exists():
            f = OUT / ext / (name + ".tran")
        assert_close(pp.read_polars(f), pp.read_polars(OUT / "psfascii_ac" / asc), rtol)


def test_psfbinf_is_float32():
    df = pp.read_polars(OUT / "psfbinf_ac/ac1.ac")
    assert pl.Struct({"re": pl.Float32, "im": pl.Float32}) in df.schema.values()


@pytest.mark.parametrize("an", ["tran1", "ac1"])
def test_nested_sweeps_agree(an):
    ref = pp.results(OUT / "sweep_psfascii").read(an)
    assert ref.columns[:2] == ["temp", "rval"]
    assert ref.select("temp", "rval").n_unique() == 9
    for fmt in ("psfbin", "psfxl"):
        r = pp.results(OUT / f"sweep_{fmt}")
        assert_close(r.read(an), ref, 1e-12)
        assert r.warnings == []


def test_big_psfbin():
    f = pp.open(OUT / "psfbin_chunked/tran1.tran.tran")
    assert f.is_complete
    t = f.read_signal(f.names[0])["time"].to_numpy()
    assert len(t) == 1_000_010 and np.all(np.diff(t) > 0)
    assert all("\x00" not in str(v) for v in f.header.values())
