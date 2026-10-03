from pathlib import Path

import polars as pl
import pytest

from polars_psf.cli import main

TD = Path(__file__).resolve().parents[2] / "testdata"
SWEEP = str(TD / "spectre25/sweep_psfbin")


def test_info_and_names(capsys):
    assert main(["info", SWEEP]) == 0
    out = capsys.readouterr().out
    assert "result directory" in out and "tran1" in out and "temp, rval" in out
    assert main(["info", str(TD / "pycircuit/resultdirs/parsweep/psf")]) == 0
    assert "schematic name map" in capsys.readouterr().out
    assert main(["names", SWEEP, "tran1", "--grep", "^n1"]) == 0
    assert capsys.readouterr().out.split() == ["n1", "n10"]
    assert main(["names", SWEEP]) == 1  # two results: name one
    assert "pass a result" in capsys.readouterr().err


def test_export_formats(tmp_path):
    out = tmp_path / "t.parquet"
    assert main(["export", SWEEP, str(out), "-r", "tran1", "-s", "n10", "-s", "n1", "-w", "temp=27"]) == 0
    df = pl.read_parquet(out)
    assert df.columns == ["rval", "time", "n10", "n1"] and df["rval"].n_unique() == 3
    csv = tmp_path / "ac.csv"
    assert main(["export", SWEEP, str(csv), "-r", "ac1", "-s", "n10", "-w", "temp=27", "-w", "rval=1000"]) == 0
    assert pl.read_csv(csv).columns == ["freq", "n10.re", "n10.im"]  # complex split for CSV
    long = tmp_path / "long.ipc"
    assert main(["export", SWEEP, str(long), "-r", "tran1", "--long", "-w", "temp=27"]) == 0
    assert pl.read_ipc(long).columns == ["temp", "rval", "time", "signal", "value"]


def test_export_rejects_unknown_format(tmp_path):
    with pytest.raises(SystemExit):
        main(["export", SWEEP, str(tmp_path / "x.xyz"), "-r", "tran1"])
