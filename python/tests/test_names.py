from pathlib import Path

import pytest

import polars_psf as pp
from polars_psf.names import NameMap

TD = Path(__file__).resolve().parents[2] / "testdata"
ADE = TD / "pycircuit/resultdirs/parsweep"


def test_ade_maps_next_to_the_results():
    r = pp.open(ADE / "psf")  # finds ../netlist/amap
    m = r.name_map
    assert m is not None and m.top == "PSFTEST" and m.delimiter == "."
    assert r.netlist_name("/vout") == "VOUT" and r.netlist_name("/gnd!") == "0"
    assert r.schematic_name("NET9") == "/net9"
    assert m.terminal("/R0/PLUS") == ("R0", 1) and m.terminal("/R0/MINUS") == ("R0", -1)
    assert m.terminal("/E0/NC+") == (None, 0)  # tied off by the netlister
    # operating point (complete in every run) by schematic name
    for vdc1, vdc2 in [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]:
        assert r.opBegin.value("/vout", VDC1=vdc1, VDC2=vdc2) == r.opBegin.value("VOUT", VDC1=vdc1, VDC2=vdc2)


def test_schematic_names_in_v_and_i():
    r = pp.open(ADE / "psf")
    w = r.srcSweep.v("/vout")
    assert w.yname == "/vout" and w.to_polars().columns[-1] == "/vout"
    plus, minus = r.srcSweep.i("/R0/PLUS"), r.srcSweep.i("/R0/MINUS")
    assert plus.yname == "/R0/PLUS" and minus.yname == "/R0/MINUS"
    assert plus.lazy.explain() != minus.lazy.explain()  # MINUS is the negated R0 current
    with pytest.raises(ValueError):
        r.srcSweep.i("/E0/NC+")
    with pytest.raises(KeyError):
        pp.open(TD / "spectre25/sweep_psfbin").tran1.v("/n10")  # no ADE maps there


def write_map(path: Path, entries: dict[str, str]) -> None:
    def q(s: str) -> str:  # PSF ASCII strings escape their quotes, as the real maps do
        return '"' + s.replace('"', '\\"') + '"'

    body = "".join(f"{q(k)} {q(v)}\n" for k, v in entries.items())
    path.write_text(f'HEADER\n"PSFversion" "1.00"\n{body}')


def test_hierarchical_paths(tmp_path):
    amap = tmp_path / "netlist" / "amap"
    amap.mkdir(parents=True)
    write_map(amap / "top_level_map.f.net", {"__NEXT_FILE__": "TOP"})
    write_map(amap / "__simulator_information__", {"__artHierarchyDelimiter__": "."})
    write_map(amap / "TOP.f.inst", {"I0v": "I0", "I0n": "AMP"})
    write_map(amap / "TOP.i.inst", {"I0v": "I0", "I0n": "AMP"})
    write_map(amap / "TOP.f.net", {"gnd!v": "0", "inv": "IN"})
    write_map(amap / "AMP.f.inst", {"M1v": "M1", "M1n": "nmos4"})
    write_map(amap / "AMP.f.net", {"voutv": "OUT"})
    write_map(amap / "AMP.i.net", {"OUTv": "vout"})
    write_map(amap / "nmos4.f.inst", {"Dv": ":d", "Sv": "(FUNCTION minus(root(\"D\")))"})
    m = NameMap.find(tmp_path / "psf")
    assert m.net("/I0/vout") == "I0.OUT" and m.net("/in") == "IN" and m.net("/I0/gnd!") == "0"
    assert m.net("/I0/unmapped") == "I0.unmapped"  # no entry: the name is kept
    assert m.schematic("I0.OUT") == "/I0/vout"
    assert m.terminal("/I0/M1/D") == ("I0.M1:d", 1) and m.terminal("/I0/M1/S") == ("I0.M1:d", -1)
    assert m.terminal("/I0/M1/G") == ("I0.M1:G", 1)  # no primitive entry: netlister default
    with pytest.raises(ValueError):
        m.terminal("/vout")
    assert NameMap.find(tmp_path / "elsewhere" / "deeper" / "x") is None
