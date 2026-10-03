import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from conftest import recorded

import polars_psf as pp

TD = Path(__file__).resolve().parents[2] / "testdata"


def complex_waveform(x, y):
    """Build a Waveform with a Struct{re, im} value column from complex numpy data."""
    y = np.asarray(y)
    df = pl.DataFrame(
        {
            "x": np.asarray(x, dtype=float),
            "y": pl.DataFrame({"re": y.real, "im": y.imag}).to_struct("y"),
        }
    )
    return pp.Waveform(df, "y")


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


def test_synthetic_ops_match_numpy():
    x = pl.Series([0.0, 1.0, 2.0, 3.0])
    y = pl.Series([1.0, 1.0, -1.0, -1.0])
    w = pp.Waveform(pl.DataFrame({"t": x, "v": y}), "v", units={"v": "V"})

    assert w.cross() == 1.5
    assert math.isnan(w.cross(type="rising"))
    assert w.cross(type="falling") == 1.5
    assert w.cross(0.0, 1, pp.falling) == 1.5  # pycircuit constants still accepted
    with pytest.raises(ValueError):
        w.cross(0.0, 0)  # OCEAN edges count from 1
    # OCEAN: xmax/xmin are the x at the extreme y (argmax/argmin are aliases)
    assert (w.ymax(), w.ymin(), w.xmax(), w.xmin(), w.argmax(), w.argmin()) == (1.0, -1.0, 0.0, 2.0, 0.0, 2.0)

    assert w.value(0.5) == 1.0  # between samples
    assert w.value(-10.0) == 1.0 and w.value(99.0) == -1.0  # clamped
    assert w.value(1.5) == 0.0
    assert w.value([0.5, 1.5]).y.to_list() == [1.0, 0.0]

    assert w.rms() == pytest.approx(np.sqrt(np.mean(np.array([1.0, 1.0, -1.0, -1.0]) ** 2)))
    assert w.average() == 0.0
    assert w.stddev() == pytest.approx(np.std([1.0, 1.0, -1.0, -1.0]))

    assert (w * 2).y.to_list() == [2.0, 2.0, -2.0, -2.0]
    assert (2 - w).y.to_list() == [1.0, 1.0, 3.0, 3.0]
    assert (w**2).y.to_list() == [1.0, 1.0, 1.0, 1.0]
    assert (-w).y.to_list() == [-1.0, -1.0, 1.0, 1.0]
    assert abs(w).y.to_list() == [1.0, 1.0, 1.0, 1.0]
    assert (w > 0).y.to_list() == [True, True, False, False]
    assert pp.value(w, 1.5) == 0.0 and pp.ymax(w) == 1.0

    # derivatives are per-interval (forward difference): one point fewer
    d = w.deriv()
    assert len(d) == 3 and d.y.to_list() == [0.0, -2.0, 0.0]

    # clip interpolates both limits
    c = w.clip(0.5, 2.5)
    assert c.x.to_list() == [0.5, 1.0, 2.0, 2.5]
    assert c.y.to_list() == [1.0, 1.0, -1.0, -1.0]
    assert w.clip(0.0).x.to_list() == [0.0, 1.0, 2.0, 3.0]


def test_x_mismatch_rejected():
    a = pp.Waveform(pl.DataFrame({"t": [0.0, 1.0], "v": [0.0, 1.0]}), "v")
    b = pp.Waveform(pl.DataFrame({"t": [0.0, 2.0], "v": [0.0, 1.0]}), "v")
    with pytest.raises(ValueError):
        a + b


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


def test_phase_margin_doctest():
    w = 2 * np.pi * np.logspace(3, 8, 41)
    H = 1.5 * (1 / (1 - 1j * w / -1e6)) ** 2
    assert f"{complex_waveform(w, H).phase_margin():.4g}" == "110.4"


def test_bandwidth_doctest():
    w = 2 * np.pi * np.logspace(3, 8)
    H = 1 / (1 - 1j * w / -1e6)
    assert pp.bandwidth(complex_waveform(w, H)) == pytest.approx(1000896.9666087811)


def test_dft_single_sided_amplitude():
    t = np.arange(0, 1.0, 1 / 64.0)
    w = pp.Waveform(pl.DataFrame({"t": t, "v": np.sin(2 * np.pi * 8 * t)}), "v")
    d = w.dft()
    assert d.x[d.y.arg_max()] == pytest.approx(8.0)
    assert d.y.max() == pytest.approx(1.0)
    assert d.xname == "freq"
    with pytest.raises(ValueError):
        pp.Waveform(pl.DataFrame({"t": [0.0, 0.1, 0.3], "v": [0.0, 1.0, 0.0]}), "v").dft()


def test_intermodulation_and_compression():
    freq = pl.Series([1e3, 2e3, 3e3, 5e3])
    out = pp.Waveform(pl.DataFrame({"f": freq, "out": [0.1, 0.02, 0.03, 0.005]}), "out")
    inp = pp.Waveform(pl.DataFrame({"f": freq, "in": [0.01] * 4}), "in")
    gain = out.value(1e3) / inp.value(1e3)
    assert pp.iip3(out, inp, 1e3, 2e3) == pytest.approx(math.sqrt(0.1 * 0.02**2 / 0.005) / gain)
    assert pp.iip2(out, inp, 1e3, 2e3) == pytest.approx(0.1 * 0.02 / 0.03 / gain)
    assert pp.im3(out, 1e3, 2e3) == pytest.approx(0.005)
    assert pp.im2(out, 1e3, 2e3) == pytest.approx(0.03)

    # response compresses 1 dB below the 1 dB/dB line through the first point
    x = pl.Series([5.0, 10.0, 15.0, 20.0])
    y = pl.Series([20.0, 20.0, 15.0, 10.0])
    w_db = pp.Waveform(pl.DataFrame({"pin": x, "gain": y}), "gain")
    assert pp.compression_point(w_db) == pytest.approx(6.0)  # line = 15 + pin


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

    w = pp.Waveform(pl.DataFrame({"t": [0.0, 1.0], "v": [0.0, 1.0]}), "v")
    with pytest.raises(ValueError):
        pp.Waveform.from_series(pl.Series([0.0, 1.0]), pl.Series([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError):
        w.clip(5.0, 1.0)  # xfrom > xto
    with pytest.raises(ValueError):
        pp.Waveform(pl.DataFrame({"t": [0.0], "v": [1.0]}), "v").deriv()  # too few points


def test_multi_index_carries_extra_columns():
    df = pl.DataFrame(
        {
            "iteration": [0, 0, 0, 1, 1, 1],
            "time": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "out": [0.0, 1.0, 0.0, 0.0, 2.0, 0.0],
        }
    )
    w = pp.Waveform(df, "out", index=["iteration", "time"])
    assert w.index == ["iteration", "time"]
    assert w.xname == "time" and w.yname == "out" and len(w) == 6
    assert w.to_polars().columns == ["iteration", "time", "out"]
    assert w.x.to_list() == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]  # sampling axis, not iteration

    assert w.groups == ["iteration"]

    d = w.deriv()  # within each iteration: no difference across the boundary
    assert d.index == ["iteration", "time"] and len(d) == 4
    assert d.to_polars()["iteration"].to_list() == [0, 0, 1, 1]
    assert d.y.to_list() == pytest.approx([1.0, -1.0, 2.0, -2.0])

    assert w.filter(pl.col("iteration") == 1).y.to_list() == [0.0, 2.0, 0.0]
    assert abs(w).to_polars()["iteration"].to_list() == [0, 0, 0, 1, 1, 1]

    # measurements: one row per iteration
    assert w.ymax().rows() == [(0, 1.0), (1, 2.0)]
    assert w.argmax().rows() == [(0, 1.0), (1, 1.0)]
    assert w.cross(0.5).rows() == [(0, 0.5), (1, 0.25)]
    assert w.cross(0.5, edge=-1).rows() == [(0, 1.5), (1, 1.75)]
    assert w.value(1.5).rows() == [(0, 0.5), (1, 1.0)]
    assert w.rms().columns == ["iteration", "rms"]
    r = w.value([0.5, 1.5])  # resampled per iteration
    assert r.index == ["iteration", "time"] and r.to_polars().rows() == [
        (0, 0.5, 0.5), (0, 1.5, 0.5), (1, 0.5, 1.0), (1, 1.5, 1.0)
    ]
    c = w.clip(0.5, 1.0)  # clipped per iteration, edges interpolated in each
    assert c.to_polars().rows() == [(0, 0.5, 0.5), (0, 1.0, 1.0), (1, 0.5, 1.0), (1, 1.0, 2.0)]

    # clamping at the ends keeps the other index columns of the boundary row
    wi = w.filter(pl.col("iteration") == 1)  # per-iteration sweep is monotonic
    c = wi.clip(0.5, 1.5)
    assert c.index == ["iteration", "time"] and c.x.to_list() == [0.5, 1.0, 1.5]
    assert c.to_polars()["iteration"].to_list() == [1, 1, 1]


def test_plot_delegates_to_polars():
    alt = pytest.importorskip("altair")
    w = pp.Waveform(
        pl.DataFrame({"iteration": [0, 0, 1, 1], "t": [0.0, 1.0, 0.0, 1.0], "v": [0.0, 1.0, 0.0, 2.0]}),
        "v",
        index=["iteration", "t"],
    )
    spec = w.plot(color="iteration").to_dict()
    assert spec["mark"]["type"] == "line"
    assert spec["encoding"]["x"]["field"] == "t" and spec["encoding"]["y"]["field"] == "v"
    assert spec["encoding"]["color"]["field"] == "iteration"
    assert w.plot(mark="point").to_dict()["mark"]["type"] == "point"
    assert isinstance(w.plot(), alt.Chart)

    with pytest.raises(ValueError):  # struct/complex values must be reduced first
        complex_waveform([0.0, 1.0], [1.0 + 0j, 2.0 + 0j]).plot()


def test_bandwidth_low_and_high():
    f = pl.Series([1.0, 2.0, 3.0, 4.0])
    lp = pp.Waveform(pl.DataFrame({"f": f, "h": [1.0, 1.0, 0.5, 0.5]}), "h")
    hp = pp.Waveform(pl.DataFrame({"f": f, "h": [0.5, 0.5, 1.0, 1.0]}), "h")
    g = 10 ** (-3 / 20)
    assert lp.bandwidth() == pytest.approx(2 + (1 - g) / 0.5)
    assert hp.bandwidth(type="high") == pytest.approx(2 + (g - 0.5) / 0.5)
    bp = pp.Waveform(pl.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0], "h": [0.1, 0.5, 1.0, 0.5, 0.1]}), "h")
    lo, hi = 2 + (g - 0.5) / 0.5, 3 + (1 - g) / 0.5
    assert bp.bandwidth(type="band") == pytest.approx(hi - lo)
    with pytest.raises(ValueError):
        lp.bandwidth(type="bogus")


def test_binop_reads_only_index_until_values_are_needed():
    from conftest import recorded

    d, rec = recorded(TD / "psf-parser/binary/myac.ac")
    h = d.wave("out") / d.wave("in")
    assert [names for _, _, names in rec.calls] == [[], []]  # sweep checks only
    assert h.is_complex and len(h.y) == len(d.wave("out").y)


def test_broadcasting_per_curve_results():
    g = pp.Waveform(
        pl.DataFrame({"it": [0, 0, 1, 1], "t": [0.0, 1.0, 0.0, 1.0], "v": [1.0, 3.0, 10.0, 20.0]}),
        "v",
        index=["it", "t"],
    )
    assert (g - g.mean()).y.to_list() == [-1.0, 1.0, -5.0, 5.0]
    assert (g / g.ymax()).y.to_list() == [pytest.approx(1 / 3), 1.0, 0.5, 1.0]
    # DataFrame - Waveform is handled by polars' DataFrame.__sub__ first: put the waveform first
    assert (-(g - g.mean())).y.to_list() == [1.0, -1.0, 5.0, -5.0]
    assert (g - g.mean()).to_polars()["it"].to_list() == [0, 0, 1, 1]  # order kept
    with pytest.raises(ValueError):
        g - pl.DataFrame({"other": [0], "x": [1.0]})

    one = g.filter(pl.col("it") == 1)  # a single curve: measurements are scalars
    assert one.ymax() == 20.0 and one.cross(15.0) == 0.5
    assert (one - one.mean()).y.to_list() == [-5.0, 5.0]

    w = pp.Waveform(pl.DataFrame({"t": [0.0, 1.0, 2.0], "v": [1.0, 2.0, 6.0]}), "v")
    assert (w - w.mean()).y.to_list() == [-2.0, -1.0, 3.0]


def test_broadcasting_complex_per_curve():
    z = pl.DataFrame({"re": [1.0, 3.0, 2.0, 2.0], "im": [0.0, 2.0, 1.0, -1.0]}).to_struct("h")
    g = pp.Waveform(pl.DataFrame({"it": [0, 0, 1, 1], "f": [1.0, 2.0, 1.0, 2.0], "h": z}), "h", index=["it", "f"])
    c = (g - g.mean()).to_numpy()
    assert c == pytest.approx(np.array([-1 - 1j, 1 + 1j, 1j, -1j]))


def test_corner_bandwidth_in_one_line():
    r = pp.open(TD / "spectre25/sweep_psfbin")
    bw = r.wave("n10", "ac1").bandwidth()
    assert bw.columns == ["temp", "rval", "bandwidth"] and bw.height == 9
    one = r.wave("n10", "ac1", temp=27, rval=1000.0).bandwidth()
    assert bw.filter((pl.col("temp") == 27) & (pl.col("rval") == 1000.0))["bandwidth"].item() == pytest.approx(one)


def test_transient_measurements():
    t = np.linspace(0.0, 10.0, 10001)
    y = 1 - np.exp(-t) * np.cos(3 * t)  # damped step response towards 1
    w = pp.Waveform(pl.DataFrame({"t": t, "v": y}), "v")
    # reference values: first crossings, linearly interpolated between samples
    def first_cross(level):
        k = int(np.argmax(y >= level))
        return t[k - 1] + (level - y[k - 1]) / (y[k] - y[k - 1]) * (t[k] - t[k - 1])

    lo, hi = y[0] + 0.1 * (y[-1] - y[0]), y[0] + 0.9 * (y[-1] - y[0])
    t10, t90 = first_cross(lo), first_cross(hi)
    assert w.rise_time() == pytest.approx(t90 - t10, rel=1e-9)
    assert w.slew_rate() == pytest.approx((hi - lo) / (t90 - t10), rel=1e-6)
    assert w.overshoot() == pytest.approx((y.max() - y[-1]) / (y[-1] - y[0]) * 100, rel=1e-3)
    out = np.flatnonzero(np.abs(y - y[-1]) > 0.01 * (y[-1] - y[0]))
    assert w.settling_time() == pytest.approx(t[out[-1]], abs=2e-3)
    with pytest.raises(ValueError):
        w.fall_time()
    assert (-w).fall_time() == pytest.approx(w.rise_time())
    assert w.integ() == pytest.approx(np.trapezoid(y, t))
    assert w.integ(0.0, 5.0) == pytest.approx(np.trapezoid(y[t <= 5.0], t[t <= 5.0]), rel=1e-6)
    assert w.iinteg().y[-1] == pytest.approx(w.integ())

    s = pp.Waveform(pl.DataFrame({"t": t, "v": np.sin(2 * np.pi * 0.5 * t)}), "v")
    assert s.frequency() == pytest.approx(0.5, rel=1e-3) and s.period() == pytest.approx(2.0, rel=1e-3)
    shifted = pp.Waveform(pl.DataFrame({"t": t, "v": np.sin(2 * np.pi * 0.5 * (t - 0.25))}), "v")
    assert s.delay(shifted, 0.0, type="falling") == pytest.approx(0.25, abs=1e-9)  # 1.0 -> 1.25
    # a sample exactly on the threshold is one crossing; a curve starting on it has none
    assert [s.cross(0.0, e) for e in (1, 2)] == pytest.approx([1.0, 2.0])
    assert [shifted.cross(0.0, e) for e in (1, 2)] == pytest.approx([0.25, 1.25])


def test_gain_margin_and_leaf():
    def loop(f):
        s = 2j * np.pi * f
        return 100 / (1 + s / 2e4) / (1 + s / 2e6) / (1 + s / 2e7)  # three poles

    f = np.logspace(3, 9, 6001)
    gw = complex_waveform(f, loop(f))
    # exact phase-crossover (loop on the negative real axis) by bisection on the transfer function
    a, b = 1e6, 1e8  # Im(loop) changes sign here, with Re(loop) < 0
    for _ in range(100):
        m = (a + b) / 2
        a, b = (m, b) if np.sign(loop(m).imag) == np.sign(loop(a).imag) else (a, m)
    exact = -20 * np.log10(abs(loop(a)))
    assert gw.gain_margin() == pytest.approx(exact, abs=0.01)

    df = pl.DataFrame({"it": [0, 0, 1, 1], "t": [0.0, 1.0, 0.0, 1.0], "v": [1.0, 2.0, 3.0, 4.0]})
    g = pp.Waveform(df, "v", index=["it", "t"])
    one = g.leaf(it=1)
    assert one.index == ["t"] and one.y.to_list() == [3.0, 4.0]
    assert pp.leaf_value(g, it=0).y.to_list() == [1.0, 2.0]
    with pytest.raises(KeyError):
        g.leaf(temp=1)


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
