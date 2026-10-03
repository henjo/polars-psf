"""Calculator functions: ``db20(w)``, ``bandwidth(w, 3, "low")``, ``rise_time(w)``.

Each forwards to the :class:`Waveform` method of the same name, so it works per curve of a
family (corners, Monte Carlo) and returns one row per curve there. The semantics follow OCEAN:
``xmax`` is the x at the largest y, and ``cross`` counts edges from 1 with
``"rising"``/``"falling"``/``"either"``. The OCEAN names (``dB20``, ``unityGainFreq``,
``riseTime``, ``leafValue``, ...) and pycircuit names (``unityGainFrequency``, ``IIP3``, ...) are
aliases.

Arguments follow OCEAN's order where it has positional ones; OCEAN's ``initType``/``finalType``
flags (initial value given as an x instead of a y) are not supported: pass y values.
"""

from __future__ import annotations

import math

from .waveform import Waveform

__all__ = [
    # results
    "wave", "waves",
    # calculator functions
    "db10", "db20", "mag", "phase", "phase_deg", "phase_rad", "real", "imag", "conjugate",
    "value", "ymax", "ymin", "xmax", "xmin", "average", "rms", "stddev",
    "cross", "deriv", "integ", "iinteg", "clip", "dft", "leaf_value",
    "bandwidth", "unity_gain_frequency", "phase_margin", "gain_margin",
    "rise_time", "fall_time", "slew_rate", "overshoot", "settling_time", "delay", "frequency", "period",
    "im2", "im3", "iip2", "iip3", "calc_extrapolation_line", "compression_point",
    # OCEAN aliases
    "dB10", "dB20", "phaseDeg", "phaseRad", "leafValue", "unityGainFreq", "phaseMargin", "gainMargin",
    "riseTime", "fallTime", "slewRate", "settlingTime",
    # pycircuit aliases
    "unityGainFrequency", "IM2", "IM3", "IIP2", "IIP3",
]


def wave(source, name: str, result: str | None = None, **params) -> Waveform:
    """One signal as a lazy :class:`Waveform`; ``source`` is a path or :class:`Dataset`."""
    return _dataset(source).wave(name, result, **params)


def waves(source, names=None, result: str | None = None, **params) -> dict:
    """``{name: Waveform}`` for several signals (decoded together on first access)."""
    return _dataset(source).waves(names, result, **params)


def _dataset(source):
    from ..dataset import Dataset

    return source if isinstance(source, Dataset) else Dataset(source)


# --- elementwise -----------------------------------------------------------------------------
def db20(w: Waveform) -> Waveform:
    return w.db20()


def db10(w: Waveform) -> Waveform:
    return w.db10()


def mag(w: Waveform) -> Waveform:
    return w.mag()


def phase(w: Waveform) -> Waveform:
    """Phase in degrees (OCEAN ``phase``)."""
    return w.phase()


def phase_deg(w: Waveform) -> Waveform:
    return w.phase(deg=True)


def phase_rad(w: Waveform) -> Waveform:
    return w.phase(deg=False)


def real(w: Waveform) -> Waveform:
    return w.real()


def imag(w: Waveform) -> Waveform:
    return w.imag()


def conjugate(w: Waveform) -> Waveform:
    return w.conj()


def deriv(w: Waveform) -> Waveform:
    return w.deriv()


def iinteg(w: Waveform) -> Waveform:
    return w.iinteg()


def clip(w: Waveform, xfrom: float, xto: float | None = None) -> Waveform:
    return w.clip(xfrom, xto)


def dft(w: Waveform) -> Waveform:
    return w.dft()


def leaf_value(w: Waveform, **values) -> Waveform:
    """One leaf of a family: ``leaf_value(w, temp=27, rval=1000)``."""
    return w.leaf(**values)


# --- measurements (scalar per curve) ----------------------------------------------------------
def value(w: Waveform, x):
    return w.value(x)


def ymax(w: Waveform):
    return w.ymax()


def ymin(w: Waveform):
    return w.ymin()


def xmax(w: Waveform):
    """x where y is largest."""
    return w.xmax()


def xmin(w: Waveform):
    """x where y is smallest."""
    return w.xmin()


def average(w: Waveform):
    return w.average()


def rms(w: Waveform):
    return w.rms()


def stddev(w: Waveform):
    return w.stddev()


def integ(w: Waveform, xfrom: float | None = None, xto: float | None = None):
    return w.integ(xfrom, xto)


def cross(w: Waveform, threshold: float = 0.0, edge: int = 1, type: str = "either"):
    """x of the ``edge``-th crossing of ``threshold`` (from 1; negative counts from the end)."""
    return w.cross(threshold, edge, type)


def bandwidth(w: Waveform, db: float = 3.0, type: str = "low"):
    return w.bandwidth(db, type)


def unity_gain_frequency(w: Waveform):
    return w.unity_gain_frequency()


def phase_margin(w: Waveform):
    return w.phase_margin()


def gain_margin(w: Waveform):
    return w.gain_margin()


def rise_time(w: Waveform, initial=None, final=None, theta1: float = 10.0, theta2: float = 90.0):
    """Rise time (OCEAN ``riseTime(wave initVal nil finalVal nil theta1 theta2)``)."""
    return w.rise_time(theta1, theta2, initial, final)


def fall_time(w: Waveform, initial=None, final=None, theta1: float = 10.0, theta2: float = 90.0):
    return w.fall_time(theta1, theta2, initial, final)


def slew_rate(w: Waveform, initial=None, final=None, theta1: float = 10.0, theta2: float = 90.0):
    return w.slew_rate(theta1, theta2, initial, final)


def overshoot(w: Waveform, initial=None, final=None):
    return w.overshoot(initial, final)


def settling_time(w: Waveform, initial=None, final=None, tolerance: float = 1.0):
    return w.settling_time(tolerance, initial, final)


def delay(
    wf1: Waveform,
    wf2: Waveform,
    value1: float = 0.0,
    value2: float | None = None,
    edge1: str = "either",
    nth1: int = 1,
    edge2: str | None = None,
    nth2: int = 1,
):
    """x of ``wf2``'s ``nth2`` crossing of ``value2`` minus that of ``wf1`` (OCEAN ``delay``
    keyword names)."""
    return wf1.delay(wf2, value1, value2, nth1, nth2, edge1, edge2)


def frequency(w: Waveform, threshold: float | None = None):
    return w.frequency(threshold)


def period(w: Waveform, threshold: float | None = None):
    return w.period(threshold)


# --- RF (pycircuit) ---------------------------------------------------------------------------
def im3(w: Waveform, fund1: float, fund2: float, fund0=None) -> float:
    """Third-order intermodulation tone at ``fund1 + 2*fund2`` of an output spectrum.

    ``fund0`` (the LO of a mixer) is accepted for pycircuit compatibility; output tones are
    measured at IF, so it does not shift them (only :func:`iip3` uses it, for the input)."""
    return abs(w).value(fund1 + 2 * fund2)


def im2(w: Waveform, fund1: float, fund2: float, fund0=None) -> float:
    """Second-order intermodulation tone at ``fund1 + fund2`` of an output spectrum (``fund0``:
    see :func:`im3`)."""
    return abs(w).value(fund1 + fund2)


def iip3(output: Waveform, input: Waveform, fund1: float, fund2: float, fund0=None) -> float:
    """Input-referred third-order intercept point."""
    s = abs(output)
    if fund0 is None:
        gain = (s / abs(input)).value(fund1)
    else:
        gain = s.value(abs(fund1)) / abs(input).value(abs(abs(fund1) + fund0))
    return math.sqrt(s.value(abs(fund1)) * s.value(abs(fund2)) ** 2 / s.value(fund1 + 2 * fund2)) / gain


def iip2(output: Waveform, input: Waveform, fund1: float, fund2: float, fund0=None) -> float:
    """Input-referred second-order intercept point."""
    s = abs(output)
    if fund0 is None:
        gain = (s / abs(input)).value(fund1)
    else:
        gain = s.value(abs(fund1)) / abs(input).value(abs(abs(fund1) + fund0))
    return s.value(abs(fund1)) * s.value(abs(fund2)) / s.value(fund1 + fund2) / gain


def calc_extrapolation_line(
    w_db: Waveform, slope: float = 1.0, extrapolation_point: float | None = None, **_
) -> Waveform:
    """Linear (in dB) extrapolation of ``w_db`` through its low end (smallest x)."""
    if extrapolation_point is None:
        extrapolation_point = w_db.x.min()
    m = w_db.value(extrapolation_point) - slope * extrapolation_point
    return m + slope * w_db.xval()


def compression_point(
    w_db: Waveform, slope: float = 1.0, compression: float = 1.0, extrapolation_point=None, **_
) -> float:
    """Input-referred compression point: where the response falls ``compression`` dB below the
    extrapolated small-signal line."""
    line = calc_extrapolation_line(w_db, slope, extrapolation_point)
    return (line - w_db).cross(compression)


# --- aliases ----------------------------------------------------------------------------------
# OCEAN
dB20, dB10 = db20, db10
phaseDeg, phaseRad = phase_deg, phase_rad
leafValue = leaf_value
unityGainFreq = unity_gain_frequency
phaseMargin, gainMargin = phase_margin, gain_margin
riseTime, fallTime, slewRate, settlingTime = rise_time, fall_time, slew_rate, settling_time
# pycircuit
unityGainFrequency = unity_gain_frequency
IM2, IM3, IIP2, IIP3 = im2, im3, iip2, iip3
