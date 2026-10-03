"""Numpy-style post-processing on top of the polars-psf readers.

A :class:`Waveform` is a value column over one or more index columns, backed by a ``pl.LazyFrame``
(and materialized into a ``pl.DataFrame`` only when a row value is needed).  It gives the
imperative ergonomics of pycircuit's ``post`` module while every operation stays in Polars::

    w = pp.wave("ac.ac", "out")          # or pp.open("ac.ac").wave("out")
    w = pp.wave("ac.ac", "out") / pp.wave("ac.ac", "in")    # transfer function
    w.db20(), w.phase(), w.bandwidth(), w.unity_gain_frequency(), w.phase_margin()
    w.value(1e6), w.cross(0.0), w.deriv(), w.rms()

Elementwise and structure-changing operations (``+ - * / ** abs phase real imag conj deriv
filter clip``) build a lazy query plan; nothing is decoded until a value observation (``.x``,
``.y``, ``len``, ``to_polars``, plots, reductions, resampling) needs rows.  Complex signals
(AC/noise) are ``Struct{re, im}`` columns; ``+ - * / **``, ``abs``, ``phase``, ``real``, ``imag``
and ``conj`` are built on the ``re``/``im`` fields.  Mixed real/complex arithmetic is supported.
The standalone functions mirror ``pycircuit.post.functions``: :func:`db20`, :func:`phase`,
:func:`cross`, :func:`bandwidth`, :func:`unity_gain_frequency`, :func:`phase_margin`,
:func:`im2`/:func:`im3`, :func:`iip2`/:func:`iip3`, :func:`compression_point`, :func:`average`,
:func:`rms`, :func:`stddev`, :func:`deriv`, :func:`clip`.

Differences from ``pycircuit.post``: no plotting of its own -- :meth:`Waveform.plot` delegates to
Polars' ``DataFrame.plot`` (Altair, requires ``polars[plot]``); a waveform may hold several curves
(index columns besides the sampling axis, e.g. Monte Carlo iterations or corners), and every
operation works per curve: measurements then return one row per curve; ``cross`` interpolates
the bracketing samples linearly, which is exactly what post's ``brenth(interp1d(...))`` converges
to.

Modules: :mod:`.waveform` (the class), :mod:`.functions` (the free functions).

A ``Waveform`` from :meth:`Dataset.wave` is itself lazy: it wraps a ``Dataset.scan`` plan
projected to one signal, so ``pp.wave("ac.ac", "out")`` costs a header read and defers decoding
until the first value access.
"""

from .functions import *  # noqa: F403
from .functions import __all__ as _functions
from .waveform import Waveform, either, falling, raising

__all__ = ["Waveform", "raising", "falling", "either", *_functions]
