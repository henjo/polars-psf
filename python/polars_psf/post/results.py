"""``wave``/``waves`` free functions on PSF files and result directories."""

from __future__ import annotations

from polars_waveform import Waveform

__all__ = ["wave", "waves"]


def wave(source, name: str, result: str | None = None, **params) -> Waveform:
    """One signal as a lazy :class:`Waveform`; ``source`` is a path or :class:`Dataset`."""
    return _dataset(source).wave(name, result, **params)


def waves(source, names=None, result: str | None = None, **params) -> dict:
    """``{name: Waveform}`` for several signals (decoded together on first access)."""
    return _dataset(source).waves(names, result, **params)


def _dataset(source):
    from ..dataset import Dataset

    return source if isinstance(source, Dataset) else Dataset(source)
