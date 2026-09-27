"""Lazy long-format scan of one PSF file, with signal selection pushed into the reader.

    lf = pp.scan_long("noise.pnoise", field="total")      # freq | signal | value
    (lf.filter(pl.col("signal").cast(pl.String).str.starts_with("xa1."))
       .group_by("freq").agg(pl.col("value").sum())
       .collect())

Polars hands the query's predicate to the source. If it only uses the ``signal`` column it is
evaluated on the list of signal names (one row per signal, not per point) and polars-psf decodes only
the matching signals. Other predicates are applied after reading. All query execution happens in
Polars' Rust engine; Python only wires the plan.
"""

from os import PathLike

import polars as pl
from polars.io.plugins import register_io_source

from ._polars_psf import PsfFile

__all__ = ["scan_long"]


def scan_long(
    source: "str | PathLike | PsfFile", field: str | None = None, *, chunk_rows: int = 1 << 22
) -> pl.LazyFrame:
    """LazyFrame ``[sweep, signal, value]`` over all traces of a swept file.

    ``field`` selects a struct member (e.g. ``"total"`` for noise contributions); traces without it
    are left out. ``signal`` is
    Categorical: compare with ``==``/``is_in`` directly, or ``cast(pl.String)`` for string functions.
    Signals are read in chunks of about ``chunk_rows`` rows (signals x sweep points), so memory
    stays bounded for long transients and ``head()`` stops early.
    """
    f = source if isinstance(source, PsfFile) else PsfFile(source)
    schema = pl.DataFrame(f.long_schema(field)).schema
    # with a field: only traces that have it (noise files also hold plain traces like "out")
    all_names = f.names_with_field(field) if field is not None else f.names
    names = pl.DataFrame({"signal": all_names}, schema={"signal": pl.Categorical})
    points = f.header.get("PSF sweep points") or 0
    if not points and f.is_psfxl:
        points = 1 << 20  # PSFXL stubs have no point count in the header; assume long transients
    chunk_signals = max(1, chunk_rows // max(1, points))

    def src(with_columns, predicate, n_rows, batch_size):
        selected = names
        pushed = False
        if predicate is not None and set(predicate.meta.root_names()) <= {"signal"}:
            try:
                selected = names.filter(predicate)
                pushed = True
            except pl.exceptions.PolarsError:
                pass  # not evaluable on the name table; filter after reading
        sel = selected["signal"].cast(pl.String).to_list()
        remaining = n_rows
        for i in range(0, len(sel), chunk_signals):
            df = f.to_polars_long(sel[i : i + chunk_signals], field)
            if predicate is not None and not pushed:
                df = df.filter(predicate)
            if with_columns is not None:
                df = df.select(with_columns)
            if remaining is not None:
                df = df.head(remaining)
                remaining -= df.height
            yield df
            if remaining == 0:
                return

    return register_io_source(src, schema=schema)
