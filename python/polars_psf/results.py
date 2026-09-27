"""Result directories: nested sweeps, Monte Carlo and ADE parametric runs as one long table.

Every row carries the outer sweep parameters, the inner sweep value and the signals::

    r = pp.results("sim.raw")
    r.analyses                       # name, type, params, leaves
    df = r.read("tran1", ["out"])    # iteration | time | out
    lf = r.scan("tran1")             # LazyFrame; filters on parameters skip whole files
    lf.filter(pl.col("iteration") <= 10).select("iteration", "time", "out").collect()
"""

import os
from os import PathLike

import polars as pl
from polars.io.plugins import register_io_source

from ._polars_psf import Results as _Results

__all__ = ["Results"]


class Results:
    """A Spectre/ADE result directory (logFile or runObjFile)."""

    def __init__(self, path: str | PathLike):
        self._r = _Results(path)

    def __repr__(self) -> str:
        return repr(self._r)

    @property
    def root(self) -> str:
        return self._r.root

    @property
    def warnings(self) -> list[str]:
        """Missing data files, rounded parameter values and other non-fatal problems."""
        return self._r.warnings

    @property
    def analyses(self) -> pl.DataFrame:
        rows = self._r.analyses()
        return pl.DataFrame(
            {
                "name": [r[0] for r in rows],
                "type": [r[1] for r in rows],
                "params": [r[2] for r in rows],
                "leaves": [r[3] for r in rows],
            },
            schema={"name": pl.String, "type": pl.String, "params": pl.List(pl.String), "leaves": pl.UInt32},
        )

    def leaves(self, analysis: str) -> pl.DataFrame:
        """One row per data file: parameters, logFile entry name and path."""
        return pl.DataFrame(self._r.leaf_table(analysis))

    def read(self, analysis: str, names: list[str] | None = None) -> pl.DataFrame:
        """Reads all leaves of an analysis (in parallel) into one long DataFrame."""
        n = self.leaves(analysis).height
        frames = [pl.DataFrame(b) for b in self._r.read_leaves(analysis, list(range(n)), names)]
        if not frames:
            return pl.DataFrame(self._r.schema(analysis, names))
        return pl.concat(frames, how="vertical", rechunk=False)

    def scan(self, analysis: str, *, batch_leaves: int | None = None) -> pl.LazyFrame:
        """Lazy long table of an analysis.

        Projection pushdown decodes only the selected signals; filters that use only parameter
        columns skip whole files. Leaves are read ``batch_leaves`` at a time in parallel.
        """
        analysis = self._r.resolve(analysis)
        schema = pl.DataFrame(self._r.schema(analysis)).schema
        table = self.leaves(analysis).with_row_index("_leaf")
        params = [c for c in table.columns if c not in ("_leaf", "leaf", "path")]
        sweep = self._r.sweep_name(analysis)  # always returned, never requested as a trace
        fixed = set(params) | {sweep}
        step = batch_leaves or max(1, os.cpu_count() or 1)
        reader = self._r

        def source(with_columns, predicate, n_rows, batch_size):
            idx = table["_leaf"].to_list()
            roots = set(predicate.meta.root_names()) if predicate is not None else set()
            if predicate is not None and roots <= set(params):
                idx = table.filter(predicate)["_leaf"].to_list()
            names = None
            if with_columns is not None:
                want = dict.fromkeys([*with_columns, *roots])
                names = [c for c in want if c not in fixed and c in schema]
            remaining = n_rows
            for i in range(0, len(idx), step):
                for b in reader.read_leaves(analysis, idx[i : i + step], names):
                    df = pl.DataFrame(b)
                    if predicate is not None:
                        df = df.filter(predicate)
                    if with_columns is not None:
                        df = df.select(with_columns)
                    if remaining is not None:
                        df = df.head(remaining)
                        remaining -= df.height
                    yield df
                    if remaining == 0:
                        return

        return register_io_source(source, schema=schema)

    def read_long(self, analysis: str, field: str | None = None) -> pl.DataFrame:
        """All leaves as one long DataFrame ``[params..., sweep, signal, value]``."""
        return self.scan_long(analysis, field).collect()

    def scan_long(
        self, analysis: str, field: str | None = None, *, max_pairs: int = 20_000_000
    ) -> pl.LazyFrame:
        """Lazy long table ``[params..., sweep, signal, value]`` over all leaves of an analysis.

        ``field`` selects a struct member (e.g. ``"total"``); traces without it are left out.
        Filters are pushed down before any data is read:

        - only parameter columns (``pl.col("temp") == 27``): other leaf files are skipped;
        - only ``signal``: each leaf decodes just the matching signals;
        - parameters and ``signal`` combined: evaluated on the (leaves x signal names) table, up to
          ``max_pairs`` rows, giving every leaf its own signal list.

        Anything else is applied by Polars after reading. Leaves are read in parallel.
        """
        analysis = self._r.resolve(analysis)
        schema = pl.DataFrame(self._r.long_schema(analysis, field)).schema
        table = self.leaves(analysis).with_row_index("_leaf")
        params = [c for c in table.columns if c not in ("_leaf", "leaf", "path")]
        leaves = table.select("_leaf", *params)
        names = pl.DataFrame({"signal": self._r.signal_names(analysis, field)}, schema={"signal": pl.Categorical})
        step = max(1, os.cpu_count() or 1)
        reader = self._r

        def plan(predicate):
            """[(leaf index, signal list or None=all)] after pushing the predicate down."""
            all_leaves = [(i, None) for i in leaves["_leaf"].to_list()]
            if predicate is None:
                return all_leaves
            roots = set(predicate.meta.root_names())
            try:
                if roots <= set(params):
                    return [(i, None) for i in leaves.filter(predicate)["_leaf"].to_list()]
                if roots <= {"signal"}:
                    sel = names.filter(predicate)["signal"].cast(pl.String).to_list()
                    return [(i, sel) for i, _ in all_leaves]
                if roots <= set(params) | {"signal"} and leaves.height * names.height <= max_pairs:
                    pairs = leaves.join(names, how="cross").filter(predicate)
                    per_leaf = pairs.group_by("_leaf", maintain_order=True).agg(pl.col("signal").cast(pl.String))
                    return list(zip(per_leaf["_leaf"].to_list(), per_leaf["signal"].to_list()))
            except pl.exceptions.PolarsError:
                pass  # not evaluable before reading
            return all_leaves

        def source(with_columns, predicate, n_rows, batch_size):
            jobs = plan(predicate)
            remaining = n_rows
            for k in range(0, len(jobs), step):
                for b in reader.read_leaves_long(analysis, jobs[k : k + step], field):
                    df = pl.DataFrame(b)
                    if predicate is not None:  # exact after pushdown, needed otherwise
                        df = df.filter(predicate)
                    if with_columns is not None:
                        df = df.select(with_columns)
                    if remaining is not None:
                        df = df.head(remaining)
                        remaining -= df.height
                    yield df
                    if remaining == 0:
                        return

        return register_io_source(source, schema=schema)
