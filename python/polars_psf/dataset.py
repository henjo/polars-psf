"""One lazy entry point for PSF files and Spectre/ADE result directories.

    d = pp.open("sim.raw")                    # result dir (logFile / runObjFile) or one PSF file
    d.results                                 # one row per analysis result: name, type, params, leaves
    lf = d.scan("tran1")                      # wide LazyFrame [params..., time, signals...]
    lf.filter(pl.col("temp") == 27).select("time", "out").collect()
    d.scan_long("tran1").filter(pl.col("signal") == "out")   # [params..., time, signal, value]
    w = d.wave("out", "tran1", temp=27, rval=1e3)             # lazy Waveform

A single PSF file is a dataset with one result and one leaf without parameters, so the
``result`` argument can be left out. Opening reads metadata only; values are decoded when a
query is collected. Polars hands each query's projection and predicate to the reader:

- only parameter columns (``pl.col("temp") == 27``): other leaf files are skipped;
- selected signals (wide) or a ``signal`` filter (long): only those signals are decoded;
- parameters and ``signal`` combined (long): evaluated on the (leaves x signal names) table, up to
  ``max_pairs`` rows, giving every leaf its own signal list.

Anything else is applied by Polars after reading. Python only plans which leaves and signals to
read (predicates are Polars expressions); reading, chunking and parallelism happen in Rust.
"""

from __future__ import annotations

from os import PathLike

import polars as pl
from polars.io.plugins import register_io_source
from polars_waveform import Waveform, cx

from ._polars_psf import PsfFile, ResultDir
from .names import NameMap

__all__ = ["Dataset", "Result", "open", "openResults"]

_FIXED = ("leaf", "path")  # leaf table columns that are not parameters
_REL_TOL = 1e-5  # parameter matching: logFiles round values to 6 significant digits


def open(path: str | PathLike, netlist: str | PathLike | None = None) -> Dataset:
    """Open a PSF file or result directory lazily (only metadata is read). ``netlist`` is ADE's
    netlist directory for schematic names; by default it is found next to the results."""
    return Dataset(path, netlist)


openResults = open  # OCEAN alias


def _units(f: PsfFile, name: str):
    try:
        return f.props(name).get("units")
    except KeyError:
        return None


def _points(f: PsfFile) -> int | None:
    """Sweep points from the header (PSFXL: the .sig metadata), None if unknown."""
    n = f.header.get("PSF sweep points")
    if not n and f.is_psfxl:
        n = f.psfxl_meta.get("cdnshsweepcount")
    return int(n) if n else None


def _frames(batches):
    """DataFrames from a Rust batch iterator (each step yields a list of record batches)."""
    for step in batches:
        for b in step:
            yield pl.DataFrame(b)


def _emit(frames, predicate, with_columns, n_rows):
    """Apply the residual predicate, projection and row limit of an io source to its frames."""
    remaining = n_rows
    for df in frames:
        if with_columns is not None:
            missing = [c for c in with_columns if c not in df.columns]
            if missing:
                raise ValueError(
                    f"a leaf file lacks {missing}: the leaves of this result do not share their signals"
                )
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


def _equal(col: str, v) -> pl.Expr:
    """``col == v``, with a relative tolerance for floats (logFile values are rounded)."""
    if isinstance(v, float) or (isinstance(v, int) and not isinstance(v, bool)):
        return (pl.col(col) - v).abs() <= _REL_TOL * abs(v) + 1e-300
    return pl.col(col) == v


class _Once:
    """A LazyFrame collected at most once (shared by the waveforms of :meth:`Dataset.waves`)."""

    def __init__(self, lf: pl.LazyFrame):
        self.lf, self.df = lf, None

    def get(self) -> pl.DataFrame:
        if self.df is None:
            self.df = self.lf.collect()
        return self.df

    def scan(self, columns: list[str]) -> pl.LazyFrame:
        def source(with_columns, predicate, n_rows, batch_size):
            yield from _emit([self.get().select(columns)], predicate, with_columns, n_rows)

        return register_io_source(source, schema=self.lf.select(columns).collect_schema())


class Dataset:
    """A PSF file or result directory. Everything value-related is a lazy Polars query."""

    def __init__(self, path: str | PathLike, netlist: str | PathLike | None = None):
        """``netlist``: ADE netlist directory with the name maps (``amap/``); by default found
        next to the results when a schematic name (``/I0/out``) is first used."""
        self._r = ResultDir(path)
        self._tables: dict[str, pl.DataFrame] = {}
        self._schemas: dict[tuple, pl.Schema] = {}
        self._files: dict[str, PsfFile] = {}
        self._netlist = netlist
        self._name_map: NameMap | bool | None = False  # False: not looked up yet

    def __repr__(self) -> str:
        kind = "file" if self._r.is_single_file else "dir"
        return f"Dataset({self.root!r}, {kind}, results={self.results['name'].to_list()})"

    def __contains__(self, result: str) -> bool:
        """True if ``result`` names a result (short or family name)."""
        try:
            self._r.resolve(result)
        except KeyError:
            return False
        return True

    # metadata ---------------------------------------------------------------------------------
    @property
    def root(self) -> str:
        return self._r.root

    @property
    def warnings(self) -> list[str]:
        """Missing data files, rounded parameter values, renamed repeated trace names (``name#2``)
        and other non-fatal problems."""
        return self._r.warnings

    @property
    def results(self) -> pl.DataFrame:
        """One row per analysis result: name (``tran1``; the family name when two share one),
        analysis type, outer sweep parameters, number of leaf files and the simulator's
        description (``Transient Analysis `tran1': time = (0 s -> 5 ms)``; null for a single
        PSF file)."""
        rows = self._r.results()
        return pl.DataFrame(
            {
                "name": [r[0] for r in rows],
                "type": [r[1] for r in rows],
                "params": [r[2] for r in rows],
                "leaves": [r[3] for r in rows],
                "description": [r[4] for r in rows],
            },
            schema={"name": pl.String, "type": pl.String, "params": pl.List(pl.String), "leaves": pl.UInt32,
                    "description": pl.String},
        )

    def _rn(self, result: str | None) -> str:
        if result is None:
            names = [r[0] for r in self._r.results()]
            if len(names) != 1:
                raise ValueError(f"pass a result, one of {names}")
            result = names[0]
        return self._r.resolve(result)

    def _table(self, rn: str) -> pl.DataFrame:
        """Leaf table with a ``_leaf`` row index (cached)."""
        if rn not in self._tables:
            self._tables[rn] = pl.DataFrame(self._r.leaf_table(rn)).with_row_index("_leaf")
        return self._tables[rn]

    def _schema(self, rn: str, field: str | None, long: bool = False) -> pl.Schema:
        """Output schema of ``scan`` / ``scan_long`` (cached; built without reading values)."""
        key = (rn, field, long)
        if key not in self._schemas:
            b = self._r.long_schema(rn, field) if long else self._r.schema(rn)
            self._schemas[key] = pl.DataFrame(b).schema
        return self._schemas[key]

    def _params(self, rn: str) -> list[str]:
        return [c for c in self._table(rn).columns if c not in ("_leaf", *_FIXED)]

    def _file(self, path: str) -> PsfFile:
        if path not in self._files:
            self._files[path] = PsfFile(path)
        return self._files[path]

    def leaves(self, result: str | None = None) -> pl.DataFrame:
        """One row per data file: parameters, logFile entry name and path."""
        return self._table(self._rn(result)).drop("_leaf")

    def names(self, result: str | None = None, field: str | None = None) -> list[str]:
        """Signal names (of the first leaf); with ``field`` only structs that have that member."""
        return self._r.signal_names(self._rn(result), field)

    def sweep_name(self, result: str | None = None) -> str | None:
        """Name of the sweep variable (e.g. ``"time"``), ``None`` if not swept."""
        return self._r.sweep_name(self._rn(result))

    def is_swept(self, result: str | None = None) -> bool:
        return self.sweep_name(result) is not None

    def _match(self, rn: str, params: dict):
        """(leaves, predicate or None) for parameter values; every leaf if ``params`` is empty.

        A value matches exactly when some leaf has it, else within a relative tolerance of 1e-5,
        so values copied from a logFile (``3.66667``) find the full-precision sweep value. The
        predicate selects the same leaves in a scan (it pushes down to the leaf table).
        """
        table = self._table(rn)
        unknown = set(params) - set(self._params(rn))
        if unknown:
            raise KeyError(f"unknown parameter(s) {sorted(unknown)}; have {self._params(rn)}")
        conds = []
        for k, v in params.items():
            exact = pl.col(k) == v
            conds.append(exact if table.filter(exact).height else _equal(k, v))
        sel = table.filter(pl.all_horizontal(conds)) if conds else table
        if sel.height == 0:
            raise KeyError(f"no leaf of {rn!r} matches {params}")
        if sel.height == table.height:
            return sel, None
        # the matched values exactly: same leaves, and evaluable on the leaf table
        pred = pl.all_horizontal([pl.col(k).is_in(sel[k].unique().implode()) for k in params])
        return sel, pred

    def _select(self, rn: str, params: dict):
        """Like :meth:`_match`, for exactly one leaf."""
        sel, pred = self._match(rn, params)
        if sel.height > 1:
            raise ValueError(f"{sel.height} leaves of {rn!r} match {params}; pass parameters to pick one")
        return pred, sel

    def file(self, result: str | None = None, **params) -> PsfFile:
        """Low-level reader of one leaf (header, props, PSFXL metadata, Arrow batches)."""
        _, sel = self._select(self._rn(result), params)
        return self._file(sel["path"][0])

    def value(self, name: str, result: str | None = None, **params):
        """One value of a non-swept leaf (operating point, info) as a Python object; schematic
        paths (``"/I0/vout"``) are translated with ADE's name maps."""
        if name.startswith("/"):
            name = self._need_map().net(name)
        return self.file(result, **params).value(name)

    def values(self, result: str | None = None, **params) -> dict:
        """All values of a non-swept leaf as ``{name: value}``."""
        return self.file(result, **params).values()

    # queries ----------------------------------------------------------------------------------
    def scan(self, result: str | None = None, *, batch_leaves: int | None = None) -> pl.LazyFrame:
        """Wide LazyFrame ``[params..., sweep, signals...]`` over all leaves of a result.

        Non-swept leaves give one row of values each. Projection pushdown decodes only the selected
        signals; filters that use only parameter columns skip whole files. Leaves are read
        ``batch_leaves`` at a time in parallel (default: one per CPU).
        """
        rn = self._rn(result)
        schema = self._schema(rn, None)
        table = self._table(rn)
        params = self._params(rn)
        fixed = set(params) | {self._r.sweep_name(rn)}
        r = self._r

        def source(with_columns, predicate, n_rows, batch_size):
            idx = table["_leaf"].to_list()
            roots = set(predicate.meta.root_names()) if predicate is not None else set()
            if predicate is not None and roots <= set(params):
                idx = table.filter(predicate)["_leaf"].to_list()
            names = None
            if with_columns is not None:
                want = dict.fromkeys([*with_columns, *roots])
                names = [c for c in want if c not in fixed and c in schema]
            batches = r.scan(rn, idx, names, batch_leaves)
            yield from _emit(_frames(batches), predicate, with_columns, n_rows)

        return register_io_source(source, schema=schema)

    def scan_long(
        self,
        result: str | None = None,
        field: str | None = None,
        *,
        max_pairs: int = 20_000_000,
        chunk_rows: int = 1 << 22,
    ) -> pl.LazyFrame:
        """Long LazyFrame ``[params..., sweep, signal, value]`` over all leaves of a result.

        ``field`` selects a struct member (e.g. ``"total"`` for noise contributions); traces
        without it are left out. ``signal`` is Categorical: compare with ``==``/``is_in``, or
        ``cast(pl.String)`` for string functions. Signals are decoded about ``chunk_rows`` rows
        (signals x sweep points) at a time per leaf, so memory stays bounded for long transients
        and ``head()`` stops early. PSFXL signals keep their own time axes.
        """
        rn = self._rn(result)
        schema = self._schema(rn, field, long=True)
        params = self._params(rn)
        leaves = self._table(rn).select("_leaf", *params)
        names = pl.DataFrame({"signal": self.names(rn, field)}, schema={"signal": pl.Categorical})
        r = self._r

        def plan(predicate):
            """[(leaf index, signal list or None=all)] after pushing the predicate down."""
            every = [(i, None) for i in leaves["_leaf"].to_list()]
            if predicate is None:
                return every
            roots = set(predicate.meta.root_names())
            try:
                if roots <= set(params):
                    return [(i, None) for i in leaves.filter(predicate)["_leaf"].to_list()]
                if roots <= {"signal"}:
                    sel = names.filter(predicate)["signal"].cast(pl.String).to_list()
                    return [(i, sel) for i, _ in every]
                if roots <= set(params) | {"signal"} and leaves.height * names.height <= max_pairs:
                    pairs = leaves.join(names, how="cross").filter(predicate)
                    per_leaf = pairs.group_by("_leaf", maintain_order=True).agg(pl.col("signal").cast(pl.String))
                    return list(zip(per_leaf["_leaf"].to_list(), per_leaf["signal"].to_list(), strict=True))
            except pl.exceptions.PolarsError:
                pass  # not evaluable before reading
            return every

        def source(with_columns, predicate, n_rows, batch_size):
            batches = r.scan_long(rn, plan(predicate), field, chunk_rows)
            # the predicate is exact after pushdown and needed otherwise
            yield from _emit(_frames(batches), predicate, with_columns, n_rows)

        return register_io_source(source, schema=schema)

    # waveforms --------------------------------------------------------------------------------
    def _wave_setup(self, rn: str, params: dict):
        """(first matched file, predicate, group columns, matched leaves) for waveforms."""
        sel, pred = self._match(rn, params)
        f = self._file(sel["path"][0])
        if not f.is_swept:
            raise ValueError(f"{f.path!r} is not swept; use Dataset.value()")
        groups = [p for p in self._params(rn) if sel[p].n_unique() > 1]
        return f, pred, groups, sel

    @property
    def name_map(self) -> NameMap | None:
        """The ADE schematic <-> netlist name map, or ``None`` if there is none."""
        if self._name_map is False:
            if self._netlist is not None:
                self._name_map = NameMap(self._netlist)
            else:
                self._name_map = NameMap.find(self._r.root)
        return self._name_map

    def _need_map(self) -> NameMap:
        if self.name_map is None:
            raise KeyError("schematic names (/...) need ADE's name maps: none found next to the results; "
                           "pass netlist= to pp.open()")
        return self.name_map

    def netlist_name(self, path: str) -> str:
        """Netlist name of the schematic net ``path`` (``/I0/vout`` -> ``I0.VOUT``)."""
        return self._need_map().net(path)

    def schematic_name(self, name: str) -> str:
        """Schematic path of the netlist net ``name`` (``I0.VOUT`` -> ``/I0/vout``)."""
        return self._need_map().schematic(name)

    def wave(self, signal: str, result: str | None = None, **params) -> Waveform:
        """One signal as a lazy :class:`~polars_psf.post.Waveform`.

        ``params`` narrow the leaves (corners, iterations); the parameters that still vary
        become group columns, so the waveform holds one curve per leaf::

            d.wave("out", "tran1", temp=27, rval=100)   # one curve
            d.wave("out", "ac1").bandwidth()            # one row per corner

        Nothing is decoded until a value is needed.
        """
        return self._wave(signal, self._rn(result), params)

    def _wave(self, signal: str, rn: str, params: dict, label: str | None = None, sign: int = 1) -> Waveform:
        """:meth:`wave`, optionally named ``label`` and negated (``sign=-1``)."""
        f, pred, groups, sel = self._wave_setup(rn, params)
        if signal not in f:
            raise KeyError(signal)
        sweep = f.sweep_name or pl.DataFrame(f.long_schema(None)).columns[0]
        # PSFXL signals may each have their own axis: read those through the long table
        lf = self.scan_long(rn) if f.is_psfxl else self.scan(rn)
        if pred is not None:  # pushed down to the leaf table: other files are skipped
            lf = lf.filter(pred)
        name = label or signal
        value = pl.col("value") if f.is_psfxl else pl.col(signal)
        if sign < 0:
            dtype = self._schema(rn, None, long=f.is_psfxl)["value" if f.is_psfxl else signal]
            value = cx.complex(-cx.re(value), -cx.im(value)) if isinstance(dtype, pl.Struct) else -value
        if f.is_psfxl:
            lf = lf.filter(pl.col("signal") == signal)
        lf = lf.select(*groups, sweep, value.alias(name))
        units = {sweep: _units(f, sweep), name: _units(f, signal)}
        n = _points(f) if sel.height == 1 else None
        return Waveform(lf, name, index=[*groups, sweep], units=units, n=n)

    def waves(self, names: list[str] | None = None, result: str | None = None, **params) -> dict:
        """``{name: Waveform}`` for several signals (parameters as in :meth:`wave`), decoded
        together on first access."""
        rn = self._rn(result)
        f, pred, groups, sel = self._wave_setup(rn, params)
        names = list(names) if names is not None else f.names
        if f.is_psfxl:
            return {n: self.wave(n, rn, **params) for n in names}
        sweep = f.sweep_name
        lf = self.scan(rn)
        if pred is not None:
            lf = lf.filter(pred)
        once = _Once(lf.select(*groups, sweep, *names))
        units = {sweep: _units(f, sweep)}
        n = _points(f) if sel.height == 1 else None
        return {
            c: Waveform(
                once.scan([*groups, sweep, c]), c, index=[*groups, sweep], units={**units, c: _units(f, c)}, n=n
            )
            for c in names
        }

    # results as objects -----------------------------------------------------------------------
    def result(self, name: str | None = None) -> Result:
        """One analysis result as an object (OCEAN ``selectResult``); also ``d.ac1``."""
        return Result(self, self._rn(name))

    def __getattr__(self, name: str) -> Result:
        # only called for names that are not attributes: result names (``d.ac1``)
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return Result(self, self._r.resolve(name))
        except KeyError:
            raise AttributeError(f"{type(self).__name__!r} has no attribute or result {name!r}") from None

    def __dir__(self):
        names = [r[0] for r in self._r.results()]
        return [*super().__dir__(), *(n for n in names if n.isidentifier())]

    def v(self, signal: str, result: str | None = None, **params) -> Waveform:
        """A node voltage (OCEAN ``v``): :meth:`wave` of ``signal``. A schematic path
        (``"/I0/vout"``) is translated with ADE's name maps; the waveform keeps that name."""
        rn = self._rn(result)
        if signal.startswith("/"):
            return self._wave(self._need_map().net(signal), rn, params, label=signal)
        return self._wave(signal, rn, params)

    def i(self, terminal: str, result: str | None = None, **params) -> Waveform:
        """A terminal current (OCEAN ``i``): a netlist name (``"V1:p"``) or a schematic terminal
        path (``"/R0/PLUS"``), translated with ADE's name maps."""
        rn = self._rn(result)
        if not terminal.startswith("/"):
            return self._wave(terminal, rn, params)
        signal, sign = self._need_map().terminal(terminal)
        if signal is None:
            raise ValueError(f"{terminal!r} is tied off by the netlister: its current is zero")
        return self._wave(signal, rn, params, label=terminal, sign=sign)


class Result:
    """One analysis result of a :class:`Dataset`: all its leaves (corners, Monte Carlo runs).

    The object form of the ``result`` argument (pycircuit ``PSFResult``, OCEAN
    ``selectResult``)::

        r = pp.open("sim.raw")
        ac = r.ac1                        # or r.result("ac1")
        ac.v("n10").bandwidth()           # one row per corner
        ac.v("n10", temp=27, rval=1e3)    # parameters narrow the leaves
    """

    def __init__(self, dataset: Dataset, rn: str):
        self._ds = dataset
        self._rn = rn

    @property
    def name(self) -> str:
        """Short result name (``ac1``; the family name when two results share it)."""
        for label in self._ds.results["name"]:
            if self._ds._r.resolve(label) == self._rn:
                return label
        return self._rn

    @property
    def type(self) -> str:
        """Analysis type (``ac``, ``tran``, ...)."""
        return self._ds.results.filter(pl.col("name") == self.name)["type"].item()

    @property
    def description(self) -> str | None:
        """The simulator's description (``AC Analysis `ac1': freq = (1 kHz -> 10 GHz)``)."""
        return self._ds.results.filter(pl.col("name") == self.name)["description"].item()

    @property
    def params(self) -> list[str]:
        """Outer sweep parameters (columns of :attr:`leaves`)."""
        return self._ds._params(self._rn)

    @property
    def leaves(self) -> pl.DataFrame:
        """One row per data file: parameters, logFile entry name and path."""
        return self._ds.leaves(self._rn)

    @property
    def names(self) -> list[str]:
        """Signal names (of the first leaf)."""
        return self._ds.names(self._rn)

    @property
    def sweep_name(self) -> str | None:
        return self._ds.sweep_name(self._rn)

    def __contains__(self, signal: str) -> bool:
        return signal in self.names

    def __repr__(self) -> str:
        return f"Result({self.name!r}, type={self.type!r}, params={self.params}, leaves={self.leaves.height})"

    def v(self, signal: str, **params) -> Waveform:
        """A node voltage (OCEAN ``v``) as a lazy :class:`Waveform`, one curve per leaf;
        schematic paths (``"/I0/vout"``) are translated (see :meth:`Dataset.v`)."""
        return self._ds.v(signal, self._rn, **params)

    def i(self, terminal: str, **params) -> Waveform:
        """A terminal current (OCEAN ``i``): ``"V1:p"`` or a schematic path ``"/R0/PLUS"``."""
        return self._ds.i(terminal, self._rn, **params)

    wave = v

    def waves(self, names: list[str] | None = None, **params) -> dict:
        return self._ds.waves(names, self._rn, **params)

    def scan(self, **kwargs) -> pl.LazyFrame:
        """Wide LazyFrame ``[params..., sweep, signals...]`` (see :meth:`Dataset.scan`)."""
        return self._ds.scan(self._rn, **kwargs)

    def scan_long(self, field: str | None = None, **kwargs) -> pl.LazyFrame:
        """Long LazyFrame ``[params..., sweep, signal, value]`` (see :meth:`Dataset.scan_long`)."""
        return self._ds.scan_long(self._rn, field, **kwargs)

    def value(self, name: str, **params):
        """One value of a non-swept leaf (operating point)."""
        return self._ds.value(name, self._rn, **params)

    def values(self, **params) -> dict:
        return self._ds.values(self._rn, **params)

    def file(self, **params) -> PsfFile:
        return self._ds.file(self._rn, **params)
