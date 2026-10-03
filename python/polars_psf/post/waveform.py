"""The :class:`Waveform`: a value column over one or more index columns, backed by a LazyFrame."""

from __future__ import annotations

import math

import polars as pl

from .. import cx

__all__ = ["Waveform", "either", "falling", "raising"]

# edge types (OCEAN uses the strings; the constants are kept for pycircuit code)
raising = 1
falling = 2
either = 3
_EDGES = {"rising": raising, "falling": falling, "either": either, raising: raising, falling: falling, either: either}

_KEEP = object()  # sentinel: keep the inherited label/unit
_RHS = "__rhs__"  # temporary column holding the right operand of a binary operation


def _as_series(v) -> pl.Series:
    return v if isinstance(v, pl.Series) else pl.Series(v)


def _is_complex_dtype(dtype) -> bool:
    return isinstance(dtype, pl.Struct)


def _to_complex(v):
    """A struct item ``{"re", "im"}`` as a Python complex (other values unchanged)."""
    return complex(v["re"], v["im"]) if isinstance(v, dict) else v


def _steps(z: pl.Series) -> pl.Series:
    """+1/-1 where ``z`` goes from below 0 to at-or-above 0 and back (0 elsewhere).

    A sample exactly on the threshold is one crossing (not two), and a curve starting on it has
    none."""
    return (z >= 0).cast(pl.Int8).diff()


def _dt(t) -> float:
    """``t2 - t1`` of a transition tuple."""
    return t[1] - t[0]


def _complex_series(name: str, values) -> pl.Series:
    if any(isinstance(v, complex) for v in values):
        return pl.Series(name, [{"re": complex(v).real, "im": complex(v).imag} for v in values])
    return pl.Series(name, values, dtype=pl.Float64)


class Waveform:
    """A value column over one or more index columns, backed by a lazy ``pl.LazyFrame``.

    >>> import polars as pl
    >>> w = Waveform(pl.DataFrame({"t": [0.0, 1.0, 2.0], "v": [0.0, 1.0, 0.0]}), "v")
    >>> w.ymax()
    1.0
    >>> w.cross(0.5)
    0.5
    >>> w.deriv().y.to_list()
    [1.0, -1.0]

    The index columns are every column except ``value``, or an explicit ``index=``; the sampling
    axis is ``index[-1]`` (override with ``sample=``). The other index columns are **groups**
    (corners, Monte Carlo iterations)::

        w = Waveform(df, "out", index=["iteration", "time"])   # one curve per iteration

    Everything works per group: elementwise and structure-changing operations (``deriv``,
    ``clip``, ``value(xs)``, ``dft``) keep the group columns, and measurements (``ymax``,
    ``cross``, ``value(x)``, ``bandwidth``, ...) return a scalar for a single curve and a
    DataFrame with one row per curve otherwise. Such a per-curve DataFrame broadcasts in
    arithmetic, joined on the group columns::

        centered = w - w.mean()          # each curve minus its own mean
        normalized = w / w.ymax()

    Keep the waveform on the left: ``w.mean() - w`` is evaluated by polars' ``DataFrame`` first
    and fails; write ``-(w - w.mean())``.

    The backing frame may be a ``pl.DataFrame`` or a ``pl.LazyFrame`` (decoded on first value
    access, then cached). Elementwise operations, including arithmetic between waveforms, build
    a lazy plan; only the index columns are read to check that two waveforms share their sweep.
    """

    def __init__(self, df, value, *, index=None, sample=None, units=None, n=None):
        if isinstance(df, pl.LazyFrame):
            self._lf = df
            self._df = None
            schema = df.collect_schema()
        else:
            if not isinstance(df, pl.DataFrame):
                df = pl.DataFrame(df)
            self._df = df
            self._lf = df.lazy()
            schema = df.schema
        names = schema.names()
        if value not in names:
            raise KeyError(f"no value column {value!r} in {names}")
        cols = [c for c in names if c != value]
        index = list(index) if index is not None else cols
        if not index:
            raise ValueError("a waveform needs at least one index column")
        missing = [c for c in index if c not in cols]
        if missing:
            raise KeyError(f"not index column(s) besides the value {value!r}: {missing}")
        self._value = value
        self._index = index
        self._sample = sample or index[-1]
        if self._sample not in index:
            raise ValueError(f"sample axis {self._sample!r} is not one of {index}")
        self._schema = schema
        self._units = {k: v for k, v in (units or {}).items() if v is not None}
        self._complex = _is_complex_dtype(schema[value])
        self._n = int(n) if n is not None else (df.height if isinstance(df, pl.DataFrame) else None)
        self._idx_frame: pl.DataFrame | None = None

    # construction -------------------------------------------------------------------------
    @classmethod
    def from_series(cls, x, y, *, xname=None, yname=None, xunit=None, yunit=None) -> Waveform:
        """Build from an ``x``/``y`` Series pair (a single index column)."""
        x, y = _as_series(x), _as_series(y)
        if len(x) != len(y):
            raise ValueError(f"x and y differ in length: {len(x)} != {len(y)}")
        xname = xname or x.name or "x"
        yname = yname or y.name or "y"
        if xname == yname:
            raise ValueError("x and y need distinct names; rename one")
        return cls(
            pl.DataFrame({xname: x, yname: y}),
            yname,
            index=[xname],
            units={xname: xunit, yname: yunit},
        )

    # materialization ----------------------------------------------------------------------
    def _frame(self) -> pl.DataFrame:
        """Materialized frame (index columns then value), collected on first use."""
        if self._df is None:
            self._df = self._lf.collect()
        return self._df

    def _index_frame(self) -> pl.DataFrame:
        """The index columns only (reads no values from a lazy source)."""
        if self._df is not None:
            return self._df.select(self._index)
        if self._idx_frame is None:
            self._idx_frame = self._lf.select(self._index).collect()
        return self._idx_frame

    # basics -------------------------------------------------------------------------------
    @property
    def df(self) -> pl.DataFrame:
        """The backing frame: index column(s) followed by the value column."""
        return self._frame()

    @property
    def index(self) -> list[str]:
        """Index column names; ``index[-1]`` is the sampling axis by default."""
        return list(self._index)

    @property
    def groups(self) -> list[str]:
        """Index columns other than the sampling axis (one curve per combination)."""
        return [c for c in self._index if c != self._sample]

    @property
    def x(self) -> pl.Series:
        """The sampling axis."""
        return self._frame()[self._sample]

    @property
    def y(self) -> pl.Series:
        """The value column."""
        return self._frame()[self._value]

    @property
    def xname(self) -> str:
        return self._sample

    @property
    def yname(self) -> str:
        return self._value

    @property
    def xunit(self):
        return self._units.get(self._sample)

    @property
    def yunit(self):
        return self._units.get(self._value)

    @property
    def is_complex(self) -> bool:
        return self._complex

    @property
    def lazy(self) -> pl.LazyFrame:
        """The backing plan (index column(s) followed by the value column)."""
        return self._lf

    @property
    def frame(self) -> pl.DataFrame:
        """The backing frame (Polars escape hatch); same as :meth:`to_polars`."""
        return self._frame()

    def to_polars(self) -> pl.DataFrame:
        """Index column(s) and the value column, group columns included."""
        return self._frame()

    def to_numpy(self):
        """``y`` as a numpy array (complex128 for complex signals)."""
        import numpy as np

        if self.is_complex:
            re = self.y.struct.field("re").to_numpy()
            im = self.y.struct.field("im").to_numpy()
            out = np.empty(len(re), dtype=np.complex128)
            out.real = re
            out.imag = im
            return out
        return self.y.to_numpy()

    def __array__(self, dtype=None, copy=None):
        a = self.to_numpy()
        return a.astype(dtype) if dtype is not None else a

    def __len__(self) -> int:
        if self._n is not None:
            return self._n
        return self._frame().height

    def __repr__(self) -> str:
        kind = ", complex" if self.is_complex else ""
        unit = f" [{self.yunit}]" if self.yunit else ""
        index = " x ".join(self._index)
        return f"Waveform({index} -> {self.yname}{unit}, {len(self)} points{kind})"

    # plotting -----------------------------------------------------------------------------
    def plot(self, mark: str = "line", **encode):
        """Altair chart of the backing frame, via Polars' ``DataFrame.plot``.

        Needs ``polars[plot]`` (``altair>=5.4``). ``mark`` is any Altair mark (``"line"``,
        ``"point"``/``"scatter"``, ``"area"``, ...) and the keywords are Altair encode channels;
        ``x`` and ``y`` default to the sampling axis and the value, and group columns can be
        encoded too::

            w.plot()                                        # time -> out
            w.plot(color="iteration")                       # one line per Monte Carlo iteration
            w.db20().plot(x="freq", tooltip=["freq", w.yname])

        Struct-valued waveforms (complex included) cannot be plotted directly; reduce them first
        with :meth:`abs`, :meth:`db20`, :meth:`phase`, :meth:`real` or :meth:`imag`.

        ``x``/``y`` default to explicit Altair field specs, so names containing ``:`` (e.g.
        ``I1:d``) are not misparsed as Altair shorthand.
        """
        if self.is_complex:
            raise ValueError(
                "cannot plot a struct-valued waveform; use abs(), db20(), phase(), real() or imag()"
            )
        encode.setdefault("x", {"field": self.xname, "type": "quantitative"})
        encode.setdefault("y", {"field": self.yname, "type": "quantitative"})
        return getattr(self.to_polars().plot, mark)(**encode)

    # internals ----------------------------------------------------------------------------
    def _parts_expr(self):
        """(re, im) lazy expressions of the value column; im is None for reals."""
        e = pl.col(self._value)
        return (cx.re(e), cx.im(e)) if self._complex else (e, None)

    def _mag_expr(self) -> pl.Expr:
        """``|y|`` as a lazy expression."""
        return cx.abs(self._value) if self._complex else pl.col(self._value).abs()

    def _index_units(self) -> dict:
        return {k: v for k, v in self._units.items() if k in self._index}

    def _derived(self, y, *, yname=_KEEP, yunit=_KEEP, df=None) -> Waveform:
        """New waveform with value ``y`` (Series or Expr) over the (unchanged) index columns."""
        name = self._value if yname is _KEEP else yname
        base = self._lf if df is None else (df.lazy() if isinstance(df, pl.DataFrame) else df)
        val = y.alias(name) if isinstance(y, pl.Expr) else _as_series(y).rename(name)
        lf = base.with_columns(val).select([pl.col(c) for c in self._index + [name]])
        unit = self._units.get(self._value) if yunit is _KEEP else yunit
        units = self._index_units()
        if unit is not None:
            units[name] = unit
        return Waveform(lf, name, index=self._index, sample=self._sample, units=units, n=self._n)

    def _over(self, e: pl.Expr) -> pl.Expr:
        """``e`` evaluated per group (window over the group columns)."""
        return e.over(self.groups) if self.groups else e

    # per-group machinery --------------------------------------------------------------------
    def _curves(self):
        """``(group values, single-curve Waveform)`` for each group (one pair if ungrouped)."""
        if not self.groups:
            yield {}, self
            return
        g = self.groups
        units = {k: v for k, v in self._units.items() if k in (self._sample, self._value)}
        for key, part in self._frame().partition_by(g, maintain_order=True, as_dict=True).items():
            w = Waveform(part.select(self._sample, self._value), self._value, index=[self._sample], units=units)
            yield dict(zip(g, key, strict=True)), w

    def _measure(self, fn, name: str):
        """``fn(curve)``: a scalar for a single curve, else a DataFrame ``[groups..., name]`` with
        one row per curve."""
        if not self.groups:
            return fn(self)
        keys, vals = [], []
        for k, w in self._curves():
            keys.append(k)
            vals.append(fn(w))
        if len(vals) == 1:
            return vals[0]
        out = pl.DataFrame(keys, schema={g: self._schema[g] for g in self.groups})
        return out.with_columns(_complex_series(name, vals))

    def _agg(self, expr: pl.Expr, name: str):
        """Aggregate ``expr`` over each curve: a scalar for a single curve, else
        ``[groups..., name]`` with one row per curve."""
        df = self._frame()
        if not self.groups:
            return _to_complex(df.select(expr).item())
        out = df.group_by(self.groups, maintain_order=True).agg(expr.alias(name))
        return _to_complex(out[name][0]) if out.height == 1 else out

    def _map_curves(self, fn) -> Waveform:
        """Apply a structure-changing ``fn(curve) -> Waveform`` per group and stack the results."""
        if not self.groups:
            return fn(self)
        frames, last = [], None
        for k, w in self._curves():
            last = fn(w)
            lits = [pl.lit(v, dtype=self._schema[g]).alias(g) for g, v in k.items()]
            frames.append(last.to_polars().with_columns(lits))
        index = self.groups + last.index
        df = pl.concat(frames).select(index + [last.yname])
        return Waveform(df, last.yname, index=index, sample=last.xname, units=last._units)

    # arithmetic -----------------------------------------------------------------------------
    def _check_x(self, other: Waveform) -> None:
        a, b = self._index_frame(), other._index_frame()
        if a.height != b.height or not a[self._sample].equals(b[other._sample]):
            raise ValueError("waveforms have different sweeps; resample them first")
        if self._index == other._index and not a.equals(b):
            raise ValueError("waveforms have different index columns")

    def _binop(self, other, op, *, reverse=False) -> Waveform:
        """Elementwise op, built lazily. Two waveforms must share their index values (only the
        index columns are read to check that)."""
        base = self._lf.select(self._index + [self._value])
        if isinstance(other, Waveform):
            self._check_x(other)
            base = pl.concat([base, other._lf.select(pl.col(other._value).alias(_RHS))], how="horizontal", strict=True)
            rhs, rcx, rname = pl.col(_RHS), other._complex, other.yname
        elif isinstance(other, pl.DataFrame):  # one value per curve, e.g. w - w.mean()
            keys = [c for c in other.columns if c in self.groups]
            vals = [c for c in other.columns if c not in self.groups]
            if sorted(keys) != sorted(self.groups) or len(vals) != 1:
                raise ValueError(
                    f"a per-curve operand needs the group columns {self.groups} and one value column; "
                    f"got {other.columns}"
                )
            per_curve = other.lazy().select(*keys, pl.col(vals[0]).alias(_RHS))
            base = base.join(per_curve, on=keys, how="left", maintain_order="left")
            rhs, rcx, rname = pl.col(_RHS), _is_complex_dtype(other.schema[vals[0]]), vals[0]
        elif isinstance(other, pl.Series):
            if len(other) != len(self):
                raise ValueError(f"series length {len(other)} != waveform length {len(self)}")
            base = base.with_columns(pl.lit(other).alias(_RHS))
            rhs, rcx, rname = pl.col(_RHS), _is_complex_dtype(other.dtype), "series"
        elif isinstance(other, complex):
            rhs, rcx, rname = cx.complex(pl.lit(other.real), pl.lit(other.imag)), True, str(other)
        else:
            rhs, rcx, rname = pl.lit(float(other)), False, str(other)
        lhs, lcx = pl.col(self._value), self._complex
        if reverse:
            lhs, lcx, rhs, rcx = rhs, rcx, lhs, lcx
        name = f"{rname} {op} {self.yname}" if reverse else f"{self.yname} {op} {rname}"
        if op in ("<", ">", "<=", ">="):
            if lcx or rcx:
                raise TypeError(f"cannot compare complex waveforms with {op!r}; use abs(w)")
            val = {"<": lhs < rhs, ">": lhs > rhs, "<=": lhs <= rhs, ">=": lhs >= rhs}[op]
            return self._derived(val, yname=name, yunit=None, df=base)
        if not lcx and not rcx:
            val = {"+": lhs + rhs, "-": lhs - rhs, "*": lhs * rhs, "/": lhs / rhs}[op]
        else:
            lhs = lhs if lcx else cx.from_real(lhs)
            rhs = rhs if rcx else cx.from_real(rhs)
            val = {"+": cx.add, "-": cx.sub, "*": cx.mul, "/": cx.div}[op](lhs, rhs)
        return self._derived(val, yname=name, yunit=self.yunit if op in "+-" else None, df=base)

    def __add__(self, other):
        return self._binop(other, "+")

    def __radd__(self, other):
        return self._binop(other, "+", reverse=True)

    def __sub__(self, other):
        return self._binop(other, "-")

    def __rsub__(self, other):
        return self._binop(other, "-", reverse=True)

    def __mul__(self, other):
        return self._binop(other, "*")

    def __rmul__(self, other):
        return self._binop(other, "*", reverse=True)

    def __truediv__(self, other):
        return self._binop(other, "/")

    def __rtruediv__(self, other):
        return self._binop(other, "/", reverse=True)

    def __neg__(self):
        re, im = self._parts_expr()
        val = cx.complex(-re, -im) if self._complex else -pl.col(self._value)
        return self._derived(val, yname=f"-{self.yname}")

    def __pos__(self):
        return self._derived(pl.col(self._value))

    def __pow__(self, other):
        if isinstance(other, Waveform):
            raise NotImplementedError("waveform ** waveform is not supported")
        if isinstance(other, complex):
            raise NotImplementedError("complex exponents are not supported")
        name = f"{self.yname} ** {other}"
        if not self._complex:
            return self._derived(pl.col(self._value) ** other, yname=name, yunit=None)
        return self._derived(cx.pow(self._value, float(other)), yname=name, yunit=None)

    def __abs__(self):
        return self._derived(self._mag_expr(), yname=f"abs({self.yname})", yunit=None)

    def abs(self) -> Waveform:
        """Magnitude (``abs(w)``)."""
        return abs(self)

    mag = abs

    def __lt__(self, other):
        return self._binop(other, "<")

    def __le__(self, other):
        return self._binop(other, "<=")

    def __gt__(self, other):
        return self._binop(other, ">")

    def __ge__(self, other):
        return self._binop(other, ">=")

    # elementwise math ---------------------------------------------------------------------
    def real(self) -> Waveform:
        if self._complex:
            return self._derived(cx.re(self._value), yname=f"real({self.yname})")
        return self._derived(pl.col(self._value))

    def imag(self) -> Waveform:
        if self._complex:
            return self._derived(cx.im(self._value), yname=f"imag({self.yname})")
        return self._derived(pl.col(self._value) * 0.0, yname=f"imag({self.yname})")

    def conj(self) -> Waveform:
        val = cx.conj(self._value) if self._complex else pl.col(self._value)
        return self._derived(val, yname=f"conj({self.yname})")

    conjugate = conj

    def phase(self, deg: bool = True) -> Waveform:
        """Argument in degrees (``deg=False`` for radians)."""
        val = cx.phase(self._value, deg) if self._complex else cx.phase(cx.from_real(self._value), deg)
        return self._derived(val, yname=f"phase({self.yname})", yunit="deg" if deg else "rad")

    def db10(self) -> Waveform:
        """``10*log10(|y|)``, for power quantities."""
        return self._derived(10.0 * self._mag_expr().log10(), yname=f"db10({self.yname})", yunit="dB")

    def db20(self) -> Waveform:
        """``20*log10(|y|)``."""
        return self._derived(20.0 * self._mag_expr().log10(), yname=f"db20({self.yname})", yunit="dB")

    # reductions (scalar per curve; DataFrame per group) -------------------------------------
    def _real_only(self, what: str) -> None:
        if self._complex:
            raise TypeError(f"{what} needs a real waveform; use abs(w) or real(w)")

    def ymax(self):
        self._real_only("ymax()")
        return self._agg(pl.col(self._value).max(), "ymax")

    def ymin(self):
        self._real_only("ymin()")
        return self._agg(pl.col(self._value).min(), "ymin")

    def xmax(self):
        """x where y is largest (OCEAN ``xmax``; the largest x is ``w.x.max()``)."""
        self._real_only("xmax()")
        return self._agg(pl.col(self._sample).get(pl.col(self._value).arg_max()), "xmax")

    def xmin(self):
        """x where y is smallest (OCEAN ``xmin``; the smallest x is ``w.x.min()``)."""
        self._real_only("xmin()")
        return self._agg(pl.col(self._sample).get(pl.col(self._value).arg_min()), "xmin")

    argmax = xmax
    argmin = xmin

    def average(self):
        if self._complex:
            re, im = self._parts_expr()
            return self._agg(pl.struct(re.mean().alias("re"), im.mean().alias("im")), "average")
        return self._agg(pl.col(self._value).mean(), "average")

    def rms(self):
        re, im = self._parts_expr()
        sq = re * re + im * im if im is not None else re * re
        return self._agg(sq.mean().sqrt(), "rms")

    mean = average

    def stddev(self):
        re, im = self._parts_expr()
        var = re.var(ddof=0) + im.var(ddof=0) if im is not None else re.var(ddof=0)
        return self._agg(var.sqrt(), "stddev")

    # sampling -----------------------------------------------------------------------------
    def _resample(self, xs: pl.Series) -> pl.Series:
        """``y`` linearly interpolated at ``xs`` on a single curve (clamped at the ends)."""
        x, y = self.x, self.y
        if not x.is_sorted():
            if not x.is_sorted(descending=True):
                raise ValueError("sweep is not monotonic (several curves? see Waveform.groups)")
            x, y = x.reverse(), y.reverse()
        n = len(x)
        if n == 1:
            return pl.Series(self._value, [y[0]] * len(xs), dtype=y.dtype)
        xc = xs.cast(pl.Float64).clip(x[0], x[-1])
        i1 = x.search_sorted(xc, side="left").clip(1, n - 1)
        i0 = i1 - 1
        x0, x1 = x.gather(i0), x.gather(i1)
        t = ((xc - x0) / (x1 - x0)).fill_nan(1.0)

        def lin(s: pl.Series) -> pl.Series:
            s0, s1 = s.gather(i0), s.gather(i1)
            return s0 + t * (s1 - s0)

        if self._complex:
            re, im = lin(y.struct.field("re")), lin(y.struct.field("im"))
            return pl.DataFrame({"re": re, "im": im}).to_struct(self._value)
        return lin(y).rename(self._value)

    def _interp(self, x0: float):
        return _to_complex(self._resample(pl.Series([float(x0)]))[0])

    def value(self, x):
        """Interpolate ``y`` at ``x`` (clamped outside the sample range).

        Scalar ``x`` gives a float/complex per curve; a sequence gives a resampled
        :class:`Waveform`.
        """
        if isinstance(x, (int, float)):
            return self._measure(lambda w: w._interp(float(x)), "value")
        xs = pl.Series(self._sample, [float(v) for v in x])

        def one(w: Waveform) -> Waveform:
            frame = pl.DataFrame({w._sample: xs, w._value: w._resample(xs)})
            return Waveform(frame, w._value, index=[w._sample], units=w._units)

        return self._map_curves(one)

    def deriv(self) -> Waveform:
        """dy/dx (forward difference over each sample interval, within each curve)."""
        if self._n is not None and self._n < 2:
            raise ValueError("need at least two points to differentiate")
        sx, v = self._sample, self._value
        nxt = lambda e: self._over(e.shift(-1))  # noqa: E731
        dx = nxt(pl.col(sx)) - pl.col(sx)
        name = f"d({self.yname})/d({self.xname})"
        if self._complex:
            re, im = self._parts_expr()
            val = cx.complex((nxt(re) - re) / dx, (nxt(im) - im) / dx).alias(name)
        else:
            val = ((nxt(pl.col(v)) - pl.col(v)) / dx).alias(name)
        lf = self._lf.with_columns(val).filter(nxt(pl.col(sx)).is_not_null())
        lf = lf.select([pl.col(c) for c in self._index] + [name])
        n = None if self.groups or self._n is None else self._n - 1
        return Waveform(lf, name, index=self._index, sample=self._sample, units=self._index_units(), n=n)

    def filter(self, predicate) -> Waveform:
        """Subset rows with a boolean ``pl.Expr`` on the frame, or a boolean Series mask."""
        keep = self._index + [self._value]
        if isinstance(predicate, pl.Expr):
            lf = self._lf.filter(predicate).select([pl.col(c) for c in keep])
            return Waveform(lf, self._value, index=self._index, sample=self._sample, units=self._units)
        df = self._frame().filter(_as_series(predicate)).select(keep)
        return Waveform(df, self._value, index=self._index, sample=self._sample, units=self._units, n=df.height)

    def clip(self, xfrom: float, xto: float | None = None) -> Waveform:
        """Restrict each curve to ``[xfrom, xto]`` (to its last x if ``xto`` is None),
        interpolating both limits when they fall between samples."""
        if xto is not None and xfrom > xto:
            raise ValueError(f"xfrom ({xfrom}) > xto ({xto})")
        return self._map_curves(lambda w: w._clip1(xfrom, xto))

    def _clip1(self, xfrom: float, xto: float | None) -> Waveform:
        xmin, xmax = self.x.min(), self.x.max()
        xto = xmax if xto is None else xto
        if xfrom > xto:
            raise ValueError(f"xfrom ({xfrom}) > xto ({xto})")
        df = self._frame()
        inside = df.filter(pl.col(self._sample).is_between(xfrom, xto))
        edges = [x0 for x0, cut in ((xfrom, xfrom > xmin), (xto, xto < xmax)) if cut and x0 not in inside[self._sample]]
        if not edges or inside.height == 0:
            return Waveform(inside, self._value, index=self._index, sample=self._sample, units=self._units)
        xs = pl.Series(self._sample, edges)
        extra = pl.DataFrame({self._sample: xs, self._value: self._resample(xs)})
        out = pl.concat([inside, extra.select(inside.columns)]).sort(self._sample)
        return Waveform(out, self._value, index=self._index, sample=self._sample, units=self._units)

    # analysis -----------------------------------------------------------------------------
    def cross(self, threshold: float = 0.0, edge: int = 1, type: str = "either"):
        """x where ``y`` crosses ``threshold`` (OCEAN ``cross``), per curve.

        ``edge`` counts crossings from 1 (negative: from the end, ``-1`` is the last); ``type``
        is ``"rising"``, ``"falling"`` or ``"either"``. ``nan`` if there is no such crossing.
        """
        self._real_only("cross()")
        if type not in _EDGES:
            raise ValueError(f"edge type must be 'rising', 'falling' or 'either', not {type!r}")
        if edge == 0:
            raise ValueError("edge numbers start at 1 (negative counts from the end)")
        n = edge - 1 if edge > 0 else edge
        return self._measure(lambda w: w._cross1(threshold, n, _EDGES[type]), "cross")

    def _cross1(self, crossval: float, n: int, crosstype: int) -> float:
        z = self.y - crossval
        d = _steps(z)
        edges = {raising: d > 0, falling: d < 0, either: d != 0}[crosstype]
        idx = edges.arg_true()
        if not -len(idx) <= n < len(idx):
            return float("nan")
        k = int(idx[n])  # z[k-1] and z[k] bracket the crossing
        x0, x1, z0, z1 = self.x[k - 1], self.x[k], z[k - 1], z[k]
        if z1 == z0:
            return x1
        return x0 + (0.0 - z0) * (x1 - x0) / (z1 - z0)

    def xval(self) -> Waveform:
        """The sampling axis as a waveform over itself (used to build straight lines)."""
        name = "x"
        while name in self._index:
            name += "_"
        return self._derived(pl.col(self._sample), yname=name, yunit=self.xunit)

    def bandwidth(self, db: float = 3.0, type: str = "low"):
        """Frequency where ``|y|`` has dropped ``db`` below its pass-band value: the value at the
        lowest frequency for ``type="low"`` (first crossing), at the highest for ``"high"`` (last
        crossing). ``"band"`` gives the width between the crossings on either side of the peak."""
        if type not in ("low", "high", "band"):
            raise ValueError(f"bandwidth type must be 'low', 'high' or 'band', not {type!r}")
        if type == "band":
            return self._measure(lambda w: w._band1(db), "bandwidth")

        def one(w: Waveform) -> float:
            mag = abs(w)
            if mag.x[0] > mag.x[-1]:  # descending sweep: work on the ascending curve
                mag = Waveform(mag.to_polars().reverse(), mag.yname, index=mag.index, units=mag._units)
            ref = mag.y[0] if type == "low" else mag.y[-1]
            return mag._cross1(ref * 10.0 ** (-db / 20.0), 0 if type == "low" else -1, either)

        return self._measure(one, "bandwidth")

    def _band1(self, db: float) -> float:
        mag = abs(self)
        df = mag.to_polars().sort(mag.xname)
        y = df[mag.yname]
        k = y.arg_max()
        if k == 0 or k == len(y) - 1:
            return float("nan")  # no pass band with a roll-off on both sides
        level = y[k] * 10.0 ** (-db / 20.0)
        lo = Waveform(df.head(k + 1), mag.yname, index=[mag.xname])._cross1(level, -1, either)
        hi = Waveform(df.slice(k), mag.yname, index=[mag.xname])._cross1(level, 0, either)
        return hi - lo

    def unity_gain_frequency(self):
        """Frequency where ``|y|`` crosses 1."""
        return self._measure(lambda w: abs(w)._cross1(1.0, 0, either), "ugf")

    def phase_margin(self):
        """Phase margin of a loop gain: ``phase(-y)`` at the unity-gain frequency."""

        def one(w: Waveform) -> float:
            f0 = abs(w)._cross1(1.0, 0, either)
            if math.isnan(f0):
                return float("nan")
            v = w._interp(f0)
            if isinstance(v, complex):
                return math.degrees(math.atan2(-v.imag, -v.real))
            return 180.0 if v < 0 else 0.0

        return self._measure(one, "phase_margin")

    def gain_margin(self):
        """Gain margin of a loop gain in dB: ``-dB20(|y|)`` where ``-y`` has zero phase (``y`` on
        the negative real axis), the sign convention of :meth:`phase_margin` (OCEAN
        ``gainMargin``)."""

        def one(w: Waveform) -> float:
            if not w._complex:
                raise TypeError("gain_margin() needs a complex loop gain")
            df = w.to_polars().sort(w.xname)
            x = df[w.xname]
            re, im = df[w.yname].struct.field("re"), df[w.yname].struct.field("im")
            for k in (_steps(im) != 0).arg_true():
                k = int(k)
                t = im[k - 1] / (im[k - 1] - im[k])
                if re[k - 1] + t * (re[k] - re[k - 1]) < 0:
                    f = x[k - 1] + t * (x[k] - x[k - 1])
                    return -20.0 * math.log10(abs(w._interp(f)))
            return float("nan")

        return self._measure(one, "gain_margin")

    # integration ----------------------------------------------------------------------------
    def integ(self, xfrom: float | None = None, xto: float | None = None):
        """Definite integral of each curve over ``[xfrom, xto]`` (trapezoidal; OCEAN ``integ``)."""

        def one(w: Waveform):
            if xfrom is not None or xto is not None:
                w = w._clip1(w.x.min() if xfrom is None else xfrom, xto)
            return _to_complex(w.iinteg().y[-1]) if len(w.y) else 0.0

        return self._measure(one, "integ")

    def iinteg(self) -> Waveform:
        """Running integral of each curve, starting at 0 (OCEAN ``iinteg``)."""
        dx = pl.col(self._sample) - self._over(pl.col(self._sample).shift(1))

        def run(e: pl.Expr) -> pl.Expr:
            area = ((e + self._over(e.shift(1))) / 2 * dx).fill_null(0.0)
            return self._over(area.cum_sum())

        re, im = self._parts_expr()
        val = cx.complex(run(re), run(im)) if self._complex else run(re)
        return self._derived(val, yname=f"iinteg({self.yname})", yunit=None)

    # transient measurements -----------------------------------------------------------------
    def _levels(self, initial, final):
        """(initial, final) of a single curve: given, or y at the first/last sample."""
        y = self.y
        return (float(y[0]) if initial is None else initial, float(y[-1]) if final is None else final)

    def _transition(self, theta1: float, theta2: float, initial, final, rising: bool | None):
        """(x at theta1 %, x at theta2 %, level 1, level 2) of the first transition of one curve."""
        self._real_only("transition measurements")
        a, b = self._levels(initial, final)
        if rising is True and b <= a:
            raise ValueError("not a rising transition (final <= initial); use fall_time()")
        if rising is False and b >= a:
            raise ValueError("not a falling transition (final >= initial); use rise_time()")
        kind = raising if b > a else falling
        l1, l2 = a + theta1 / 100 * (b - a), a + theta2 / 100 * (b - a)
        t1 = self._cross1(l1, 0, kind)
        if math.isnan(t1):
            return t1, t1, l1, l2
        after = Waveform(self._frame().filter(pl.col(self._sample) >= t1), self._value, index=self._index,
                         sample=self._sample)
        return t1, after._cross1(l2, 0, kind), l1, l2

    def rise_time(self, theta1: float = 10.0, theta2: float = 90.0, initial=None, final=None):
        """Time from ``theta1`` % to ``theta2`` % of the transition from ``initial`` to ``final``
        (default: the first and last value), per curve (OCEAN ``riseTime``)."""
        return self._measure(lambda w: _dt(w._transition(theta1, theta2, initial, final, True)), "rise_time")

    def fall_time(self, theta1: float = 10.0, theta2: float = 90.0, initial=None, final=None):
        """Like :meth:`rise_time`, for a falling transition (OCEAN ``fallTime``)."""
        return self._measure(lambda w: _dt(w._transition(theta1, theta2, initial, final, False)), "fall_time")

    def slew_rate(self, theta1: float = 10.0, theta2: float = 90.0, initial=None, final=None):
        """Average slope between ``theta1`` % and ``theta2`` % of the transition (OCEAN
        ``slewRate``; negative for falling edges)."""

        def one(w: Waveform) -> float:
            t1, t2, l1, l2 = w._transition(theta1, theta2, initial, final, None)
            return (l2 - l1) / (t2 - t1)

        return self._measure(one, "slew_rate")

    def overshoot(self, initial=None, final=None):
        """Overshoot past ``final`` in percent of the step ``final - initial`` (OCEAN
        ``overshoot``); 0 if the curve never passes ``final``."""

        def one(w: Waveform) -> float:
            w._real_only("overshoot()")
            a, b = w._levels(initial, final)
            peak = w.y.max() if b >= a else w.y.min()
            return max(0.0, (peak - b) / (b - a) * 100.0)

        return self._measure(one, "overshoot")

    def settling_time(self, tolerance: float = 1.0, initial=None, final=None):
        """Time from the first sample until the curve stays within ``tolerance`` % of the step
        around ``final`` (OCEAN ``settlingTime``); ``nan`` if it never settles."""

        def one(w: Waveform) -> float:
            w._real_only("settling_time()")
            a, b = w._levels(initial, final)
            band = abs(tolerance / 100.0 * (b - a))
            out = (w.y - b).abs() > band
            if not out.any():
                return 0.0
            k = int(out.arg_true()[-1])
            if k == len(out) - 1:
                return float("nan")
            x0, x1, e0, e1 = w.x[k], w.x[k + 1], abs(w.y[k] - b), abs(w.y[k + 1] - b)
            t = x1 if e0 == e1 else x0 + (e0 - band) / (e0 - e1) * (x1 - x0)
            return t - w.x[0]

        return self._measure(one, "settling_time")

    def delay(
        self,
        other: Waveform,
        threshold: float = 0.0,
        other_threshold: float | None = None,
        edge: int = 1,
        other_edge: int = 1,
        type: str = "either",
        other_type: str | None = None,
    ):
        """x of ``other``'s crossing minus x of this waveform's crossing, per curve (OCEAN
        ``delay``). ``other``'s threshold and edge type default to this waveform's."""
        a = self.cross(threshold, edge, type)
        b = other.cross(threshold if other_threshold is None else other_threshold, other_edge, other_type or type)
        if isinstance(a, pl.DataFrame) and isinstance(b, pl.DataFrame):
            joined = a.join(b, on=self.groups, suffix="_other")
            return joined.select(*self.groups, delay=pl.col("cross_other") - pl.col("cross"))
        if isinstance(a, pl.DataFrame) or isinstance(b, pl.DataFrame):
            raise ValueError("delay() needs both waveforms to hold the same curves")
        return b - a

    def frequency(self, threshold: float | None = None):
        """Average frequency from the rising crossings of ``threshold`` (default: halfway
        between min and max), per curve (OCEAN ``frequency``)."""

        def one(w: Waveform) -> float:
            w._real_only("frequency()")
            level = (w.y.max() + w.y.min()) / 2 if threshold is None else threshold
            z = w.y - level
            xs = [w._cross_at(int(k), z) for k in (_steps(z) > 0).arg_true()]
            if len(xs) < 2:
                return float("nan")
            return (len(xs) - 1) / (xs[-1] - xs[0])

        return self._measure(one, "frequency")

    def period(self, threshold: float | None = None):
        """``1 / frequency()`` (OCEAN ``period``)."""
        f = self.frequency(threshold)
        if isinstance(f, pl.DataFrame):
            return f.select(*self.groups, period=1.0 / pl.col("frequency"))
        return 1.0 / f

    def _cross_at(self, k: int, z: pl.Series) -> float:
        x0, x1, z0, z1 = self.x[k - 1], self.x[k], z[k - 1], z[k]
        return x1 if z1 == z0 else x0 - z0 * (x1 - x0) / (z1 - z0)

    # families -------------------------------------------------------------------------------
    def leaf(self, **values) -> Waveform:
        """The curves at the given group values, with those group columns dropped (OCEAN
        ``leafValue``)::

            w.leaf(temp=27, rval=1000.0)   # one curve of a corner family

        Floats match within a relative tolerance of 1e-5.
        """
        unknown = set(values) - set(self.groups)
        if unknown:
            raise KeyError(f"not group column(s) {sorted(unknown)}; groups are {self.groups}")
        conds = []
        for k, v in values.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                conds.append((pl.col(k) - v).abs() <= 1e-5 * abs(v) + 1e-300)
            else:
                conds.append(pl.col(k) == v)
        index = [c for c in self._index if c not in values]
        lf = self._lf.filter(pl.all_horizontal(conds)).select(*index, self._value)
        return Waveform(lf, self._value, index=index, sample=self._sample, units=self._units)

    def dft(self) -> Waveform:
        """Single-sided amplitude spectrum of each curve (needs uniform sampling; uses numpy's
        FFT, so numpy is imported only here). DC and the Nyquist bin are not doubled."""
        return self._map_curves(lambda w: w._dft1())

    def _dft1(self) -> Waveform:
        import numpy as np

        if len(self) < 2:
            raise ValueError("need at least two points for a dft")
        t = self.x.to_numpy()
        dt = np.diff(t)
        if not np.allclose(dt, dt[0]):
            raise ValueError("dft() requires uniformly sampled x")
        y = self.to_numpy()
        n = len(y)
        amp = np.abs(np.fft.rfft(y)) / n
        amp[1 : -1 if n % 2 == 0 else None] *= 2
        name = f"dft({self.yname})"
        frame = pl.DataFrame({"freq": pl.Series(np.fft.rfftfreq(n, dt[0])), name: pl.Series(amp)})
        return Waveform(frame, name, index=["freq"], units={"freq": "Hz", name: self.yunit})
