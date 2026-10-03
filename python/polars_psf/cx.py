"""Complex-number helpers for Struct{re, im} columns.

Registers a ``cx`` expression namespace on import::

    pl.col("out").cx.abs()      # magnitude
    pl.col("out").cx.db20()     # 20*log10(|z|)
    pl.col("out").cx.phase()    # degrees (deg=False for radians)

The same functions are available as ``pp.cx.abs("out")`` etc. (accept a column name or expression).
Arithmetic on two complex columns: ``cx.mul("a", "b")``, ``cx.div``, ``cx.add``, ``cx.sub`` and
``cx.pow(x, n)`` for a real exponent; ``cx.from_real(x)`` lifts a real column to ``Struct{re, im}``.
"""

import math

import polars as pl

__all__ = [
    "abs",
    "add",
    "complex",
    "conj",
    "db10",
    "db20",
    "div",
    "from_real",
    "im",
    "mul",
    "phase",
    "pow",
    "re",
    "sub",
]


def _e(x) -> pl.Expr:
    return pl.col(x) if isinstance(x, str) else x


def re(x) -> pl.Expr:
    return _e(x).struct.field("re")


def im(x) -> pl.Expr:
    return _e(x).struct.field("im")


def abs(x) -> pl.Expr:
    return (re(x) ** 2 + im(x) ** 2).sqrt()


def db20(x) -> pl.Expr:
    return 20 * abs(x).log10()


def db10(x) -> pl.Expr:
    """10*log10(|z|), for power quantities."""
    return 10 * abs(x).log10()


def phase(x, deg: bool = True) -> pl.Expr:
    p = pl.arctan2(im(x), re(x))
    return p * (180 / math.pi) if deg else p


def conj(x) -> pl.Expr:
    return pl.struct(re(x).alias("re"), (-im(x)).alias("im"))


def complex(re_expr, im_expr) -> pl.Expr:
    """Build a Struct{re, im} column."""
    return pl.struct(_e(re_expr).alias("re"), _e(im_expr).alias("im"))


def from_real(x) -> pl.Expr:
    """A real column as ``Struct{re, im}`` with ``im = 0``."""
    return complex(x, _e(x) * 0.0)


def add(a, b) -> pl.Expr:
    return complex(re(a) + re(b), im(a) + im(b))


def sub(a, b) -> pl.Expr:
    return complex(re(a) - re(b), im(a) - im(b))


def mul(a, b) -> pl.Expr:
    ar, ai, br, bi = re(a), im(a), re(b), im(b)
    return complex(ar * br - ai * bi, ar * bi + ai * br)


def div(a, b) -> pl.Expr:
    ar, ai, br, bi = re(a), im(a), re(b), im(b)
    den = br * br + bi * bi
    return complex((ar * br + ai * bi) / den, (ai * br - ar * bi) / den)


def pow(x, n: float) -> pl.Expr:
    """``x ** n`` for a real exponent (principal branch)."""
    mag = abs(x) ** n
    th = pl.arctan2(im(x), re(x)) * n
    return complex(mag * th.cos(), mag * th.sin())


@pl.api.register_expr_namespace("cx")
class _CxNamespace:
    def __init__(self, expr: pl.Expr):
        self._e = expr

    def re(self) -> pl.Expr:
        return re(self._e)

    def im(self) -> pl.Expr:
        return im(self._e)

    def abs(self) -> pl.Expr:
        return abs(self._e)

    def db20(self) -> pl.Expr:
        return db20(self._e)

    def db10(self) -> pl.Expr:
        return db10(self._e)

    def phase(self, deg: bool = True) -> pl.Expr:
        return phase(self._e, deg)

    def conj(self) -> pl.Expr:
        return conj(self._e)

    def mul(self, other) -> pl.Expr:
        return mul(self._e, other)

    def div(self, other) -> pl.Expr:
        return div(self._e, other)

    def add(self, other) -> pl.Expr:
        return add(self._e, other)

    def sub(self, other) -> pl.Expr:
        return sub(self._e, other)

    def pow(self, n: float) -> pl.Expr:
        return pow(self._e, n)
