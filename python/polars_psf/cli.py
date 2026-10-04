"""Command line: ``polars-psf info|names|export`` (also ``python -m polars_psf``).

    polars-psf info sim.raw                        # results, parameters, sweeps, signal counts
    polars-psf names sim.raw tran1 --grep '^n'     # signal names of a result
    polars-psf export sim.raw out.parquet -r tran1 -s out -s /I0/vout --where temp=27

``export`` writes by extension: ``.parquet``, ``.csv``, ``.ipc``/``.arrow``, ``.ndjson``.
Complex signals are ``Struct{re, im}``; CSV cannot hold them, so they are written as two columns
``<name>.re`` and ``<name>.im`` there.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import polars as pl

from .dataset import Dataset

__all__ = ["main"]


def _parse_value(text: str):
    try:
        return float(text)
    except ValueError:
        return text


def _info(args) -> int:
    d = Dataset(args.path, args.netlist)
    kind = "PSF file" if d._r.is_single_file else "result directory"
    print(f"{args.path}: {kind}")
    rows = []
    for r in d.results.iter_rows(named=True):
        sweep = d.sweep_name(r["name"]) if r["leaves"] else None
        signals = len(d.names(r["name"])) if r["leaves"] else 0
        description = r.pop("description") or ""
        rows.append({**r, "params": ", ".join(r["params"]), "sweep": sweep or "-", "signals": signals,
                     "description": description})  # fmt: skip
    with pl.Config(tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True, tbl_rows=-1, fmt_str_lengths=100,
                   tbl_width_chars=250):  # fmt: skip
        print(pl.DataFrame(rows))
    if d.name_map is not None:
        print(f"schematic name map: {d.name_map.dir}")
    for w in d.warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0


def _names(args) -> int:
    d = Dataset(args.path, args.netlist)
    names = d.names(args.result, args.field)
    if args.grep:
        pat = re.compile(args.grep)
        names = [n for n in names if pat.search(n)]
    for n in names:
        print(n)
    return 0


_FORMATS = (".parquet", ".csv", ".ipc", ".arrow", ".feather", ".ndjson")


def _export(args) -> int:
    out = Path(args.out)
    ext = out.suffix.lower()
    if ext not in _FORMATS:  # before reading anything
        raise SystemExit(f"unknown output format {ext!r}: use .parquet, .csv, .ipc or .ndjson")
    d = Dataset(args.path, args.netlist)
    res = d.result(args.result)
    where = {}
    for item in args.where or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--where expects NAME=VALUE, got {item!r}")
        where[key] = _parse_value(value)
    if args.long:
        if args.signals:
            raise SystemExit("--long exports every signal; filter the output instead")
        lf = res.scan_long(args.field)
        for key, value in where.items():
            lf = lf.filter(pl.col(key) == value)
    elif args.signals:
        waves = [res.v(s, **where) for s in args.signals]
        index = waves[0].index
        lf = pl.concat([w.lazy for w in waves], how="align") if len(waves) > 1 else waves[0].lazy
        lf = lf.select(*index, *(w.yname for w in waves))
    else:
        lf = res.scan()
        for key, value in where.items():
            lf = lf.filter(pl.col(key) == value)
    df = lf.collect()
    if ext == ".parquet":
        df.write_parquet(out)
    elif ext in (".ipc", ".arrow", ".feather"):
        df.write_ipc(out)
    elif ext == ".ndjson":
        df.write_ndjson(out)
    elif ext == ".csv":
        structs = [c for c, t in df.schema.items() if isinstance(t, pl.Struct)]
        df = df.with_columns(
            pl.col(c).struct.rename_fields([f"{c}.{f.name}" for f in df.schema[c].fields]) for c in structs
        ).unnest(*structs) if structs else df
        df.write_csv(out)
    print(f"wrote {out} ({df.height} rows, {df.width} columns)", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="polars-psf", description="Inspect and export Spectre PSF results.")
    parser.add_argument("--netlist", help="ADE netlist directory with name maps (default: found next to the results)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="results, parameters, sweeps and signal counts")
    p.add_argument("path")
    p.set_defaults(fn=_info)

    p = sub.add_parser("names", help="signal names of a result")
    p.add_argument("path")
    p.add_argument("result", nargs="?", help="result name (needed for directories with several)")
    p.add_argument("--field", help="only structs with this member (e.g. total for noise)")
    p.add_argument("--grep", help="only names matching this regular expression")
    p.set_defaults(fn=_names)

    p = sub.add_parser("export", help="write a result to Parquet, CSV, IPC or NDJSON")
    p.add_argument("path")
    p.add_argument("out", help="output file; the format follows the extension")
    p.add_argument("-r", "--result", help="result name (needed for directories with several)")
    p.add_argument("-s", "--signal", dest="signals", action="append",
                   help="signal to export (repeatable; schematic paths like /I0/vout work)")
    p.add_argument("-w", "--where", action="append", help="parameter filter NAME=VALUE (repeatable)")
    p.add_argument("--long", action="store_true", help="long format: [params..., sweep, signal, value]")
    p.add_argument("--field", help="struct member for --long (e.g. total for noise)")
    p.set_defaults(fn=_export)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except (KeyError, ValueError, OSError) as e:
        print(f"polars-psf: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
