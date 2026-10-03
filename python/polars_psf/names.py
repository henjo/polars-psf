"""Schematic <-> simulator names, from the name maps ADE writes next to the results.

ADE netlists a schematic with its own names (``/I0/vout`` in the schematic is ``I0.VOUT`` or
``I0.vout`` in the netlist, depending on the simulator) and keeps the translation in
``<netlist dir>/amap/``: one small PSF-ASCII file per cell and kind (``<cell>.f.net``,
``<cell>.f.inst``, ``<cell>.i.net``, ...; ``f`` = schematic -> netlist, ``i`` = the inverse).
``top_level_map`` names the top cell's files, and ``__simulator_information__`` the hierarchy
delimiter. In an ``.inst`` file, ``<inst>v`` is an instance's netlist name and ``<inst>n`` the
map of its master, so a hierarchical path is translated level by level; a primitive's ``.inst``
file maps its terminals to currents (``""`` = the instance's current, ``(FUNCTION minus(...))``
= the negative of another terminal's current).

:class:`NameMap` reads these maps lazily. ``Dataset.v("/I0/vout")`` and ``Dataset.i("/R0/PLUS")``
use it for names that start with ``/``.
"""

from __future__ import annotations

import re
from os import PathLike
from pathlib import Path

from ._polars_psf import PsfFile

__all__ = ["NameMap"]

_FUNC = re.compile(r'^\(FUNCTION\s+(\w+)\(root\("([^"]*)"\)\)\)$')
_SKIP = {"PSFversion", "__NEXT_FILE__"}


class NameMap:
    """The ADE name maps of one netlist directory (``<netlist>/amap``)."""

    def __init__(self, netlist_dir: str | PathLike):
        self.dir = Path(netlist_dir) / "amap"
        if not (self.dir / "top_level_map.f.net").is_file():
            raise FileNotFoundError(f"no ADE name map in {self.dir}")
        self._tables: dict[tuple[str, str, str], dict[str, str]] = {}
        self.top = str(PsfFile(self.dir / "top_level_map.f.net").header.get("__NEXT_FILE__", ""))
        info = self.dir / "__simulator_information__"
        header = PsfFile(info).header if info.is_file() else {}
        self.delimiter = str(header.get("__artHierarchyDelimiter__", "."))

    @classmethod
    def find(cls, results_dir: str | PathLike) -> NameMap | None:
        """The name map for a result directory: ``netlist/`` next to or above it (ADE keeps
        ``<run>/psf`` and ``<run>/netlist`` side by side); ``None`` if there is none."""
        root = Path(results_dir)
        if root.is_file():
            root = root.parent
        for d in (root / "netlist", root.parent / "netlist", root.parent.parent / "netlist"):
            if (d / "amap" / "top_level_map.f.net").is_file():
                return cls(d)
        return None

    def _table(self, cell: str | None, kind: str, direction: str = "f") -> dict[str, str]:
        if not cell:
            return {}
        key = (cell, kind, direction)
        if key not in self._tables:
            path = self.dir / f"{cell}.{direction}.{kind}"
            header = PsfFile(path).header if path.is_file() else {}
            self._tables[key] = {k: str(v) for k, v in header.items() if k not in _SKIP}
        return self._tables[key]

    @staticmethod
    def _parts(path: str) -> list[str]:
        parts = [p for p in path.strip().split("/") if p]
        if not parts:
            raise ValueError(f"empty schematic path {path!r}")
        return parts

    def _walk(self, instances: list[str], direction: str = "f") -> tuple[list[str], str | None]:
        """Translated instance names down a hierarchy, and the map of the last master."""
        cell, out = self.top, []
        for inst in instances:
            table = self._table(cell, "inst", direction)
            out.append(table.get(f"{inst}v", inst))
            cell = table.get(f"{inst}n")
        return out, cell

    def net(self, path: str) -> str:
        """Netlist name of the schematic net ``path`` (``/I0/vout`` -> ``I0.VOUT``). Names
        without a map entry are kept as they are."""
        *instances, net = self._parts(path)
        names, cell = self._walk(instances)
        mapped = self._table(cell, "net").get(f"{net}v")
        if mapped is None and instances:  # global nets (gnd!) are mapped at the top
            mapped = self._table(self.top, "net").get(f"{net}v") if net.endswith("!") else None
            if mapped is not None:
                return mapped
        return self.delimiter.join([*names, net if mapped is None else mapped])

    def terminal(self, path: str) -> tuple[str | None, int]:
        """``(netlist signal, sign)`` of the current into the schematic terminal ``path``
        (``/R0/PLUS`` -> ``("R0", 1)``, ``/R0/MINUS`` -> ``("R0", -1)``). The signal is ``None``
        for terminals the netlister ties off (zero current)."""
        *instances, term = self._parts(path)
        if not instances:
            raise ValueError(f"{path!r} is not a terminal path (/instance/terminal)")
        names, master = self._walk(instances)
        inst = self.delimiter.join(names)
        table = self._table(master, "inst")
        sign, seen = 1, set()
        expr = table.get(f"{term}v")
        while True:
            if expr is None:  # no primitive map: the netlister's default terminal name
                return f"{inst}:{term}", sign
            m = _FUNC.match(expr)
            if m is None:
                return inst + expr, sign  # "" = the instance's current, else a suffix
            fn, other = m.groups()
            if fn == "zero":
                return None, 0
            if fn != "minus" or other in seen:
                raise ValueError(f"cannot map terminal {path!r}: {expr}")
            seen.add(other)
            sign, expr = -sign, table.get(f"{other}v")

    def schematic(self, name: str) -> str:
        """Schematic path of the netlist net ``name`` (``I0.VOUT`` -> ``/I0/vout``)."""
        *instances, net = name.split(self.delimiter) if self.delimiter else [name]
        names, cell = self._walk(instances, "i")
        mapped = self._table(cell, "net", "i").get(f"{net}v", net)
        return "/" + "/".join([*names, mapped])
