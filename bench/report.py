"""Render bench/run.py JSON as a Markdown table: python bench/report.py RESULTS.json"""
import json
import sys

rows = json.load(open(sys.argv[1]))


def cell(r):
    if "error" in r:
        e = r["error"]
        if "RuntimeError" in e or "std::exception" in e:
            return "fails (exception)"
        return "fails (" + e.split(":")[0] + ")"
    return f"{r['total_ms']:,.1f} ms"


def speedup(lib, r):
    if "error" in lib or "error" in r:
        return ""
    return f" ({lib['total_ms'] / r['total_ms']:,.1f}x)"


print("| case | size | N | libpsf 0.1.4 | polars-psf, 1 thread | polars-psf, 12 threads | max rel diff |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    lib = r["libpsf"]
    size = r["bytes"] / 1e6
    size = f"{size:,.0f} MB" if size >= 10 else f"{size:.1f} MB"
    print(
        f"| {r['case']} | {size} | {r['n']} | {cell(lib)} | {cell(r['polars-psf 1T'])}{speedup(lib, r['polars-psf 1T'])} "
        f"| {cell(r['polars-psf 12T'])}{speedup(lib, r['polars-psf 12T'])} | {r.get('max_rel_diff', '')} |"
    )
