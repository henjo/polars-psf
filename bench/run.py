"""Benchmark harness: python bench/run.py OUT.json  (run in an env with libpsf and polars-psf installed)."""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SYN = Path(os.environ.get("SYNTH_DIR", "/tmp"))
S = "samplegen/out/"
T = "testdata/pycircuit/psf/"
CASES = [
    # (label, file, N values)
    ("Spectre 25.1 tran psfbin 2.4 GB, 302 sig", S + "psfbin_chunked/tran1.tran.tran", [1, 10, 302]),
    ("Spectre 25.1 tran PSFXL 411 MB, 102 sig", S + "psfxl_long/tran1.tran.tran", [1, 10, 102]),
    ("Spectre 25.1 tran psfbin 0.3 MB, 11 sig (swept leaf)", S + "sweep_psfbin/swp_t-001_swp_r-001_tran1.tran.tran", [11]),
    ("Spectre 25.1 ac psfbin 5.4 MB complex, 22 sig", S + "psfxl_ac/ac1.ac", [22]),
    ("Spectre 25.1 ac psfbinf (float32) 3.6 MB, 22 sig", S + "psfbinf_ac/ac1.ac", [22]),
    ("Spectre 25.1 ac psfascii 12 MB, 22 sig", S + "psfascii_ac/ac1.ac", [22]),
    ("Spectre 4.4 tran psfbin 0.8 MB, 32 sig", T + "tran.tran", [32]),
    ("Eldo tran psfbin 0.6 MB (1 window), 144 sig", T + "timeSweep", [144]),
    ("Spectre dc 0.6 MB, 1432 sig", T + "dc.dc", [1432]),
    ("Spectre pnoise (struct traces), 26 sig", T + "pnoise0.pnoise", [26]),
    ("Spectre op info 0.7 MB, 3800 values", T + "dcOpInfo.info", [1, 3800]),
    ("synthetic row-record sweep 643 MB, 200 sig", str(SYN / "big_simple.psf"), [1, 10, 200]),
    ("synthetic tran windowed 804 MB, 200 sig", str(SYN / "big_win.psf"), [1, 10, 200]),
]
RUNS = [("libpsf", "libpsf", {}), ("polars-psf 1T", "polars-psf", {"RAYON_NUM_THREADS": "1"}), ("polars-psf 12T", "polars-psf", {})]


def one(lib, path, n, env):
    try:
        p = subprocess.run([sys.executable, str(HERE / "bench_one.py"), lib, path, str(n)], capture_output=True,
                           text=True, timeout=900, env={**os.environ, **env})
    except subprocess.TimeoutExpired:
        return {"error": "timeout (900 s)"}
    if p.returncode != 0:
        err = (p.stderr.strip().splitlines() or ["?"])[-1]
        kind = {-11: "segfault", -6: "abort"}.get(p.returncode, f"exit {p.returncode}")
        return {"error": f"{kind}: {err[:120]}" if p.returncode > 0 else kind + (f" ({err[:80]})" if err else "")}
    return json.loads(p.stdout)


results = []
only = os.environ.get("BENCH_ONLY")  # substring filter on case labels
for label, path, ns in CASES:
    if only and only not in label:
        continue
    if not Path(path).exists():
        print("skip (missing)", path)
        continue
    subprocess.run(["cat", path], stdout=subprocess.DEVNULL)  # warm page cache
    size = Path(path).stat().st_size + sum(Path(path + e).stat().st_size for e in (".psfxl",) if Path(path + e).exists())
    for n in ns:
        row = {"case": label, "file": path, "bytes": size, "n": n}
        for name, lib, env in RUNS:
            best = None
            for _ in range(3):
                r = one(lib, path, n, env)
                if "error" in r:
                    best = r
                    break
                if best is None or r["total_ms"] < best["total_ms"]:
                    best = r
            row[name] = best
        if "error" not in row["libpsf"]:
            c = subprocess.run([sys.executable, str(HERE / "check_equal.py"), path, str(min(n, 50))], capture_output=True, text=True, timeout=900)
            row["max_rel_diff"] = c.stdout.strip() if c.returncode == 0 else f"check failed: {(c.stderr.strip().splitlines() or ['?'])[-1][:80]}"
        results.append(row)
        fmt = lambda r: r.get("error") or f"{r['total_ms']:.1f} ms"
        print(f"{label} N={n}: " + " | ".join(f"{k} {fmt(row[k])}" for k, _, _ in RUNS), row.get("max_rel_diff", ""), flush=True)
Path(sys.argv[1]).write_text(json.dumps(results, indent=1))
