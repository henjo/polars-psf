"""Synthetic pnoise-like psfbin: many per-device struct traces over a frequency sweep.

Usage: gen_noise.py OUT NDEV NFREQ
Half the devices are "mos" structs {id, fn, rd, rs, total}, half "res" {rn, fn, total}
(Spectre noise contributions differ per device type). Row-record layout (0x200) like Spectre pnoise.
"""
import struct
import sys

import numpy as np

out, ndev, nfreq = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])


def u32(*v):
    return struct.pack(">" + "I" * len(v), *v)


def s(x):
    b = x.encode()
    return u32(len(b)) + b + b"\0" * ((4 - len(b) % 4) % 4)


def pi(k, v):
    return u32(34) + s(k) + u32(v)


def ps(k, v):
    return u32(33) + s(k) + s(v)


MOS = ["id", "fn", "rd", "rs", "total"]
RES = ["rn", "fn", "total"]
buf = bytearray(u32(0x200))
toc = []
hp = ps("PSFversion", "1.1") + pi("PSF sweeps", 1) + pi("PSF sweep points", nfreq) + pi("PSF traces", ndev)
toc.append((0, len(buf)))
buf += u32(21, len(buf) + 8 + len(hp) + 4) + hp
# types: 1 = f64 sweep, 2 = mos struct, 3 = res struct
buf += u32(1)
toc.append((1, len(buf)))
start = len(buf)


def struct_type(tid, name, fields, base):
    body = u32(16, tid) + s(name) + u32(0, 16)
    for k, f in enumerate(fields):
        body += u32(16, base + k) + s(f) + u32(0, 11)
    return body + u32(18)


body = u32(16, 1) + s("freq") + u32(0, 11) + struct_type(2, "mos", MOS, 10) + struct_type(3, "res", RES, 20)
buf += u32(21, 0, 22, start + 16 + len(body)) + body + u32(19, 0)
buf[start + 4:start + 8] = u32(len(buf) + 4)
# sweep
buf += u32(2)
toc.append((2, len(buf)))
body = u32(16, 5) + s("freq") + u32(1)
buf += u32(21, len(buf) + 8 + len(body) + 4) + body
# traces
kinds = [2 if i % 2 == 0 else 3 for i in range(ndev)]
names = [f"x{i // 1000}.{'m' if k == 2 else 'r'}{i}" for i, k in enumerate(kinds)]
buf += u32(3)
toc.append((3, len(buf)))
start = len(buf)
body = b"".join(u32(16, 100 + i) + s(names[i]) + u32(kinds[i]) for i in range(ndev))
buf += u32(21, 0, 22, start + 16 + len(body)) + body + u32(19, 0)
buf[start + 4:start + 8] = u32(len(buf) + 4)
# values: per freq point: 16,5,freq then per device 16,id,struct doubles
buf += u32(4)
toc.append((4, len(buf)))
vstart = len(buf)
buf += u32(21, 0)
freq = np.logspace(3, 9, nfreq)
rng = np.random.default_rng(1)
rec = []
for j in range(nfreq):
    parts = [u32(16, 5), struct.pack(">d", freq[j])]
    vals = rng.random((ndev, 5)) * 1e-18
    for i in range(ndev):
        nf = 5 if kinds[i] == 2 else 3
        v = vals[i, :nf].copy()
        v[-1] = v[:-1].sum()  # total = sum of contributions
        parts.append(u32(16, 100 + i) + v.astype(">f8").tobytes())
    rec.append(b"".join(parts))
buf += b"".join(rec)
buf[vstart + 4:vstart + 8] = u32(len(buf) + 4)
buf += u32(15)
toc_at = len(buf)
buf += b"".join(u32(a, b) for a, b in toc) + b"Clarissa" + u32(toc_at)
open(out, "wb").write(buf)
print(f"{out}: {len(buf) / 1e6:.0f} MB, {ndev} devices x {nfreq} freq")
