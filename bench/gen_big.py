"""Generate a large synthetic windowed (tran-like) or simple-sweep PSF file for throughput tests.
Usage: gen_big.py OUT NTRACES NPOINTS [windowed|simple]"""
import struct, sys
import numpy as np

out, ntr, npts = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
mode = sys.argv[4] if len(sys.argv) > 4 else "windowed"
WS = 4096  # bytes per trace per window
PER = WS // 8

def u32(*v): return struct.pack(">" + "I" * len(v), *v)
def s(x):
    b = x.encode(); return u32(len(b)) + b + b"\0" * ((4 - len(b) % 4) % 4)
def pi(k, v): return u32(34) + s(k) + u32(v)
def ps(k, v): return u32(33) + s(k) + s(v)

buf = bytearray()
toc = []
buf += u32(1024 if mode == "windowed" else 512)
# header
hp = ps("PSFversion", "1.1") + pi("PSF sweeps", 1) + pi("PSF sweep points", npts) + pi("PSF traces", ntr)
if mode == "windowed":
    hp += pi("PSF window size", WS)
toc.append((0, len(buf)))
start = len(buf)
buf += u32(21, start + 8 + len(hp) + 4) + hp
# types: id 1 = f64
buf += u32(1)
toc.append((1, len(buf)))
body = u32(16, 1) + s("V") + u32(0, 11)
start = len(buf)
buf += u32(21, 0, 22, start + 16 + len(body)) + body + u32(19, 0)
end = len(buf); buf[start + 4:start + 8] = u32(end + 4)
# sweeps: id 2
buf += u32(2)
toc.append((2, len(buf)))
body = u32(16, 2) + s("time") + u32(1)
start = len(buf)
buf += u32(21, start + 8 + len(body) + 4) + body
# traces: group id 3 with members 100..
buf += u32(3)
toc.append((3, len(buf)))
members = b"".join(u32(16, 100 + i) + s(f"sig{i}") + u32(1) for i in range(ntr))
# windowed: traces in a group (as Spectre writes tran); simple: plain traces (as dc/ac)
body = (u32(17, 3) + s("group") + u32(ntr) + members) if mode == "windowed" else members
start = len(buf)
buf += u32(21, 0, 22, start + 16 + len(body)) + body + u32(19, 0)
end = len(buf); buf[start + 4:start + 8] = u32(end + 4)
# values
buf += u32(4)
toc.append((4, len(buf)))
vstart = len(buf)
buf += u32(21, 0)
t = np.arange(npts, dtype=">f8") * 1e-12
data = [np.sin(np.arange(npts) * (i + 1) * 1e-4).astype(">f8") for i in range(ntr)]
chunks = [bytes(buf)]
if mode == "windowed":
    chunks.append(u32(20, 4) + b"\0" * 4)
    for w in range(0, npts, PER):
        n = min(PER, npts - w)
        parts = [u32(16, (PER - n) << 16 | n), t[w:w + n].tobytes()]
        for d in data:
            parts.append(b"\0" * (WS - n * 8) + d[w:w + n].tobytes())
        chunks.append(b"".join(parts))
else:
    ids = np.arange(100, 100 + ntr, dtype=">u4")
    rec = np.zeros(npts, dtype=[("c", ">u4"), ("id", ">u4"), ("t", ">f8")] +
                   [(f"c{i}", ">u4") for i in range(0)])
    # per point: 16,2,t then per trace 16,id,v
    row = np.zeros((npts, 1 + ntr), dtype=[("c", ">u4"), ("id", ">u4"), ("v", ">f8")])
    row["c"] = 16
    row["id"][:, 0] = 2; row["v"][:, 0] = t
    row["id"][:, 1:] = ids;
    for i, d in enumerate(data): row["v"][:, 1 + i] = d
    chunks.append(row.tobytes())
body = b"".join(chunks)
vend = len(body)
body = bytearray(body); body[vstart + 4:vstart + 8] = u32(vend + 4)
body += u32(15)
tocb = b"".join(u32(a, b) for a, b in toc)
datasize = len(body)
body += tocb + b"Clarissa" + u32(datasize)
open(out, "wb").write(body)
print(f"{out}: {len(body)/1e6:.0f} MB")
