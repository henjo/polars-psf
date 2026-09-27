# PSF file formats

## psfbin

Big-endian throughout. The file is a sequence of 32-bit words; every item is padded to 4 bytes.

### Primitives

| name | encoding |
|---|---|
| `u32` / `i32` | 4 bytes |
| `f64` | 8 bytes IEEE 754 |
| string | `u32 len`, `len` bytes, zero padding to a multiple of 4. Some writers count bytes after a NUL into `len` (seen in `ENV_VAR_*` header properties); the reader cuts at the first NUL. |

### Chunk codes

| code | meaning |
|---|---|
| `0x10` | declaration (type, sweep, trace, value record) |
| `0x11` | group (trace section) / group value record (value section) |
| `0x12` | end of struct members |
| `0x14` | zero padding: `0x14, u32 n`, then `n` bytes (windowed value sections) |
| `0x15` | forward link: `0x15, u32 offset` |
| `0x16` | backward link: `0x16, u32 offset` |
| `0x21` | string property: `name`, `value` (strings) |
| `0x22` | integer property: `name`, `i32` |
| `0x23` | double property: `name`, `f64` |
| `0x24` | PSFXL index record (in trace declarations of PSFXL stubs, see below) |

A property list is a run of `0x21`/`0x22`/`0x23` chunks; the first other word ends it.

### File layout

```
section*            code, links, contents
[end table]         0xF, (u32 section, u32 offset)*, "Clarissa", u32 offset of the 0xF word
```

Section codes: header (first word `0x0`, `0x200`, `0x300`, `0x400` or `0x500`), `1` types, `2` sweeps,
`3` traces, `4` values, `0xF` end table. The header code matches the value layout in every file
seen (242 files):

| header code | files |
|---|---|
| `0x0`, `0x500` | non-swept |
| `0x200`, `0x300` | swept, row records |
| `0x400` | windowed transients and PSFXL stubs |

The reader does not rely on it: it derives the layout from the declarations and header properties.

After a section code come link chunks. The forward link points **just past the next section's code
word**; offsets in the end table point to the same place. The reader walks sections through the
forward links and never needs the end table, which matters because:

- transient psfbin files written by Spectre 25.1 have no end table and no `Clarissa` trailer (the
  file ends in unused window-buffer space);
- files of interrupted runs end anywhere; PSFXL stubs carry no value section.

The backward link of the type, trace and value sections points to the end of that section's
contents.

All files seen have `PSFversion` = `"1.1"` as the first header property. Header properties the
reader uses: `PSF window size` (windowed layout), `PSF sweep points` (expected point count),
`stop` (planned end of a transient, for completeness checks).

### Type declarations (section 1)

```
0x10, u32 id, string name, u32 ndims, u32 dims[ndims], u32 code, [members], props
```

| code | type | size in values |
|---|---|---|
| 1 | int8 | 4 (full 32-bit slot, sign-extended) |
| 5 | int32 | 4 |
| 9 | float32 | 4 |
| 10 | complex float32 | 8 (re, im) |
| 11 | float64 | 8 |
| 12 | complex float64 | 16 (re, im) |
| 16 | struct: member declarations (same syntax) until `0x12` | sum of members |

`ndims > 0` makes the type a fixed-size array of `prod(dims)` elements (seen: `ndims = 1`, e.g.
`ARRAY ( 3 ) FLOAT DOUBLE` in the psfascii twin). Codes 9/10 appear in single-precision
(`psfbinf`) output. Other codes have not been seen; the reader reports them as unsupported.

### Sweep and trace declarations (sections 2 and 3)

```
sweep / trace:  0x10, u32 id, string name, u32 type id, [0x24 index], props
group:          0x11, u32 id, string name, u32 n, n trace declarations
```

Group members are ordinary traces; the group only affects how values are stored.

### Values (section 4)

Three layouts occur.

**Non-swept** (no sweep declared; operating points, model/instance info):

```
0x10, u32 id, string name, u32 type id, value, props      (per value)
```

**Row records** (swept, no `PSF window size`): one record per sweep point, repeated to the end of
the section:

```
0x10, u32 sweep id, sweep value
0x10, u32 trace id, value        per ungrouped trace
0x11, u32 group id, values       per group: member values back to back, no tags
```

For fixed-size types every record has the same size, so a column can be read with a fixed stride.
Parent sweep files may declare traces without storing values for them.

**Windowed** (swept, header `PSF window size` = `W`; transients):

```
[0x14, u32 n, n bytes]*                      optional zero padding
0x10, u32 k, k sweep values                  window header; n = k & 0xffff (upper bits unknown)
per trace, in declaration order: W bytes     the window's n values, right-aligned in the block
```

Windows repeat until the end of the section or until a word that is neither `0x10` nor `0x14`
(unused buffer tail).

### PSFXL stubs

A PSFXL result is a psfbin *stub* (`tran1.tran.tran`) plus a data file (`<stub>.psfxl`) and a
metadata file (`<stub>.sig`). The stub has the usual declarations but no values (`PSF sweep
points = 0`); each trace declaration carries an index record:

```
0x24, u64 offset, u64 npoints, f64 x_min, f64 x_max, f64 y_min, f64 y_max
```

`offset` is the position in the `.psfxl` file of the signal's **last** chunk record; `x_max`
shows how far the run got (an interrupted run's stub has `x_max` < header `stop`).

## PSFXL data file (`.psfxl`)

Little-endian. Seen version: `ngpsf-1.4`.

```
"\0!ngpsf-1.4:0:ffffffffffffffff:4:20\n"    text line; fields after the version: meaning unknown
00 00 00 00  08 07 06 05 04 03 02 01 ...     byte-order marker (u64 0x0102030405060708, LE)
... records ...
```

Each signal is a chain of records, one per flush, linked backwards:

```
"\0" "type:sigidx:prev:npts:paysize:flags:X:ysize\n"  (hex fields), zero padding, payload
```

| field | meaning |
|---|---|
| `type` | `3` = float64 (only type seen) |
| `sigidx` | signal index |
| `prev` | offset of the previous record of this signal, `ffffffffffffffff` for the first |
| `npts` | points in this record |
| `paysize` | payload bytes |
| `flags` | `22`: time axis stored in this record; `a2`: time axis shared |
| `X` | flags `22`: `xsize`, the time buffer precedes the value buffer. flags `a2`: `back.xsize`, the time buffer starts `back` bytes before this record |
| `ysize` | size of the value buffer |

Payloads are [Blosc](https://www.blosc.org) (c-blosc 1) buffers of float64 (`typesize` 8).
Typically the first signal carries the time axis and the others reference it, so the axis is
decoded once. An interrupted run's `.psfxl` is an exact prefix of the complete one.

### Metadata (`.sig`)

Text lines `*key 26 value`, separated by runs of `0x01` bytes. Seen keys include `cdnshsweeprange`,
`cdnshsweepcount`, `cdnshnumflushes` and `cdnshcompressiontype` (`1` in all samples). The meaning
of `26` is unknown.

## psfascii

Text with sections `HEADER`, `TYPE`, `SWEEP`, `TRACE`, `VALUE` and `END` (`END` may be missing in
header-only files). Names are quoted; inside quotes a backslash is literal (escaped netlist names
such as `M1\&2`) except `\"`.

```
TYPE
"sweep" FLOAT DOUBLE PROP(
"key" "sweep"
)
"V" COMPLEX DOUBLE PROP(
"units" "V"
)
VALUE
"freq" 1.000000000000000e+03
"n1" (9.999999848008102e-01 -6.283184885072565e-05)
```

Types seen: `FLOAT DOUBLE`, `COMPLEX DOUBLE`, `INT BYTE`, `INT LONG`, `STRING *`,
`ARRAY ( n ) <type>`, `ARRAY ( * ) <type>`, `STRUCT( ... )`. Complex values are `(re im)`, struct
values `( ... )`. Swept files repeat `"sweep name" value` followed by one line per trace; groups
and structs nest in parentheses.

## Result directories

**`logFile`** (psfascii, Spectre): one `analysisInst` value per analysis, with fields
`analysisType`, `dataFile`, `format`, `parent`, `sweepVariable` and `description`. Nested sweeps and
Monte Carlo appear as `sweep` / `montecarlo` entries whose children name them as `parent`; data
files of children are named `<parent>-NNN_<name>`. The children's properties carry the outer
sweep value **rounded**; the full-precision values are in the parent's data file (`.sweep` /
`.montecarlo`, a psfbin or psfascii file with the sweep variable as its sweep). polars-psf takes
values from there and uses the rounded ones only to match children to sweep points.

**`runObjFile`** (psfascii, ADE parametric runs): `runObject` values with `logName` (paths of the
leaves' `logFile`s), `parent` and `sweepVariable`; each node's properties hold its value of the
parent's sweep variable (and `dataDir`).
