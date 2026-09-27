# Test data

Third-party PSF files used as golden references. Each directory keeps its original license.

| dir | source | license |
|---|---|---|
| `pycircuit/` | pycircuit `pycircuit/post/cds/test` (Spectre 2001–2013, Eldo) | BSD-3, see `pycircuit/LICENSE` |
| `psf-parser/` | https://github.com/ToonBettens/psf-parser `tests/data` (Spectre 2025) | MIT, see `psf-parser/LICENSE` |
| `spectre25/` | Spectre 25.1 output of `samplegen/` netlists (RC ladders, generated for this project) | MIT (this repo) |

Binary files have psfascii twins (`pycircuit/psfasc/*.asc`, `psf-parser/ascii/*`) used for value comparison.
Known quirk: psf-parser `mymonte-*` ascii/binary/psfxl twins come from different Monte Carlo runs; `mytran`
ascii differs from its binary/psfxl twin (6002 vs 65 points).

`pycircuit/resultdirs/parsweep/VDC*/psf` were symlinks to `.` in the original; they are replaced by
directories with copies of the referenced files so the tree also checks out on Windows.

`spectre25/`: `sweep_psfbin/`, `sweep_psfxl/` are complete `temp x rval` sweep directories (tran + ac) from
`samplegen/sweep_tran.scs`; `sweep_psfascii/` holds the logFile and one leaf pair for value comparison;
`psfbinf_ac/tran1.tran` and `psfxl_ac/tran1.tran.tran*` come from `samplegen/ladder_ac.scs` in two formats.
The transient psfbin files have no end table (as written by Spectre 25.1). The encoded `ENV_VAR_*` header
properties were overwritten with `0` characters of the same length.
