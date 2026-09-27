#!/usr/bin/env bash
# Generate PSF samples for polars-psf. Run on a *licensed* Spectre install.
# Output: ./out/<case>/ (raw dirs). Then: tar czf polars_psf_samples.tgz out/
set -euo pipefail
cd "$(dirname "$0")"
S="${SPECTRE:-spectre}"
run() { local name=$1 fmt=$2 net=$3; shift 3
  echo "== $name ($fmt)"; rm -rf out/$name
  "$@" "$S" -64 +escchars +log out/$name.log -format "$fmt" -raw out/$name "$net" || echo "   (exit $?)"; }
mkdir -p out
# 1. long PSFXL tran: multi-flush chains
run psfxl_long      psfxl   ladder_tran_long.scs
# 2. killed PSFXL run: partial data / lastgoodpoint
run psfxl_killed    psfxl   ladder_tran_long.scs timeout -s INT 20
# 3. psfbinf (single precision): ac (complex f32) + tran (f32)
run psfbinf_ac      psfbinf ladder_ac.scs
# 4. complex signals in PSFXL (ac) - may be written as psfbin; that is also an answer
run psfxl_ac        psfxl   ladder_ac.scs
# 5. big psfbin (>2 GiB). Note: Spectre 25.1 ignored PSF_MAX_FILE_SIZE and wrote one 2.4 GB file
run psfbin_chunked  psfbin  ladder_big_psfbin.scs env PSF_MAX_FILE_SIZE=100000000
# 6. nested sweeps (temp x rval) around tran/ac + montecarlo, in each format
run sweep_psfxl     psfxl    sweep_tran.scs
run sweep_psfbin    psfbin   sweep_tran.scs
run sweep_psfascii  psfascii sweep_tran.scs
# 7. reference ascii for (3) to verify f32 decoding
run psfascii_ac     psfascii ladder_ac.scs
ls -laR out | head -200
