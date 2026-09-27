//! Regression tests on Spectre 25.1 output (testdata/spectre25, see testdata/README.md).

use std::path::{Path, PathBuf};

use psfkit::{Column, PsfFile};

fn td(p: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/spectre25")
        .join(p)
}

/// Max difference relative to each column's largest magnitude.
fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let scale = a
        .iter()
        .chain(b)
        .fold(0f64, |m, x| m.max(x.abs()))
        .max(1e-300);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / scale)
        .fold(0.0, f64::max)
}

fn f64s(c: &Column) -> Vec<f64> {
    c.to_f64().unwrap_or_else(|| {
        let (re, im) = c.as_complex_f64().expect("real or complex");
        re.iter().chain(im).copied().collect()
    })
}

fn assert_same(a: &PsfFile, b: &PsfFile, tol: f64) {
    let (da, db) = (a.read_all().unwrap(), b.read_all().unwrap());
    assert_eq!(da.sweep_name, db.sweep_name);
    assert!(max_rel(&f64s(&da.sweep), &f64s(&db.sweep)) <= tol);
    assert_eq!(da.traces.len(), db.traces.len());
    for ((na, ca), (nb, cb)) in da.traces.iter().zip(&db.traces) {
        assert_eq!(na, nb);
        let e = max_rel(&f64s(ca), &f64s(cb));
        assert!(e <= tol, "{na}: {e:e}");
    }
}

/// Spectre 25.1 writes transient psfbin without the end table (no "Clarissa" trailer).
#[test]
fn transient_psfbin_without_end_table() {
    let p = td("sweep_psfbin/swp_t-001_swp_r-001_tran1.tran.tran");
    let bytes = std::fs::read(&p).unwrap();
    assert_ne!(&bytes[bytes.len() - 12..bytes.len() - 4], b"Clarissa");
    let b = PsfFile::open(&p).unwrap();
    assert!(b.is_complete());
    let a = PsfFile::open(td("sweep_psfascii/swp_t-001_swp_r-001_tran1.tran.tran")).unwrap();
    assert_same(&b, &a, 1e-12);
}

#[test]
fn ac_psfbin_matches_ascii() {
    let b = PsfFile::open(td("sweep_psfbin/swp_t-001_swp_r-001_ac1.ac")).unwrap();
    let a = PsfFile::open(td("sweep_psfascii/swp_t-001_swp_r-001_ac1.ac")).unwrap();
    assert_same(&b, &a, 1e-12);
}

#[cfg(feature = "psfxl")]
#[test]
fn psfxl_leaf_matches_psfbin() {
    for leaf in ["swp_t-000_swp_r-000", "swp_t-002_swp_r-002"] {
        let x = PsfFile::open(td(&format!("sweep_psfxl/{leaf}_tran1.tran.tran"))).unwrap();
        let b = PsfFile::open(td(&format!("sweep_psfbin/{leaf}_tran1.tran.tran"))).unwrap();
        assert!(x.is_psfxl_stub() && x.is_complete());
        assert_same(&x, &b, 0.0);
    }
}

/// Compared against the PSFXL output of the same netlist.
#[cfg(feature = "psfxl")]
#[test]
fn psfbinf_is_float32_and_matches() {
    let f = PsfFile::open(td("psfbinf_ac/tran1.tran")).unwrap();
    let d = f.read_all().unwrap();
    assert!(
        d.traces
            .iter()
            .all(|(_, c)| matches!(c, Column::Float32(_)))
    );
    let x = PsfFile::open(td("psfxl_ac/tran1.tran.tran")).unwrap();
    assert_same(&f, &x, 1e-6);
}
