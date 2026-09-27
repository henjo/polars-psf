//! Golden tests: psfbin files vs their psfascii twins, and smoke tests over the whole corpus.

use std::path::{Path, PathBuf};

use psfkit::{Column, PsfFile, Value};

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

/// Binary/ascii twins. Known-different pairs (different simulation runs) are excluded.
fn pairs() -> Vec<(PathBuf, PathBuf)> {
    let td = testdata();
    let mut v = Vec::new();
    for e in std::fs::read_dir(td.join("pycircuit/psf")).unwrap() {
        let p = e.unwrap().path();
        let a = td
            .join("pycircuit/psfasc")
            .join(format!("{}.asc", p.file_name().unwrap().to_str().unwrap()));
        if a.exists() {
            v.push((p, a));
        }
    }
    for e in std::fs::read_dir(td.join("psf-parser/binary")).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_str().unwrap().to_owned();
        let a = td.join("psf-parser/ascii").join(&name);
        let quirk =
            (name.starts_with("mymonte-") && name.contains("tran")) || name == "mytran.tran.tran";
        if a.exists() && !quirk {
            v.push((p, a));
        }
    }
    v.sort();
    assert!(v.len() > 40, "corpus missing? found {} pairs", v.len());
    v
}

fn close(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= 1e-5 * a.abs().max(b.abs()) + 1e-30 || (a.is_nan() && b.is_nan())
}

fn value_close(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Complex(ar, ai), Value::Complex(br, bi)) => close(*ar, *br) && close(*ai, *bi),
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Struct(x), Value::Struct(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y)
                    .all(|((ka, va), (kb, vb))| ka == kb && value_close(va, vb))
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| value_close(p, q))
        }
        _ => match (a.as_f64(), b.as_f64()) {
            (Some(x), Some(y)) => close(x, y),
            _ => false,
        },
    }
}

fn column_close(what: &str, a: &Column, b: &Column) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("{what}: {} vs {} points", a.len(), b.len()));
    }
    for i in 0..a.len() {
        let (x, y) = (a.get(i).unwrap(), b.get(i).unwrap());
        if !value_close(&x, &y) {
            return Err(format!("{what}[{i}]: {x:?} vs {y:?}"));
        }
    }
    Ok(())
}

fn compare(bin: &Path, asc: &Path) -> Result<(), String> {
    let b = PsfFile::open(bin).map_err(|e| format!("bin: {e}"))?;
    let a = PsfFile::open(asc).map_err(|e| format!("asc: {e}"))?;
    assert_eq!(b.format(), psfkit::Format::Binary);
    assert_eq!(a.format(), psfkit::Format::Ascii);
    if b.names().unwrap() != a.names().unwrap() {
        return Err(format!(
            "names differ: {:?}.. vs {:?}..",
            &b.names().unwrap()[..3.min(b.names().unwrap().len())],
            &a.names().unwrap()[..3.min(a.names().unwrap().len())]
        ));
    }
    if b.is_swept() {
        let (db, da) = (
            b.read_all().map_err(|e| format!("bin read: {e}"))?,
            a.read_all().map_err(|e| e.to_string())?,
        );
        column_close(&db.sweep_name, &db.sweep, &da.sweep)?;
        for ((n, cb), (_, ca)) in db.traces.iter().zip(&da.traces) {
            column_close(n, cb, ca)?;
        }
    } else {
        for (vb, va) in b.values().unwrap().iter().zip(a.values().unwrap()) {
            if !value_close(&vb.value, &va.value) {
                return Err(format!("{}: {:?} vs {:?}", vb.name, vb.value, va.value));
            }
            if vb.props.len() != va.props.len() {
                return Err(format!(
                    "{}: {} vs {} props",
                    vb.name,
                    vb.props.len(),
                    va.props.len()
                ));
            }
        }
    }
    Ok(())
}

#[test]
fn binary_matches_ascii() {
    let mut fails = Vec::new();
    let pairs = pairs();
    for (b, a) in &pairs {
        if let Err(e) = compare(b, a) {
            fails.push(format!("{}: {e}", b.display()));
        }
    }
    assert!(
        fails.is_empty(),
        "{} of {} pairs differ:\n{}",
        fails.len(),
        pairs.len(),
        fails.join("\n")
    );
}

fn all_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.symlink_metadata().unwrap().file_type().is_symlink() {
            continue;
        }
        if p.is_dir() {
            all_files(&p, out);
        } else {
            out.push(p);
        }
    }
}

/// Every PSF file in the corpus opens and its values decode.
#[test]
fn whole_corpus_reads() {
    let mut files = Vec::new();
    all_files(&testdata(), &mut files);
    let mut n = 0;
    let mut fails = Vec::new();
    for p in files {
        let bytes = std::fs::read(&p).unwrap();
        let is_psf = bytes.starts_with(&[0, 0, 2, 0])
            || bytes.starts_with(&[0, 0, 5, 0])
            || bytes.starts_with(&[0, 0, 0, 0])
            || bytes.starts_with(&[0, 0, 3, 0])
            || bytes.starts_with(&[0, 0, 4, 0])
            || bytes.starts_with(b"HEADER");
        if !is_psf || p.extension().is_some_and(|e| e == "psfxl" || e == "sig") {
            continue;
        }
        n += 1;
        let r = PsfFile::open(&p).and_then(|f| {
            if f.is_swept() && !f.is_psfxl_stub() {
                f.read_all().map(|_| ())
            } else {
                Ok(())
            }
        });
        if let Err(e) = r {
            fails.push(format!("{}: {e}", p.display()));
        }
    }
    assert!(n > 150, "only {n} PSF files found");
    assert!(
        fails.is_empty(),
        "{} of {n} files fail:\n{}",
        fails.len(),
        fails.join("\n")
    );
}
