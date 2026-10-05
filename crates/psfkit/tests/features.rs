//! Targeted behaviour tests on corpus files.

use std::path::{Path, PathBuf};

use psfkit::{Column, PsfFile, Scalar, Value};

fn td(p: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(p)
}

#[test]
fn windowed_transient() {
    let f = PsfFile::open(td("pycircuit/psf/tran.tran")).unwrap();
    assert!(f.is_swept() && f.is_complete());
    let d = f.read(&["I1:d", "C0:1"]).unwrap();
    assert_eq!(d.sweep_name, "time");
    assert_eq!(d.sweep.len(), 2757);
    assert_eq!(f.header().get_i64("PSF sweep points"), Some(2757));
    let t = d.sweep.as_f64().unwrap();
    assert_eq!(t[0], 0.0);
    assert_eq!(*t.last().unwrap(), 2e-8);
    assert_eq!(d.traces[1].0, "C0:1");
    assert_eq!(d.traces[1].1.as_f64().unwrap()[1], 5.397923751829672e-7);
}

#[test]
fn eldo_group_records() {
    // layout 0x300: group values tagged 0x11
    let f = PsfFile::open(td("pycircuit/psf/srcSweep")).unwrap();
    assert_eq!(f.groups().len(), 1);
    let d = f.read_all().unwrap();
    assert_eq!(d.sweep.as_f64().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    let names: Vec<_> = d.traces.iter().map(|t| t.0.as_str()).collect();
    assert_eq!(names, ["VOUT", "VIN", "R0"]);
    assert_eq!(d.traces[0].1.as_f64().unwrap(), &[-6.0, -4.0, -2.0, 0.0]);
}

#[test]
fn complex_ac() {
    let f = PsfFile::open(td("psf-parser/binary/myac.ac")).unwrap();
    let d = f.read_all().unwrap();
    assert!(
        d.traces
            .iter()
            .any(|(_, c)| matches!(c, Column::ComplexFloat64 { .. }))
    );
    assert!(f.traces().iter().all(|t| t.dtype.as_scalar().is_some()));
}

#[test]
fn struct_traces_pnoise() {
    let f = PsfFile::open(td("pycircuit/psf/pnoise0.pnoise")).unwrap();
    let d = f.read(&["xi1.qi71.ql", "xi1.r11"]).unwrap();
    let bjt = &d.traces[0].1;
    let res = &d.traces[1].1;
    assert_eq!(
        bjt.field("total").unwrap().as_f64().unwrap()[0],
        4.745056346150785e-10
    );
    // struct layouts differ per device type
    assert!(bjt.field("rb").is_some() && res.field("rb").is_none() && res.field("rn").is_some());
    assert_eq!(bjt.len(), 21);
}

#[test]
fn nonswept_info() {
    let f = PsfFile::open(td("pycircuit/psf/dcOpInfo.info")).unwrap();
    assert!(!f.is_swept());
    let v = f.value("IREG21U_0.MP5.b1").unwrap();
    assert!(
        (v.value.field("betadc").unwrap().as_f64().unwrap() - 4.795_701_449_943_476).abs() < 1e-12
    );
    assert!(f.read_all().is_err());
}

#[test]
fn logfile_ascii() {
    let f = PsfFile::open(td("pycircuit/pardcsweep.raw/logFile")).unwrap();
    assert_eq!(f.format(), psfkit::Format::Ascii);
    let v = f.value("sweep1-000_sweep2-001_dc1-dc").unwrap();
    assert_eq!(
        v.value.field("dataFile").and_then(Value::as_str),
        Some("sweep1-000_sweep2-001_dc1.dc")
    );
    assert_eq!(
        v.value.field("sweepVariable"),
        Some(&Value::Array(vec![Value::String("vdc1".into())]))
    );
    assert_eq!(v.props.get_f64("vdc2"), Some(3.66667));
}

#[test]
fn nested_sweep_values_full_precision() {
    let f = PsfFile::open(td("pycircuit/pardcsweep.raw/sweep1-000_sweep2_dc1.sweep")).unwrap();
    let d = f.read_all().unwrap();
    assert_eq!(d.sweep.as_f64().unwrap()[1], 3.6666666666666665);
}

#[test]
fn truncated_simple_sweep() {
    // killed Eldo run: value link never patched, first record incomplete
    let f = PsfFile::open(td("pycircuit/resultdirs/parsweep/VDC1=0,VDC2=0/srcSweep")).unwrap();
    assert!(!f.is_complete());
    let d = f.read_all().unwrap();
    assert_eq!(d.sweep.len(), 0);
    assert!(d.traces.iter().all(|t| t.1.is_empty()));
}

#[test]
fn truncated_windowed_keeps_complete_windows() {
    let full = std::fs::read(td("pycircuit/psf/tran.tran")).unwrap();
    let whole = PsfFile::from_bytes(full.clone())
        .unwrap()
        .read_all()
        .unwrap();
    let cut = PsfFile::from_bytes(full[..full.len() * 2 / 3].to_vec()).unwrap();
    assert!(!cut.is_complete());
    let part = cut.read_all().unwrap();
    let n = part.sweep.len();
    assert!(n > 0 && n < whole.sweep.len(), "{n}");
    assert_eq!(
        &whole.sweep.as_f64().unwrap()[..n],
        part.sweep.as_f64().unwrap()
    );
    for ((_, a), (_, b)) in whole.traces.iter().zip(&part.traces) {
        assert_eq!(&a.as_f64().unwrap()[..n], b.as_f64().unwrap());
    }
}

#[test]
fn parent_sweep_file_without_trace_values() {
    let f = PsfFile::open(td("pycircuit/psf/bwswp_acbw.sweep")).unwrap();
    let d = f.read_all().unwrap();
    assert_eq!(d.sweep.len(), 5);
    assert!(d.traces.iter().all(|t| t.1.is_empty()));
}

#[test]
fn psfxl_stub_index() {
    let f = PsfFile::open(td("psf-parser/psfxl/tran1.tran.tran")).unwrap();
    assert!(f.is_psfxl_stub());
    let xs: Vec<_> = f.traces().iter().map(|t| t.xl.unwrap()).collect();
    assert_eq!(xs.len(), 2);
    assert_eq!(
        (xs[0].offset, xs[0].npoints, xs[1].offset),
        (0x48, 65, 0x377)
    );
    assert_eq!(xs[0].x_max, 0.005);
    // stub alone (no .psfxl attached) cannot provide values
    let bytes = std::fs::read(td("psf-parser/psfxl/tran1.tran.tran")).unwrap();
    let e = PsfFile::from_bytes(bytes).unwrap().read_all().unwrap_err();
    let want = if cfg!(feature = "psfxl") {
        "not found"
    } else {
        "disabled"
    };
    assert!(e.to_string().contains(want), "{e}");
    assert_eq!(f.sweeps()[0].dtype.as_scalar(), Some(Scalar::Float64));
}

#[test]
fn unknown_name() {
    let f = PsfFile::open(td("pycircuit/psf/srcSweep")).unwrap();
    assert!(matches!(f.read(&["nope"]), Err(psfkit::Error::NotFound(_))));
}

/// Cutting any binary file at arbitrary points must give an error or a partial result, never a panic.
#[test]
fn truncation_never_panics() {
    let dirs = [
        "pycircuit/psf",
        "psf-parser/binary",
        "psf-parser/psfxl",
        "pycircuit/pardcsweep.raw",
    ];
    for d in dirs {
        for e in std::fs::read_dir(td(d)).unwrap() {
            let bytes = std::fs::read(e.unwrap().path()).unwrap();
            let step = (bytes.len() / 97).max(1);
            for cut in (0..bytes.len())
                .step_by(step)
                .chain([bytes.len().saturating_sub(1)])
            {
                if let Ok(f) = PsfFile::from_bytes(bytes[..cut].to_vec()) {
                    if f.is_swept() {
                        let _ = f.read_all();
                    }
                }
            }
        }
    }
}

#[cfg(feature = "psfxl")]
fn assert_close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= 1e-14 * x.abs().max(y.abs()),
            "[{i}] {x} vs {y}"
        );
    }
}

#[cfg(feature = "psfxl")]
#[test]
fn psfxl_matches_ascii() {
    let x = PsfFile::open(td("psf-parser/psfxl/tran1.tran.tran")).unwrap();
    let a = PsfFile::open(td("psf-parser/ascii/tran1.tran.tran")).unwrap();
    let (dx, da) = (x.read_all().unwrap(), a.read_all().unwrap());
    // psfascii prints 16 significant digits
    assert_close(dx.sweep.as_f64().unwrap(), da.sweep.as_f64().unwrap());
    for ((nx, cx), (na, ca)) in dx.traces.iter().zip(&da.traces) {
        assert_eq!(nx, na);
        assert_close(cx.as_f64().unwrap(), ca.as_f64().unwrap());
    }
    assert_eq!(x.psfxl_meta().get_i64("cdnshsweepcount"), Some(65));
    let (t, y) = x.read_signal("out").unwrap();
    assert_eq!((t.len(), y.len()), (65, 65));
}

/// Monte Carlo PSFXL: deterministic input matches ascii; stub min/max match decoded data.
#[cfg(feature = "psfxl")]
#[test]
fn psfxl_monte_carlo() {
    for i in 1..=3 {
        let x = PsfFile::open(td(&format!(
            "psf-parser/psfxl/mymonte-00{i}_tran1.tran.tran"
        )))
        .unwrap();
        let d = x.read_all().unwrap();
        for (t, (_, c)) in x.traces().iter().zip(&d.traces) {
            let v = c.as_f64().unwrap();
            let xl = t.xl.unwrap();
            assert_eq!(v.len() as u64, xl.npoints);
            assert_eq!(v.iter().cloned().fold(f64::INFINITY, f64::min), xl.y_min);
            assert_eq!(
                v.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                xl.y_max
            );
        }
        let a = PsfFile::open(td(&format!(
            "psf-parser/ascii/mymonte-00{i}_tran1.tran.tran"
        )))
        .unwrap();
        let (_, ya) = a.read_signal("in").unwrap();
        let (_, yx) = x.read_signal("in").unwrap();
        assert_close(ya.as_f64().unwrap(), yx.as_f64().unwrap());
    }
}

/// Reading only the sweep of a PSFXL file gives the time axis (it is stored with the signals).
#[cfg(feature = "psfxl")]
#[test]
fn psfxl_sweep_only() {
    let f = PsfFile::open(td("psf-parser/psfxl/tran1.tran.tran")).unwrap();
    let d = f.read(&[]).unwrap();
    assert_eq!(d.sweep.len(), 65);
    assert_eq!(d.sweep.as_f64().unwrap()[64], 0.005);
}

/// psfascii transient with trace name `a` declared twice (Spectre pnoise td_ppv does this).
const REPEATED: &str = r#"HEADER
"PSFversion" "1.00"
TYPE
"V" FLOAT DOUBLE PROP( "units" "V" )
"s" FLOAT DOUBLE PROP( "units" "s" )
SWEEP
"time" "s"
TRACE
"a" "V"
"b" "V"
"a" "V"
"a#2" "V"
"a#3" "V"
VALUE
"time" 0
"a" 1
"b" 2
"a" 3
"a#2" 4
"a#3" 0
"time" 1
"a" 5
"b" 6
"a" 7
"a#2" 8
"a#3" 0
END
"#;

#[test]
fn repeated_trace_names_are_renamed() {
    let f = PsfFile::from_bytes(REPEATED.as_bytes().to_vec()).unwrap();
    // psfascii has no trace IDs: the position (2) is used; a#2 is taken
    assert_eq!(f.names().unwrap(), ["a", "b", "a#2_", "a#2", "a#3"]);
    assert_eq!(f.renamed_traces(), [(2, "a".to_owned())]);
    let d = f.read_all().unwrap();
    let v: Vec<Vec<f64>> = d.traces.iter().map(|t| t.1.to_f64().unwrap()).collect();
    assert_eq!(
        v,
        [[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0], [0.0, 0.0]]
    );
    assert_eq!(
        f.read(&["a#2_"]).unwrap().traces[0].1.to_f64().unwrap(),
        [3.0, 7.0]
    );
}
