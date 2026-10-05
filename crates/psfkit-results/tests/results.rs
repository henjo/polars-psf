use std::path::{Path, PathBuf};

use psfkit_results::{Param, ResultDir};

fn td(p: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(p)
}

fn f(p: &Param) -> f64 {
    p.as_f64().unwrap()
}

#[test]
fn spectre_nested_sweep() {
    // sweep1 (vdc3) { sweep2 (vdc2) { dc1, dc2 (vdc1) } }
    let r = ResultDir::open(td("pycircuit/pardcsweep.raw")).unwrap();
    let a = r.results().unwrap();
    let names: Vec<_> = a.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(names, ["sweep1_dc1", "sweep1_dc2"]);
    let labels: Vec<_> = a.iter().map(|a| a.label.as_str()).collect();
    assert_eq!(labels, ["dc1", "dc2"]);
    assert_eq!(a[0].params, ["vdc3", "vdc2"]);
    assert_eq!(a[0].leaves, 16);
    let leaves = r.leaves("dc1").unwrap();
    assert_eq!(leaves.len(), 16);
    // full precision from .sweep files, not the rounded logFile value 3.66667
    let l = leaves
        .iter()
        .find(|l| l.name == "sweep1-000_sweep2-001_dc1-dc")
        .unwrap();
    assert_eq!(f(&l.params[0].1), 10.0);
    assert_eq!(f(&l.params[1].1), 3.6666666666666665);
    assert!(l.path.ends_with("sweep1-000_sweep2-001_dc1.dc"));
    assert!(r.warnings().is_empty(), "{:?}", r.warnings());
}

#[test]
fn modern_logfile_sweeps_and_monte_carlo() {
    let r = ResultDir::open(td("psf-parser/binary/logFile")).unwrap();
    let a = r.results().unwrap();
    let get = |n: &str| {
        a.iter()
            .find(|x| x.name == n)
            .unwrap_or_else(|| panic!("{n} in {a:?}"))
    };
    assert_eq!(get("mysweep_ac2").params, ["R1:r", "C1:c"]);
    assert_eq!(get("mysweep_ac2").leaves, 9);
    assert_eq!(
        get("mysweep_ac2").description,
        "AC Analysis `mysweep-000_mynestedsweep-000_ac2': freq = (1 Hz -> 1 MHz)"
    );
    assert_eq!(get("mysweep_dc1").params, ["R1:r"]);
    assert_eq!(get("mymonte_tran1").params, ["iteration"]);
    assert_eq!(get("mymonte_tran1").leaves, 3);
    assert!(get("tran1").params.is_empty());
    let l = r.leaves("ac2").unwrap();
    let pts: Vec<(f64, f64)> = l
        .iter()
        .map(|l| (f(&l.params[0].1), f(&l.params[1].1)))
        .collect();
    assert_eq!(pts[0], (1000.0, 1e-6));
    assert_eq!(pts[5], (2000.0, 3e-6));
    let it: Vec<f64> = r
        .leaves("mymonte_tran1")
        .unwrap()
        .iter()
        .map(|l| f(&l.params[0].1))
        .collect();
    assert_eq!(it, [1.0, 2.0, 3.0]);
    assert!(matches!(
        r.leaves("nope"),
        Err(psfkit_results::Error::UnknownResult(_))
    ));
}

#[test]
fn ade_parametric_runobjfile() {
    let r = ResultDir::open(td("pycircuit/resultdirs/parsweep/psf")).unwrap();
    let a = r.results().unwrap();
    let op = a.iter().find(|a| a.name == "opBegin").unwrap();
    assert_eq!(op.params, ["VDC1", "VDC2"]);
    assert_eq!(op.leaves, 9);
    // test data is inconsistent: 3 logFiles list srcSweep, only 1 of those has the file;
    // logFile is authoritative, missing files are reported (lazily, when resolved), not fatal
    assert_eq!(a.iter().find(|a| a.name == "srcSweep").unwrap().leaves, 3);
    assert!(r.warnings().is_empty());
    assert_eq!(r.leaves("srcSweep").unwrap().len(), 1);
    assert_eq!(
        r.warnings()
            .iter()
            .filter(|w| w.contains("missing data file"))
            .count(),
        2
    );
    let pts: Vec<(f64, f64)> = r
        .leaves("opBegin")
        .unwrap()
        .iter()
        .map(|l| (f(&l.params[0].1), f(&l.params[1].1)))
        .collect();
    assert!(pts.contains(&(2.0, 1.0)));
}

#[test]
fn single_run_dir() {
    let r = ResultDir::open(td("pycircuit/resultdirs/simple")).unwrap();
    let names: Vec<_> = r.results().unwrap().into_iter().map(|a| a.name).collect();
    assert!(
        names.contains(&"srcSweep".to_owned()) && names.contains(&"opBegin".to_owned()),
        "{names:?}"
    );
}

/// open() and results() must not touch parent sweep files or data files.
#[test]
fn lazy_resolution() {
    let tmp = std::env::temp_dir().join(format!("psfkit-lazy-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    for e in std::fs::read_dir(td("pycircuit/pardcsweep.raw")).unwrap() {
        let p = e.unwrap().path();
        std::fs::copy(&p, tmp.join(p.file_name().unwrap())).unwrap();
    }
    std::fs::remove_file(tmp.join("sweep1_dc1.sweep")).unwrap();
    std::fs::remove_file(tmp.join("sweep1-000_sweep2-000_dc2.dc")).unwrap();
    let r = ResultDir::open(&tmp).unwrap();
    assert_eq!(r.results().unwrap().len(), 2);
    assert!(r.warnings().is_empty(), "{:?}", r.warnings());
    // dc2 resolution notices only its own missing leaf
    assert_eq!(r.leaves("dc2").unwrap().len(), 15);
    assert_eq!(r.warnings().len(), 1);
    // dc1 falls back to the rounded logFile values for vdc3
    let dc1 = r.leaves("dc1").unwrap();
    assert_eq!(dc1.len(), 16);
    assert!(
        r.warnings().iter().any(|w| w.contains("rounded")),
        "{:?}",
        r.warnings()
    );
    std::fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn single_file_is_one_result() {
    let r = ResultDir::open(td("psf-parser/binary/myac.ac")).unwrap();
    assert!(r.is_single_file());
    let a = r.results().unwrap();
    assert_eq!(a.len(), 1);
    assert_eq!(
        (a[0].label.as_str(), a[0].analysis_type.as_str()),
        ("myac", "ac")
    );
    assert!(a[0].params.is_empty());
    assert_eq!(r.resolve("ac").unwrap(), "myac");
    assert!(r.resolve("tran").is_err());
    let leaves = r.leaves("myac").unwrap();
    assert_eq!(leaves.len(), 1);
    assert!(leaves[0].params.is_empty());
    // the cached handle is shared
    let f1 = r.file(&leaves[0].path).unwrap();
    let f2 = r.file(&leaves[0].path).unwrap();
    assert!(std::sync::Arc::ptr_eq(&f1, &f2));
}

#[test]
fn logfile_path_opens_the_directory() {
    let r = ResultDir::open(td("pycircuit/pardcsweep.raw/logFile")).unwrap();
    assert!(!r.is_single_file());
    assert_eq!(r.results().unwrap().len(), 2);
}

#[test]
fn multi_output_analyses_are_separate_results() {
    // pss + pnoise with ppv=yes: six root entries for two analyses (issue #1)
    let dir = std::env::temp_dir().join(format!("psfkit-multi-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let entries = [
        ("pss-td.pss", "td.pss", "pss.td.pss", "time"),
        ("pss-fd.pss", "fd.pss", "pss.fd.pss", "freq"),
        ("pss-fi.pss", "fi.pss", "pss.fi.pss", "freq"),
        (
            "pnoise-td_ppv.pnoise",
            "td_ppv.pnoise",
            "pnoise.td_ppv.pnoise",
            "time",
        ),
        (
            "pnoise-pnoise",
            "pnoise",
            "pnoise.pnoise",
            "relative frequency",
        ),
        (
            "pnoise-pm.pnoise",
            "pm.pnoise",
            "pnoise.pm.pnoise",
            "relative frequency",
        ),
    ];
    let mut log = String::from(
        "HEADER\n\"PSFversion\" \"1.00\"\nTYPE\n\"analysisInst\" STRUCT(\n\"analysisType\" STRING *\n\
         \"dataFile\" STRING *\n\"format\" STRING *\n\"parent\" STRING *\n\
         \"sweepVariable\" ARRAY ( * ) STRING *\n\"description\" STRING *\n)\nVALUE\n",
    );
    for (name, atype, file, sweep) in entries {
        log += &format!(
            "\"{name}\" \"analysisInst\" (\n\"{atype}\"\n\"{file}\"\n\"PSF\"\n\"\"\n(\"{sweep}\")\n\"\"\n)\n"
        );
        std::fs::write(dir.join(file), "").unwrap();
    }
    log += "END\n";
    std::fs::write(dir.join("logFile"), log).unwrap();

    let r = ResultDir::open(&dir).unwrap();
    let a = r.results().unwrap();
    let names: Vec<_> = a.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "pss_td",
            "pss_fd",
            "pss_fi",
            "pnoise_td_ppv",
            "pnoise",
            "pnoise_pm"
        ]
    );
    assert!(a.iter().all(|a| a.leaves == 1 && a.label == a.name));
    for (name, atype, file, _) in entries {
        let n = &a.iter().find(|x| x.analysis_type == atype).unwrap().name;
        let l = r.leaves(n).unwrap();
        assert_eq!(l.len(), 1, "{n}");
        assert_eq!(l[0].name, name);
        assert!(l[0].path.ends_with(file));
        assert_eq!(r.resolve(atype).unwrap(), *n); // by analysis type
    }
    assert_eq!(r.resolve("pm").unwrap(), "pnoise_pm");
    assert!(r.warnings().is_empty(), "{:?}", r.warnings());
    std::fs::remove_dir_all(&dir).unwrap();
}
