//! Result directories written by Spectre 25.1 (testdata/spectre25).

use std::path::{Path, PathBuf};

use psfkit_results::Results;

fn td(p: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/spectre25")
        .join(p)
}

#[test]
fn nested_temp_rval_sweeps() {
    let mut points = Vec::new();
    for fmt in ["sweep_psfbin", "sweep_psfxl"] {
        let r = Results::open(td(fmt)).unwrap();
        let a = r.analyses().unwrap();
        let names: Vec<_> = a.iter().map(|x| x.name.as_str()).collect();
        assert_eq!(names, ["swp_t_tran1", "swp_t_ac1"], "{fmt}");
        assert!(
            a.iter()
                .all(|x| x.params == ["temp", "rval"] && x.leaves == 9)
        );
        let leaves = r.leaves("tran1").unwrap();
        assert_eq!(leaves.len(), 9);
        let p: Vec<(f64, f64)> = leaves
            .iter()
            .map(|l| {
                (
                    l.params[0].1.as_f64().unwrap(),
                    l.params[1].1.as_f64().unwrap(),
                )
            })
            .collect();
        assert!(r.warnings().is_empty(), "{fmt}: {:?}", r.warnings());
        points.push(p);
    }
    assert_eq!(points[0], points[1]);
    assert_eq!(points[0][0], (-40.0, 500.0));
    assert_eq!(points[0][8], (125.0, 2000.0));
}
