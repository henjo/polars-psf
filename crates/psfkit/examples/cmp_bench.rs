//! `cmp_bench FILE N`: open + names, then read N signals (swept) or N values (non-swept).
fn main() -> Result<(), psfkit::Error> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n: usize = a[1].parse().unwrap();
    let t0 = std::time::Instant::now();
    let f = psfkit::PsfFile::open(&a[0])?;
    let names: Vec<String> = f.names()?.iter().map(|s| s.to_string()).collect();
    let t1 = t0.elapsed();
    let n = n.min(names.len());
    if f.is_swept() {
        let idx: Vec<usize> = (0..n).collect();
        let d = f.read_indices(&idx)?;
        std::hint::black_box(&d);
    } else {
        for name in &names[..n] {
            std::hint::black_box(f.value(name)?);
        }
    }
    let t2 = t0.elapsed();
    println!(
        "psfkit: open {:.1} ms | {n} signals {:.1} ms",
        t1.as_secs_f64() * 1e3,
        (t2 - t1).as_secs_f64() * 1e3
    );
    Ok(())
}
