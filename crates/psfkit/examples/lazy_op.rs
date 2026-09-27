//! `lazy_op FILE NAME`: time open / single value / all values of a non-swept file.
fn main() -> Result<(), psfkit::Error> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let t = std::time::Instant::now();
    let f = psfkit::PsfFile::open(&a[0])?;
    let t1 = t.elapsed();
    let v = f.value(&a[1])?;
    let t2 = t.elapsed();
    let n = f.values()?.len();
    let t3 = t.elapsed();
    println!(
        "open {t1:?} | first value (index + decode 1) {:?} | all {n} values {:?} | {:?}",
        t2 - t1,
        t3 - t2,
        v.value.field("betadc")
    );
    Ok(())
}
