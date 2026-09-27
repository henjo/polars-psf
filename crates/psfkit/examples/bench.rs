//! `bench FILE...`: time open + read_all.
fn main() -> Result<(), psfkit::Error> {
    for p in std::env::args().skip(1) {
        let t0 = std::time::Instant::now();
        let f = psfkit::PsfFile::open(&p)?;
        let t1 = t0.elapsed();
        let d = f.read_all()?;
        let t2 = t0.elapsed();
        let mb = std::fs::metadata(&p).unwrap().len() as f64 / 1e6;
        println!(
            "{p}: {mb:.0} MB, {} traces x {} pts, open {t1:?}, read_all {:?} ({:.2} GB/s)",
            d.traces.len(),
            d.sweep.len(),
            t2 - t1,
            mb / 1e3 / (t2 - t1).as_secs_f64()
        );
    }
    Ok(())
}
