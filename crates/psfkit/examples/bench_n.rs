//! `bench_n FILE N`: read the first N traces, report time.
fn main() -> Result<(), psfkit::Error> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let f = psfkit::PsfFile::open(&a[0])?;
    let n: usize = a[1].parse().unwrap();
    let idx: Vec<usize> = (0..n.min(f.traces().len())).collect();
    let t = std::time::Instant::now();
    let d = f.read_indices(&idx)?;
    let dt = t.elapsed();
    let bytes = (d.traces.len() + 1) * d.sweep.len() * 8;
    println!(
        "{n} traces: {dt:?}  ({:.2} GB/s output)",
        bytes as f64 / 1e9 / dt.as_secs_f64()
    );
    Ok(())
}
