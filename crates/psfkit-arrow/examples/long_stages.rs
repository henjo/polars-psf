//! `long_stages FILE FIELD`: time open / read_all_field / long batch build.
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let t = std::time::Instant::now();
    let f = psfkit::PsfFile::open(&a[0]).unwrap();
    let t1 = t.elapsed();
    let d = f.read_all_field(&a[1]).unwrap();
    let t2 = t.elapsed();
    let b = psfkit_arrow::to_long_record_batch(d, None).unwrap();
    let t3 = t.elapsed();
    println!(
        "open {t1:?} | read_field {:?} | long batch {:?} | rows {}",
        t2 - t1,
        t3 - t2,
        b.num_rows()
    );
}
