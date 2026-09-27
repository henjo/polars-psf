//! Prints a PSF file summary: `psfdump FILE [SIGNAL...]` (`--names` lists all signal names).
fn main() -> Result<(), psfkit::Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let f = psfkit::PsfFile::open(&args[0])?;
    if args.get(1).is_some_and(|a| a == "--names") {
        for n in f.names()? {
            println!("{n}");
        }
        return Ok(());
    }
    println!(
        "format={:?} swept={} psfxl_stub={} types={} sweeps={} traces={} groups={} values={}",
        f.format(),
        f.is_swept(),
        f.is_psfxl_stub(),
        f.types().len(),
        f.sweeps().len(),
        f.traces().len(),
        f.groups().len(),
        f.values().map_or(0, |v| v.len())
    );
    if std::env::var("PSFDUMP_TYPES").is_ok() {
        for t in f.types() {
            println!("  type {} {:?} = {:?}", t.id, t.name, t.dtype);
        }
    }
    for (k, v) in f.header().iter() {
        println!("  {k} = {v}");
    }
    let names: Vec<&str> = args[1..].iter().map(String::as_str).collect();
    if f.is_swept() && !names.is_empty() {
        let d = f.read(&names)?;
        println!(
            "{}: {:?}",
            d.sweep_name,
            (0..d.sweep.len().min(5))
                .map(|i| d.sweep.get(i).unwrap())
                .collect::<Vec<_>>()
        );
        for (n, c) in &d.traces {
            println!(
                "{n} [{}]: {:?}",
                c.len(),
                (0..c.len().min(5))
                    .map(|i| c.get(i).unwrap())
                    .collect::<Vec<_>>()
            );
        }
    } else {
        for n in names {
            println!("{n} = {:?}", f.value(n).map(|v| v.value));
        }
    }
    Ok(())
}
