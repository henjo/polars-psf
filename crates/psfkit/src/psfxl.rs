//! PSFXL data files (`*.psfxl`, "ngpsf" format), referenced by a psfbin stub.
//!
//! Layout (little-endian), as observed in Spectre output:
//! - text line `\0!ngpsf-1.4:...\n`, then a preamble with byte-order marker `08 07 .. 01`.
//! - one record per signal chunk at the offset from the stub's `0x24` index:
//!   `\0` + ASCII `type:sigidx:prev:npts:paysize:flags:X:ysize\n`, zero padding, then Blosc buffers.
//!   `X` is `xsize` (time axis stored inline before y) or `back.xsize` (shared time axis located
//!   `back` bytes before this record). `prev` links to the previous chunk of the same signal.
//! - type 3 = f64.

use crate::column::Column;
use crate::error::{Error, Result, malformed};
use crate::types::{PropValue, Properties};

const NONE: u64 = u64::MAX;

/// True if `b` starts like an ngpsf data file.
pub(crate) fn sniff(b: &[u8]) -> bool {
    b.starts_with(b"\0!ngpsf-")
}

pub(crate) fn check_header(b: &[u8]) -> Result<()> {
    if !sniff(b) {
        return malformed(0, "not an ngpsf (PSFXL) data file");
    }
    let line_end = b.iter().position(|&c| c == b'\n').unwrap_or(0);
    let line = String::from_utf8_lossy(&b[2..line_end]);
    let version = line
        .split(':')
        .next()
        .unwrap_or("")
        .trim_start_matches("ngpsf-");
    if !version.starts_with("1.") {
        return Err(Error::Unsupported(format!("ngpsf version {version}")));
    }
    // byte-order marker: u64 0x0102030405060708 stored little-endian
    let bom = b"\x08\x07\x06\x05\x04\x03\x02\x01";
    if !b[line_end..(line_end + 32).min(b.len())]
        .windows(8)
        .any(|w| w == bom)
    {
        return Err(Error::Unsupported(
            "big-endian or unknown ngpsf byte order".into(),
        ));
    }
    Ok(())
}

struct Record {
    npts: usize,
    prev: u64,
    /// absolute offset of the x (time) Blosc buffer
    x_at: usize,
    y_at: usize,
}

fn hex(s: &str, at: usize) -> Result<u64> {
    u64::from_str_radix(s, 16)
        .or_else(|_| malformed(at, format!("bad hex field {s:?} in ngpsf record")))
}

fn record(b: &[u8], off: usize) -> Result<Record> {
    if b.get(off) != Some(&0) {
        return malformed(off, "ngpsf record does not start with NUL");
    }
    let rest = b.get(off + 1..).unwrap_or(&[]);
    let nl = rest
        .iter()
        .take(256)
        .position(|&c| c == b'\n')
        .ok_or(Error::UnexpectedEof { offset: off })?;
    let hdr = std::str::from_utf8(&rest[..nl]).map_err(|_| Error::Malformed {
        offset: off,
        msg: "non-ASCII record header".into(),
    })?;
    let f: Vec<&str> = hdr.split(':').collect();
    if f.len() != 8 {
        return malformed(
            off,
            format!("ngpsf record header {hdr:?}: expected 8 fields"),
        );
    }
    if f[0] != "3" {
        return Err(Error::Unsupported(format!(
            "ngpsf data type {} (only 3 = f64 seen so far)",
            f[0]
        )));
    }
    let npts = hex(f[3], off)? as usize;
    let ysize = hex(f[7], off)? as usize;
    // payload: skip zero padding up to the first Blosc header (version byte is non-zero)
    let mut p = off + 1 + nl + 1;
    let pad_end = (p + 16).min(b.len());
    while p < pad_end && b[p] == 0 {
        p += 1;
    }
    let (x_at, y_at) = match f[6].split_once('.') {
        Some((back, _xsize)) => {
            let back = hex(back, off)? as usize;
            (
                off.checked_sub(back).ok_or(Error::Malformed {
                    offset: off,
                    msg: "x back-reference before file start".into(),
                })?,
                p,
            )
        }
        None => (p, p + hex(f[6], off)? as usize),
    };
    if y_at.checked_add(ysize).is_none_or(|e| e > b.len()) {
        return Err(Error::UnexpectedEof { offset: y_at });
    }
    Ok(Record {
        npts,
        prev: hex(f[2], off)?,
        x_at,
        y_at,
    })
}

/// Decompresses the Blosc buffer at `at` into f64 values.
fn blosc_f64(b: &[u8], at: usize, expect: usize) -> Result<Vec<f64>> {
    let buf = b
        .get(at..)
        .filter(|s| s.len() >= 16)
        .ok_or(Error::UnexpectedEof { offset: at })?;
    // header: version, versionlz, flags, typesize, nbytes, blocksize, cbytes (u32 LE)
    let nbytes = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    let cbytes = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
    if !(1..=4).contains(&buf[0])
        || buf[3] != 8
        || nbytes != expect * 8
        || cbytes > buf.len()
        || cbytes < 16
    {
        return malformed(
            at,
            format!(
                "unexpected Blosc header (nbytes {nbytes}, want {})",
                expect * 8
            ),
        );
    }
    let mut out = vec![0f64; expect];
    // SAFETY: src holds cbytes valid bytes (checked above); dest has nbytes capacity.
    let n = unsafe {
        blosc_src::blosc_decompress_ctx(buf.as_ptr().cast(), out.as_mut_ptr().cast(), nbytes, 1)
    };
    if n < 0 || n as usize != nbytes {
        return malformed(at, format!("Blosc decompression failed ({n})"));
    }
    Ok(out)
}

/// One signal: identity of its time axis (offsets of the x buffers) and values.
pub(crate) struct Signal {
    pub x_key: Vec<(usize, usize)>,
    pub y: Vec<f64>,
}

/// Reads the chunk chain ending at `last` (oldest chunk first); the time axis is decoded
/// separately with [`read_axis`] so shared axes are decompressed once.
pub(crate) fn read_signal(b: &[u8], last: u64, npoints: u64) -> Result<Signal> {
    let mut chain = Vec::new();
    let mut off = last;
    while off != NONE {
        if chain.len() > 1 << 24 {
            return malformed(off as usize, "ngpsf chunk chain too long (cycle?)");
        }
        let off_us =
            usize::try_from(off).map_err(|_| Error::UnexpectedEof { offset: usize::MAX })?;
        let r = record(b, off_us)?;
        if r.prev != NONE && r.prev >= off {
            return malformed(off_us, "ngpsf prev link does not point backwards");
        }
        off = r.prev;
        chain.push(r);
    }
    chain.reverse();
    let total: usize = chain.iter().map(|r| r.npts).sum();
    if total as u64 != npoints {
        return malformed(
            last as usize,
            format!("ngpsf chain has {total} points, index says {npoints}"),
        );
    }
    let mut y = Vec::with_capacity(total);
    for r in &chain {
        y.extend(blosc_f64(b, r.y_at, r.npts)?);
    }
    Ok(Signal {
        x_key: chain.iter().map(|r| (r.x_at, r.npts)).collect(),
        y,
    })
}

/// Decodes a time axis identified by [`Signal::x_key`].
pub(crate) fn read_axis(b: &[u8], key: &[(usize, usize)]) -> Result<Vec<f64>> {
    let mut x = Vec::with_capacity(key.iter().map(|k| k.1).sum());
    for &(at, n) in key {
        x.extend(blosc_f64(b, at, n)?);
    }
    Ok(x)
}

/// The time axis shared by all signals (decoded once per distinct buffer set).
pub(crate) fn common_axis(b: &[u8], sigs: &[Signal]) -> Result<Column> {
    let Some(first) = sigs.first() else {
        return Ok(Column::Float64(Vec::new()));
    };
    let x = read_axis(b, &first.x_key)?;
    let mut checked: Vec<&[(usize, usize)]> = vec![&first.x_key];
    for s in &sigs[1..] {
        if checked.contains(&s.x_key.as_slice()) {
            continue;
        }
        if read_axis(b, &s.x_key)? != x {
            return Err(Error::Unsupported(
                "PSFXL signals have different time axes; read them one by one with read_signal()"
                    .into(),
            ));
        }
        checked.push(&s.x_key);
    }
    Ok(Column::Float64(x))
}

/// Parses the `.sig` metadata file (`*key 26 value` lines, 0x01 padding).
pub(crate) fn sig_properties(b: &[u8]) -> Properties {
    let text: String = b.iter().filter(|&&c| c != 1).map(|&c| c as char).collect();
    let mut props = Vec::new();
    for line in text.lines() {
        let Some(l) = line.strip_prefix('*') else {
            continue;
        };
        let mut it = l.splitn(3, ' ');
        if let (Some(k), Some(_), Some(v)) = (it.next(), it.next(), it.next()) {
            let v = v.trim();
            let val = v
                .parse::<i64>()
                .map(PropValue::Int)
                .unwrap_or_else(|_| PropValue::String(v.to_owned()));
            props.push((k.to_owned(), val));
        }
    }
    Properties(props)
}
