//! Reader for Cadence Spectre PSF simulation results.
//!
//! Supports psfbin (all value layouts: non-swept, row records incl. groups, windowed transient) and
//! psfascii, and PSFXL (stub + `.psfxl` data file, feature `psfxl`).
//!
//! ```no_run
//! use psfkit::PsfFile;
//! let f = PsfFile::open("ac.ac")?;
//! for t in f.traces() {
//!     println!("{} ({})", t.name, t.type_name);
//! }
//! let d = f.read(&["out"])?;
//! let (re, im) = d.traces[0].1.as_complex_f64().unwrap();
//! # Ok::<(), psfkit::Error>(())
//! ```

mod ascii;
mod binary;
mod column;
mod error;
mod file;
#[cfg(feature = "psfxl")]
mod psfxl;
mod types;

pub use column::Column;
pub use error::{Error, Result};
pub use file::{Format, Layout, PsfFile, SweptData};
pub use types::{
    DataType, Field, Group, NamedValue, PropValue, Properties, Scalar, TypeDef, Value, Variable,
    XlIndex,
};
