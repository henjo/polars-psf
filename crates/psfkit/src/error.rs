use std::path::PathBuf;

/// Errors returned while reading PSF data.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// The file ended before a complete structure could be read.
    #[error("unexpected end of data at offset {offset}")]
    UnexpectedEof { offset: usize },
    /// A structure did not match the expected layout.
    #[error("malformed PSF at offset {offset}: {msg}")]
    Malformed { offset: usize, msg: String },
    /// The file uses a valid but unsupported feature.
    #[error("unsupported PSF feature: {0}")]
    Unsupported(String),
    /// Syntax error in a psfascii file.
    #[error("psfascii syntax error at line {line}: {msg}")]
    Ascii { line: usize, msg: String },
    /// No trace/value with the requested name.
    #[error("no signal named {0:?}")]
    NotFound(String),
    /// The operation needs a swept (or non-swept) file.
    #[error("{0}")]
    WrongKind(&'static str),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

pub(crate) fn malformed<T>(offset: usize, msg: impl Into<String>) -> Result<T> {
    Err(Error::Malformed {
        offset,
        msg: msg.into(),
    })
}
