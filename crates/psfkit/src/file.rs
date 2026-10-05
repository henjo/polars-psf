use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::ascii;
use crate::binary::{self, Values};
use crate::column::Column;
use crate::error::{Error, Result};
use crate::types::{Group, NamedValue, Properties, TypeDef, Variable};

/// On-disk encoding of a PSF file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// psfbin (big-endian binary).
    Binary,
    /// psfascii.
    Ascii,
}

enum Data {
    Mmap(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for Data {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Data::Mmap(m) => m,
            Data::Owned(v) => v,
        }
    }
}

enum Body {
    Binary(binary::Values),
    /// Offset/line of the VALUE section, parsed on first use.
    Ascii(Option<(usize, usize)>),
}

/// How values are laid out on disk; decides whether reading one signal is selective.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Layout {
    /// Operating point / info values (non-swept).
    NonSwept,
    /// psfbin row records (dc, ac, noise, ...): every signal is on every page.
    Rows,
    /// psfbin windowed transient: each signal has its own blocks.
    Windowed,
    /// PSFXL: per-signal compressed records in a separate data file.
    Psfxl,
    /// psfascii (parsed as a whole).
    Ascii,
    /// No values.
    Empty,
}

/// Swept data: the sweep column and one column per requested trace.
#[derive(Clone, Debug, PartialEq)]
pub struct SweptData {
    pub sweep_name: String,
    pub sweep: Column,
    /// `(trace name, values)` in request order.
    pub traces: Vec<(String, Column)>,
}

/// Initializes `cell` with `f` unless already set; errors are returned, not cached.
fn lazy<T>(cell: &OnceLock<T>, f: impl FnOnce() -> Result<T>) -> Result<&T> {
    if let Some(v) = cell.get() {
        return Ok(v);
    }
    let v = f()?;
    Ok(cell.get_or_init(|| v))
}

/// A PSF file (psfbin, psfascii, or a PSFXL stub with its data file).
///
/// Everything is lazy: opening maps the file and parses only the declarations (header, types,
/// sweep and trace names). Values are decoded on request and only for the requested signals;
/// non-swept values are indexed on first lookup and decoded one by one.
///
/// ```no_run
/// let f = psfkit::PsfFile::open("tran.tran")?;
/// let data = f.read(&["out"])?;
/// let t = data.sweep.as_f64().unwrap();
/// # Ok::<(), psfkit::Error>(())
/// ```
pub struct PsfFile {
    data: Data,
    format: Format,
    header: Properties,
    types: Vec<TypeDef>,
    sweeps: Vec<Variable>,
    traces: Vec<Variable>,
    /// Traces renamed because their name was taken: (trace index, name in the file).
    renamed: Vec<(usize, String)>,
    groups: Vec<Group>,
    body: Body,
    by_name: OnceLock<HashMap<String, usize>>,
    ns_index: OnceLock<Vec<binary::NsEntry>>,
    ns_values: OnceLock<Vec<NamedValue>>,
    ascii_values: OnceLock<ascii::AsciiValues>,
    /// PSFXL data file (`<stub>.psfxl`) and lazily read `.sig` metadata.
    xl_data: Option<Data>,
    sig_path: Option<PathBuf>,
    xl_meta: OnceLock<Properties>,
}

impl PsfFile {
    /// Opens and memory-maps a file; the format is detected from its content.
    pub fn open(path: impl AsRef<Path>) -> Result<PsfFile> {
        let path = path.as_ref();
        let map = map_file(path)?;
        let mut f = Self::from_data(Data::Mmap(map))?;
        if f.is_psfxl_stub() {
            let sibling = |ext: &str| {
                let mut p = path.as_os_str().to_owned();
                p.push(ext);
                PathBuf::from(p)
            };
            let data = sibling(".psfxl");
            if data.exists() {
                f.attach_psfxl(Data::Mmap(map_file(&data)?))?;
            }
            f.sig_path = Some(sibling(".sig"));
        }
        Ok(f)
    }

    /// Attaches the PSFXL data file of a stub opened with [`Self::from_bytes`].
    pub fn with_psfxl_bytes(mut self, bytes: Vec<u8>) -> Result<PsfFile> {
        self.attach_psfxl(Data::Owned(bytes))?;
        Ok(self)
    }

    fn attach_psfxl(&mut self, data: Data) -> Result<()> {
        #[cfg(feature = "psfxl")]
        crate::psfxl::check_header(&data)?;
        self.xl_data = Some(data);
        Ok(())
    }

    /// PSFXL `.sig` metadata (`cdnshsweepcount`, `cdnshnumflushes`, ...), read on first use;
    /// empty for other formats.
    pub fn psfxl_meta(&self) -> &Properties {
        self.xl_meta.get_or_init(|| {
            #[cfg(feature = "psfxl")]
            if let Some(sig) = self.sig_path.as_ref().and_then(|p| std::fs::read(p).ok()) {
                return crate::psfxl::sig_properties(&sig);
            }
            Properties::default()
        })
    }

    /// Parses a file held in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<PsfFile> {
        Self::from_data(Data::Owned(bytes))
    }

    fn from_data(data: Data) -> Result<PsfFile> {
        let (format, header, types, sweeps, mut traces, groups, body) = if binary::sniff(&data) {
            let p = binary::parse(&data)?;
            (
                Format::Binary,
                p.header,
                p.types,
                p.sweeps,
                p.traces,
                p.groups,
                Body::Binary(p.values),
            )
        } else if ascii::sniff(&data) {
            let a = ascii::parse(&data)?;
            let at = a.values_at;
            (
                Format::Ascii,
                a.header,
                a.types,
                a.sweeps,
                a.traces,
                a.groups,
                Body::Ascii(at),
            )
        } else {
            return Err(Error::Malformed {
                offset: 0,
                msg: "neither psfbin nor psfascii".into(),
            });
        };
        let renamed = unique_names(&mut traces);
        Ok(PsfFile {
            data,
            format,
            header,
            types,
            sweeps,
            traces,
            renamed,
            groups,
            body,
            by_name: OnceLock::new(),
            ns_index: OnceLock::new(),
            ns_values: OnceLock::new(),
            ascii_values: OnceLock::new(),
            xl_data: None,
            sig_path: None,
            xl_meta: OnceLock::new(),
        })
    }

    pub fn format(&self) -> Format {
        self.format
    }

    /// Header properties (`PSFversion`, `simulator`, `analysis type`, ...).
    pub fn header(&self) -> &Properties {
        &self.header
    }

    pub fn types(&self) -> &[TypeDef] {
        &self.types
    }

    /// Sweep variables (0 for non-swept files, normally 1 otherwise).
    pub fn sweeps(&self) -> &[Variable] {
        &self.sweeps
    }

    /// Trace declarations, group members flattened in file order.
    pub fn traces(&self) -> &[Variable] {
        &self.traces
    }

    /// Traces whose name occurs earlier in the file (Spectre can write one name twice): they are
    /// renamed `name#<trace id>` in [`Self::traces`] (psfascii: `name#<position>`). Pairs of (trace index, name in the
    /// file); empty for most files.
    pub fn renamed_traces(&self) -> &[(usize, String)] {
        &self.renamed
    }

    pub fn groups(&self) -> &[Group] {
        &self.groups
    }

    pub fn is_swept(&self) -> bool {
        !self.sweeps.is_empty()
    }

    /// False if the value section is cut short (killed or still running simulation); reads then
    /// return the complete points only.
    pub fn is_complete(&self) -> bool {
        if self.is_psfxl_stub() {
            return self.psfxl_complete();
        }
        !matches!(
            self.body,
            Body::Binary(
                Values::Simple {
                    truncated: true,
                    ..
                } | Values::Windowed {
                    truncated: true,
                    ..
                }
            )
        )
    }

    /// PSFXL: compares the time reached (stub index `x_max`) with the planned `stop` in the
    /// header. Without a `stop` property the file is assumed complete.
    fn psfxl_complete(&self) -> bool {
        let reached = self
            .traces
            .iter()
            .filter_map(|t| t.xl.map(|x| x.x_max))
            .fold(f64::NEG_INFINITY, f64::max);
        match self.header.get_f64("stop") {
            Some(stop) if reached.is_finite() => reached >= stop - 1e-9 * stop.abs(),
            _ => true,
        }
    }

    /// Value layout; reading a few signals is selective only for `Windowed` and `Psfxl`.
    pub fn layout(&self) -> Layout {
        match &self.body {
            Body::Ascii(_) => Layout::Ascii,
            Body::Binary(Values::NonSwept { .. }) => Layout::NonSwept,
            Body::Binary(Values::Simple { .. }) => Layout::Rows,
            Body::Binary(Values::Windowed { .. }) => Layout::Windowed,
            Body::Binary(Values::External) => Layout::Psfxl,
            Body::Binary(Values::None) => Layout::Empty,
        }
    }

    /// True if the values are stored in a separate PSFXL data file.
    pub fn is_psfxl_stub(&self) -> bool {
        matches!(self.body, Body::Binary(Values::External))
    }

    fn ascii(&self) -> Result<&ascii::AsciiValues> {
        let Body::Ascii(at) = self.body else {
            return Err(Error::WrongKind("not a psfascii file"));
        };
        lazy(&self.ascii_values, || {
            let d = ascii::Decls {
                types: &self.types,
                sweeps: &self.sweeps,
                traces: &self.traces,
                renamed: &self.renamed,
                groups: &self.groups,
            };
            ascii::parse_values(&self.data, at, &d)
        })
    }

    fn types_by_id(&self) -> HashMap<u32, &TypeDef> {
        self.types.iter().map(|t| (t.id, t)).collect()
    }

    /// Non-swept psfbin index (names + offsets), built on first use without decoding values.
    fn ns_index(&self) -> Result<&[binary::NsEntry]> {
        let Body::Binary(Values::NonSwept { start, end }) = self.body else {
            return Ok(&[]);
        };
        lazy(&self.ns_index, || {
            binary::nonswept_index(&self.data, start, end, &self.types_by_id())
        })
        .map(Vec::as_slice)
    }

    /// Signal names: traces for swept files, value names for non-swept files.
    pub fn names(&self) -> Result<Vec<&str>> {
        if self.is_swept() {
            return Ok(self.traces.iter().map(|t| t.name.as_str()).collect());
        }
        Ok(match self.body {
            Body::Ascii(_) => self
                .ascii()?
                .nonswept
                .iter()
                .map(|v| v.name.as_str())
                .collect(),
            _ => self.ns_index()?.iter().map(|e| e.name.as_str()).collect(),
        })
    }

    /// Index of a trace (swept) or value (non-swept) by name.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        let map = lazy(&self.by_name, || {
            let names = self.names()?;
            let mut m = HashMap::with_capacity(names.len());
            for (i, n) in names.into_iter().enumerate() {
                m.entry(n.to_owned()).or_insert(i);
            }
            Ok(m)
        })
        .ok()?;
        map.get(name).copied()
    }

    /// All values of a non-swept file (operating point, info, logFile), decoded on first call.
    /// Empty for swept files. Use [`Self::value`] to decode single values.
    pub fn values(&self) -> Result<&[NamedValue]> {
        if self.is_swept() {
            return Ok(&[]);
        }
        match self.body {
            Body::Ascii(_) => Ok(&self.ascii()?.nonswept),
            _ => lazy(&self.ns_values, || {
                let types = self.types_by_id();
                self.ns_index()?
                    .iter()
                    .map(|e| binary::nonswept_value(&self.data, e, types[&e.type_id]))
                    .collect()
            })
            .map(Vec::as_slice),
        }
    }

    /// One non-swept value, decoded on demand.
    pub fn value(&self, name: &str) -> Result<NamedValue> {
        let not_found = || Error::NotFound(name.to_owned());
        if self.is_swept() {
            return Err(Error::WrongKind("file is swept; use read()"));
        }
        let i = self.index_of(name).ok_or_else(not_found)?;
        if let Some(all) = self.ns_values.get() {
            return Ok(all[i].clone());
        }
        match self.body {
            Body::Ascii(_) => Ok(self.ascii()?.nonswept[i].clone()),
            _ => {
                let e = &self.ns_index()?[i];
                binary::nonswept_value(&self.data, e, self.types_by_id()[&e.type_id])
            }
        }
    }

    /// Reads the sweep and the named traces in one pass. Unknown names give [`Error::NotFound`].
    pub fn read(&self, names: &[&str]) -> Result<SweptData> {
        let idx = names
            .iter()
            .map(|n| {
                self.index_of(n)
                    .ok_or_else(|| Error::NotFound((*n).to_owned()))
            })
            .collect::<Result<Vec<_>>>()?;
        self.read_indices(&idx)
    }

    /// Reads the sweep and all traces.
    pub fn read_all(&self) -> Result<SweptData> {
        let idx: Vec<usize> = (0..self.traces.len()).collect();
        self.read_indices(&idx)
    }

    /// Reads the sweep and struct member `field` of the named traces (e.g. `"total"` of noise
    /// contributions). On row-record files only that member is decoded.
    pub fn read_field(&self, names: &[&str], field: &str) -> Result<SweptData> {
        let idx = names
            .iter()
            .map(|n| {
                self.index_of(n)
                    .ok_or_else(|| Error::NotFound((*n).to_owned()))
            })
            .collect::<Result<Vec<_>>>()?;
        self.read_impl(&idx, Some(field))
    }

    /// [`Self::read_field`] for all traces.
    pub fn read_all_field(&self, field: &str) -> Result<SweptData> {
        let idx: Vec<usize> = (0..self.traces.len()).collect();
        self.read_impl(&idx, Some(field))
    }

    /// Reads the sweep and the traces at the given indices (into [`Self::traces`]).
    pub fn read_indices(&self, idx: &[usize]) -> Result<SweptData> {
        self.read_impl(idx, None)
    }

    fn read_impl(&self, idx: &[usize], field: Option<&str>) -> Result<SweptData> {
        if !self.is_swept() {
            return Err(Error::WrongKind("file is not swept; use values()"));
        }
        if let Some(&bad) = idx.iter().find(|&&i| i >= self.traces.len()) {
            return Err(Error::NotFound(format!("trace index {bad}")));
        }
        let (sweep, cols) = match &self.body {
            Body::Binary(Values::External) => self.read_psfxl(idx)?,
            Body::Binary(values) => {
                // Selective reads of a mapped file: disable read-ahead (it would pull in the
                // neighbouring traces' blocks) and prefetch exactly the needed ranges instead.
                #[cfg(unix)]
                let selective = match &self.data {
                    // hints only help when data must come from disk; on a cached file they
                    // just disable fault-around
                    // only windowed (transient) files let a few traces skip most pages; row-record
                    // files interleave all traces on every page
                    Data::Mmap(m)
                        if matches!(values, Values::Windowed { .. })
                            && idx.len() * 2 < self.traces.len()
                            && !mostly_resident(m) =>
                    {
                        Some(m)
                    }
                    _ => None,
                };
                #[cfg(not(unix))]
                let selective: Option<&memmap2::Mmap> = None;
                #[cfg(unix)]
                let prefetch_fn = |off: usize, len: usize| {
                    if let Some(m) = selective {
                        let _ = m.advise_range(memmap2::Advice::WillNeed, off, len);
                    }
                };
                #[cfg(not(unix))]
                let prefetch_fn = |_: usize, _: usize| {};
                #[cfg(unix)]
                if let Some(m) = selective {
                    let _ = m.advise(memmap2::Advice::Random);
                }
                let p = binary::View {
                    sweeps: &self.sweeps,
                    traces: &self.traces,
                    groups: &self.groups,
                    values,
                    expected_points: self
                        .header
                        .get_i64("PSF sweep points")
                        .and_then(|n| usize::try_from(n).ok()),
                    prefetch: selective.map(|_| &prefetch_fn as &(dyn Fn(usize, usize) + Sync)),
                };
                let s = match field {
                    Some(fl) => binary::read_swept_field(&self.data, &p, idx, fl),
                    None => binary::read_swept(&self.data, &p, idx),
                };
                #[cfg(unix)]
                if let Some(m) = selective {
                    let _ = m.advise(memmap2::Advice::Normal);
                }
                let s = s?;
                (s.sweep, s.traces)
            }
            Body::Ascii(_) => {
                let v = self.ascii()?;
                let sweep = v
                    .sweep_col
                    .clone()
                    .ok_or(Error::WrongKind("file has no values"))?;
                (
                    sweep,
                    idx.iter().map(|&i| v.trace_cols[i].clone()).collect(),
                )
            }
        };
        // binary files decode only the member; other formats extract it here
        let cols: Vec<Column> = match field {
            Some(fl) if !matches!(self.body, Body::Binary(ref v) if !matches!(v, Values::External)) => {
                cols.into_iter()
                    .zip(idx)
                    .map(|(c, &i)| {
                        if c.is_empty() {
                            Ok(c)
                        } else {
                            c.into_field(fl).ok_or_else(|| {
                                Error::NotFound(format!("{}.{fl}", self.traces[i].name))
                            })
                        }
                    })
                    .collect::<Result<_>>()?
            }
            _ => cols,
        };
        Ok(SweptData {
            sweep_name: self.sweeps[0].name.clone(),
            sweep,
            traces: idx
                .iter()
                .map(|&i| self.traces[i].name.clone())
                .zip(cols)
                .collect(),
        })
    }

    /// Reads one trace with its own sweep axis. Works for every format; needed for PSFXL files
    /// whose signals do not share a time axis.
    pub fn read_signal(&self, name: &str) -> Result<(Column, Column)> {
        let i = self
            .index_of(name)
            .ok_or_else(|| Error::NotFound(name.to_owned()))?;
        if matches!(self.body, Body::Binary(Values::External)) {
            let s = self.psfxl_signal(i)?;
            return Ok((Column::Float64(s.0), Column::Float64(s.1)));
        }
        let mut d = self.read_indices(&[i])?;
        Ok((d.sweep, d.traces.pop().unwrap().1))
    }

    #[cfg(feature = "psfxl")]
    fn psfxl_signal(&self, i: usize) -> Result<(Vec<f64>, Vec<f64>)> {
        let s = self.psfxl_read(&[i])?.pop().unwrap();
        let x = crate::psfxl::read_axis(self.xl_data.as_deref().unwrap(), &s.x_key)?;
        Ok((x, s.y))
    }

    #[cfg(not(feature = "psfxl"))]
    fn psfxl_signal(&self, _i: usize) -> Result<(Vec<f64>, Vec<f64>)> {
        Err(Error::Unsupported(
            "PSFXL support disabled (feature `psfxl`)".into(),
        ))
    }

    #[cfg(feature = "psfxl")]
    fn psfxl_read(&self, idx: &[usize]) -> Result<Vec<crate::psfxl::Signal>> {
        let data = self
            .xl_data
            .as_deref()
            .ok_or_else(|| Error::Unsupported("PSFXL data file (<stub>.psfxl) not found".into()))?;
        let bytes = idx
            .iter()
            .filter_map(|&i| self.traces[i].xl)
            .map(|x| x.npoints as usize * 16)
            .sum();
        binary::map_columns(idx, bytes, |&i| {
            let t = &self.traces[i];
            let xl = t.xl.ok_or_else(|| {
                Error::Unsupported(format!("trace {:?} has no PSFXL index", t.name))
            })?;
            crate::psfxl::read_signal(data, xl.offset, xl.npoints)
        })
    }

    #[cfg(feature = "psfxl")]
    fn read_psfxl(&self, idx: &[usize]) -> Result<(Column, Vec<Column>)> {
        if idx.is_empty() && !self.traces.is_empty() {
            // sweep only: PSFXL has no separate sweep, take the first signal's time axis
            let s = self.psfxl_read(&[0])?.pop().unwrap();
            let x = crate::psfxl::read_axis(self.xl_data.as_deref().unwrap(), &s.x_key)?;
            return Ok((Column::Float64(x), Vec::new()));
        }
        let sigs = self.psfxl_read(idx)?;
        let sweep = crate::psfxl::common_axis(self.xl_data.as_deref().unwrap(), &sigs)?;
        Ok((
            sweep,
            sigs.into_iter().map(|s| Column::Float64(s.y)).collect(),
        ))
    }

    #[cfg(not(feature = "psfxl"))]
    fn read_psfxl(&self, _idx: &[usize]) -> Result<(Column, Vec<Column>)> {
        Err(Error::Unsupported(
            "PSFXL support disabled (feature `psfxl`)".into(),
        ))
    }
}

fn map_file(path: &Path) -> Result<memmap2::Mmap> {
    let io = |source| Error::Io {
        path: path.to_owned(),
        source,
    };
    let file = std::fs::File::open(path).map_err(io)?;
    // SAFETY: read-only map; concurrent truncation by another process is the usual mmap caveat.
    unsafe { memmap2::Mmap::map(&file) }.map_err(io)
}

/// True if at least 90% of a sample of 256 pages of the mapping is in the page cache
/// (`mincore` over the whole mapping would cost ~90 ms for a 2.4 GB file).
#[cfg(unix)]
fn mostly_resident(m: &memmap2::Mmap) -> bool {
    // SAFETY: sysconf has no preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as usize;
    let pages = m.len() / page;
    if pages == 0 {
        return true;
    }
    let samples = pages.min(256);
    let mut resident = 0;
    for k in 0..samples {
        let off = (k * pages / samples) * page;
        let mut v = 0u8;
        // SAFETY: one page inside our own mapping; v receives one byte.
        let r = unsafe {
            libc::mincore(
                m.as_ptr().add(off) as *mut libc::c_void,
                page,
                (&mut v as *mut u8).cast(),
            )
        };
        if r != 0 {
            return false;
        }
        resident += (v & 1) as usize;
    }
    resident * 10 >= samples * 9
}

/// Renames traces whose name is taken by an earlier trace to `name#<id>` (trace ID; psfascii has
/// none, there it is the trace's position); returns (index, original name) of the renamed ones.
fn unique_names(traces: &mut [Variable]) -> Vec<(usize, String)> {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut dups = Vec::new();
    for (i, t) in traces.iter().enumerate() {
        if !seen.insert(t.name.clone()) {
            dups.push(i);
        }
    }
    dups.into_iter()
        .map(|i| {
            let orig = traces[i].name.clone();
            let mut name = format!("{orig}#{}", traces[i].id);
            while seen.contains(&name) {
                name.push('_'); // a trace already has that name (unlikely)
            }
            seen.insert(name.clone());
            traces[i].name = name;
            (i, orig)
        })
        .collect()
}
