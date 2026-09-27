//! psfbin reader.
//!
//! Sections are parsed front to back. Each section starts with its code followed by a forward link
//! (`0x15, u32 offset`) that points just past the next section's code word, so the end table is not
//! needed (truncated files and PSFXL stubs have none).

use std::collections::HashMap;

use crate::column::Column;
use crate::error::{Error, Result, malformed};
use crate::types::{
    DataType, Field, Group, NamedValue, PropValue, Properties, Scalar, TypeDef, Value, Variable,
    XlIndex,
};

const DECL: u32 = 0x10;
const GROUP: u32 = 0x11;
const STRUCT_END: u32 = 0x12;
const ZERO_PAD: u32 = 0x14;
const LINK_FWD: u32 = 0x15;
const LINK_BACK: u32 = 0x16;
const XL_INDEX: u32 = 0x24;
const PROP_STRING: u32 = 0x21;
const PROP_INT: u32 = 0x22;
const PROP_DOUBLE: u32 = 0x23;
const SEC_END: u32 = 0xF;

/// True if `b` looks like a psfbin file (first word is a header code).
pub(crate) fn sniff(b: &[u8]) -> bool {
    b.len() >= 8
        && matches!(rd_u32(b, 0), 0 | 0x200 | 0x300 | 0x400 | 0x500)
        && rd_u32(b, 4) == LINK_FWD
}

#[inline]
fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_be_bytes(b[o..o + 4].try_into().unwrap())
}
#[inline]
fn rd_u64(b: &[u8], o: usize) -> u64 {
    u64::from_be_bytes(b[o..o + 8].try_into().unwrap())
}
#[inline]
fn rd_f32(b: &[u8], o: usize) -> f32 {
    f32::from_be_bytes(b[o..o + 4].try_into().unwrap())
}
#[inline]
fn rd_f64(b: &[u8], o: usize) -> f64 {
    f64::from_be_bytes(b[o..o + 8].try_into().unwrap())
}

/// Bounds-checked big-endian cursor.
struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn need(&self, n: usize) -> Result<()> {
        if self.p.checked_add(n).is_some_and(|e| e <= self.b.len()) {
            Ok(())
        } else {
            Err(Error::UnexpectedEof { offset: self.p })
        }
    }
    fn u32(&mut self) -> Result<u32> {
        self.need(4)?;
        let v = rd_u32(self.b, self.p);
        self.p += 4;
        Ok(v)
    }
    fn peek(&self) -> Result<u32> {
        self.need(4)?;
        Ok(rd_u32(self.b, self.p))
    }
    fn peek_is(&self, v: u32) -> bool {
        self.peek().is_ok_and(|x| x == v)
    }
    fn f64(&mut self) -> Result<f64> {
        self.need(8)?;
        let v = rd_f64(self.b, self.p);
        self.p += 8;
        Ok(v)
    }
    fn u64(&mut self) -> Result<u64> {
        self.need(8)?;
        let v = rd_u64(self.b, self.p);
        self.p += 8;
        Ok(v)
    }
    /// Length-prefixed string padded to 4 bytes.
    fn string(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        self.need(n)?;
        let raw = &self.b[self.p..self.p + n];
        // Spectre sometimes counts trailing bytes after a NUL into the length (ENV_VAR_* props)
        let raw = raw.iter().position(|&c| c == 0).map_or(raw, |z| &raw[..z]);
        let s = String::from_utf8_lossy(raw).into_owned();
        self.p += n;
        // padding to 4 bytes; tolerated at end of data
        self.p = (self.p + (4 - n % 4) % 4).min(self.b.len());
        Ok(s)
    }
    fn expect(&mut self, v: u32, what: &str) -> Result<()> {
        let at = self.p;
        let got = self.u32()?;
        if got != v {
            return malformed(at, format!("expected {what} (0x{v:x}), got 0x{got:x}"));
        }
        Ok(())
    }
    /// Skips link chunks; returns (forward, backward) link targets.
    fn links(&mut self) -> Result<(Option<u32>, Option<u32>)> {
        let (mut fwd, mut back) = (None, None);
        loop {
            match self.peek() {
                Ok(LINK_FWD) => {
                    self.p += 4;
                    fwd = Some(self.u32()?);
                }
                Ok(LINK_BACK) => {
                    self.p += 4;
                    back = Some(self.u32()?);
                }
                _ => return Ok((fwd, back)),
            }
        }
    }
}

fn props(c: &mut Cur) -> Result<Properties> {
    let mut v = Vec::new();
    while let Ok(tag @ (PROP_STRING | PROP_INT | PROP_DOUBLE)) = c.peek() {
        c.p += 4;
        let name = c.string()?;
        let val = match tag {
            PROP_STRING => PropValue::String(c.string()?),
            PROP_INT => PropValue::Int(c.u32()? as i32 as i64),
            _ => PropValue::Double(c.f64()?),
        };
        v.push((name, val));
    }
    Ok(Properties(v))
}

/// Decoded psfbin file (declarations + value location).
pub(crate) struct Parsed {
    pub header: Properties,
    pub types: Vec<TypeDef>,
    pub sweeps: Vec<Variable>,
    pub traces: Vec<Variable>,
    pub groups: Vec<Group>,
    pub values: Values,
}

#[derive(Clone)]
pub(crate) enum Values {
    /// No value section (or an empty one).
    None,
    /// Non-swept value declarations `[start, end)`, decoded on demand.
    NonSwept { start: usize, end: usize },
    /// Row-major tagged records `[start, end)`; `truncated` if the section has no valid end.
    Simple {
        start: usize,
        end: usize,
        truncated: bool,
    },
    /// Windowed (transient) records `[start, end)`, `window` bytes per trace per window.
    Windowed {
        start: usize,
        end: usize,
        window: usize,
        truncated: bool,
    },
    /// Values live in a PSFXL data file.
    External,
}

/// Raw type declaration before resolving it into a DataType.
struct RawType {
    id: u32,
    name: String,
    dtype: RawDtype,
    props: Properties,
}

enum RawDtype {
    Scalar(Scalar),
    Struct(Vec<RawType>),
    Array(usize, Box<RawDtype>),
}

fn raw_type(c: &mut Cur) -> Result<RawType> {
    c.expect(DECL, "type declaration")?;
    let id = c.u32()?;
    let name = c.string()?;
    let ndims = c.u32()? as usize;
    if ndims > 16 {
        return malformed(c.p - 4, format!("type {name:?}: {ndims} array dimensions"));
    }
    let mut len = 1usize;
    for _ in 0..ndims {
        len = len.saturating_mul(c.u32()? as usize);
    }
    let at = c.p;
    let code = c.u32()?;
    let mut dtype = match code {
        16 => {
            let mut members = Vec::new();
            while !c.peek_is(STRUCT_END) {
                members.push(raw_type(c)?);
            }
            c.p += 4;
            RawDtype::Struct(members)
        }
        _ => RawDtype::Scalar(
            Scalar::from_code(code)
                .ok_or_else(|| Error::Unsupported(format!("type code {code} at offset {at}")))?,
        ),
    };
    if ndims > 0 {
        dtype = RawDtype::Array(len, Box::new(dtype));
    }
    let props = props(c)?;
    Ok(RawType {
        id,
        name,
        dtype,
        props,
    })
}

fn resolve(raw: &RawDtype, depth: usize) -> Result<DataType> {
    if depth > 32 {
        return Err(Error::Unsupported("type nesting too deep".into()));
    }
    Ok(match raw {
        RawDtype::Scalar(s) => DataType::Scalar(*s),
        RawDtype::Array(n, e) => DataType::Array {
            len: Some(*n),
            elem: Box::new(resolve(e, depth + 1)?),
        },
        RawDtype::Struct(m) => DataType::Struct(
            m.iter()
                .map(|f| {
                    Ok(Field {
                        name: f.name.clone(),
                        dtype: resolve(&f.dtype, depth + 1)?,
                        props: f.props.clone(),
                    })
                })
                .collect::<Result<_>>()?,
        ),
    })
}

struct Decls<'t> {
    types: &'t HashMap<u32, TypeDef>,
    shared: &'t HashMap<u32, std::sync::Arc<DataType>>,
}

impl Decls<'_> {
    fn var(&self, c: &mut Cur, group: Option<usize>) -> Result<Variable> {
        let id = c.u32()?;
        let name = c.string()?;
        let at = c.p;
        let type_id = c.u32()?;
        let t = self.types.get(&type_id).ok_or_else(|| Error::Malformed {
            offset: at,
            msg: format!("{name:?} references unknown type id {type_id}"),
        })?;
        let mut xl = None;
        if c.peek_is(XL_INDEX) {
            c.p += 4;
            xl = Some(XlIndex {
                offset: c.u64()?,
                npoints: c.u64()?,
                x_min: c.f64()?,
                x_max: c.f64()?,
                y_min: c.f64()?,
                y_max: c.f64()?,
            });
        }
        let props = props(c)?;
        Ok(Variable {
            id,
            name,
            type_name: t.name.clone(),
            dtype: self.shared[&type_id].clone(),
            props,
            group,
            xl,
        })
    }
}

pub(crate) fn parse(b: &[u8]) -> Result<Parsed> {
    let mut c = Cur { b, p: 0 };
    let mut out = Parsed {
        header: Properties::default(),
        types: Vec::new(),
        sweeps: Vec::new(),
        traces: Vec::new(),
        groups: Vec::new(),
        values: Values::None,
    };
    let mut types_by_id: HashMap<u32, TypeDef> = HashMap::new();
    let mut shared: HashMap<u32, std::sync::Arc<DataType>> = HashMap::new();
    let mut first = true;
    while c.p + 4 <= b.len() {
        let sec_at = c.p;
        let code = c.u32()?;
        let (fwd, back) = c.links()?;
        // forward link points just past the next section's code word
        let next = fwd
            .map(|f| f as usize)
            .filter(|&f| f > c.p && f <= b.len())
            .map(|f| f - 4);
        let limit = next.unwrap_or(b.len());
        match code {
            0 | 0x200 | 0x300 | 0x400 | 0x500 if first => {
                out.header = props(&mut c)?;
            }
            1 => {
                let end = back.map_or(limit, |x| x as usize).min(limit);
                let mut raws = Vec::new();
                while c.p < end && c.peek_is(DECL) {
                    raws.push(raw_type(&mut c)?);
                }
                for r in &raws {
                    let t = TypeDef {
                        id: r.id,
                        name: r.name.clone(),
                        dtype: resolve(&r.dtype, 0)?,
                        props: r.props.clone(),
                    };
                    shared.insert(r.id, std::sync::Arc::new(t.dtype.clone()));
                    types_by_id.insert(r.id, t.clone());
                    out.types.push(t);
                }
            }
            2 => {
                let d = Decls {
                    types: &types_by_id,
                    shared: &shared,
                };
                while c.p < limit && c.peek_is(DECL) {
                    c.p += 4;
                    out.sweeps.push(d.var(&mut c, None)?);
                }
            }
            3 => {
                let end = back.map_or(limit, |x| x as usize).min(limit);
                let d = Decls {
                    types: &types_by_id,
                    shared: &shared,
                };
                while c.p < end {
                    match c.peek()? {
                        DECL => {
                            c.p += 4;
                            out.traces.push(d.var(&mut c, None)?);
                        }
                        GROUP => {
                            c.p += 4;
                            let id = c.u32()?;
                            let name = c.string()?;
                            let n = c.u32()? as usize;
                            let gi = out.groups.len();
                            let mut members = Vec::with_capacity(n.min(1 << 20));
                            for _ in 0..n {
                                c.expect(DECL, "group member")?;
                                members.push(out.traces.len());
                                out.traces.push(d.var(&mut c, Some(gi))?);
                            }
                            out.groups.push(Group { id, name, members });
                        }
                        _ => break,
                    }
                }
            }
            4 => {
                out.values = if c.p + 4 > b.len() {
                    Values::None
                } else if out.sweeps.is_empty() {
                    let end = back.map_or(limit, |x| x as usize).min(limit);
                    Values::NonSwept { start: c.p, end }
                } else if let Some(w) = out.header.get_i64("PSF window size") {
                    Values::Windowed {
                        start: c.p,
                        end: limit,
                        window: w as usize,
                        truncated: next.is_none(),
                    }
                } else {
                    Values::Simple {
                        start: c.p,
                        end: limit,
                        truncated: next.is_none(),
                    }
                };
            }
            SEC_END => break,
            _ if first => {
                return malformed(sec_at, format!("not a psfbin file (first word 0x{code:x})"));
            }
            _ => return malformed(sec_at, format!("unknown section code 0x{code:x}")),
        }
        first = false;
        match next {
            Some(n) if n > sec_at => c.p = n,
            // no valid forward link: continue after what was parsed, unless inside values
            _ if code == 4 => break,
            _ => {}
        }
    }
    if matches!(out.values, Values::None) && out.traces.iter().any(|t| t.xl.is_some()) {
        out.values = Values::External;
    }
    Ok(out)
}

/// Location of one non-swept value: name, type and offset of the encoded value.
pub(crate) struct NsEntry {
    pub name: String,
    pub type_id: u32,
    pub value_at: usize,
}

/// Scans non-swept declarations without decoding values (values are skipped by size).
pub(crate) fn nonswept_index(
    b: &[u8],
    start: usize,
    end: usize,
    types: &HashMap<u32, &TypeDef>,
) -> Result<Vec<NsEntry>> {
    let mut c = Cur { b, p: start };
    let mut out = Vec::new();
    while c.p < end && c.peek_is(DECL) {
        c.p += 4;
        let _id = c.u32()?;
        let name = c.string()?;
        let at = c.p;
        let type_id = c.u32()?;
        let t = types.get(&type_id).ok_or_else(|| Error::Malformed {
            offset: at,
            msg: format!("{name:?}: unknown type id {type_id}"),
        })?;
        let value_at = c.p;
        skip_value(&mut c, &t.dtype)?;
        props(&mut c)?;
        out.push(NsEntry {
            name,
            type_id,
            value_at,
        });
    }
    Ok(out)
}

/// Decodes one indexed non-swept value and its properties.
pub(crate) fn nonswept_value(b: &[u8], e: &NsEntry, t: &TypeDef) -> Result<NamedValue> {
    let mut c = Cur { b, p: e.value_at };
    let value = decode_value(&mut c, &t.dtype)?;
    let props = props(&mut c)?;
    Ok(NamedValue {
        name: e.name.clone(),
        type_name: t.name.clone(),
        value,
        props,
    })
}

fn skip_value(c: &mut Cur, dt: &DataType) -> Result<()> {
    if let Some(n) = dt.encoded_size() {
        c.need(n)?;
        c.p += n;
        return Ok(());
    }
    match dt {
        DataType::Scalar(_) => {
            let n = c.u32()? as usize;
            c.need(n)?;
            c.p = (c.p + n + (4 - n % 4) % 4).min(c.b.len());
        }
        DataType::Array { len, elem } => {
            let n = len.ok_or_else(|| Error::Unsupported("unbounded array in psfbin".into()))?;
            for _ in 0..n {
                skip_value(c, elem)?;
            }
        }
        DataType::Struct(fields) => {
            for f in fields {
                skip_value(c, &f.dtype)?;
            }
        }
    }
    Ok(())
}

fn decode_value(c: &mut Cur, dt: &DataType) -> Result<Value> {
    Ok(match dt {
        DataType::Scalar(Scalar::String) => Value::String(c.string()?),
        DataType::Scalar(s) => {
            let n = s.encoded_size().unwrap();
            c.need(n)?;
            let v = scalar_at(c.b, c.p, *s);
            c.p += n;
            v
        }
        DataType::Array { len, elem } => {
            let n = len.ok_or_else(|| Error::Unsupported("unbounded array in psfbin".into()))?;
            Value::Array(
                (0..n)
                    .map(|_| decode_value(c, elem))
                    .collect::<Result<_>>()?,
            )
        }
        DataType::Struct(fields) => Value::Struct(
            fields
                .iter()
                .map(|f| Ok((f.name.clone(), decode_value(c, &f.dtype)?)))
                .collect::<Result<_>>()?,
        ),
    })
}

/// Decodes a fixed-size scalar at `o` (caller checked bounds).
fn scalar_at(b: &[u8], o: usize, s: Scalar) -> Value {
    match s {
        Scalar::Int8 => Value::Int(rd_u32(b, o) as i32 as i8 as i64),
        Scalar::Int32 => Value::Int(rd_u32(b, o) as i32 as i64),
        Scalar::Float32 => Value::Float(rd_f32(b, o) as f64),
        Scalar::ComplexFloat32 => Value::Complex(rd_f32(b, o) as f64, rd_f32(b, o + 4) as f64),
        Scalar::Float64 => Value::Float(rd_f64(b, o)),
        Scalar::ComplexFloat64 => Value::Complex(rd_f64(b, o), rd_f64(b, o + 8)),
        Scalar::String => unreachable!("strings are variable size"),
    }
}

// ---------------------------------------------------------------------------------------------
// Swept values

/// Byte offsets of one column's points. All offsets are validated to be in bounds before use.
#[derive(Clone)]
enum Offsets<'w> {
    Strided {
        start: usize,
        stride: usize,
        n: usize,
    },
    /// Per window: offset of the first point and point count; points are `elem` bytes apart.
    Windows {
        wins: &'w [(usize, usize)],
        elem: usize,
    },
}

impl Offsets<'_> {
    #[inline]
    fn for_each(&self, delta: usize, mut f: impl FnMut(usize)) {
        match *self {
            Offsets::Strided { start, stride, n } => {
                let mut o = start + delta;
                for _ in 0..n {
                    f(o);
                    o += stride;
                }
            }
            Offsets::Windows { wins, elem } => {
                for &(first, n) in wins {
                    let mut o = first + delta;
                    for _ in 0..n {
                        f(o);
                        o += elem;
                    }
                }
            }
        }
    }

    fn count(&self) -> usize {
        match self {
            Offsets::Strided { n, .. } => *n,
            Offsets::Windows { wins, .. } => wins.iter().map(|w| w.1).sum(),
        }
    }
}

/// Fills `col` with the fixed-size values of `dt` at `offs + delta`.
fn gather(col: &mut Column, dt: &DataType, b: &[u8], offs: &Offsets, delta: usize) {
    macro_rules! go {
        ($v:expr, $e:expr) => {{
            let v = $v;
            v.reserve(offs.count());
            offs.for_each(delta, |o| v.push($e(o)));
        }};
    }
    match (col, dt) {
        (Column::Int8(v), _) => go!(v, |o| rd_u32(b, o) as i8),
        (Column::Int32(v), _) => go!(v, |o| rd_u32(b, o) as i32),
        (Column::Float32(v), _) => go!(v, |o| rd_f32(b, o)),
        (Column::Float64(v), _) => go!(v, |o| rd_f64(b, o)),
        (Column::ComplexFloat32 { re, im }, _) => {
            go!(re, |o| rd_f32(b, o));
            go!(im, |o| rd_f32(b, o + 4));
        }
        (Column::ComplexFloat64 { re, im }, _) => {
            go!(re, |o| rd_f64(b, o));
            go!(im, |o| rd_f64(b, o + 8));
        }
        (Column::Struct(cols), DataType::Struct(fields)) => {
            let mut fo = 0;
            for ((_, c), f) in cols.iter_mut().zip(fields) {
                gather(c, &f.dtype, b, offs, delta + fo);
                fo += f.dtype.encoded_size().expect("fixed-size struct");
            }
        }
        (Column::Array { width, values }, DataType::Array { elem, .. }) => {
            let es = elem.encoded_size().expect("fixed-size array");
            let w = *width;
            offs.for_each(delta, |o| {
                let one = Offsets::Strided {
                    start: o,
                    stride: es,
                    n: w,
                };
                gather(values, elem, b, &one, 0);
            });
        }
        (Column::String(_), _) => unreachable!("strings are not gathered"),
        (c, dt) => unreachable!("column/type mismatch {c:?} {dt:?}"),
    }
}

/// Borrowed declarations needed to decode swept values.
pub(crate) struct View<'a> {
    pub sweeps: &'a [Variable],
    pub traces: &'a [Variable],
    pub groups: &'a [Group],
    pub values: &'a Values,
    /// Header `PSF sweep points`, if present.
    pub expected_points: Option<usize>,
    /// Called with byte ranges about to be read when only some traces are wanted (lets an mmap
    /// prefetch exactly those ranges instead of relying on read-ahead).
    pub prefetch: Option<&'a (dyn Fn(usize, usize) + Sync)>,
}

/// Swept values of selected traces, plus the sweep column.
pub(crate) struct Swept {
    pub sweep: Column,
    pub traces: Vec<Column>,
}

/// Maps columns in parallel when there is enough work (`bytes` to decode, estimated); small
/// reads stay on the calling thread (thread-pool startup costs ~ms in a fresh process).
pub(crate) fn map_columns<T: Sync, R: Send>(
    items: &[T],
    bytes: usize,
    f: impl Fn(&T) -> Result<R> + Sync + Send,
) -> Result<Vec<R>> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        if items.len() > 1 && bytes >= 1 << 20 {
            return items.par_iter().map(f).collect();
        }
    }
    items.iter().map(f).collect()
}

/// Reads the sweep and the traces `want` (indices into `p.traces`).
/// Byte offset and type of struct member `field` inside `dt`.
fn field_at<'a>(dt: &'a DataType, field: &str) -> Option<(usize, &'a DataType)> {
    let DataType::Struct(fields) = dt else {
        return None;
    };
    let mut off = 0;
    for f in fields {
        if f.name == field {
            return Some((off, &f.dtype));
        }
        off += f.dtype.encoded_size()?;
    }
    None
}

/// Like [`read_swept`], but only struct member `field` of each trace is decoded.
pub(crate) fn read_swept_field(b: &[u8], p: &View, want: &[usize], field: &str) -> Result<Swept> {
    let missing = |i: usize| Error::NotFound(format!("{}.{field}", p.traces[i].name));
    if let (Values::Simple { start, end, .. }, [sweep]) = (p.values, p.sweeps) {
        if let Some(plan) = plan_simple(b, p, sweep, *start, *end)? {
            let n = plan.n;
            let at = |off: usize| Offsets::Strided {
                start: start + off,
                stride: plan.stride,
                n,
            };
            let mut sc = Column::new(&sweep.dtype, n)?;
            gather(&mut sc, &sweep.dtype, b, &at(plan.sweep_off), 0);
            let bytes = n * want.len() * 8;
            let traces = map_columns(want, bytes, |&i| {
                let (off, dt) = field_at(&p.traces[i].dtype, field).ok_or_else(|| missing(i))?;
                let mut c = Column::new(dt, n)?;
                gather(&mut c, dt, b, &at(plan.trace_off[i]), off);
                Ok(c)
            })?;
            return Ok(Swept { sweep: sc, traces });
        }
    }
    // other layouts: decode whole structs, keep the member
    let s = read_swept(b, p, want)?;
    let traces = s
        .traces
        .into_iter()
        .zip(want)
        .map(|(c, &i)| {
            if c.is_empty() {
                Ok(c)
            } else {
                c.into_field(field).ok_or_else(|| missing(i))
            }
        })
        .collect::<Result<_>>()?;
    Ok(Swept {
        sweep: s.sweep,
        traces,
    })
}

pub(crate) fn read_swept(b: &[u8], p: &View, want: &[usize]) -> Result<Swept> {
    let [sweep] = p.sweeps else {
        return Err(Error::Unsupported(format!(
            "{} sweep variables in one file",
            p.sweeps.len()
        )));
    };
    match *p.values {
        Values::Simple {
            start,
            end,
            truncated,
        } => match plan_simple(b, p, sweep, start, end)? {
            Some(plan) => {
                let n = plan.n;
                let at = |off: usize| Offsets::Strided {
                    start: start + off,
                    stride: plan.stride,
                    n,
                };
                let mut sc = Column::new(&sweep.dtype, n)?;
                gather(&mut sc, &sweep.dtype, b, &at(plan.sweep_off), 0);
                let bytes = n * plan.stride.min(1 << 20);
                let traces = map_columns(want, bytes, |&i| {
                    let t = &p.traces[i];
                    let mut c = Column::new(&t.dtype, n)?;
                    gather(&mut c, &t.dtype, b, &at(plan.trace_off[i]), 0);
                    Ok(c)
                })?;
                Ok(Swept { sweep: sc, traces })
            }
            None => read_simple_generic(b, p, sweep, start, end, truncated, want),
        },
        Values::Windowed {
            start,
            end,
            window,
            truncated,
        } => read_windowed(b, p, sweep, start, end, window, truncated, want),
        Values::External => Err(Error::Unsupported(
            "values are in a PSFXL file; open it with the psfxl reader".into(),
        )),
        Values::None => Ok(Swept {
            sweep: Column::new(&sweep.dtype, 0)?,
            traces: want
                .iter()
                .map(|&i| Column::new(&p.traces[i].dtype, 0))
                .collect::<Result<_>>()?,
        }),
        Values::NonSwept { .. } => Err(Error::WrongKind("file is not swept")),
    }
}

struct SimplePlan {
    n: usize,
    stride: usize,
    sweep_off: usize,
    /// Offset of each trace's value inside a record (indexed like `p.traces`).
    trace_off: Vec<usize>,
}

/// Layout of fixed-size row records, validated against the data; None if not applicable.
fn plan_simple(
    b: &[u8],
    p: &View,
    sweep: &Variable,
    start: usize,
    end: usize,
) -> Result<Option<SimplePlan>> {
    let Some(ss) = sweep.dtype.encoded_size() else {
        return Ok(None);
    };
    if p.traces.iter().any(|t| t.dtype.encoded_size().is_none()) || end <= start + 8 + ss {
        return Ok(None);
    }
    let by_id: HashMap<u32, usize> = p
        .traces
        .iter()
        .enumerate()
        .map(|(i, t)| (t.id, i))
        .collect();
    let groups: HashMap<u32, usize> = p
        .groups
        .iter()
        .enumerate()
        .map(|(i, g)| (g.id, i))
        .collect();
    // walk the first record
    if rd_u32(b, start) != DECL || rd_u32(b, start + 4) != sweep.id {
        return Ok(None);
    }
    let mut o = start + 8 + ss;
    let mut trace_off = vec![usize::MAX; p.traces.len()];
    let mut tags = Vec::new(); // (relative offset, tag, id) of entry headers
    while o + 8 <= end {
        let (tag, id) = (rd_u32(b, o), rd_u32(b, o + 4));
        if tag == DECL && id == sweep.id {
            break;
        }
        tags.push((o - start, tag, id));
        o += 8;
        match tag {
            DECL => {
                let Some(&i) = by_id.get(&id) else {
                    return Ok(None);
                };
                trace_off[i] = o - start;
                o += p.traces[i].dtype.encoded_size().unwrap();
            }
            GROUP => {
                let Some(&g) = groups.get(&id) else {
                    return Ok(None);
                };
                for &i in &p.groups[g].members {
                    trace_off[i] = o - start;
                    o += p.traces[i].dtype.encoded_size().unwrap();
                }
            }
            _ => return Ok(None),
        }
    }
    let stride = o - start;
    if trace_off.contains(&usize::MAX) || (end - start) % stride != 0 {
        return Ok(None);
    }
    let n = (end - start) / stride;
    // every record must start with the sweep tag; entry tags checked on a sample of records
    for r in 0..n {
        let ro = start + r * stride;
        if rd_u32(b, ro) != DECL || rd_u32(b, ro + 4) != sweep.id {
            return Ok(None);
        }
        if (r % 1024 == 0 || r == n - 1)
            && tags
                .iter()
                .any(|&(to, tag, id)| rd_u32(b, ro + to) != tag || rd_u32(b, ro + to + 4) != id)
        {
            return Ok(None);
        }
    }
    Ok(Some(SimplePlan {
        n,
        stride,
        sweep_off: 8,
        trace_off,
    }))
}

/// Tag-driven fallback for records with variable-size values or irregular layout.
fn read_simple_generic(
    b: &[u8],
    p: &View,
    sweep: &Variable,
    start: usize,
    end: usize,
    truncated: bool,
    want: &[usize],
) -> Result<Swept> {
    let mut slot = vec![None; p.traces.len()];
    for (k, &i) in want.iter().enumerate() {
        slot[i] = Some(k);
    }
    let by_id: HashMap<u32, usize> = p
        .traces
        .iter()
        .enumerate()
        .map(|(i, t)| (t.id, i))
        .collect();
    let groups: HashMap<u32, usize> = p
        .groups
        .iter()
        .enumerate()
        .map(|(i, g)| (g.id, i))
        .collect();
    let mut sweep_col = Column::new(&sweep.dtype, 0)?;
    let mut cols: Vec<Column> = want
        .iter()
        .map(|&i| Column::new(&p.traces[i].dtype, 0))
        .collect::<Result<_>>()?;
    let mut c = Cur {
        b: &b[..end],
        p: start,
    };
    let mut body = || -> Result<()> {
        let mut one = |c: &mut Cur, i: usize| -> Result<()> {
            let v = decode_value(c, &p.traces[i].dtype)?;
            if let Some(k) = slot[i] {
                cols[k].push_value(&v)?;
            }
            Ok(())
        };
        while c.p + 8 <= end {
            let at = c.p;
            let tag = c.u32()?;
            let id = c.u32()?;
            match tag {
                DECL if id == sweep.id => {
                    sweep_col.push_value(&decode_value(&mut c, &sweep.dtype)?)?
                }
                DECL => match by_id.get(&id) {
                    Some(&i) => one(&mut c, i)?,
                    None => return malformed(at, format!("value for unknown trace id {id}")),
                },
                GROUP => match groups.get(&id) {
                    Some(&g) => {
                        for &i in &p.groups[g].members {
                            one(&mut c, i)?;
                        }
                    }
                    None => return malformed(at, format!("value for unknown group id {id}")),
                },
                _ => break,
            }
        }
        Ok(())
    };
    match body() {
        Err(Error::UnexpectedEof { .. }) if truncated => {}
        r => r?,
    }
    let n = sweep_col.len();
    if truncated {
        // keep complete points only
        let m = cols.iter().map(Column::len).fold(n, usize::min);
        sweep_col.truncate(m);
        cols.iter_mut().for_each(|c| c.truncate(m));
    } else if let Some((k, col)) = cols
        .iter()
        .enumerate()
        .find(|(_, col)| col.len() != n && !col.is_empty())
    {
        // traces without any value (e.g. parent sweep files) are returned empty
        return malformed(
            start,
            format!(
                "trace {:?} has {} values, sweep has {n}",
                p.traces[want[k]].name,
                col.len()
            ),
        );
    }
    Ok(Swept {
        sweep: sweep_col,
        traces: cols,
    })
}

/// Windowed layout: `DECL, u32 (x<<16 | n), n sweep values, then per trace a `window`-byte block
/// with its n values right-aligned`. Zero-pad chunks may appear between windows.
#[allow(clippy::too_many_arguments)]
fn read_windowed(
    b: &[u8],
    p: &View,
    sweep: &Variable,
    start: usize,
    end: usize,
    window: usize,
    truncated: bool,
    want: &[usize],
) -> Result<Swept> {
    let ss = sweep
        .dtype
        .encoded_size()
        .ok_or_else(|| Error::Unsupported("variable-size sweep in windowed file".into()))?;
    let sizes: Vec<usize> = p
        .traces
        .iter()
        .map(|t| {
            t.dtype.encoded_size().ok_or_else(|| {
                Error::Unsupported(format!("variable-size trace {:?} in windowed file", t.name))
            })
        })
        .collect::<Result<_>>()?;
    let ntr = p.traces.len();
    let end = end.min(b.len());
    // (offset of first sweep value, n, offset of first trace block)
    let mut wins: Vec<(usize, usize, usize)> = Vec::new();
    let mut o = start;
    while o + 8 <= end {
        match rd_u32(b, o) {
            ZERO_PAD => {
                let n = rd_u32(b, o + 4) as usize;
                o = o.saturating_add(8 + n);
            }
            DECL => {
                let n = (rd_u32(b, o + 4) & 0xffff) as usize;
                let sv = o + 8;
                let blocks = sv + n * ss;
                let next = blocks + ntr * window;
                if n * sizes.iter().copied().max().unwrap_or(0) > window {
                    return malformed(
                        o,
                        format!("window with {n} points exceeds window size {window}"),
                    );
                }
                if next > end {
                    break; // truncated file: drop incomplete window
                }
                wins.push((sv, n, blocks));
                o = next;
            }
            // unused tail of Spectre's window buffer
            _ => break,
        }
    }
    let got: usize = wins.iter().map(|w| w.1).sum();
    if let Some(exp) = p.expected_points.filter(|&e| !truncated && e != got) {
        return malformed(
            start,
            format!("windowed data has {got} points, header says {exp}"),
        );
    }
    if let Some(pf) = p.prefetch {
        for &(sv, n, blocks) in &wins {
            pf(sv, n * ss);
            for &i in want {
                pf(blocks + i * window + window - n * sizes[i], n * sizes[i]);
            }
        }
    }
    let sw: Vec<(usize, usize)> = wins.iter().map(|&(sv, n, _)| (sv, n)).collect();
    let mut sc = Column::new(&sweep.dtype, 0)?;
    gather(
        &mut sc,
        &sweep.dtype,
        b,
        &Offsets::Windows {
            wins: &sw,
            elem: ss,
        },
        0,
    );
    let bytes =
        wins.iter().map(|w| w.1).sum::<usize>() * want.iter().map(|&i| sizes[i]).sum::<usize>();
    let traces = map_columns(want, bytes, |&i| {
        let size = sizes[i];
        let tw: Vec<(usize, usize)> = wins
            .iter()
            .map(|&(_, n, blocks)| (blocks + i * window + window - n * size, n))
            .collect();
        let mut c = Column::new(&p.traces[i].dtype, 0)?;
        gather(
            &mut c,
            &p.traces[i].dtype,
            b,
            &Offsets::Windows {
                wins: &tw,
                elem: size,
            },
            0,
        );
        Ok(c)
    })?;
    Ok(Swept { sweep: sc, traces })
}
