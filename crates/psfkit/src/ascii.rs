//! psfascii reader. Files are small in practice (logFile, runObjFile, info), so values are decoded
//! eagerly into columns.

use std::collections::HashMap;

use crate::column::Column;
use crate::error::{Error, Result};
use crate::types::{
    DataType, Field, Group, NamedValue, PropValue, Properties, Scalar, TypeDef, Value, Variable,
};

/// True if `b` looks like a psfascii file.
pub(crate) fn sniff(b: &[u8]) -> bool {
    let s = b
        .iter()
        .position(|c| !c.is_ascii_whitespace())
        .map_or(&b[..0], |i| &b[i..]);
    s.starts_with(b"HEADER")
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Str(String),
    /// Number, keeping the raw text to distinguish ints from floats.
    Num(f64, bool),
    Word(String),
    LParen,
    RParen,
    Star,
}

struct Lexer<'a> {
    s: &'a [u8],
    p: usize,
    line: usize,
    peeked: Option<(Tok, usize)>,
}

impl<'a> Lexer<'a> {
    fn err<T>(&self, msg: impl Into<String>) -> Result<T> {
        Err(Error::Ascii {
            line: self.line,
            msg: msg.into(),
        })
    }

    fn lex(&mut self) -> Result<Option<Tok>> {
        let s = self.s;
        while self.p < s.len() && s[self.p].is_ascii_whitespace() {
            if s[self.p] == b'\n' {
                self.line += 1;
            }
            self.p += 1;
        }
        if self.p >= s.len() {
            return Ok(None);
        }
        let c = s[self.p];
        Ok(Some(match c {
            b'(' => {
                self.p += 1;
                Tok::LParen
            }
            b')' => {
                self.p += 1;
                Tok::RParen
            }
            b'*' => {
                self.p += 1;
                Tok::Star
            }
            b'"' => {
                self.p += 1;
                let mut out = Vec::new();
                loop {
                    match s.get(self.p) {
                        None => return self.err("unterminated string"),
                        Some(b'"') => {
                            self.p += 1;
                            break;
                        }
                        // backslashes are literal (escaped Spectre names like `M1\&2`), except `\"`
                        Some(b'\\') if s.get(self.p + 1) == Some(&b'"') => {
                            out.push(b'"');
                            self.p += 2;
                        }
                        Some(&ch) => {
                            if ch == b'\n' {
                                self.line += 1;
                            }
                            out.push(ch);
                            self.p += 1;
                        }
                    }
                }
                Tok::Str(String::from_utf8_lossy(&out).into_owned())
            }
            _ => {
                let st = self.p;
                while self.p < s.len()
                    && !s[self.p].is_ascii_whitespace()
                    && !b"()\"".contains(&s[self.p])
                {
                    self.p += 1;
                }
                let w = std::str::from_utf8(&s[st..self.p]).unwrap_or("");
                if w.as_bytes()[0].is_ascii_alphabetic()
                    && !matches!(w.to_ascii_lowercase().as_str(), "nan" | "inf" | "infinity")
                {
                    Tok::Word(w.to_owned())
                } else {
                    match w.parse::<f64>() {
                        Ok(v) => Tok::Num(v, !w.contains(['.', 'e', 'E', 'n', 'N', 'i', 'I'])),
                        Err(_) => return self.err(format!("bad token {w:?}")),
                    }
                }
            }
        }))
    }

    fn peek(&mut self) -> Result<Option<&Tok>> {
        if self.peeked.is_none() {
            let line = self.line;
            if let Some(t) = self.lex()? {
                self.peeked = Some((t, line));
            }
        }
        Ok(self.peeked.as_ref().map(|(t, _)| t))
    }

    fn next(&mut self) -> Result<Tok> {
        self.peek()?;
        match self.peeked.take() {
            Some((t, _)) => Ok(t),
            None => self.err("unexpected end of file"),
        }
    }

    fn word_is(&mut self, w: &str) -> Result<bool> {
        Ok(matches!(self.peek()?, Some(Tok::Word(x)) if x == w))
    }

    fn expect_word(&mut self, w: &str) -> Result<()> {
        match self.next()? {
            Tok::Word(x) if x == w => Ok(()),
            t => self.err(format!("expected {w}, got {t:?}")),
        }
    }

    fn expect(&mut self, t: Tok) -> Result<()> {
        let got = self.next()?;
        if got != t {
            return self.err(format!("expected {t:?}, got {got:?}"));
        }
        Ok(())
    }

    fn string(&mut self) -> Result<String> {
        match self.next()? {
            Tok::Str(s) => Ok(s),
            t => self.err(format!("expected string, got {t:?}")),
        }
    }

    fn is_str(&mut self) -> Result<bool> {
        Ok(matches!(self.peek()?, Some(Tok::Str(_))))
    }
}

pub(crate) struct AsciiFile {
    pub header: Properties,
    pub types: Vec<TypeDef>,
    pub sweeps: Vec<Variable>,
    pub traces: Vec<Variable>,
    pub groups: Vec<Group>,
    /// Byte offset and line just after `VALUE`; values are parsed on demand.
    pub values_at: Option<(usize, usize)>,
}

/// Declarations needed to parse the VALUE section.
pub(crate) struct Decls<'a> {
    pub types: &'a [TypeDef],
    pub sweeps: &'a [Variable],
    pub traces: &'a [Variable],
    /// Renamed traces: (index, name in the file); see `PsfFile::renamed_traces`.
    pub renamed: &'a [(usize, String)],
    pub groups: &'a [Group],
}

/// Parsed VALUE section.
pub(crate) struct AsciiValues {
    pub sweep_col: Option<Column>,
    pub trace_cols: Vec<Column>,
    pub nonswept: Vec<NamedValue>,
}

/// `"name" value` pairs until the next non-string token.
fn prop_list(lx: &mut Lexer) -> Result<Properties> {
    let mut v = Vec::new();
    while lx.is_str()? {
        let k = lx.string()?;
        let val = match lx.next()? {
            Tok::Str(s) => PropValue::String(s),
            Tok::Num(x, true) if x.abs() < 9.0e15 => PropValue::Int(x as i64),
            Tok::Num(x, _) => PropValue::Double(x),
            t => return lx.err(format!("bad property value {t:?}")),
        };
        v.push((k, val));
    }
    Ok(Properties(v))
}

/// Optional `PROP( ... )`.
fn opt_props(lx: &mut Lexer) -> Result<Properties> {
    if lx.word_is("PROP")? {
        lx.next()?;
        lx.expect(Tok::LParen)?;
        let p = prop_list(lx)?;
        lx.expect(Tok::RParen)?;
        Ok(p)
    } else {
        Ok(Properties::default())
    }
}

fn typespec(lx: &mut Lexer) -> Result<DataType> {
    let w = match lx.next()? {
        Tok::Word(w) => w,
        t => return lx.err(format!("expected type, got {t:?}")),
    };
    let sub = |lx: &mut Lexer| -> Result<String> {
        match lx.next()? {
            Tok::Word(w) => Ok(w),
            Tok::Star => Ok("*".into()),
            t => lx.err(format!("bad type spec {t:?}")),
        }
    };
    Ok(match w.as_str() {
        "FLOAT" => match sub(lx)?.as_str() {
            "DOUBLE" => DataType::Scalar(Scalar::Float64),
            o => return lx.err(format!("FLOAT {o}")),
        },
        "COMPLEX" => match sub(lx)?.as_str() {
            "DOUBLE" => DataType::Scalar(Scalar::ComplexFloat64),
            o => return lx.err(format!("COMPLEX {o}")),
        },
        "INT" => match sub(lx)?.as_str() {
            "BYTE" => DataType::Scalar(Scalar::Int8),
            "LONG" => DataType::Scalar(Scalar::Int32),
            o => return lx.err(format!("INT {o}")),
        },
        "STRING" => {
            sub(lx)?;
            DataType::Scalar(Scalar::String)
        }
        "ARRAY" => {
            lx.expect(Tok::LParen)?;
            let len = match lx.next()? {
                Tok::Star => None,
                Tok::Num(n, true) => Some(n as usize),
                t => return lx.err(format!("bad array length {t:?}")),
            };
            lx.expect(Tok::RParen)?;
            DataType::Array {
                len,
                elem: Box::new(typespec(lx)?),
            }
        }
        "STRUCT" => {
            lx.expect(Tok::LParen)?;
            let mut fields = Vec::new();
            while lx.is_str()? {
                let name = lx.string()?;
                let dtype = typespec(lx)?;
                let props = opt_props(lx)?;
                fields.push(Field { name, dtype, props });
            }
            lx.expect(Tok::RParen)?;
            DataType::Struct(fields)
        }
        o => return lx.err(format!("unknown type {o}")),
    })
}

/// Untyped value tree: number, string or parenthesised list.
fn raw_value(lx: &mut Lexer) -> Result<Value> {
    Ok(match lx.next()? {
        Tok::Num(x, true) if x.abs() < 9.0e15 => Value::Int(x as i64),
        Tok::Num(x, _) => Value::Float(x),
        Tok::Str(s) => Value::String(s),
        Tok::LParen => {
            let mut items = Vec::new();
            while !matches!(lx.peek()?, Some(Tok::RParen)) {
                items.push(raw_value(lx)?);
            }
            lx.next()?;
            Value::Array(items)
        }
        t => return lx.err(format!("bad value {t:?}")),
    })
}

/// Shapes an untyped value according to its declared type.
fn typed(v: Value, dt: &DataType) -> Value {
    match (v, dt) {
        (Value::Int(x), DataType::Scalar(Scalar::Float32 | Scalar::Float64)) => {
            Value::Float(x as f64)
        }
        (Value::Float(x), DataType::Scalar(s))
            if !matches!(s, Scalar::Float32 | Scalar::Float64 | Scalar::String)
                && !s.is_complex() =>
        {
            Value::Int(x as i64)
        }
        (Value::Array(p), DataType::Scalar(s)) if s.is_complex() && p.len() == 2 => Value::Complex(
            p[0].as_f64().unwrap_or(f64::NAN),
            p[1].as_f64().unwrap_or(f64::NAN),
        ),
        (Value::Array(items), DataType::Struct(fields)) if items.len() == fields.len() => {
            Value::Struct(
                fields
                    .iter()
                    .zip(items)
                    .map(|(f, v)| (f.name.clone(), typed(v, &f.dtype)))
                    .collect(),
            )
        }
        (Value::Array(items), DataType::Array { elem, .. }) => {
            Value::Array(items.into_iter().map(|v| typed(v, elem)).collect())
        }
        (v, _) => v,
    }
}

pub(crate) fn parse(b: &[u8]) -> Result<AsciiFile> {
    let mut lx = Lexer {
        s: b,
        p: 0,
        line: 1,
        peeked: None,
    };
    lx.expect_word("HEADER")?;
    let header = prop_list(&mut lx)?;
    let mut f = AsciiFile {
        header,
        types: Vec::new(),
        sweeps: Vec::new(),
        traces: Vec::new(),
        groups: Vec::new(),
        values_at: None,
    };
    let mut types: HashMap<String, TypeDef> = HashMap::new();
    let lookup = |types: &HashMap<String, TypeDef>, lx: &Lexer, n: &str| -> Result<DataType> {
        types
            .get(n)
            .map(|t| t.dtype.clone())
            .ok_or_else(|| Error::Ascii {
                line: lx.line,
                msg: format!("unknown type {n:?}"),
            })
    };
    if lx.word_is("TYPE")? {
        lx.next()?;
        while lx.is_str()? {
            let name = lx.string()?;
            let dtype = typespec(&mut lx)?;
            let props = opt_props(&mut lx)?;
            let t = TypeDef {
                id: f.types.len() as u32,
                name: name.clone(),
                dtype,
                props,
            };
            types.insert(name, t.clone());
            f.types.push(t);
        }
    }
    // one shared dtype per type name
    let shared: HashMap<String, std::sync::Arc<DataType>> = types
        .iter()
        .map(|(k, t)| (k.clone(), std::sync::Arc::new(t.dtype.clone())))
        .collect();
    let var = |lx: &mut Lexer, name: String, id: usize, group: Option<usize>| -> Result<Variable> {
        let type_name = lx.string()?;
        let dtype = match shared.get(&type_name) {
            Some(d) => d.clone(),
            None => std::sync::Arc::new(lookup(&types, lx, &type_name)?),
        };
        let props = opt_props(lx)?;
        Ok(Variable {
            id: id as u32,
            name,
            type_name,
            dtype,
            props,
            group,
            xl: None,
        })
    };
    if lx.word_is("SWEEP")? {
        lx.next()?;
        while lx.is_str()? {
            let name = lx.string()?;
            let v = var(&mut lx, name, f.sweeps.len(), None)?;
            f.sweeps.push(v);
        }
    }
    if lx.word_is("TRACE")? {
        lx.next()?;
        while lx.is_str()? {
            let name = lx.string()?;
            if lx.word_is("GROUP")? {
                lx.next()?;
                let n = match lx.next()? {
                    Tok::Num(n, true) => n as usize,
                    t => return lx.err(format!("bad group size {t:?}")),
                };
                let gi = f.groups.len();
                let mut members = Vec::with_capacity(n);
                for _ in 0..n {
                    let m = lx.string()?;
                    members.push(f.traces.len());
                    let v = var(&mut lx, m, f.traces.len(), Some(gi))?;
                    f.traces.push(v);
                }
                f.groups.push(Group {
                    id: gi as u32,
                    name,
                    members,
                });
            } else {
                let v = var(&mut lx, name, f.traces.len(), None)?;
                f.traces.push(v);
            }
        }
    }
    if lx.word_is("VALUE")? {
        lx.next()?;
        f.values_at = Some((lx.p, lx.line));
        return Ok(f);
    }
    // header-only files (e.g. ADE amap files) end without END
    if lx.peek()?.is_some() {
        lx.expect_word("END")?;
    }
    Ok(f)
}

/// Parses the VALUE section of a file declared by [`parse`].
pub(crate) fn parse_values(b: &[u8], at: Option<(usize, usize)>, f: &Decls) -> Result<AsciiValues> {
    let mut out = AsciiValues {
        sweep_col: None,
        trace_cols: Vec::new(),
        nonswept: Vec::new(),
    };
    let Some((p, line)) = at else { return Ok(out) };
    let mut lx = Lexer {
        s: b,
        p,
        line,
        peeked: None,
    };
    let types: HashMap<&str, &DataType> = f
        .types
        .iter()
        .map(|t| (t.name.as_str(), &t.dtype))
        .collect();
    if f.sweeps.is_empty() {
        while lx.is_str()? {
            let name = lx.string()?;
            let type_name = lx.string()?;
            let dt = *types.get(type_name.as_str()).ok_or_else(|| Error::Ascii {
                line: lx.line,
                msg: format!("unknown type {type_name:?}"),
            })?;
            let value = typed(raw_value(&mut lx)?, dt);
            let props = opt_props(&mut lx)?;
            out.nonswept.push(NamedValue {
                name,
                type_name,
                value,
                props,
            });
        }
    } else {
        swept_values(&mut lx, f, &mut out)?;
    }
    if lx.peek()?.is_some() {
        lx.expect_word("END")?;
    }
    Ok(out)
}

fn swept_values(lx: &mut Lexer, f: &Decls, out: &mut AsciiValues) -> Result<()> {
    let [sweep] = f.sweeps else {
        return Err(Error::Unsupported(format!(
            "{} sweep variables in one file",
            f.sweeps.len()
        )));
    };
    let mut sweep_col = Column::new(&sweep.dtype, 0)?;
    let mut cols: Vec<Column> = f
        .traces
        .iter()
        .map(|t| Column::new(&t.dtype, 0))
        .collect::<Result<_>>()?;
    // name in the file -> traces (several if the file repeats a name)
    let renamed: HashMap<usize, &str> = f.renamed.iter().map(|(i, n)| (*i, n.as_str())).collect();
    let mut by_name: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, t) in f
        .traces
        .iter()
        .enumerate()
        .filter(|(_, t)| t.group.is_none())
    {
        let name = renamed.get(&i).copied().unwrap_or(&t.name);
        by_name.entry(name).or_default().push(i);
    }
    let groups: HashMap<&str, usize> = f
        .groups
        .iter()
        .enumerate()
        .map(|(i, g)| (g.name.as_str(), i))
        .collect();
    let err = |lx: &Lexer, e: Error| match e {
        Error::Malformed { msg, .. } => Error::Ascii { line: lx.line, msg },
        e => e,
    };
    while lx.is_str()? {
        let name = lx.string()?;
        if name == sweep.name {
            let v = typed(raw_value(lx)?, &sweep.dtype);
            sweep_col.push_value(&v).map_err(|e| err(lx, e))?;
        } else if let Some(idx) = by_name.get(name.as_str()) {
            // a repeated name: its values come in declaration order at every point
            let i = *idx
                .iter()
                .min_by_key(|&&i| cols[i].len())
                .expect("non-empty");
            let v = typed(raw_value(lx)?, &f.traces[i].dtype);
            cols[i].push_value(&v).map_err(|e| err(lx, e))?;
        } else if let Some(&g) = groups.get(name.as_str()) {
            for &i in &f.groups[g].members {
                let v = typed(raw_value(lx)?, &f.traces[i].dtype);
                cols[i].push_value(&v).map_err(|e| err(lx, e))?;
            }
        } else {
            return lx.err(format!("value for unknown signal {name:?}"));
        }
    }
    // psfascii omits a trace at points where it has no value; only full columns are consistent
    let n = sweep_col.len();
    if let Some((i, c)) = cols.iter().enumerate().find(|(_, c)| c.len() != n) {
        return lx.err(format!(
            "trace {:?} has {} values, sweep has {n}",
            f.traces[i].name,
            c.len()
        ));
    }
    out.sweep_col = Some(sweep_col);
    out.trace_cols = cols;
    Ok(())
}
