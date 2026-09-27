//! Columnar storage for swept data.

use crate::error::{Error, Result};
use crate::types::{DataType, Scalar, Value};

/// Values of one sweep variable or trace, one entry per sweep point.
///
/// Complex numbers are stored as separate real/imaginary vectors (struct-of-arrays), which maps
/// directly to an Arrow/Polars `Struct{re, im}` column.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Column {
    Int8(Vec<i8>),
    Int32(Vec<i32>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    ComplexFloat32 {
        re: Vec<f32>,
        im: Vec<f32>,
    },
    ComplexFloat64 {
        re: Vec<f64>,
        im: Vec<f64>,
    },
    String(Vec<String>),
    /// Fixed-size array per point, flattened: point `i` is `values[i*width..(i+1)*width]`.
    Array {
        width: usize,
        values: Box<Column>,
    },
    Struct(Vec<(String, Column)>),
}

impl Column {
    /// Empty column for a data type.
    pub fn new(dtype: &DataType, capacity: usize) -> Result<Column> {
        Ok(match dtype {
            DataType::Scalar(s) => Self::new_scalar(*s, capacity),
            DataType::Array { len: Some(n), elem } => Column::Array {
                width: *n,
                values: Box::new(Column::new(elem, capacity * n)?),
            },
            DataType::Array { len: None, .. } => {
                return Err(Error::Unsupported(
                    "variable-length array in a column".into(),
                ));
            }
            DataType::Struct(fields) => Column::Struct(
                fields
                    .iter()
                    .map(|f| Ok((f.name.clone(), Column::new(&f.dtype, capacity)?)))
                    .collect::<Result<_>>()?,
            ),
        })
    }

    fn new_scalar(s: Scalar, n: usize) -> Column {
        match s {
            Scalar::Int8 => Column::Int8(Vec::with_capacity(n)),
            Scalar::Int32 => Column::Int32(Vec::with_capacity(n)),
            Scalar::Float32 => Column::Float32(Vec::with_capacity(n)),
            Scalar::Float64 => Column::Float64(Vec::with_capacity(n)),
            Scalar::ComplexFloat32 => Column::ComplexFloat32 {
                re: Vec::with_capacity(n),
                im: Vec::with_capacity(n),
            },
            Scalar::ComplexFloat64 => Column::ComplexFloat64 {
                re: Vec::with_capacity(n),
                im: Vec::with_capacity(n),
            },
            Scalar::String => Column::String(Vec::with_capacity(n)),
        }
    }

    /// Builds a column from decoded values (e.g. one row per non-swept value).
    pub fn from_values<'a>(
        dtype: &DataType,
        values: impl IntoIterator<Item = &'a Value>,
    ) -> Result<Column> {
        let mut c = Column::new(dtype, 0)?;
        for v in values {
            c.push_value(v)?;
        }
        Ok(c)
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        match self {
            Column::Int8(v) => v.len(),
            Column::Int32(v) => v.len(),
            Column::Float32(v) => v.len(),
            Column::Float64(v) => v.len(),
            Column::ComplexFloat32 { re, .. } => re.len(),
            Column::ComplexFloat64 { re, .. } => re.len(),
            Column::String(v) => v.len(),
            Column::Array { width, values } => values.len().checked_div(*width).unwrap_or(0),
            Column::Struct(f) => f.first().map_or(0, |(_, c)| c.len()),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Keeps the first `n` points.
    pub fn truncate(&mut self, n: usize) {
        match self {
            Column::Int8(v) => v.truncate(n),
            Column::Int32(v) => v.truncate(n),
            Column::Float32(v) => v.truncate(n),
            Column::Float64(v) => v.truncate(n),
            Column::ComplexFloat32 { re, im } => {
                re.truncate(n);
                im.truncate(n);
            }
            Column::ComplexFloat64 { re, im } => {
                re.truncate(n);
                im.truncate(n);
            }
            Column::String(v) => v.truncate(n),
            Column::Array { width, values } => values.truncate(n * *width),
            Column::Struct(f) => f.iter_mut().for_each(|(_, c)| c.truncate(n)),
        }
    }

    /// Real f64 data, if this is a `Float64` column.
    pub fn as_f64(&self) -> Option<&[f64]> {
        match self {
            Column::Float64(v) => Some(v),
            _ => None,
        }
    }

    /// Real and imaginary parts, if this is a `ComplexFloat64` column.
    pub fn as_complex_f64(&self) -> Option<(&[f64], &[f64])> {
        match self {
            Column::ComplexFloat64 { re, im } => Some((re, im)),
            _ => None,
        }
    }

    /// Takes struct member `name` out of a struct column (no copy).
    pub fn into_field(self, name: &str) -> Option<Column> {
        match self {
            Column::Struct(f) => f.into_iter().find(|(k, _)| k == name).map(|(_, c)| c),
            _ => None,
        }
    }

    /// Struct member column by name.
    pub fn field(&self, name: &str) -> Option<&Column> {
        match self {
            Column::Struct(f) => f.iter().find(|(k, _)| k == name).map(|(_, c)| c),
            _ => None,
        }
    }

    /// Converts to f64 when lossless enough for numeric use (ints, floats). None for complex etc.
    pub fn to_f64(&self) -> Option<Vec<f64>> {
        Some(match self {
            Column::Int8(v) => v.iter().map(|&x| x as f64).collect(),
            Column::Int32(v) => v.iter().map(|&x| x as f64).collect(),
            Column::Float32(v) => v.iter().map(|&x| x as f64).collect(),
            Column::Float64(v) => v.clone(),
            _ => return None,
        })
    }

    /// Value at point `i`.
    pub fn get(&self, i: usize) -> Option<Value> {
        if i >= self.len() {
            return None;
        }
        Some(match self {
            Column::Int8(v) => Value::Int(v[i].into()),
            Column::Int32(v) => Value::Int(v[i].into()),
            Column::Float32(v) => Value::Float(v[i].into()),
            Column::Float64(v) => Value::Float(v[i]),
            Column::ComplexFloat32 { re, im } => Value::Complex(re[i].into(), im[i].into()),
            Column::ComplexFloat64 { re, im } => Value::Complex(re[i], im[i]),
            Column::String(v) => Value::String(v[i].clone()),
            Column::Array { width, values } => Value::Array(
                (i * width..(i + 1) * width)
                    .filter_map(|k| values.get(k))
                    .collect(),
            ),
            Column::Struct(f) => Value::Struct(
                f.iter()
                    .filter_map(|(k, c)| Some((k.clone(), c.get(i)?)))
                    .collect(),
            ),
        })
    }

    /// Appends a decoded value (used by the psfascii reader).
    pub(crate) fn push_value(&mut self, v: &Value) -> Result<()> {
        fn bad(v: &Value) -> Error {
            Error::Malformed {
                offset: 0,
                msg: format!("value {v:?} does not match column type"),
            }
        }
        let int = |v: &Value| -> Result<i64> {
            match v {
                Value::Int(x) => Ok(*x),
                Value::Float(x) if x.fract() == 0.0 => Ok(*x as i64),
                _ => Err(bad(v)),
            }
        };
        let float = |v: &Value| v.as_f64().ok_or_else(|| bad(v));
        let cplx = |v: &Value| -> Result<(f64, f64)> {
            match v {
                Value::Complex(a, b) => Ok((*a, *b)),
                Value::Array(p) if p.len() == 2 => Ok((float(&p[0])?, float(&p[1])?)),
                Value::Struct(p) if p.len() == 2 => Ok((float(&p[0].1)?, float(&p[1].1)?)),
                other => Ok((float(other)?, 0.0)),
            }
        };
        match self {
            Column::Int8(c) => c.push(int(v)? as i8),
            Column::Int32(c) => c.push(int(v)? as i32),
            Column::Float32(c) => c.push(float(v)? as f32),
            Column::Float64(c) => c.push(float(v)?),
            Column::ComplexFloat32 { re, im } => {
                let (a, b) = cplx(v)?;
                re.push(a as f32);
                im.push(b as f32);
            }
            Column::ComplexFloat64 { re, im } => {
                let (a, b) = cplx(v)?;
                re.push(a);
                im.push(b);
            }
            Column::String(c) => c.push(v.as_str().ok_or_else(|| bad(v))?.to_owned()),
            Column::Array { width, values } => match v {
                Value::Array(items) if items.len() == *width => {
                    for it in items {
                        values.push_value(it)?;
                    }
                }
                _ => return Err(bad(v)),
            },
            Column::Struct(fields) => match v {
                Value::Struct(items) if items.len() == fields.len() => {
                    for ((_, c), (_, it)) in fields.iter_mut().zip(items) {
                        c.push_value(it)?;
                    }
                }
                Value::Array(items) if items.len() == fields.len() => {
                    for ((_, c), it) in fields.iter_mut().zip(items) {
                        c.push_value(it)?;
                    }
                }
                _ => return Err(bad(v)),
            },
        }
        Ok(())
    }
}
