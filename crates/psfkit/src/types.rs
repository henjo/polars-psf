//! Metadata model: properties, data types and declarations.

use std::fmt;

/// Value of a header/trace property.
#[derive(Clone, Debug, PartialEq)]
pub enum PropValue {
    String(String),
    Int(i64),
    Double(f64),
}

impl PropValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            PropValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            PropValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Numeric value as f64 (ints are converted).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            PropValue::Int(v) => Some(*v as f64),
            PropValue::Double(v) => Some(*v),
            PropValue::String(_) => None,
        }
    }
}

impl fmt::Display for PropValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PropValue::String(s) => write!(f, "{s:?}"),
            PropValue::Int(v) => write!(f, "{v}"),
            PropValue::Double(v) => write!(f, "{v:e}"),
        }
    }
}

/// Ordered property list (order as in the file; duplicate keys are kept).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Properties(pub Vec<(String, PropValue)>);

impl Properties {
    /// First property with the given name.
    pub fn get(&self, key: &str) -> Option<&PropValue> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(PropValue::as_str)
    }

    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(PropValue::as_i64)
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(PropValue::as_f64)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &PropValue)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Scalar element types.
///
/// Int8 occupies a full 4-byte slot in psfbin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Scalar {
    Int8,
    Int32,
    Float32,
    ComplexFloat32,
    Float64,
    ComplexFloat64,
    String,
}

impl Scalar {
    /// psfbin type code -> scalar (codes seen in real files only). Struct (16) and unknown codes return None.
    pub(crate) fn from_code(code: u32) -> Option<Scalar> {
        Some(match code {
            1 => Scalar::Int8,
            5 => Scalar::Int32,
            9 => Scalar::Float32,
            10 => Scalar::ComplexFloat32,
            11 => Scalar::Float64,
            12 => Scalar::ComplexFloat64,
            _ => return None,
        })
    }

    /// Encoded size in psfbin, None for variable-size strings.
    pub fn encoded_size(self) -> Option<usize> {
        Some(match self {
            Scalar::Int8 | Scalar::Int32 | Scalar::Float32 => 4,
            Scalar::ComplexFloat32 | Scalar::Float64 => 8,
            Scalar::ComplexFloat64 => 16,
            Scalar::String => return None,
        })
    }

    pub fn is_complex(self) -> bool {
        matches!(self, Scalar::ComplexFloat32 | Scalar::ComplexFloat64)
    }
}

/// Data type of a sweep, trace or value.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum DataType {
    Scalar(Scalar),
    /// Fixed-size array; `len` is the product of all dimensions (None = unbounded `ARRAY ( * )`).
    Array {
        len: Option<usize>,
        elem: Box<DataType>,
    },
    Struct(Vec<Field>),
}

impl DataType {
    /// Encoded size in psfbin if every part has a fixed size.
    pub fn encoded_size(&self) -> Option<usize> {
        match self {
            DataType::Scalar(s) => s.encoded_size(),
            DataType::Array { len, elem } => Some(len.as_ref()? * elem.encoded_size()?),
            DataType::Struct(fields) => fields.iter().map(|f| f.dtype.encoded_size()).sum(),
        }
    }

    pub fn as_scalar(&self) -> Option<Scalar> {
        match self {
            DataType::Scalar(s) => Some(*s),
            _ => None,
        }
    }
}

/// Struct member.
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub name: String,
    pub dtype: DataType,
    pub props: Properties,
}

/// Named type from the TYPE section.
#[derive(Clone, Debug, PartialEq)]
pub struct TypeDef {
    pub id: u32,
    pub name: String,
    pub dtype: DataType,
    pub props: Properties,
}

/// Index record of a PSFXL trace (chunk 0x24 in the stub file).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct XlIndex {
    /// Offset of the (last) signal record in the `.psfxl` file.
    pub offset: u64,
    pub npoints: u64,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

/// Sweep variable or trace declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct Variable {
    pub id: u32,
    pub name: String,
    /// Name of the referenced type (e.g. "node", "sweep").
    pub type_name: String,
    /// Shared between traces of the same type (Spectre files have 10^4+ traces of few types).
    pub dtype: std::sync::Arc<DataType>,
    pub props: Properties,
    /// Group this trace belongs to (index into [`crate::PsfFile::groups`]).
    pub group: Option<usize>,
    /// PSFXL location, when the values live in a `.psfxl` file.
    pub xl: Option<XlIndex>,
}

/// Trace group (compound signal). Members are indices into [`crate::PsfFile::traces`].
#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub id: u32,
    pub name: String,
    pub members: Vec<usize>,
}

/// A single decoded value (non-swept values, scalars).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Value {
    Int(i64),
    Float(f64),
    Complex(f64, f64),
    String(String),
    Array(Vec<Value>),
    Struct(Vec<(String, Value)>),
}

impl Value {
    /// Numeric value as f64 (None for complex, strings and composites).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(v) => Some(*v as f64),
            Value::Float(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Struct member by name.
    pub fn field(&self, name: &str) -> Option<&Value> {
        match self {
            Value::Struct(m) => m.iter().find(|(k, _)| k == name).map(|(_, v)| v),
            _ => None,
        }
    }
}

/// Entry of a non-swept VALUE section (operating points, info files, logFile).
#[derive(Clone, Debug, PartialEq)]
pub struct NamedValue {
    pub name: String,
    pub type_name: String,
    pub value: Value,
    pub props: Properties,
}
