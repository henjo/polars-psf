//! Converts [`psf`] columns to Apache Arrow arrays.
//!
//! Numeric vectors are moved into Arrow buffers without copying. Complex values become
//! `Struct{re, im}`, PSF structs become Arrow structs and fixed-size arrays become
//! `FixedSizeList`.
//!
//! ```no_run
//! let f = psfkit::PsfFile::open("ac.ac")?;
//! let batch = psfkit_arrow::to_record_batch(&f, f.read(&["out"])?)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, ArrowPrimitiveType, FixedSizeListArray, PrimitiveArray, RecordBatch,
    RecordBatchOptions, StringArray, StructArray, types::*,
};
use arrow_buffer::{Buffer, ScalarBuffer};
use arrow_schema::{ArrowError, DataType, Field, Fields, Schema};
use psfkit::{Column, PsfFile, Scalar, SweptData};

fn prim<T: ArrowPrimitiveType>(v: Vec<T::Native>) -> ArrayRef {
    // Vec -> Buffer takes ownership of the allocation (no copy)
    Arc::new(PrimitiveArray::<T>::new(
        ScalarBuffer::from(Buffer::from_vec(v)),
        None,
    ))
}

fn complex_fields(elem: DataType) -> Fields {
    Fields::from(vec![
        Field::new("re", elem.clone(), false),
        Field::new("im", elem, false),
    ])
}

/// Arrow type of a PSF data type.
pub fn arrow_type(dt: &psfkit::DataType) -> Result<DataType, ArrowError> {
    Ok(match dt {
        psfkit::DataType::Scalar(s) => match s {
            Scalar::Int8 => DataType::Int8,
            Scalar::Int32 => DataType::Int32,
            Scalar::Float32 => DataType::Float32,
            Scalar::Float64 => DataType::Float64,
            Scalar::ComplexFloat32 => DataType::Struct(complex_fields(DataType::Float32)),
            Scalar::ComplexFloat64 => DataType::Struct(complex_fields(DataType::Float64)),
            Scalar::String => DataType::Utf8,
            s => return Err(ArrowError::NotYetImplemented(format!("PSF scalar {s:?}"))),
        },
        psfkit::DataType::Array { len: Some(n), elem } => DataType::FixedSizeList(
            Arc::new(Field::new("item", arrow_type(elem)?, false)),
            i32::try_from(*n)
                .map_err(|_| ArrowError::InvalidArgumentError("array too wide".into()))?,
        ),
        psfkit::DataType::Array { len: None, .. } => {
            return Err(ArrowError::NotYetImplemented("unbounded PSF array".into()));
        }
        psfkit::DataType::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|f| Ok(Field::new(&f.name, arrow_type(&f.dtype)?, false)))
                .collect::<Result<Fields, ArrowError>>()?,
        ),
        dt => return Err(ArrowError::NotYetImplemented(format!("PSF type {dt:?}"))),
    })
}

/// Converts a column, moving its buffers.
pub fn column_to_arrow(col: Column) -> Result<ArrayRef, ArrowError> {
    Ok(match col {
        Column::Int8(v) => prim::<Int8Type>(v),
        Column::Int32(v) => prim::<Int32Type>(v),
        Column::Float32(v) => prim::<Float32Type>(v),
        Column::Float64(v) => prim::<Float64Type>(v),
        Column::ComplexFloat32 { re, im } => Arc::new(StructArray::try_new(
            complex_fields(DataType::Float32),
            vec![prim::<Float32Type>(re), prim::<Float32Type>(im)],
            None,
        )?),
        Column::ComplexFloat64 { re, im } => Arc::new(StructArray::try_new(
            complex_fields(DataType::Float64),
            vec![prim::<Float64Type>(re), prim::<Float64Type>(im)],
            None,
        )?),
        Column::String(v) => Arc::new(StringArray::from(v)),
        Column::Array { width, values } => {
            let values = column_to_arrow(*values)?;
            let item = Arc::new(Field::new("item", values.data_type().clone(), false));
            let w = i32::try_from(width)
                .map_err(|_| ArrowError::InvalidArgumentError("array too wide".into()))?;
            Arc::new(FixedSizeListArray::try_new(item, w, values, None)?)
        }
        Column::Struct(members) => {
            let len = members.first().map_or(0, |(_, c)| c.len());
            let (fields, arrays): (Vec<Field>, Vec<ArrayRef>) = members
                .into_iter()
                .map(|(name, c)| {
                    let a = column_to_arrow(c)?;
                    Ok((Field::new(name, a.data_type().clone(), false), a))
                })
                .collect::<Result<Vec<_>, ArrowError>>()?
                .into_iter()
                .unzip();
            if fields.is_empty() {
                Arc::new(StructArray::new_empty_fields(len, None))
            } else {
                Arc::new(StructArray::try_new(Fields::from(fields), arrays, None)?)
            }
        }
        c => return Err(ArrowError::NotYetImplemented(format!("column {c:?}"))),
    })
}

/// Field metadata for a trace: `psf:type` and every string/number property (e.g. `units`).
fn metadata(v: &psfkit::Variable) -> HashMap<String, String> {
    let mut m: HashMap<String, String> = v
        .props
        .iter()
        .map(|(k, p)| {
            let s = match p {
                psfkit::PropValue::String(s) => s.clone(),
                other => other.to_string(),
            };
            (format!("psf:{k}"), s)
        })
        .collect();
    m.insert("psf:type".into(), v.type_name.clone());
    m
}

/// Record batch with the sweep as first column followed by the traces.
///
/// Traces without values (e.g. declared-only traces of parent sweep files) are dropped.
/// PSF properties are attached as field metadata (`psf:units`, ...).
pub fn to_record_batch(file: &PsfFile, data: SweptData) -> Result<RecordBatch, ArrowError> {
    to_record_batch_with(file, data, true)
}

/// Like [`to_record_batch`]; `metadata = false` skips PSF properties as field metadata (cheaper
/// for many columns, and consumers like Polars drop it anyway).
pub fn to_record_batch_with(
    file: &PsfFile,
    data: SweptData,
    metadata_on: bool,
) -> Result<RecordBatch, ArrowError> {
    let n = data.sweep.len();
    let meta = |name: &str, sweep: bool| {
        if !metadata_on {
            return HashMap::new();
        }
        let v = if sweep {
            file.sweeps().iter().find(|v| v.name == name)
        } else {
            file.index_of(name).and_then(|i| file.traces().get(i))
        };
        v.map(metadata).unwrap_or_default()
    };
    let mut fields = Vec::with_capacity(data.traces.len() + 1);
    let mut cols = Vec::with_capacity(data.traces.len() + 1);
    let sweep = column_to_arrow(data.sweep)?;
    fields.push(
        Field::new(&data.sweep_name, sweep.data_type().clone(), false)
            .with_metadata(meta(&data.sweep_name, true)),
    );
    cols.push(sweep);
    for (name, c) in data.traces {
        if c.len() != n {
            continue;
        }
        let a = column_to_arrow(c)?;
        fields.push(
            Field::new(&name, a.data_type().clone(), false).with_metadata(meta(&name, false)),
        );
        cols.push(a);
    }
    RecordBatch::try_new_with_options(
        Arc::new(Schema::new(fields)),
        cols,
        &RecordBatchOptions::new().with_row_count(Some(n)),
    )
}

/// One-row record batch with the values of a non-swept file (operating point, info).
///
/// `names` selects values (all if None). Values whose type has no Arrow mapping are skipped.
pub fn values_to_record_batch(
    file: &PsfFile,
    names: Option<&[&str]>,
) -> Result<RecordBatch, ArrowError> {
    let types: HashMap<&str, &psfkit::DataType> = file
        .types()
        .iter()
        .map(|t| (t.name.as_str(), &t.dtype))
        .collect();
    let mut fields = Vec::new();
    let mut cols = Vec::new();
    let mut push = |v: &psfkit::NamedValue| -> Result<(), ArrowError> {
        let Some(dt) = types.get(v.type_name.as_str()) else {
            return Ok(());
        };
        let Ok(col) = Column::from_values(dt, [&v.value]) else {
            return Ok(());
        };
        let a = column_to_arrow(col)?;
        let mut md: HashMap<String, String> = v
            .props
            .iter()
            .map(|(k, p)| {
                (
                    format!("psf:{k}"),
                    match p {
                        psfkit::PropValue::String(s) => s.clone(),
                        other => other.to_string(),
                    },
                )
            })
            .collect();
        md.insert("psf:type".into(), v.type_name.clone());
        fields.push(Field::new(&v.name, a.data_type().clone(), false).with_metadata(md));
        cols.push(a);
        Ok(())
    };
    match names {
        Some(ns) => {
            for n in ns {
                let v = file
                    .value(n)
                    .map_err(|e| ArrowError::InvalidArgumentError(e.to_string()))?;
                push(&v)?;
            }
        }
        None => {
            let mut seen = std::collections::HashSet::new();
            let all = file
                .values()
                .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
            for v in all {
                if seen.insert(v.name.as_str()) {
                    push(v)?;
                }
            }
        }
    }
    RecordBatch::try_new_with_options(
        Arc::new(Schema::new(fields)),
        cols,
        &RecordBatchOptions::new().with_row_count(Some(1)),
    )
}

/// Long ("tidy") record batch: one row per (signal, sweep point) with columns
/// `[sweep, signal, value]`.
///
/// Built for many small signals (e.g. tens of thousands of noise contributions), where a wide
/// table would carry per-column overhead. `field` selects a struct member (e.g. `"total"`); all
/// selected signals must then have that member and the same value type. `signal` is dictionary
/// encoded (Polars `Categorical`).
pub fn to_long_record_batch(
    data: SweptData,
    field: Option<&str>,
) -> Result<RecordBatch, ArrowError> {
    use arrow_array::{Array, DictionaryArray, UInt32Array, types::UInt32Type};
    let n = data.sweep.len();
    let mut names = Vec::with_capacity(data.traces.len());
    let mut values: Vec<Column> = Vec::with_capacity(data.traces.len());
    for (name, c) in data.traces {
        if c.len() != n {
            continue; // declared-only traces
        }
        let c = match field {
            Some(f) => c.field(f).cloned().ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!("signal {name:?} has no field {f:?}"))
            })?,
            None => c,
        };
        names.push(name);
        values.push(c);
    }
    let m = names.len();
    let value = concat_columns(values)?;
    let sweep = match data.sweep {
        // common case: repeat f64 values directly
        Column::Float64(v) => {
            let mut out = Vec::with_capacity(v.len() * m);
            for _ in 0..m {
                out.extend_from_slice(&v);
            }
            column_to_arrow(Column::Float64(out))?
        }
        other => arrow_select_tile(&column_to_arrow(other)?, m)?,
    };
    let keys = UInt32Array::from_iter_values((0..m as u32).flat_map(|k| std::iter::repeat_n(k, n)));
    let dict = DictionaryArray::<UInt32Type>::try_new(keys, Arc::new(StringArray::from(names)))?;
    let value = column_to_arrow(value)?;
    let fields = vec![
        Field::new(&data.sweep_name, sweep.data_type().clone(), false),
        Field::new("signal", dict.data_type().clone(), false),
        Field::new("value", value.data_type().clone(), false),
    ];
    RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        vec![sweep, Arc::new(dict), value],
    )
}

/// Repeats an array `times` times (one `take`, not `times` concatenations).
fn arrow_select_tile(a: &ArrayRef, times: usize) -> Result<ArrayRef, ArrowError> {
    use arrow_array::Array;
    let n = a.len() as u32;
    let idx = arrow_array::UInt32Array::from_iter_values((0..times).flat_map(|_| 0..n));
    arrow_select::take::take(a.as_ref(), &idx, None)
}

/// Concatenates columns of the same type end to end.
fn concat_columns(cols: Vec<Column>) -> Result<Column, ArrowError> {
    let total: usize = cols.iter().map(Column::len).sum();
    let mut it = cols.into_iter();
    let Some(first) = it.next() else {
        return Ok(Column::Float64(Vec::new()));
    };
    let mut acc = first;
    // reserve once instead of growing through 10^4 appends
    match &mut acc {
        Column::Float64(a) => a.reserve(total - a.len()),
        Column::Float32(a) => a.reserve(total - a.len()),
        Column::ComplexFloat64 { re, im } => {
            re.reserve(total - re.len());
            im.reserve(total - im.len());
        }
        _ => {}
    }
    for c in it {
        match (&mut acc, c) {
            (Column::Float64(a), Column::Float64(b)) => a.extend(b),
            (Column::Float32(a), Column::Float32(b)) => a.extend(b),
            (Column::ComplexFloat64 { re, im }, Column::ComplexFloat64 { re: r2, im: i2 }) => {
                re.extend(r2);
                im.extend(i2);
            }
            (Column::ComplexFloat32 { re, im }, Column::ComplexFloat32 { re: r2, im: i2 }) => {
                re.extend(r2);
                im.extend(i2);
            }
            (Column::Int32(a), Column::Int32(b)) => a.extend(b),
            (a, b) => {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "long format needs one value type; got {} and {}",
                    type_label(a),
                    type_label(&b)
                )));
            }
        }
    }
    Ok(acc)
}

fn type_label(c: &Column) -> &'static str {
    match c {
        Column::Float64(_) => "f64",
        Column::Float32(_) => "f32",
        Column::ComplexFloat64 { .. } => "complex f64",
        Column::ComplexFloat32 { .. } => "complex f32",
        Column::Struct(_) => "struct (pass field=...)",
        _ => "other",
    }
}

/// Empty record batch with the schema [`to_record_batch`] would produce for `names`
/// (all traces if None), without reading values.
pub fn swept_schema(file: &PsfFile, names: Option<&[&str]>) -> Result<RecordBatch, ArrowError> {
    let sweep = file
        .sweeps()
        .first()
        .ok_or_else(|| ArrowError::InvalidArgumentError("file is not swept".into()))?;
    let mut fields = vec![
        Field::new(&sweep.name, arrow_type(&sweep.dtype)?, false).with_metadata(metadata(sweep)),
    ];
    let traces: Vec<&psfkit::Variable> = match names {
        Some(ns) => ns
            .iter()
            .map(|n| {
                file.traces().iter().find(|t| t.name == *n).ok_or_else(|| {
                    ArrowError::InvalidArgumentError(format!("no trace named {n:?}"))
                })
            })
            .collect::<Result<_, _>>()?,
        None => file.traces().iter().collect(),
    };
    for t in traces {
        fields.push(Field::new(&t.name, arrow_type(&t.dtype)?, false).with_metadata(metadata(t)));
    }
    Ok(RecordBatch::new_empty(Arc::new(Schema::new(fields))))
}
