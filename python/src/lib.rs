//! Python module `polars_psf._polars_psf`.

use pyo3::create_exception;
use pyo3::exceptions::{PyKeyError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyDict, PyList};
use pyo3_arrow::PyRecordBatch;

create_exception!(
    _polars_psf,
    PsfError,
    PyValueError,
    "Malformed or unsupported PSF data."
);

fn err(e: psfkit::Error) -> PyErr {
    match e {
        psfkit::Error::NotFound(n) => PyKeyError::new_err(n),
        psfkit::Error::Io { .. } => PyOSError::new_err(e.to_string()),
        e => PsfError::new_err(e.to_string()),
    }
}

fn arrow_err(e: impl std::fmt::Display) -> PyErr {
    PsfError::new_err(e.to_string())
}

fn prop_to_py<'py>(py: Python<'py>, v: &psfkit::PropValue) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        psfkit::PropValue::String(s) => s.into_pyobject(py)?.into_any(),
        psfkit::PropValue::Int(i) => i.into_pyobject(py)?.into_any(),
        psfkit::PropValue::Double(d) => d.into_pyobject(py)?.into_any(),
    })
}

fn props_to_py<'py>(py: Python<'py>, p: &psfkit::Properties) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in p.iter() {
        if !d.contains(k)? {
            d.set_item(k, prop_to_py(py, v)?)?;
        }
    }
    Ok(d)
}

fn value_to_py<'py>(py: Python<'py>, v: &psfkit::Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        psfkit::Value::Int(i) => i.into_pyobject(py)?.into_any(),
        psfkit::Value::Float(f) => f.into_pyobject(py)?.into_any(),
        psfkit::Value::Complex(re, im) => PyComplex::from_doubles(py, *re, *im).into_any(),
        psfkit::Value::String(s) => s.into_pyobject(py)?.into_any(),
        psfkit::Value::Array(items) => PyList::new(
            py,
            items
                .iter()
                .map(|x| value_to_py(py, x))
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .into_any(),
        psfkit::Value::Struct(fields) => {
            let d = PyDict::new(py);
            for (k, x) in fields {
                d.set_item(k, value_to_py(py, x)?)?;
            }
            d.into_any()
        }
        other => return Err(PsfError::new_err(format!("unsupported value {other:?}"))),
    })
}

/// A PSF result file (psfbin, psfascii or PSFXL stub + data).
#[pyclass(name = "PsfFile", module = "polars_psf", frozen)]
struct PyPsfFile {
    inner: psfkit::PsfFile,
    path: String,
}

impl PyPsfFile {
    fn batch(
        &self,
        py: Python<'_>,
        names: Option<Vec<String>>,
        metadata: bool,
    ) -> PyResult<PyRecordBatch> {
        let names: Option<Vec<&str>> = names
            .as_ref()
            .map(|n| n.iter().map(String::as_str).collect());
        let batch = py.detach(|| file_batch(&self.inner, names.as_deref(), metadata))?;
        Ok(PyRecordBatch::new(batch))
    }
}

#[pymethods]
impl PyPsfFile {
    #[new]
    fn new(py: Python<'_>, path: std::path::PathBuf) -> PyResult<Self> {
        let inner = py.detach(|| psfkit::PsfFile::open(&path)).map_err(err)?;
        Ok(PyPsfFile {
            inner,
            path: path.display().to_string(),
        })
    }

    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    /// "binary" or "ascii".
    #[getter]
    fn format(&self) -> &'static str {
        match self.inner.format() {
            psfkit::Format::Binary => "binary",
            _ => "ascii",
        }
    }

    #[getter]
    fn is_swept(&self) -> bool {
        self.inner.is_swept()
    }

    /// False for killed/running simulations (only complete points are returned).
    #[getter]
    fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    /// "rows" (dc/ac/noise: all signals on every page), "windowed" (transient), "psfxl",
    /// "nonswept", "ascii" or "empty".
    #[getter]
    fn layout(&self) -> &'static str {
        match self.inner.layout() {
            psfkit::Layout::Rows => "rows",
            psfkit::Layout::Windowed => "windowed",
            psfkit::Layout::Psfxl => "psfxl",
            psfkit::Layout::NonSwept => "nonswept",
            psfkit::Layout::Ascii => "ascii",
            _ => "empty",
        }
    }

    #[getter]
    fn is_psfxl(&self) -> bool {
        self.inner.is_psfxl_stub()
    }

    /// Name of the sweep variable (e.g. "time", "freq"), None for non-swept files.
    #[getter]
    fn sweep_name(&self) -> Option<&str> {
        self.inner.sweeps().first().map(|s| s.name.as_str())
    }

    /// Signal names (traces, or values of non-swept files).
    #[getter]
    fn names(&self) -> PyResult<Vec<&str>> {
        self.inner.names().map_err(err)
    }

    #[getter]
    fn header<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        props_to_py(py, self.inner.header())
    }

    /// PSFXL `.sig` metadata (empty for other formats).
    #[getter]
    fn psfxl_meta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        props_to_py(py, self.inner.psfxl_meta())
    }

    /// Properties of a sweep, trace or value (e.g. {"units": "V"}), plus "type".
    fn props<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
        let f = &self.inner;
        let (props, type_name) =
            if let Some(v) = f.sweeps().iter().chain(f.traces()).find(|v| v.name == name) {
                (v.props.clone(), v.type_name.clone())
            } else if f.is_swept() {
                return Err(PyKeyError::new_err(name.to_owned()));
            } else {
                let v = f.value(name).map_err(err)?;
                (v.props, v.type_name)
            };
        let d = props_to_py(py, &props)?;
        d.set_item("type", type_name)?;
        Ok(d)
    }

    /// Units of a signal, if declared.
    fn units(&self, py: Python<'_>, name: &str) -> PyResult<Option<String>> {
        Ok(self
            .props(py, name)?
            .get_item("units")?
            .map(|u| u.to_string()))
    }

    /// Empty batch with the schema of `read(names)` / `to_polars(names)` (no values are read).
    #[pyo3(signature = (names=None))]
    fn schema(&self, names: Option<Vec<String>>) -> PyResult<PyRecordBatch> {
        let names: Option<Vec<&str>> = names
            .as_ref()
            .map(|n| n.iter().map(String::as_str).collect());
        Ok(PyRecordBatch::new(file_schema(
            &self.inner,
            names.as_deref(),
        )?))
    }

    /// Sweep + traces (non-swept: one row of values) as an Arrow record batch (PyCapsule
    /// interface; pass to polars/pyarrow).
    #[pyo3(signature = (names=None))]
    fn read(&self, py: Python<'_>, names: Option<Vec<String>>) -> PyResult<PyRecordBatch> {
        self.batch(py, names, true)
    }

    /// Sweep + traces as a polars DataFrame (complex signals are Struct{re, im}).
    #[pyo3(signature = (names=None))]
    fn to_polars<'py>(
        &self,
        py: Python<'py>,
        names: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // polars drops Arrow field metadata; skip building it
        let batch = self.batch(py, names, false)?;
        py.import("polars")?.call_method1("DataFrame", (batch,))
    }

    /// Long ("tidy") polars DataFrame `[sweep, signal, value]`, one row per signal and sweep
    /// point, built in Rust. For many small signals (noise contributions) this avoids the
    /// per-column overhead of a wide table. `field` picks a struct member, e.g. "total".
    #[pyo3(signature = (names=None, field=None))]
    fn to_polars_long<'py>(
        &self,
        py: Python<'py>,
        names: Option<Vec<String>>,
        field: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let f = &self.inner;
        let batch = py.detach(|| {
            let names: Option<Vec<&str>> = names
                .as_ref()
                .map(|n| n.iter().map(String::as_str).collect());
            let data = match (&names, &field) {
                (Some(n), Some(fl)) => f.read_field(n, fl),
                // all traces that have the member (noise files also hold plain traces like "out")
                (None, Some(fl)) => f.read_field(&with_field(f, fl), fl),
                (Some(n), None) => f.read(n),
                (None, None) => f.read_all(),
            }
            .map_err(err)?;
            psfkit_arrow::to_long_record_batch(data, None).map_err(arrow_err)
        })?;
        py.import("polars")?
            .call_method1("DataFrame", (PyRecordBatch::new(batch),))
    }

    /// Names of the traces that are structs with member `field` (e.g. noise contributions).
    fn names_with_field(&self, field: &str) -> Vec<&str> {
        with_field(&self.inner, field)
    }

    /// Empty batch with the schema of `to_polars_long(field=...)` (no values are read).
    #[pyo3(signature = (field=None))]
    fn long_schema(&self, field: Option<&str>) -> PyResult<PyRecordBatch> {
        Ok(PyRecordBatch::new(long_schema_of(&self.inner, field)?))
    }

    /// One trace with its own sweep axis, as a 2-column polars DataFrame. Needed for PSFXL
    /// files whose signals have different time axes.
    fn read_signal<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        let f = &self.inner;
        let (x, y) = py.detach(|| f.read_signal(name)).map_err(err)?;
        let sweep = self.sweep_name().unwrap_or("x").to_owned();
        let data = psfkit::SweptData {
            sweep_name: sweep,
            sweep: x,
            traces: vec![(name.to_owned(), y)],
        };
        let batch = py
            .detach(|| psfkit_arrow::to_record_batch(f, data))
            .map_err(arrow_err)?;
        py.import("polars")?
            .call_method1("DataFrame", (PyRecordBatch::new(batch),))
    }

    /// Values of a non-swept file as {name: value}; structs become dicts, complex -> complex.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for v in self.inner.values().map_err(err)? {
            if !d.contains(&v.name)? {
                d.set_item(&v.name, value_to_py(py, &v.value)?)?;
            }
        }
        Ok(d)
    }

    /// One non-swept value.
    fn value<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        let v = self.inner.value(name).map_err(err)?;
        value_to_py(py, &v.value)
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.inner.names().map_err(err)?.len())
    }

    fn __contains__(&self, name: &str) -> bool {
        self.inner.index_of(name).is_some()
    }

    fn __repr__(&self) -> String {
        let f = &self.inner;
        match f.sweeps().first() {
            Some(s) => format!(
                "PsfFile({:?}, sweep={:?}, traces={})",
                self.path,
                s.name,
                f.traces().len()
            ),
            None => format!("PsfFile({:?}, non-swept)", self.path),
        }
    }
}

fn has_field(dt: &psfkit::DataType, field: &str) -> bool {
    matches!(dt, psfkit::DataType::Struct(fs) if fs.iter().any(|f| f.name == field))
}

fn with_field<'a>(f: &'a psfkit::PsfFile, field: &str) -> Vec<&'a str> {
    f.traces()
        .iter()
        .filter(|t| has_field(&t.dtype, field))
        .map(|t| t.name.as_str())
        .collect()
}

fn res_err(e: psfkit_results::Error) -> PyErr {
    match e {
        psfkit_results::Error::Psf(e) => err(e),
        e @ (psfkit_results::Error::UnknownResult(_) | psfkit_results::Error::Ambiguous { .. }) => {
            PyKeyError::new_err(e.to_string())
        }
        e @ psfkit_results::Error::NotResults(_) => PyOSError::new_err(e.to_string()),
        e => PsfError::new_err(e.to_string()),
    }
}

/// Constant column for an outer parameter.
fn param_array(p: &psfkit_results::Param, n: usize) -> arrow_array::ArrayRef {
    use std::sync::Arc;
    match p {
        psfkit_results::Param::Float(v) => Arc::new(arrow_array::Float64Array::from(vec![*v; n])),
        psfkit_results::Param::Str(s) => {
            Arc::new(arrow_array::StringArray::from(vec![s.as_str(); n]))
        }
    }
}

/// Prepends parameter columns to a leaf batch.
fn with_params(
    params: &[(String, psfkit_results::Param)],
    b: arrow_array::RecordBatch,
) -> Result<arrow_array::RecordBatch, arrow_schema::ArrowError> {
    use arrow_schema::{Field, Schema};
    use std::sync::Arc;
    let n = b.num_rows();
    let mut fields: Vec<Field> = params
        .iter()
        .map(|(k, v)| {
            let dt = match v {
                psfkit_results::Param::Float(_) => arrow_schema::DataType::Float64,
                psfkit_results::Param::Str(_) => arrow_schema::DataType::Utf8,
            };
            Field::new(k, dt, false)
        })
        .collect();
    let mut cols: Vec<arrow_array::ArrayRef> =
        params.iter().map(|(_, v)| param_array(v, n)).collect();
    fields.extend(b.schema().fields().iter().map(|f| f.as_ref().clone()));
    cols.extend(b.columns().iter().cloned());
    arrow_array::RecordBatch::try_new_with_options(
        Arc::new(Schema::new(fields)),
        cols,
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(n)),
    )
}

/// Schema of the long table of `f` (`[sweep, signal, value]`), without reading values.
fn long_schema_of(f: &psfkit::PsfFile, field: Option<&str>) -> PyResult<arrow_array::RecordBatch> {
    use std::sync::Arc;
    let sweep = f
        .sweeps()
        .first()
        .ok_or_else(|| PsfError::new_err("file is not swept"))?;
    let t = match field {
        Some(fl) => f.traces().iter().find(|t| has_field(&t.dtype, fl)),
        None => f.traces().first(),
    }
    .ok_or_else(|| PsfError::new_err("no matching traces"))?;
    let dt: &psfkit::DataType = match field {
        None => &t.dtype,
        Some(fl) => match &*t.dtype {
            psfkit::DataType::Struct(fs) => {
                &fs.iter()
                    .find(|x| x.name == fl)
                    .ok_or_else(|| PyKeyError::new_err(format!("{}.{fl}", t.name)))?
                    .dtype
            }
            _ => return Err(PsfError::new_err(format!("{:?} is not a struct", t.name))),
        },
    };
    let dict = arrow_schema::DataType::Dictionary(
        Box::new(arrow_schema::DataType::UInt32),
        Box::new(arrow_schema::DataType::Utf8),
    );
    let schema = arrow_schema::Schema::new(vec![
        arrow_schema::Field::new(
            &sweep.name,
            psfkit_arrow::arrow_type(&sweep.dtype).map_err(arrow_err)?,
            false,
        ),
        arrow_schema::Field::new("signal", dict, false),
        arrow_schema::Field::new(
            "value",
            psfkit_arrow::arrow_type(dt).map_err(arrow_err)?,
            false,
        ),
    ]);
    Ok(arrow_array::RecordBatch::new_empty(Arc::new(schema)))
}

/// Swept files -> sweep + traces, non-swept -> one row of values.
fn file_batch(
    f: &psfkit::PsfFile,
    names: Option<&[&str]>,
    metadata: bool,
) -> PyResult<arrow_array::RecordBatch> {
    if !f.is_swept() {
        return psfkit_arrow::values_to_record_batch(f, names).map_err(arrow_err);
    }
    let data = match names {
        Some(n) => f.read(n),
        None => f.read_all(),
    }
    .map_err(err)?;
    psfkit_arrow::to_record_batch_with(f, data, metadata).map_err(arrow_err)
}

/// Empty batch with the schema of [`file_batch`], without reading swept values.
fn file_schema(f: &psfkit::PsfFile, names: Option<&[&str]>) -> PyResult<arrow_array::RecordBatch> {
    let b = if f.is_swept() {
        // polars drops Arrow field metadata; skip building it
        psfkit_arrow::swept_schema_with(f, names, false)
    } else {
        psfkit_arrow::values_to_record_batch(f, names).map(|b| b.slice(0, 0))
    };
    b.map_err(arrow_err)
}

/// (name, analysis_type, param names, listed leaves, description)
type ResultRow = (String, String, Vec<String>, usize, Option<String>);

use psfkit_results::{Leaf, ResultDir};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Wide batch of one leaf: `[params..., sweep, traces...]` (non-swept: one row of values).
fn leaf_batch(
    dir: &ResultDir,
    leaf: &Leaf,
    names: Option<&[String]>,
) -> PyResult<arrow_array::RecordBatch> {
    let f = dir.file(&leaf.path).map_err(res_err)?;
    let names: Option<Vec<&str>> = names.map(|n| n.iter().map(String::as_str).collect());
    // polars drops Arrow field metadata; skip building it
    let b = file_batch(&f, names.as_deref(), false)?;
    with_params(&leaf.params, b).map_err(arrow_err)
}

/// Long batch of one leaf: `[params..., sweep, signal, value]`. `names` not present in this
/// leaf are skipped (leaves of a sweep normally share their signals). PSFXL signals keep their
/// own time axes.
fn leaf_batch_long(
    dir: &ResultDir,
    leaf: &Leaf,
    names: Option<&[String]>,
    field: Option<&str>,
) -> PyResult<arrow_array::RecordBatch> {
    let f = dir.file(&leaf.path).map_err(res_err)?;
    let sel: Vec<&str> = match (names, field) {
        (Some(n), _) => n
            .iter()
            .map(String::as_str)
            .filter(|n| f.index_of(n).is_some())
            .collect(),
        (None, Some(fl)) => with_field(&f, fl),
        (None, None) => f.traces().iter().map(|t| t.name.as_str()).collect(),
    };
    let b = if f.is_psfxl_stub() && field.is_none() {
        let sweep = f.sweeps().first().map_or("time", |s| s.name.as_str());
        let signals = sel
            .iter()
            .map(|n| {
                let (x, y) = f.read_signal(n).map_err(err)?;
                Ok(((*n).to_owned(), x, y))
            })
            .collect::<PyResult<Vec<_>>>()?;
        psfkit_arrow::to_long_record_batch_signals(sweep, signals)
    } else {
        let data = match field {
            Some(fl) => f.read_field(&sel, fl),
            None => f.read(&sel),
        }
        .map_err(err)?;
        psfkit_arrow::to_long_record_batch(data, None)
    }
    .map_err(arrow_err)?;
    with_params(&leaf.params, b).map_err(arrow_err)
}

/// Sweep points of a file: header, else PSFXL `.sig` count, else unknown.
fn sweep_points(f: &psfkit::PsfFile) -> Option<usize> {
    let n = f.header().get_i64("PSF sweep points").or_else(|| {
        f.psfxl_meta().get_i64("cdnshsweepcount").or_else(|| {
            f.psfxl_meta()
                .get_str("cdnshsweepcount")
                .and_then(|s| s.parse().ok())
        })
    })?;
    usize::try_from(n).ok().filter(|&n| n > 0)
}

enum Job {
    Wide(usize),
    Long(usize, Option<Vec<String>>),
}

/// Iterator over the batches of a scan: each step reads up to `step` jobs (leaves, or leaf x
/// signal chunks) in parallel and yields their record batches.
#[pyclass(module = "polars_psf", frozen)]
struct BatchIter {
    dir: Arc<ResultDir>,
    leaves: Arc<Vec<Leaf>>,
    names: Option<Vec<String>>,
    field: Option<String>,
    jobs: Mutex<VecDeque<Job>>,
    step: usize,
}

#[pymethods]
impl BatchIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<Vec<PyRecordBatch>>> {
        use rayon::prelude::*;
        let jobs: Vec<Job> = {
            let mut q = self.jobs.lock().unwrap();
            let n = self.step.min(q.len());
            q.drain(..n).collect()
        };
        if jobs.is_empty() {
            return Ok(None);
        }
        let batches = py.detach(|| {
            jobs.par_iter()
                .map(|job| match job {
                    Job::Wide(i) => leaf_batch(&self.dir, &self.leaves[*i], self.names.as_deref()),
                    Job::Long(i, sel) => leaf_batch_long(
                        &self.dir,
                        &self.leaves[*i],
                        sel.as_deref(),
                        self.field.as_deref(),
                    ),
                })
                .collect::<PyResult<Vec<_>>>()
        })?;
        Ok(Some(batches.into_iter().map(PyRecordBatch::new).collect()))
    }
}

/// A Spectre/ADE result directory (logFile, runObjFile; nested sweeps, Monte Carlo) or a single
/// PSF file opened as a directory with one result and one leaf.
#[pyclass(name = "ResultDir", module = "polars_psf", frozen)]
struct PyResultDir {
    inner: Arc<ResultDir>,
}

impl PyResultDir {
    fn leaves(&self, result: &str) -> PyResult<Arc<Vec<Leaf>>> {
        self.inner.leaves(result).map_err(res_err)
    }

    fn first(&self, result: &str) -> PyResult<(Arc<Vec<Leaf>>, Arc<psfkit::PsfFile>)> {
        let leaves = self.leaves(result)?;
        let leaf = leaves
            .first()
            .ok_or_else(|| PsfError::new_err(format!("result {result:?} has no leaves")))?;
        let f = self.inner.file(&leaf.path).map_err(res_err)?;
        Ok((leaves, f))
    }

    fn check(leaves: &[Leaf], idx: &[usize]) -> PyResult<()> {
        match idx.iter().find(|&&i| i >= leaves.len()) {
            Some(&bad) => Err(pyo3::exceptions::PyIndexError::new_err(bad)),
            None => Ok(()),
        }
    }
}

#[pymethods]
impl PyResultDir {
    #[new]
    fn new(py: Python<'_>, path: std::path::PathBuf) -> PyResult<Self> {
        let inner = py.detach(|| ResultDir::open(&path)).map_err(res_err)?;
        Ok(PyResultDir {
            inner: Arc::new(inner),
        })
    }

    #[getter]
    fn root(&self) -> String {
        self.inner.root().display().to_string()
    }

    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.inner.warnings()
    }

    #[getter]
    fn is_single_file(&self) -> bool {
        self.inner.is_single_file()
    }

    /// [(label, analysis_type, [param names], n_leaves, description)] in file order; labels are the short
    /// analysis names (`tran1`) unless two families share one.
    fn results(&self, py: Python<'_>) -> PyResult<Vec<ResultRow>> {
        let a = py.detach(|| self.inner.results()).map_err(res_err)?;
        Ok(a.into_iter()
            .map(|a| {
                let description = (!a.description.is_empty()).then_some(a.description);
                (a.label, a.analysis_type, a.params, a.leaves, description)
            })
            .collect())
    }

    /// Full result (family) name for a label or (possibly suffix) name.
    fn resolve(&self, result: &str) -> PyResult<String> {
        self.inner.resolve(result).map_err(res_err)
    }

    /// Parameter columns + `leaf` (logFile entry) + `path`, one row per leaf.
    fn leaf_table(&self, result: &str) -> PyResult<PyRecordBatch> {
        use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray};
        let leaves = self.leaves(result)?;
        let names: Vec<&str> = leaves.iter().map(|l| l.name.as_str()).collect();
        let paths: Vec<String> = leaves
            .iter()
            .map(|l| l.path.display().to_string())
            .collect();
        let mut cols: Vec<(String, ArrayRef)> = Vec::new();
        // parameter columns: same names for every leaf of a result
        if let Some(first) = leaves.first() {
            for (k, (name, v0)) in first.params.iter().enumerate() {
                let col: ArrayRef = match v0 {
                    psfkit_results::Param::Float(_) => Arc::new(Float64Array::from(
                        leaves
                            .iter()
                            .map(|l| l.params.get(k).and_then(|p| p.1.as_f64()))
                            .collect::<Vec<_>>(),
                    )),
                    psfkit_results::Param::Str(_) => Arc::new(StringArray::from(
                        leaves
                            .iter()
                            .map(|l| match l.params.get(k) {
                                Some((_, psfkit_results::Param::Str(s))) => Some(s.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>(),
                    )),
                };
                cols.push((name.clone(), col));
            }
        }
        cols.push(("leaf".into(), Arc::new(StringArray::from(names))));
        cols.push(("path".into(), Arc::new(StringArray::from(paths))));
        let b = RecordBatch::try_from_iter(cols).map_err(arrow_err)?;
        Ok(PyRecordBatch::new(b))
    }

    /// Empty batch with the wide output schema (params + sweep + traces), from the first leaf.
    #[pyo3(signature = (result, names=None))]
    fn schema(&self, result: &str, names: Option<Vec<String>>) -> PyResult<PyRecordBatch> {
        let (leaves, f) = self.first(result)?;
        let names: Option<Vec<&str>> = names
            .as_ref()
            .map(|n| n.iter().map(String::as_str).collect());
        let b = file_schema(&f, names.as_deref())?;
        Ok(PyRecordBatch::new(
            with_params(&leaves[0].params, b).map_err(arrow_err)?,
        ))
    }

    /// Empty batch with the long schema (`[params..., sweep, signal, value]`) from the first leaf.
    #[pyo3(signature = (result, field=None))]
    fn long_schema(&self, result: &str, field: Option<&str>) -> PyResult<PyRecordBatch> {
        let (leaves, f) = self.first(result)?;
        let b = long_schema_of(&f, field)?;
        Ok(PyRecordBatch::new(
            with_params(&leaves[0].params, b).map_err(arrow_err)?,
        ))
    }

    /// Signal names of the first leaf (only traces with member `field`, if given; values of
    /// non-swept files).
    #[pyo3(signature = (result, field=None))]
    fn signal_names(&self, result: &str, field: Option<&str>) -> PyResult<Vec<String>> {
        let (_, f) = self.first(result)?;
        Ok(match field {
            Some(fl) => with_field(&f, fl).into_iter().map(String::from).collect(),
            None => f
                .names()
                .map_err(err)?
                .into_iter()
                .map(String::from)
                .collect(),
        })
    }

    /// Sweep variable of the leaves (e.g. "time"), None if they are not swept.
    fn sweep_name(&self, result: &str) -> PyResult<Option<String>> {
        let (_, f) = self.first(result)?;
        Ok(f.sweeps().first().map(|s| s.name.clone()))
    }

    /// Path of a leaf (index into leaf_table).
    fn leaf_path(&self, result: &str, leaf: usize) -> PyResult<String> {
        let leaves = self.leaves(result)?;
        Self::check(&leaves, &[leaf])?;
        Ok(leaves[leaf].path.display().to_string())
    }

    /// Wide batches of the given leaves (indices into leaf_table); `names` selects traces
    /// (all if None). Each iteration step reads `batch` leaves in parallel.
    #[pyo3(signature = (result, leaves, names=None, batch=None))]
    fn scan(
        &self,
        result: &str,
        leaves: Vec<usize>,
        names: Option<Vec<String>>,
        batch: Option<usize>,
    ) -> PyResult<BatchIter> {
        let all = self.leaves(result)?;
        Self::check(&all, &leaves)?;
        Ok(BatchIter {
            dir: self.inner.clone(),
            leaves: all,
            names,
            field: None,
            jobs: Mutex::new(leaves.into_iter().map(Job::Wide).collect()),
            step: batch.unwrap_or_else(rayon::current_num_threads).max(1),
        })
    }

    /// Long batches for `jobs` = [(leaf index, signal names or None for all)]. Jobs with more
    /// signals than fit in about `chunk_rows` rows (signals x sweep points) are split, so memory
    /// stays bounded for long transients. A single file is read one chunk at a time; leaves of
    /// a directory `batch` at a time in parallel.
    #[pyo3(signature = (result, jobs, field=None, chunk_rows=1 << 22, batch=None))]
    fn scan_long(
        &self,
        result: &str,
        jobs: Vec<(usize, Option<Vec<String>>)>,
        field: Option<String>,
        chunk_rows: usize,
        batch: Option<usize>,
    ) -> PyResult<BatchIter> {
        let (all, f) = self.first(result)?;
        let idx: Vec<usize> = jobs.iter().map(|j| j.0).collect();
        Self::check(&all, &idx)?;
        // PSFXL stubs may lack a point count; assume long transients
        let points = sweep_points(&f).unwrap_or(if f.is_psfxl_stub() { 1 << 20 } else { 1 });
        let chunk = (chunk_rows / points).max(1);
        let names: Vec<String> = match field.as_deref() {
            Some(fl) => with_field(&f, fl).into_iter().map(String::from).collect(),
            None => f.traces().iter().map(|t| t.name.clone()).collect(),
        };
        let mut queue = VecDeque::new();
        for (i, sel) in jobs {
            let full = sel.as_ref().unwrap_or(&names);
            if full.len() <= chunk {
                queue.push_back(Job::Long(i, sel));
            } else {
                for c in full.chunks(chunk) {
                    queue.push_back(Job::Long(i, Some(c.to_vec())));
                }
            }
        }
        let step = if self.inner.is_single_file() {
            1
        } else {
            batch.unwrap_or_else(rayon::current_num_threads).max(1)
        };
        Ok(BatchIter {
            dir: self.inner.clone(),
            leaves: all,
            names: None,
            field,
            jobs: Mutex::new(queue),
            step,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ResultDir({:?}, results={})",
            self.root(),
            self.inner.results().map_or(0, |a| a.len())
        )
    }
}

#[pymodule]
fn _polars_psf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPsfFile>()?;
    m.add_class::<PyResultDir>()?;
    m.add_class::<BatchIter>()?;
    m.add("PsfError", m.py().get_type::<PsfError>())?;
    Ok(())
}
