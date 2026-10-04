//! Result directories of Spectre and ADE: resolves `logFile` / `runObjFile` trees (nested sweeps,
//! Monte Carlo, ADE parametric runs) into leaf PSF files, each tagged with the values of all its
//! outer sweep parameters. A single PSF file opens as a directory with one result and one
//! parameter-free leaf, so callers handle both the same way.
//!
//! ```no_run
//! let r = psfkit_results::ResultDir::open("sim.raw")?;
//! for a in r.results()? {
//!     println!("{} ({}): params {:?}, {} leaves", a.label, a.analysis_type, a.params, a.leaves);
//! }
//! for leaf in r.leaves("ac2")?.iter() {
//!     let f = psfkit::PsfFile::open(&leaf.path)?;
//!     println!("{:?} -> {}", leaf.params, f.traces().len());
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Everything is lazy: [`ResultDir::open`] reads only the top-level `runObjFile`/`logFile`;
//! logFiles are parsed on the first [`ResultDir::results`] call; parent sweep files are read and
//! leaf files checked only when [`ResultDir::leaves`] is called for that result (then cached).
//!
//! Outer sweep values come from the parent `.sweep` / `.montecarlo` files (full precision); the
//! rounded values in `logFile` properties are only used to match children to sweep points, or as a
//! fallback (reported in [`ResultDir::warnings`]).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use psfkit::{PsfFile, Value};

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error(transparent)]
    Psf(#[from] psfkit::Error),
    #[error("{0}: no logFile or runObjFile found")]
    NotResults(PathBuf),
    #[error("{path}: {msg}")]
    Invalid { path: PathBuf, msg: String },
    #[error("no result {0:?}")]
    UnknownResult(String),
    #[error("result name {name:?} is ambiguous: {candidates:?}")]
    Ambiguous {
        name: String,
        candidates: Vec<String>,
    },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Value of an outer sweep parameter.
#[derive(Clone, Debug, PartialEq)]
pub enum Param {
    Float(f64),
    Str(String),
}

impl Param {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Param::Float(v) => Some(*v),
            Param::Str(_) => None,
        }
    }
}

/// One PSF data file and the outer parameters it was simulated at.
#[derive(Clone, Debug, PartialEq)]
pub struct Leaf {
    /// Result family (root entry name without the `-type` suffix, e.g. `mysweep_ac2`).
    pub result: String,
    /// Spectre analysis type of the data file (`dc`, `ac`, `tran`, ...).
    pub analysis_type: String,
    /// logFile entry name.
    pub name: String,
    /// Outer parameters, outermost first.
    pub params: Vec<(String, Param)>,
    pub path: PathBuf,
}

/// Summary of the result of one analysis: its family of leaf files over all sweep points (from
/// the logFiles only; data files are not checked).
#[derive(Clone, Debug, PartialEq)]
pub struct ResultInfo {
    /// Family name (root logFile entry without `-type`, e.g. `swp_t_tran1`).
    pub name: String,
    /// Name for listings: the analysis' own name (`tran1`) when no other family shares it,
    /// otherwise [`Self::name`]. Accepted by [`ResultDir::resolve`].
    pub label: String,
    pub analysis_type: String,
    pub params: Vec<String>,
    /// Leaves listed in the logFiles (some may turn out missing in [`ResultDir::leaves`]).
    pub leaves: usize,
    /// The simulator's description of the first leaf in its logFile (``Transient Analysis
    /// `tran1': time = (0 s -> 5 ms)``); empty if there is none (single PSF files).
    pub description: String,
}

/// A logFile to read, with the outer parameters of its ADE run (empty for plain Spectre).
struct Source {
    log: PathBuf,
    outer: Vec<(String, Param)>,
}

/// Parsed logFile.
struct Tree {
    dir: PathBuf,
    outer: Vec<(String, Param)>,
    entries: Vec<Entry>,
    children: HashMap<String, Vec<usize>>,
}

struct Entry {
    name: String,
    atype: String,
    data_file: String,
    description: String,
    sweep: Option<String>,
    props: psfkit::Properties,
}

/// The analysis' own name: the longest `_`-delimited tail shared by the family name and the
/// first leaf entry (`swp_t_tran1` + `swp_t-000_swp_r-000_tran1-tran` -> `tran1`).
fn own_name<'a>(family: &'a str, leaf: &Entry) -> &'a str {
    let entry = leaf
        .name
        .strip_suffix(&format!("-{}", leaf.atype))
        .unwrap_or(&leaf.name);
    let common = family
        .bytes()
        .rev()
        .zip(entry.bytes().rev())
        .take_while(|(a, b)| a == b)
        .count();
    if common == family.len() {
        return family; // not nested (or identical names)
    }
    let tail = &family[family.len() - common..];
    // cut at a `_` that both names have right before the tail
    match tail.find('_') {
        Some(i) if i + 1 < tail.len() => &tail[i + 1..],
        _ => family,
    }
}

impl Tree {
    fn kids(&self, name: &str) -> &[usize] {
        self.children.get(name).map(Vec::as_slice).unwrap_or(&[])
    }

    fn roots(&self) -> &[usize] {
        self.kids("")
    }

    fn family(&self, root: usize) -> String {
        let e = &self.entries[root];
        e.name
            .strip_suffix(&format!("-{}", e.atype))
            .unwrap_or(&e.name)
            .to_owned()
    }

    /// Leaf entries under `i` (file order), with the sweep variables on the path.
    fn leaf_entries(
        &self,
        i: usize,
        vars: &mut Vec<String>,
        out: &mut Vec<(usize, Vec<String>)>,
        depth: usize,
    ) {
        let e = &self.entries[i];
        let kids = self.kids(&e.name);
        if depth > 64 {
            return;
        }
        if kids.is_empty() {
            if !matches!(e.atype.as_str(), "sweep" | "montecarlo") {
                out.push((i, vars.clone()));
            }
            return;
        }
        if let Some(v) = &e.sweep {
            vars.push(v.clone());
        }
        for &k in kids {
            self.leaf_entries(k, vars, out, depth + 1);
        }
        if e.sweep.is_some() {
            vars.pop();
        }
    }
}

/// Data files kept open by [`ResultDir::file`] (least recently used are dropped first).
const OPEN_FILES: usize = 64;

/// A result directory. Cheap to open; see the crate docs for what is read when.
pub struct ResultDir {
    root: PathBuf,
    sources: Vec<Source>,
    trees: OnceLock<Vec<Tree>>,
    families: Mutex<HashMap<String, Arc<Vec<Leaf>>>>,
    warnings: Mutex<Vec<String>>,
    /// A single PSF file opened as a one-leaf directory.
    single: Option<(ResultInfo, Arc<Vec<Leaf>>)>,
    /// Open data files, most recently used last.
    files: Mutex<Vec<(PathBuf, Arc<PsfFile>)>>,
}

const DIR_FILES: [&str; 3] = ["runObjFile", "logFile", "logFile.tmp"];

impl ResultDir {
    /// Opens a result directory (containing `runObjFile` or `logFile`), one of those files, or
    /// a single PSF data file (see [`Self::from_file`]). Reads only that file.
    pub fn open(path: impl AsRef<Path>) -> Result<ResultDir> {
        let path = path.as_ref();
        let is_dir_file = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| DIR_FILES.contains(&n));
        if !path.is_dir() && !is_dir_file {
            return Self::from_file(path);
        }
        let (dir, file) = if path.is_dir() {
            let pick = DIR_FILES.iter().map(|n| path.join(n)).find(|p| p.is_file());
            (
                path.to_owned(),
                pick.ok_or_else(|| Error::NotResults(path.to_owned()))?,
            )
        } else {
            (
                path.parent().unwrap_or(Path::new(".")).to_owned(),
                path.to_owned(),
            )
        };
        let mut warnings = Vec::new();
        let sources = if file.file_name().and_then(|n| n.to_str()) == Some("runObjFile") {
            run_obj(&file, &mut warnings)?
        } else {
            vec![Source {
                log: file,
                outer: Vec::new(),
            }]
        };
        Ok(ResultDir {
            root: dir,
            sources,
            trees: OnceLock::new(),
            families: Mutex::new(HashMap::new()),
            warnings: Mutex::new(warnings),
            single: None,
            files: Mutex::new(Vec::new()),
        })
    }

    /// Opens one PSF data file as a result directory with a single result (named after the
    /// header's `analysis name`, else the file name up to the first `.`) and one leaf without
    /// parameters. Reads the file's declarations.
    pub fn from_file(path: impl AsRef<Path>) -> Result<ResultDir> {
        let path = path.as_ref();
        let f = Arc::new(PsfFile::open(path)?);
        let header = f.header();
        let name = header
            .get_str("analysis name")
            .map(str::to_owned)
            .unwrap_or_else(|| {
                let file = path
                    .file_name()
                    .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
                file.split('.').next().unwrap_or_default().to_owned()
            });
        let analysis_type = header
            .get_str("analysis type")
            .unwrap_or_default()
            .to_owned();
        let info = ResultInfo {
            name: name.clone(),
            label: name.clone(),
            analysis_type: analysis_type.clone(),
            params: Vec::new(),
            leaves: 1,
            description: String::new(),
        };
        let leaf = Leaf {
            result: name.clone(),
            analysis_type,
            name,
            params: Vec::new(),
            path: path.to_owned(),
        };
        Ok(ResultDir {
            root: path.to_owned(),
            sources: Vec::new(),
            trees: OnceLock::new(),
            families: Mutex::new(HashMap::new()),
            warnings: Mutex::new(Vec::new()),
            single: Some((info, Arc::new(vec![leaf]))),
            files: Mutex::new(vec![(path.to_owned(), f)]),
        })
    }

    /// True if this was opened from a single PSF data file.
    pub fn is_single_file(&self) -> bool {
        self.single.is_some()
    }

    /// An opened data file, shared between calls. The last [`OPEN_FILES`] files stay open.
    pub fn file(&self, path: &Path) -> Result<Arc<PsfFile>> {
        {
            let mut files = self.files.lock().unwrap();
            if let Some(i) = files.iter().position(|(p, _)| p == path) {
                let hit = files.remove(i);
                let f = hit.1.clone();
                files.push(hit);
                return Ok(f);
            }
        }
        // open outside the lock: leaves are read in parallel
        let f = Arc::new(PsfFile::open(path)?);
        let mut files = self.files.lock().unwrap();
        if files.len() >= OPEN_FILES {
            files.remove(0);
        }
        files.push((path.to_owned(), f.clone()));
        Ok(f)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Problems found so far that did not stop resolution (missing logFiles or data files,
    /// rounded parameter values). Grows as more of the directory is resolved.
    pub fn warnings(&self) -> Vec<String> {
        self.warnings.lock().unwrap().clone()
    }

    fn warn(&self, w: impl IntoIterator<Item = String>) {
        self.warnings.lock().unwrap().extend(w);
    }

    /// Parses all logFiles (once).
    fn trees(&self) -> Result<&[Tree]> {
        if let Some(t) = self.trees.get() {
            return Ok(t);
        }
        let mut trees = Vec::with_capacity(self.sources.len());
        let mut missing = Vec::new();
        for s in &self.sources {
            if !s.log.is_file() {
                missing.push(format!("missing logFile {}", s.log.display()));
                continue;
            }
            trees.push(parse_log(&s.log, &s.outer)?);
        }
        // report only from the thread whose result is kept
        if self.trees.set(trees).is_ok() {
            self.warn(missing);
        }
        Ok(self.trees.get().expect("set above"))
    }

    /// Result families in first-seen order. Parses the logFiles on first call.
    pub fn results(&self) -> Result<Vec<ResultInfo>> {
        if let Some((info, _)) = &self.single {
            return Ok(vec![info.clone()]);
        }
        let mut out: Vec<ResultInfo> = Vec::new();
        for t in self.trees()? {
            for &root in t.roots() {
                let mut leaves = Vec::new();
                t.leaf_entries(root, &mut Vec::new(), &mut leaves, 0);
                let Some((first, vars)) = leaves.first() else {
                    continue;
                };
                let name = t.family(root);
                match out.iter_mut().find(|a| a.name == name) {
                    Some(a) => a.leaves += leaves.len(),
                    None => out.push(ResultInfo {
                        label: own_name(&name, &t.entries[*first]).to_owned(),
                        name,
                        analysis_type: t.entries[*first].atype.clone(),
                        params: t
                            .outer
                            .iter()
                            .map(|p| p.0.clone())
                            .chain(vars.iter().cloned())
                            .collect(),
                        leaves: leaves.len(),
                        description: t.entries[*first].description.clone(),
                    }),
                }
            }
        }
        // own names shared by several families (`tran1` and `mymonte_tran1`): keep family names
        let mut count: HashMap<String, usize> = HashMap::new();
        for a in &out {
            *count.entry(a.label.clone()).or_default() += 1;
        }
        for a in &mut out {
            if count[&a.label] > 1 {
                a.label = a.name.clone();
            }
        }
        Ok(out)
    }

    /// Resolves a result name: exact family name or label, or a unique family whose name ends
    /// with `_<name>` (so `ac2` finds `mysweep_ac2`).
    pub fn resolve(&self, name: &str) -> Result<String> {
        if let Some((info, _)) = &self.single {
            // a single file also answers to its analysis type ("ac", "tran")
            return if name == info.name || name == info.analysis_type {
                Ok(info.name.clone())
            } else {
                Err(Error::UnknownResult(name.to_owned()))
            };
        }
        let all = self.results()?;
        if let Some(a) = all.iter().find(|a| a.name == name || a.label == name) {
            return Ok(a.name.clone());
        }
        let suffix = format!("_{name}");
        let c: Vec<String> = all
            .into_iter()
            .map(|a| a.name)
            .filter(|n| n.ends_with(&suffix))
            .collect();
        match c.len() {
            1 => Ok(c.into_iter().next().unwrap()),
            0 => Err(Error::UnknownResult(name.to_owned())),
            _ => Err(Error::Ambiguous {
                name: name.to_owned(),
                candidates: c,
            }),
        }
    }

    /// Existing leaves of one result family (see [`Self::resolve`]). Reads that family's parent
    /// sweep files and checks its data files on first call; cached afterwards.
    pub fn leaves(&self, result: &str) -> Result<Arc<Vec<Leaf>>> {
        let name = self.resolve(result)?;
        if let Some((_, leaves)) = &self.single {
            return Ok(leaves.clone());
        }
        if let Some(l) = self.families.lock().unwrap().get(&name) {
            return Ok(l.clone());
        }
        let mut leaves = Vec::new();
        let mut warnings = Vec::new();
        for t in self.trees()? {
            for &root in t.roots() {
                if t.family(root) == name {
                    let mut w = Walk {
                        t,
                        family: &name,
                        leaves: &mut leaves,
                        warnings: &mut warnings,
                    };
                    w.walk(root, t.outer.clone(), 0)?;
                }
            }
        }
        self.warn(warnings);
        let leaves = Arc::new(leaves);
        self.families.lock().unwrap().insert(name, leaves.clone());
        Ok(leaves)
    }
}

/// Resolution of one result family.
struct Walk<'a> {
    t: &'a Tree,
    family: &'a str,
    leaves: &'a mut Vec<Leaf>,
    warnings: &'a mut Vec<String>,
}

impl Walk<'_> {
    fn walk(&mut self, i: usize, params: Vec<(String, Param)>, depth: usize) -> Result<()> {
        let t = self.t;
        if depth > 64 {
            return Err(Error::Invalid {
                path: t.dir.clone(),
                msg: "logFile parent cycle".into(),
            });
        }
        let e = &t.entries[i];
        let kids = t.kids(&e.name);
        let path = t.dir.join(&e.data_file);
        if kids.is_empty() {
            if matches!(e.atype.as_str(), "sweep" | "montecarlo") {
                return Ok(()); // sweep without results
            }
            if !path.is_file() {
                self.warnings
                    .push(format!("{}: missing data file {}", e.name, path.display()));
                return Ok(());
            }
            self.leaves.push(Leaf {
                result: self.family.to_owned(),
                analysis_type: e.atype.clone(),
                name: e.name.clone(),
                params,
                path,
            });
            return Ok(());
        }
        let Some(var) = e.sweep.clone() else {
            // parent without sweep variable: pass through
            for &k in kids {
                self.walk(k, params.clone(), depth + 1)?;
            }
            return Ok(());
        };
        let values = sweep_values(&path);
        if let Err(err) = &values {
            self.warnings.push(format!(
                "{}: cannot read sweep values ({err}); using logFile values",
                e.name
            ));
        }
        let assigned = self.assign(kids, values.ok(), &var, &e.name);
        for (&k, v) in kids.iter().zip(assigned) {
            let Some(v) = v else { continue };
            let mut p = params.clone();
            p.push((var.clone(), v));
            self.walk(k, p, depth + 1)?;
        }
        Ok(())
    }

    /// Matches children to sweep points: by order when counts agree and the rounded logFile
    /// values confirm it, otherwise by nearest value.
    fn assign(
        &mut self,
        kids: &[usize],
        values: Option<Vec<Param>>,
        var: &str,
        parent: &str,
    ) -> Vec<Option<Param>> {
        let t = self.t;
        let hints: Vec<Option<Param>> = kids
            .iter()
            .map(|&k| prop_param(&t.entries[k].props, var))
            .collect();
        let Some(values) = values else {
            if hints.iter().any(Option::is_none) {
                self.warnings.push(format!(
                    "{parent}: children without value for {var:?} skipped"
                ));
            } else {
                self.warnings
                    .push(format!("{parent}: {var:?} values rounded (from logFile)"));
            }
            return hints;
        };
        let agrees = |v: &Param, h: &Option<Param>| h.as_ref().is_none_or(|h| param_close(v, h));
        if values.len() == kids.len() && values.iter().zip(&hints).all(|(v, h)| agrees(v, h)) {
            return values.into_iter().map(Some).collect();
        }
        hints
            .iter()
            .enumerate()
            .map(|(i, h)| {
                let found = h
                    .as_ref()
                    .and_then(|h| values.iter().find(|v| param_close(v, h)).cloned());
                if found.is_none() {
                    self.warnings.push(format!(
                        "{parent}: child {} has no matching {var:?} value; skipped",
                        t.entries[kids[i]].name
                    ));
                }
                found
            })
            .collect()
    }
}

/// ADE `runObjFile`: tree of run objects; leaves list logFiles, outer values are in props.
fn run_obj(file: &Path, warnings: &mut Vec<String>) -> Result<Vec<Source>> {
    let f = PsfFile::open(file)?;
    let dir = file.parent().unwrap_or(Path::new("."));
    struct Run<'a> {
        name: &'a str,
        logs: Vec<&'a str>,
        parent: &'a str,
        sweep: Option<&'a str>,
        props: &'a psfkit::Properties,
    }
    let values = f.values()?;
    let runs: Vec<Run> = values
        .iter()
        .map(|v| Run {
            name: &v.name,
            logs: str_list(v.value.field("logName")),
            parent: v
                .value
                .field("parent")
                .and_then(Value::as_str)
                .unwrap_or(""),
            sweep: str_list(v.value.field("sweepVariable")).into_iter().next(),
            props: &v.props,
        })
        .collect();
    let by_name: HashMap<&str, &Run> = runs.iter().map(|r| (r.name, r)).collect();
    let mut sources = Vec::new();
    for run in &runs {
        let logs = run.logs.iter().filter(|l| {
            matches!(
                Path::new(l).file_name().and_then(|n| n.to_str()),
                Some("logFile" | "logFile.tmp")
            )
        });
        let mut params = Vec::new();
        let mut node = run;
        let mut depth = 0;
        // outer params: each node stores its value of the parent's sweep variable
        while let Some(parent) = by_name.get(node.parent) {
            if let Some(var) = parent.sweep {
                match prop_param(node.props, var) {
                    Some(v) => params.push((var.to_owned(), v)),
                    None => warnings.push(format!("run {:?}: no value for {var:?}", node.name)),
                }
            }
            node = parent;
            depth += 1;
            if depth > 64 {
                return Err(Error::Invalid {
                    path: file.to_owned(),
                    msg: "runObjFile parent cycle".into(),
                });
            }
        }
        params.reverse();
        for log in logs {
            sources.push(Source {
                log: dir.join(log),
                outer: params.clone(),
            });
        }
    }
    Ok(sources)
}

/// Spectre `logFile`: analysis entries with parent links (data files are not touched).
fn parse_log(file: &Path, outer: &[(String, Param)]) -> Result<Tree> {
    let f = PsfFile::open(file)?;
    let mut entries = Vec::new();
    let mut parents = Vec::new();
    for v in f.values()?.iter().filter(|v| v.type_name == "analysisInst") {
        let s = |k: &str| {
            v.value
                .field(k)
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned()
        };
        parents.push(s("parent"));
        entries.push(Entry {
            name: v.name.clone(),
            atype: s("analysisType"),
            data_file: s("dataFile"),
            description: s("description"),
            sweep: str_list(v.value.field("sweepVariable"))
                .first()
                .map(|s| s.to_string()),
            props: v.props.clone(),
        });
    }
    let mut children: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, p) in parents.into_iter().enumerate() {
        children.entry(p).or_default().push(i);
    }
    Ok(Tree {
        dir: file.parent().unwrap_or(Path::new(".")).to_owned(),
        outer: outer.to_vec(),
        entries,
        children,
    })
}

fn str_list(v: Option<&Value>) -> Vec<&str> {
    match v {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .filter(|s| !s.is_empty())
            .collect(),
        Some(Value::String(s)) if !s.is_empty() => vec![s],
        _ => Vec::new(),
    }
}

fn prop_param(p: &psfkit::Properties, key: &str) -> Option<Param> {
    Some(match p.get(key)? {
        psfkit::PropValue::String(s) => Param::Str(s.clone()),
        other => Param::Float(other.as_f64()?),
    })
}

/// logFile values have ~6 significant digits.
fn param_close(a: &Param, b: &Param) -> bool {
    match (a, b) {
        (Param::Float(x), Param::Float(y)) => (x - y).abs() <= 1e-5 * x.abs().max(y.abs()) + 1e-300,
        (Param::Str(x), Param::Str(y)) => x == y,
        _ => false,
    }
}

/// Sweep values of a parent `.sweep` / `.montecarlo` file.
fn sweep_values(path: &Path) -> Result<Vec<Param>> {
    let f = PsfFile::open(path)?;
    let d = f.read(&[])?;
    if let Some(v) = d.sweep.to_f64() {
        return Ok(v.into_iter().map(Param::Float).collect());
    }
    Ok((0..d.sweep.len())
        .map(|i| match d.sweep.get(i) {
            Some(Value::String(s)) => Param::Str(s),
            v => Param::Str(format!("{v:?}")),
        })
        .collect())
}
