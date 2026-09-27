//! Result directories of Spectre and ADE: resolves `logFile` / `runObjFile` trees (nested sweeps,
//! Monte Carlo, ADE parametric runs) into leaf PSF files, each tagged with the values of all its
//! outer sweep parameters.
//!
//! ```no_run
//! let r = psfkit_results::Results::open("sim.raw")?;
//! for a in r.analyses()? {
//!     println!("{} ({}): params {:?}, {} leaves", a.name, a.analysis_type, a.params, a.leaves);
//! }
//! for leaf in r.leaves("ac2")?.iter() {
//!     let f = psfkit::PsfFile::open(&leaf.path)?;
//!     println!("{:?} -> {}", leaf.params, f.traces().len());
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Everything is lazy: [`Results::open`] reads only the top-level `runObjFile`/`logFile`;
//! logFiles are parsed on the first [`Results::analyses`] call; parent sweep files are read and
//! leaf files checked only when [`Results::leaves`] is called for that analysis (then cached).
//!
//! Outer sweep values come from the parent `.sweep` / `.montecarlo` files (full precision); the
//! rounded values in `logFile` properties are only used to match children to sweep points, or as a
//! fallback (reported in [`Results::warnings`]).

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
    #[error("no analysis {0:?}")]
    UnknownAnalysis(String),
    #[error("analysis name {name:?} is ambiguous: {candidates:?}")]
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
    /// Analysis family (root entry name without the `-type` suffix, e.g. `mysweep_ac2`).
    pub analysis: String,
    /// Spectre analysis type of the data file (`dc`, `ac`, `tran`, ...).
    pub analysis_type: String,
    /// logFile entry name.
    pub name: String,
    /// Outer parameters, outermost first.
    pub params: Vec<(String, Param)>,
    pub path: PathBuf,
}

/// Summary of an analysis family (from the logFiles only; data files are not checked).
#[derive(Clone, Debug, PartialEq)]
pub struct Analysis {
    pub name: String,
    pub analysis_type: String,
    pub params: Vec<String>,
    /// Leaves listed in the logFiles (some may turn out missing in [`Results::leaves`]).
    pub leaves: usize,
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
    sweep: Option<String>,
    props: psfkit::Properties,
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

/// A result directory. Cheap to open; see the crate docs for what is read when.
pub struct Results {
    root: PathBuf,
    sources: Vec<Source>,
    trees: OnceLock<Vec<Tree>>,
    families: Mutex<HashMap<String, Arc<Vec<Leaf>>>>,
    warnings: Mutex<Vec<String>>,
}

impl Results {
    /// Opens a result directory (containing `runObjFile` or `logFile`), or one of those files.
    /// Reads only that file.
    pub fn open(path: impl AsRef<Path>) -> Result<Results> {
        let path = path.as_ref();
        let (dir, file) = if path.is_dir() {
            let pick = ["runObjFile", "logFile", "logFile.tmp"]
                .iter()
                .map(|n| path.join(n))
                .find(|p| p.is_file());
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
        Ok(Results {
            root: dir,
            sources,
            trees: OnceLock::new(),
            families: Mutex::new(HashMap::new()),
            warnings: Mutex::new(warnings),
        })
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

    /// Analysis families in first-seen order. Parses the logFiles on first call.
    pub fn analyses(&self) -> Result<Vec<Analysis>> {
        let mut out: Vec<Analysis> = Vec::new();
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
                    None => out.push(Analysis {
                        name,
                        analysis_type: t.entries[*first].atype.clone(),
                        params: t
                            .outer
                            .iter()
                            .map(|p| p.0.clone())
                            .chain(vars.iter().cloned())
                            .collect(),
                        leaves: leaves.len(),
                    }),
                }
            }
        }
        Ok(out)
    }

    /// Resolves an analysis name: exact family name, or a unique family whose name ends with
    /// `_<name>` (so `ac2` finds `mysweep_ac2`).
    pub fn resolve(&self, name: &str) -> Result<String> {
        let all = self.analyses()?;
        if all.iter().any(|a| a.name == name) {
            return Ok(name.to_owned());
        }
        let suffix = format!("_{name}");
        let c: Vec<String> = all
            .into_iter()
            .map(|a| a.name)
            .filter(|n| n.ends_with(&suffix))
            .collect();
        match c.len() {
            1 => Ok(c.into_iter().next().unwrap()),
            0 => Err(Error::UnknownAnalysis(name.to_owned())),
            _ => Err(Error::Ambiguous {
                name: name.to_owned(),
                candidates: c,
            }),
        }
    }

    /// Existing leaves of one analysis family (see [`Self::resolve`]). Reads that family's parent
    /// sweep files and checks its data files on first call; cached afterwards.
    pub fn leaves(&self, analysis: &str) -> Result<Arc<Vec<Leaf>>> {
        let name = self.resolve(analysis)?;
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

/// Resolution of one analysis family.
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
                analysis: self.family.to_owned(),
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
