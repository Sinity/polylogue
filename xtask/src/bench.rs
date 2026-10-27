//! `xtask bench`: the first build measured on a fixed, representative
//! sample of the operator's corpus instead of the whole of it.
//!
//! `bench sample` draws the sample (devkit's stratified draw: by origin and
//! file kind, by year for session files, with each stratum's size spread)
//! and writes its manifest under `analysis/bench/` of the project home.
//! `bench run` copies the sample into the bench directory once, runs the
//! release `first_build` over it under the real-data lock, records the
//! per-stage result with periodic host-load observations, and prints an
//! uncalibrated linear input-byte estimate. `--incremental` appends the rest
//! of every session file and measures the incremental pass. `--name` keeps
//! samples of several sizes apart; `bench estimate` fits each stage over the
//! latest runs of one build at two or more sizes and predicts the whole
//! corpus stage by stage, and `bench validate` compares that prediction with
//! a full run's result, stage by stage. `bench pair` runs two builds at once
//! over one sample, so both see the same load. `bench tails` times the last
//! pages of the largest sessions, and any named, of a sample run's archive.
//!
//! `--target largest` does the same over the largest Claude Code session and
//! the largest Codex database, which a sample of a few GB cannot hold; it
//! measures peak memory and throughput on one huge input and predicts
//! nothing.
mod fit;
mod reads;
mod tails;

use devkit::bench::{self, Candidate as SampleFile, LoadMonitor, Manifest};
use devkit::history::Store;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    time::UNIX_EPOCH,
};

/// The project home's bench files: manifests here, sample copies and
/// archives beside them. Operator data stays on this machine, outside Git.
fn bench_home() -> PathBuf {
    env::var_os("POLYLOGUE_BENCH_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/realm/project/polylogue/bench"))
}

/// What a bench runs over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Target {
    /// The stratified sample; its run permits an uncalibrated linear estimate.
    Sample,
    /// The largest Claude Code session and the largest Codex database,
    /// which the sample cannot hold: memory and throughput on one huge
    /// input. Nothing is predicted from it.
    Largest,
}

impl Target {
    fn name(self) -> &'static str {
        match self {
            Target::Sample => "sample",
            Target::Largest => "largest",
        }
    }

    /// The name runs are recorded under.
    fn bench(self) -> &'static str {
        match self {
            Target::Sample => "first_build",
            Target::Largest => "first_build-largest",
        }
    }
}

/// A drawn sample: its target and the name its manifest, copy and runs go
/// under (the target's name unless several sizes are kept).
#[derive(Debug, Clone)]
struct Drawn {
    target: Target,
    name: String,
}

impl Drawn {
    fn new(target: Target, name: Option<String>) -> Self {
        let name = name.unwrap_or_else(|| target.name().to_owned());
        Self { target, name }
    }

    fn dir(&self) -> PathBuf {
        bench_home().join(&self.name)
    }

    fn manifest(&self) -> PathBuf {
        env::var_os("POLYLOGUE_BENCH_MANIFEST_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/realm/project/polylogue/analysis/bench"))
            .join(format!("polylogue-{}.json", self.name))
    }
}

const LOCK: &str = "/realm/tmp/work/realdata.lock";

#[derive(Debug, clap::Subcommand)]
pub enum Bench {
    /// Draw a representative sample of the corpus and write its manifest.
    Sample {
        #[arg(long, value_enum, default_value = "sample")]
        target: Target,
        /// The sample's name, for keeping samples of several sizes
        /// (default: the target's name).
        #[arg(long)]
        name: Option<String>,
        /// Target size of the sample, in MB.
        #[arg(long, default_value_t = 3000)]
        mb: u64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Session files are cut to this fraction for the first build; the
        /// rest is appended for the incremental pass.
        #[arg(long, default_value_t = 0.9)]
        cut: f64,
    },
    /// Run the first build over the sample and record it.
    Run {
        #[arg(long, value_enum, default_value = "sample")]
        target: Target,
        #[arg(long)]
        name: Option<String>,
        /// A `first_build` built elsewhere, instead of this checkout's.
        #[arg(long, requires = "build")]
        binary: Option<PathBuf>,
        /// The name the run's build is recorded and fitted under (default:
        /// the checkout's commit).
        #[arg(long)]
        build: Option<String>,
        /// `disk` keeps the archive on /realm like the real one; `memory`
        /// puts it in /dev/shm and measures the pipeline alone.
        #[arg(long, default_value = "disk")]
        storage: String,
        /// Also append the rest of each session file and measure the
        /// incremental pass.
        #[arg(long)]
        incremental: bool,
    },
    /// Two builds' first builds at once over one sample, so both see the same
    /// load; records both and prints their differences per stage.
    Pair {
        #[arg(long)]
        name: Option<String>,
        /// `LABEL=PATH` of the first `first_build`.
        #[arg(long)]
        a: String,
        /// `LABEL=PATH` of the second.
        #[arg(long)]
        b: String,
    },
    /// Fits each stage over the latest runs of one build at two or more sample
    /// sizes and predicts the whole corpus stage by stage.
    Estimate {
        /// The build (default: the checkout's commit).
        #[arg(long)]
        build: Option<String>,
    },
    /// Cold and warm times of the last pages of a sample run's largest
    /// sessions and of any named ones, through the daemon and CLI.
    Tails {
        #[arg(long)]
        name: Option<String>,
        /// Largest sessions by message count to read.
        #[arg(long, default_value_t = 5)]
        largest: usize,
        /// A session to read besides the largest.
        #[arg(long)]
        session: Vec<String>,
        /// A `polylogue` built elsewhere, instead of this checkout's.
        #[arg(long)]
        binary: Option<PathBuf>,
    },
    /// Native cold/warm CLI and bounded session-page measurements on synthetic inputs.
    Reads {
        /// Warm repetitions after each operation's first request.
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..))]
        repetitions: u32,
        /// Traverse every page rather than bounded first/middle/tail windows.
        #[arg(long)]
        full_walk: bool,
    },
    /// Compare a build's estimate with a supplied full-run result, stage by
    /// stage: the fit over two or more sample sizes, or else the latest
    /// linear estimate. That the full run used the same build and input is
    /// not checked.
    Validate {
        result: PathBuf,
        /// The build (default: the checkout's commit).
        #[arg(long)]
        build: Option<String>,
    },
    /// Recorded runs and comparisons (devkit's shared commands).
    #[command(flatten)]
    Shared(devkit::cli::Bench),
}

pub fn execute(command: Bench) -> ExitCode {
    let result = match command {
        Bench::Sample {
            target,
            name,
            mb,
            seed,
            cut,
        } => sample(&Drawn::new(target, name), mb * 1_000_000, seed, cut),
        Bench::Run {
            target,
            name,
            binary,
            build,
            storage,
            incremental,
        } => run(
            &Drawn::new(target, name),
            binary,
            build,
            &storage,
            incremental,
        ),
        Bench::Pair { name, a, b } => pair(&Drawn::new(Target::Sample, name), &a, &b),
        Bench::Estimate { build } => fit::estimate(build),
        Bench::Tails {
            name,
            largest,
            session,
            binary,
        } => tails::run(&Drawn::new(Target::Sample, name), largest, &session, binary),
        Bench::Reads {
            repetitions,
            full_walk,
        } => return reads::execute(repetitions, full_walk),
        Bench::Validate { result, build } => fit::validate(&result, build),
        Bench::Shared(b) => return devkit::cli::bench("polylogue", b),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("bench: {e}");
            ExitCode::FAILURE
        }
    }
}

fn home() -> PathBuf {
    env::var_os("HOME").map(PathBuf::from).unwrap_or_default()
}

/// The tools' own directories, as their override variables name them.
fn claude_dir() -> PathBuf {
    env::var_os("CLAUDE_CONFIG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(".claude"))
}

fn codex_dir() -> PathBuf {
    env::var_os("CODEX_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(".codex"))
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let Ok(kind) = e.file_type() else { continue };
        if kind.is_dir() {
            walk(&e.path(), out);
        } else if kind.is_file() {
            out.push(e.path());
        }
    }
}

fn year(meta: &fs::Metadata) -> String {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        // Seconds to calendar years, close enough for an era bucket.
        .map(|d| (1970 + d.as_secs() / 31_556_952).to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// Every file the first build reads, with its stratum, in the layout the
/// engine expects under a home directory (`.claude/…`, `.codex/…`).
pub fn input_files() -> Vec<SampleFile> {
    let mut out = Vec::new();
    let claude = claude_dir();
    let mut files = Vec::new();
    for sub in [
        "projects",
        "todos",
        "plans",
        "tasks",
        "jobs",
        "file-history",
    ] {
        walk(&claude.join(sub), &mut files);
    }
    files.push(claude.join("history.jsonl"));
    for path in files {
        let Ok(meta) = fs::metadata(&path) else {
            continue;
        };
        let rel = path.strip_prefix(&claude).unwrap_or(&path);
        let s = rel.to_string_lossy();
        let (stratum, grows) = if s.contains("/tool-results/") {
            ("claude/tool-results".to_owned(), false)
        } else if s.contains("/subagents/") && s.ends_with(".jsonl") {
            (format!("claude/subagent@{}", year(&meta)), true)
        } else if s.starts_with("projects/")
            && s.ends_with(".jsonl")
            && rel.components().count() == 3
        {
            (format!("claude/main@{}", year(&meta)), true)
        } else {
            ("claude/side".to_owned(), false)
        };
        out.push(SampleFile {
            rel: Path::new(".claude").join(rel),
            path,
            stratum,
            size: meta.len(),
            grows,
        });
    }
    let codex = codex_dir();
    let mut files = Vec::new();
    walk(&codex.join("sessions"), &mut files);
    if let Ok(entries) = fs::read_dir(&codex) {
        files.extend(entries.flatten().map(|e| e.path()).filter(|p| p.is_file()));
    }
    for path in files {
        let Ok(meta) = fs::metadata(&path) else {
            continue;
        };
        let rel = path.strip_prefix(&codex).unwrap_or(&path);
        let s = rel.to_string_lossy();
        let stratum = if s.starts_with("sessions/") && s.ends_with(".jsonl") {
            format!("codex/rollout@{}", year(&meta))
        } else if s.ends_with(".sqlite") {
            "codex/database".to_owned()
        } else if s.ends_with(".jsonl") {
            "codex/index".to_owned()
        } else {
            // Configuration, logs, WAL side files: the engine does not read
            // them as inputs.
            continue;
        };
        out.push(SampleFile {
            rel: Path::new(".codex").join(rel),
            path,
            grows: stratum.starts_with("codex/rollout"),
            stratum,
            size: meta.len(),
        });
    }
    out
}

/// The largest file of each of these stratum families.
const LARGEST: &[&str] = &["claude/main", "codex/database"];

fn largest(all: &[SampleFile]) -> Vec<SampleFile> {
    LARGEST
        .iter()
        .filter_map(|family| {
            all.iter()
                .filter(|c| c.stratum.starts_with(family))
                .max_by_key(|c| c.size)
                .cloned()
        })
        .collect()
}

fn sample(drawn: &Drawn, bytes: u64, seed: u64, cut: f64) -> Result<(), String> {
    let all = input_files();
    let chosen = match drawn.target {
        Target::Sample => bench::select(&all, bytes, seed),
        Target::Largest => largest(&all),
    };
    // Hash copies, so a session that keeps growing cannot invalidate the
    // sample between drawing it and starting the measured build.
    let chosen = copy_sample(&drawn.dir().join("source"), &chosen)?;
    let manifest = Manifest::build("polylogue", &all, &chosen, bytes, seed, cut)
        .map_err(|e| format!("hash the sample: {e}"))?;
    let path = drawn.manifest();
    manifest
        .write(&path)
        .map_err(|e| format!("write {}: {e}", path.display()))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "manifest": path,
            "id": manifest.id(),
            "files": manifest.entries.len(),
            "representativeness": manifest.representativeness(),
        }))
        .unwrap_or_default()
    );
    Ok(())
}

/// Copies the selected files before the manifest hashes them. Its source
/// paths then name these fixed bytes, rather than files writers still change.
fn copy_sample(directory: &Path, chosen: &[SampleFile]) -> Result<Vec<SampleFile>, String> {
    chosen
        .iter()
        .map(|file| {
            let path = directory.join(&file.rel);
            fs::create_dir_all(path.parent().unwrap_or(directory)).map_err(|e| e.to_string())?;
            let copied = Command::new("cp")
                .args(["--reflink=auto", "--preserve=timestamps"])
                .arg(&file.path)
                .arg(&path)
                .output()
                .map_err(|e| e.to_string())?;
            if !copied.status.success() {
                return Err(format!(
                    "copy {}: {}",
                    file.path.display(),
                    String::from_utf8_lossy(&copied.stderr)
                ));
            }
            let mut file = file.clone();
            file.size = fs::metadata(&path).map_err(|e| e.to_string())?.len();
            file.path = path;
            Ok(file)
        })
        .collect()
}

fn first_build_binary() -> Result<PathBuf, String> {
    let ok = devkit::cargo_command()
        .args([
            "build",
            "--locked",
            "--release",
            "-p",
            "polylogue-engine",
            "--example",
            "first_build",
        ])
        .status()
        .map_err(|e| e.to_string())?
        .success();
    if !ok {
        return Err("building first_build failed".into());
    }
    let target = env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target"));
    Ok(target.join("release/examples/first_build"))
}

/// Runs `first_build` under the real-data lock (`flock -o`: the lock is not
/// inherited by anything the run starts) and returns its JSON line.
fn first_build(
    binary: &Path,
    home: &Path,
    out: &Path,
    messages_required: bool,
) -> Result<Value, String> {
    first_build_locked(binary, home, out, messages_required, false)
}

/// As [`first_build`], sharing the lock with the other run of a pair.
fn first_build_shared(binary: &Path, home: &Path, out: &Path) -> Result<Value, String> {
    first_build_locked(binary, home, out, true, true)
}

fn first_build_locked(
    binary: &Path,
    home: &Path,
    out: &Path,
    messages_required: bool,
    shared: bool,
) -> Result<Value, String> {
    // AgentCTL queues the outer benchmark job; the lock excludes other
    // benchmarks over real data without queueing nested work.
    let output = Command::new("flock")
        .args(if shared {
            ["-s", "-o"].as_slice()
        } else {
            ["-o"].as_slice()
        })
        .arg(LOCK)
        .arg(binary)
        .arg("--home")
        .arg(home)
        .arg("--dir")
        .arg(out)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "first_build failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    first_build_result(&String::from_utf8_lossy(&output.stdout), messages_required)
}

fn first_build_result(stdout: &str, messages_required: bool) -> Result<Value, String> {
    let result = stdout
        .lines()
        .rev()
        .find_map(|line| serde_json::from_str::<Value>(line).ok())
        .ok_or_else(|| "first_build printed no result".to_owned())?;
    let errors = result["errors"]
        .as_u64()
        .ok_or("first_build omitted its ingestion error count")?;
    if errors != 0 {
        return Err(format!(
            "first_build reported {errors} ingestion errors: {}",
            result["first_errors"]
        ));
    }
    let unreadable = result["unreadable_root_count"]
        .as_u64()
        .ok_or("first_build omitted its unreadable-root count")?;
    if unreadable != 0 {
        return Err(format!(
            "first_build could not read {unreadable} source roots"
        ));
    }
    for field in ["input_mb", "seconds"] {
        if !result[field]
            .as_f64()
            .is_some_and(|n| n.is_finite() && n > 0.0)
        {
            return Err(format!("first_build has no positive {field} measurement"));
        }
    }
    let messages = result["messages_written"]
        .as_u64()
        .ok_or("first_build omitted its message write count")?;
    if messages_required && messages == 0 {
        return Err("first_build wrote no messages".into());
    }
    Ok(result)
}

fn stages_of(result: &Value) -> BTreeMap<String, f64> {
    result
        .get("stages")
        .and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| Some((k.clone(), v.as_f64()?)))
                .collect()
        })
        .unwrap_or_default()
}

/// Bytes free in the filesystem holding `path`.
fn available(path: &Path) -> Option<u64> {
    let out = Command::new("df")
        .args(["--output=avail", "-B1"])
        .arg(path)
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .nth(1)?
        .trim()
        .parse()
        .ok()
}

/// The sample's manifest and its copy, made once.
fn materialized(drawn: &Drawn) -> Result<(Manifest, PathBuf), String> {
    let path = drawn.manifest();
    let manifest = Manifest::read(&path).map_err(|e| {
        format!(
            "{}: {e} (run `xtask bench sample --target {} --name {}` first)",
            path.display(),
            drawn.target.name(),
            drawn.name
        )
    })?;
    let home = drawn.dir().join("home");
    let fresh = manifest
        .materialize(&home)
        .map_err(|e| format!("materialize the sample: {e}"))?;
    eprintln!(
        "sample {} {} ({} files, {:.0} MB){}",
        drawn.name,
        manifest.id(),
        manifest.entries.len(),
        manifest.bytes() as f64 / 1e6,
        if fresh { ", copied" } else { "" }
    );
    Ok((manifest, home))
}

/// The checkout's commit, the default name of its build.
fn head() -> Result<String, String> {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err("git rev-parse HEAD failed".into());
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

/// The metrics a sample run records: the result, its build, the linear
/// estimate and the corpus input it stands for.
fn run_metrics(drawn: &Drawn, manifest: &Manifest, build: &str, result: &Value) -> Value {
    let mut metrics = result.clone();
    metrics["build"] = json!(build);
    if drawn.target == Target::Sample {
        let scale = manifest.representativeness()["scale"].as_f64();
        metrics["linear_estimate"] =
            bench::linear_estimate(result["seconds"].as_f64(), &stages_of(result), scale);
        metrics["corpus_input_mb"] =
            json!(scale.zip(result["input_mb"].as_f64()).map(|(s, mb)| s * mb));
    }
    metrics
}

fn run(
    drawn: &Drawn,
    binary: Option<PathBuf>,
    build: Option<String>,
    storage: &str,
    incremental: bool,
) -> Result<(), String> {
    let (manifest, sample_home) = materialized(drawn)?;
    let out = match storage {
        "disk" => drawn.dir().join("out"),
        "memory" => PathBuf::from("/dev/shm/polylogue-bench-out"),
        other => return Err(format!("storage is disk or memory, not {other}")),
    };
    // An archive in memory takes about its input's size again; filling
    // /dev/shm would push the desktop into swap.
    if storage == "memory"
        && available(Path::new("/dev/shm")).is_some_and(|free| free < manifest.bytes() * 3)
    {
        return Err(format!(
            "/dev/shm has less than three times the input ({:.0} MB) free; use --storage disk",
            manifest.bytes() as f64 / 1e6
        ));
    }
    if out.exists() {
        fs::remove_dir_all(&out).map_err(|e| e.to_string())?;
    }
    let binary = match binary {
        Some(binary) => binary,
        None => first_build_binary()?,
    };
    let build = match build {
        Some(build) => build,
        None => head()?,
    };
    let load = LoadMonitor::start()?;
    let result = first_build(&binary, &sample_home, &out, true)?;
    let load = load.finish()?;
    let metrics = run_metrics(drawn, &manifest, &build, &result);
    let store = Store::open().map_err(|e| e.to_string())?;
    let checkout = env::current_dir().unwrap_or_default();
    let id = bench::record(
        &store,
        &checkout,
        "polylogue",
        drawn.target.bench(),
        &manifest.id(),
        &out,
        &load,
        &metrics,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "{}",
        json!({"run": id, "build": build, "storage": bench::storage_of(&out),
               "load_checks_passed": load.checks_passed(), "load": load.to_json(),
               "result": result, "linear_estimate": metrics.get("linear_estimate")})
    );
    if incremental {
        let appended = manifest
            .grow(&sample_home)
            .map_err(|e| format!("append: {e}"))?;
        let load = LoadMonitor::start()?;
        let result = first_build(&binary, &sample_home, &out, false)?;
        let load = load.finish()?;
        let seconds = result["seconds"]
            .as_f64()
            .filter(|n| n.is_finite() && *n > 0.0);
        let mut metrics = result.clone();
        metrics["build"] = json!(build);
        metrics["appended_mb"] = json!(appended as f64 / 1e6);
        metrics["appended_mb_per_s"] =
            json!(seconds.map(|seconds| appended as f64 / 1e6 / seconds));
        let id = bench::record(
            &store,
            &checkout,
            "polylogue",
            &format!("{}-incremental", drawn.target.bench()),
            &manifest.id(),
            &out,
            &load,
            &metrics,
        )
        .map_err(|e| e.to_string())?;
        println!(
            "{}",
            json!({"run": id, "incremental": metrics, "load": load.to_json()})
        );
    }
    if storage == "memory" {
        let _ = fs::remove_dir_all(&out);
    }
    Ok(())
}

/// Splits `LABEL=PATH`.
fn labelled(spec: &str) -> Result<(String, PathBuf), String> {
    let (label, path) = spec
        .split_once('=')
        .filter(|(label, path)| !label.is_empty() && !path.is_empty())
        .ok_or_else(|| format!("{spec:?} is not LABEL=PATH"))?;
    Ok((label.to_owned(), PathBuf::from(path)))
}

/// Both builds' first builds at once over one sample, on disk.
fn pair(drawn: &Drawn, a: &str, b: &str) -> Result<(), String> {
    let builds = [labelled(a)?, labelled(b)?];
    if builds[0].0 == builds[1].0 {
        return Err("the two builds need different labels".into());
    }
    let (manifest, sample_home) = materialized(drawn)?;
    let outs: Vec<PathBuf> = builds
        .iter()
        .map(|(label, _)| drawn.dir().join(format!("out-{label}")))
        .collect();
    for out in &outs {
        if out.exists() {
            fs::remove_dir_all(out).map_err(|e| e.to_string())?;
        }
    }
    let load = LoadMonitor::start()?;
    // Both run at once, so both see the same host load; the real-data lock
    // is shared between them.
    let results: Vec<Result<Value, String>> = std::thread::scope(|scope| {
        let runs: Vec<_> = builds
            .iter()
            .zip(&outs)
            .map(|((_, binary), out)| {
                let home = &sample_home;
                scope.spawn(move || first_build_shared(binary, home, out))
            })
            .collect();
        runs.into_iter()
            .map(|run| {
                run.join()
                    .unwrap_or_else(|_| Err("a build panicked".into()))
            })
            .collect()
    });
    let load = load.finish()?;
    let store = Store::open().map_err(|e| e.to_string())?;
    let checkout = env::current_dir().unwrap_or_default();
    let mut recorded = Vec::new();
    for (((label, _), out), result) in builds.iter().zip(&outs).zip(results) {
        let result = result?;
        let mut metrics = run_metrics(drawn, &manifest, label, &result);
        metrics["paired"] = json!(true);
        let id = bench::record(
            &store,
            &checkout,
            "polylogue",
            "first_build-pair",
            &manifest.id(),
            out,
            &load,
            &metrics,
        )
        .map_err(|e| e.to_string())?;
        recorded.push((label.clone(), id, result));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "runs": recorded.iter().map(|(label, id, _)| json!({"build": label, "run": id})).collect::<Vec<_>>(),
            "load_checks_passed": load.checks_passed(), "load": load.to_json(),
            "difference": fit::difference(&recorded[0].2, &recorded[1].2),
            "results": recorded.iter().map(|(label, _, result)| json!({"build": label, "result": result})).collect::<Vec<_>>(),
        }))
        .unwrap_or_default()
    );
    Ok(())
}

#[cfg(test)]
mod tests;
