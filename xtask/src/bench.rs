//! `xtask bench`: the first build measured on a frozen, representative
//! sample of the operator's corpus instead of the whole of it.
//!
//! `bench sample` draws the sample (devkit's stratified draw: by origin and
//! file kind, by year for session files, with each stratum's size spread)
//! and writes its manifest under `analysis/bench/` of the project home.
//! `bench run` copies the sample into the bench directory once, runs the
//! release `first_build` over it under the real-data lock, records the
//! per-stage result with periodic host-load observations, and prints an
//! uncalibrated linear input-byte estimate. `--incremental` appends the rest
//! of every session file and measures the incremental pass. `bench validate`
//! compares that assumption with a supplied full run; it does not establish
//! calibration across sizes or check candidate/input identity.
//!
//! `--target largest` does the same over the largest Claude Code session and
//! the largest Codex database, which a sample of a few GB cannot hold; it
//! measures peak memory and throughput on one huge input and predicts
//! nothing.
mod reads;

use devkit::bench::{self, Candidate, LoadMonitor, Manifest};
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

    fn dir(self) -> PathBuf {
        bench_home().join(self.name())
    }
}

fn manifest_path(target: Target) -> PathBuf {
    PathBuf::from(format!(
        "/realm/project/polylogue/analysis/bench/polylogue-{}.json",
        target.name()
    ))
}

const LOCK: &str = "/realm/tmp/work/realdata.lock";

#[derive(Debug, clap::Subcommand)]
pub enum Bench {
    /// Draw a representative sample of the corpus and write its manifest.
    Sample {
        #[arg(long, value_enum, default_value = "sample")]
        target: Target,
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
        /// `disk` keeps the archive on /realm like the real one; `memory`
        /// puts it in /dev/shm and measures the pipeline alone.
        #[arg(long, default_value = "disk")]
        storage: String,
        /// Also append the rest of each session file and measure the
        /// incremental pass.
        #[arg(long)]
        incremental: bool,
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
    /// Compare the latest linear estimate with a supplied full-run result.
    /// Candidate/input matching and multi-size calibration remain unchecked.
    Validate { result: PathBuf },
    /// Recorded runs and comparisons (devkit's shared commands).
    #[command(flatten)]
    Shared(devkit::cli::Bench),
}

pub fn execute(command: Bench) -> ExitCode {
    let result = match command {
        Bench::Sample {
            target,
            mb,
            seed,
            cut,
        } => sample(target, mb * 1_000_000, seed, cut),
        Bench::Run {
            target,
            storage,
            incremental,
        } => run(target, &storage, incremental),
        Bench::Reads {
            repetitions,
            full_walk,
        } => return reads::execute(repetitions, full_walk),
        Bench::Validate { result } => validate(&result),
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
pub fn candidates() -> Vec<Candidate> {
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
        out.push(Candidate {
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
        out.push(Candidate {
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

fn largest(all: &[Candidate]) -> Vec<Candidate> {
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

fn sample(target: Target, bytes: u64, seed: u64, cut: f64) -> Result<(), String> {
    let all = candidates();
    let chosen = match target {
        Target::Sample => bench::select(&all, bytes, seed),
        Target::Largest => largest(&all),
    };
    let manifest = Manifest::build("polylogue", &all, &chosen, bytes, seed, cut)
        .map_err(|e| format!("hash the sample: {e}"))?;
    let path = manifest_path(target);
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
    // AgentCTL admits the outer benchmark job; the lock excludes other
    // benchmarks over real data without queueing nested work.
    let output = Command::new("flock")
        .arg("-o")
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

fn run(target: Target, storage: &str, incremental: bool) -> Result<(), String> {
    let path = manifest_path(target);
    let manifest = Manifest::read(&path).map_err(|e| {
        format!(
            "{}: {e} (run `xtask bench sample --target {}` first)",
            path.display(),
            target.name()
        )
    })?;
    let sample_home = target.dir().join("home");
    let fresh = manifest
        .materialize(&sample_home)
        .map_err(|e| format!("materialize the sample: {e}"))?;
    eprintln!(
        "sample {} ({} files, {:.0} MB){}",
        manifest.id(),
        manifest.entries.len(),
        manifest.bytes() as f64 / 1e6,
        if fresh { ", copied" } else { "" }
    );
    let out = match storage {
        "disk" => target.dir().join("out"),
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
    let binary = first_build_binary()?;
    let load = LoadMonitor::start()?;
    let result = first_build(&binary, &sample_home, &out, true)?;
    let load = load.finish()?;
    let mut metrics = result.clone();
    let estimate = (target == Target::Sample).then(|| {
        bench::linear_estimate(
            result["seconds"].as_f64(),
            &stages_of(&result),
            manifest.representativeness()["scale"].as_f64(),
        )
    });
    if let Some(value) = &estimate {
        metrics["linear_estimate"] = value.clone();
    }
    let store = Store::open().map_err(|e| e.to_string())?;
    let checkout = env::current_dir().unwrap_or_default();
    let id = bench::record(
        &store,
        &checkout,
        "polylogue",
        target.bench(),
        &manifest.id(),
        &out,
        &load,
        &metrics,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "{}",
        json!({"run": id, "storage": bench::storage_of(&out),
               "load_checks_passed": load.checks_passed(), "load": load.to_json(),
               "result": result, "linear_estimate": estimate})
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
        metrics["appended_mb"] = json!(appended as f64 / 1e6);
        metrics["appended_mb_per_s"] =
            json!(seconds.map(|seconds| appended as f64 / 1e6 / seconds));
        let id = bench::record(
            &store,
            &checkout,
            "polylogue",
            &format!("{}-incremental", target.bench()),
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

fn validate(path: &Path) -> Result<(), String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let actual = text
        .lines()
        .rev()
        .find_map(|l| {
            serde_json::from_str::<Value>(l.trim())
                .ok()
                .filter(|v| v.get("seconds").is_some())
        })
        .ok_or("no first_build result line in that file")?;
    let store = Store::open().map_err(|e| e.to_string())?;
    let runs = bench::runs(&store, "polylogue", 50).map_err(|e| e.to_string())?;
    let run = runs
        .iter()
        .find(|r| r.bench == "first_build" && r.storage == "disk")
        .ok_or("no recorded first_build bench run on disk")?;
    let mut report = validation_report(&run.metrics["linear_estimate"], &actual)?;
    report["bench_run"] = json!(run.id);
    report["sample_load_checks_passed"] = json!(run.load_checks_passed);
    report["sample_load"] = run.load.clone();
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}

fn validation_report(estimate: &Value, actual: &Value) -> Result<Value, String> {
    let seconds = |v: &Value| v.as_f64().filter(|n| n.is_finite() && *n >= 0.0);
    if estimate["method"] != "linear_in_input_bytes" {
        return Err(
            "the recorded run has no explicit linear estimate; make a new sample run".into(),
        );
    }
    let estimated =
        seconds(&estimate["estimate"]["seconds"]).ok_or("the recorded estimate is unavailable")?;
    let real = seconds(&actual["seconds"]).ok_or("the supplied run has no valid seconds value")?;
    Ok(json!({
        "estimated_seconds": estimated, "actual_seconds": real,
        "relative_error": (real > 0.0).then(|| (estimated - real) / real),
        "scale": estimate["scale"], "assumptions": estimate["assumptions"],
        "actual_input_mb": actual["input_mb"], "calibrated": false,
        "candidate_match": "not_verified", "input_match": "not_verified",
        "full_run_load": actual.get("load"),
        "limitation": "One supplied result does not establish multi-size calibration or matching work."
    }))
}

#[cfg(test)]
mod tests;
