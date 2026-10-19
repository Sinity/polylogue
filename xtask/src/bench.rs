//! `xtask bench`: the first build measured on a frozen, representative
//! sample of the operator's corpus instead of the whole of it.
//!
//! `bench sample` draws the sample (devkit's stratified draw: by origin and
//! file kind, by year for session files, with each stratum's size spread)
//! and writes its manifest under `analysis/bench/` of the project home.
//! `bench run` copies the sample into the bench directory once, runs the
//! release `first_build` over it under the real-data lock, records the
//! per-stage result with the host's load, and prints the prediction for the
//! full corpus. `--incremental` then appends the rest of every session file
//! and measures the incremental pass. `bench validate` compares the latest
//! prediction with a full run's result.
//!
//! `--target largest` does the same over the largest Claude Code session and
//! the largest Codex database, which a sample of a few GB cannot hold; it
//! measures peak memory and throughput on one huge input and predicts
//! nothing.
use devkit::bench::{self, Candidate, HostLoad, Manifest};
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
    /// The stratified sample; its run predicts the full corpus.
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

/// Machine-wide build slots: a command run through this script waits for
/// one of a fixed number of slots, so builds and bench runs don't pile up on
/// one host. Skipped where the script doesn't exist.
fn slotted(program: &str) -> Command {
    let slot = env::var_os("BUILD_SLOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/realm/tmp/work/build-slot"));
    if slot.exists() {
        let mut c = Command::new(slot);
        c.arg(program);
        c
    } else {
        Command::new(program)
    }
}

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
    /// Compare the latest prediction with a full run's result (the JSON line
    /// `first_build` prints, inside any file).
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
    let ok = slotted("cargo")
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
fn first_build(binary: &Path, home: &Path, out: &Path) -> Result<Value, String> {
    // A build slot, then the real-data lock: the run neither competes with
    // builds nor with other runs over real data.
    let output = slotted("flock")
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
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .rev()
        .find_map(|l| serde_json::from_str::<Value>(l).ok())
        .ok_or_else(|| "first_build printed no result".into())
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
    let load = HostLoad::now();
    let result = first_build(&binary, &sample_home, &out)?;
    let mut metrics = result.clone();
    let prediction = (target == Target::Sample).then(|| {
        let scale = manifest.representativeness()["scale"]
            .as_f64()
            .unwrap_or(1.0);
        let seconds = result["seconds"].as_f64().unwrap_or(0.0);
        bench::predict(seconds, &stages_of(&result), scale)
    });
    if let Some(p) = &prediction {
        metrics["prediction"] = p.clone();
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
        load,
        &metrics,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "{}",
        json!({"run": id, "storage": bench::storage_of(&out), "quiet": load.quiet(),
               "load": {"load1": load.load1, "io_full": load.io_full, "memory_full": load.memory_full},
               "result": result, "prediction": prediction})
    );
    if incremental {
        let appended = manifest
            .grow(&sample_home)
            .map_err(|e| format!("append: {e}"))?;
        let load = HostLoad::now();
        let result = first_build(&binary, &sample_home, &out)?;
        let seconds = result["seconds"].as_f64().unwrap_or(0.0);
        let mut metrics = result.clone();
        metrics["appended_mb"] = json!(appended as f64 / 1e6);
        metrics["appended_mb_per_s"] = json!(appended as f64 / 1e6 / seconds.max(1e-9));
        let id = bench::record(
            &store,
            &checkout,
            "polylogue",
            &format!("{}-incremental", target.bench()),
            &manifest.id(),
            &out,
            load,
            &metrics,
        )
        .map_err(|e| e.to_string())?;
        println!("{}", json!({"run": id, "incremental": metrics}));
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
    let predicted = run.metrics["prediction"]["seconds"].as_f64().unwrap_or(0.0);
    let real = actual["seconds"].as_f64().unwrap_or(0.0);
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "bench_run": run.id,
            "bench_quiet": run.quiet,
            "predicted_seconds": predicted,
            "actual_seconds": real,
            "error": if real > 0.0 { (predicted - real) / real } else { f64::NAN },
            "actual_input_mb": actual["input_mb"],
            "predicted_scale": run.metrics["prediction"]["scale"],
        }))
        .unwrap_or_default()
    );
    Ok(())
}
