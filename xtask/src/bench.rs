//! `xtask bench`: builds of `first_build` compared on a fixed sample of the
//! operator's real inputs, under a memory limit.
//!
//! `bench sample` lists the inputs with the engine's own discovery
//! (`first_build --list-inputs`), groups each Claude Code session with its
//! subagents and tool results, draws whole groups with devkit's stratified
//! draw (by origin and year), copies them into the sample's home in the
//! default layout, and writes the manifest. The sample is drawn once and
//! reused: it is the fixed corpus for iteration and for comparisons.
//! `bench run` measures this checkout's build, and `bench pair` two given
//! builds, each repetition a fresh first build over the sample in its own
//! systemd unit with `MemoryMax`, the builds in turn (A B A B); the result,
//! with every repetition's unit accounting and host load and the comparison
//! per figure, is recorded in the history store. `bench tails` times the
//! last pages of the largest sessions in a run's archive, and `bench reads`
//! the interactive reads over synthetic inputs.
mod reads;
mod tails;

use devkit::bench::{self, Arm, Candidate, Figure, Limit, Manifest, Process, Repetition};
use devkit::history::Store;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    time::UNIX_EPOCH,
};

/// One root for manifests, sample homes, archives and the benchmark lock.
fn bench_root(explicit: Option<PathBuf>) -> PathBuf {
    explicit
        .or_else(|| env::var_os("POLYLOGUE_BENCH_HOME").map(PathBuf::from))
        .unwrap_or_else(|| {
            env::var_os("XDG_DATA_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| home().join(".local/share"))
                .join("polylogue-bench")
        })
}

fn sample_name(value: &str) -> Result<String, String> {
    let mut components = Path::new(value).components();
    if matches!(components.next(), Some(std::path::Component::Normal(name)) if name == std::ffi::OsStr::new(value))
        && components.next().is_none()
    {
        Ok(value.to_owned())
    } else {
        Err("a sample name must be one directory name".into())
    }
}

/// Where a run keeps its archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Storage {
    /// Under the benchmark root.
    Disk,
    /// In /dev/shm, measuring the pipeline alone. The archive's pages count
    /// against the unit's memory limit.
    Memory,
}

/// A drawn sample: the name its manifest, home and archives go under.
#[derive(Debug, Clone)]
struct Drawn {
    root: PathBuf,
    name: String,
}

impl Drawn {
    fn new(root: &Path, name: Option<String>) -> Self {
        Self {
            root: root.to_owned(),
            name: name.unwrap_or_else(|| "sample".into()),
        }
    }

    fn dir(&self) -> PathBuf {
        self.root.join(&self.name)
    }

    /// The sample's files, laid out as a home whose tools keep their
    /// default directories.
    fn home(&self) -> PathBuf {
        self.dir().join("home")
    }

    fn manifest(&self) -> PathBuf {
        self.root
            .join("manifests")
            .join(format!("polylogue-{}.json", self.name))
    }
}

/// What every measured repetition runs under.
#[derive(Debug, clap::Args)]
pub struct Conditions {
    #[arg(long)]
    #[arg(value_parser = sample_name)]
    name: Option<String>,
    /// `disk` keeps each archive under the benchmark root; `memory` puts it
    /// in /dev/shm and measures the pipeline alone.
    #[arg(long, value_enum, default_value = "disk")]
    storage: Storage,
    /// Measured first builds per build.
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..))]
    repetitions: u32,
    /// The memory limit of each measured build (systemd `MemoryMax`, no
    /// swap); a build that exceeds it is killed and reported.
    #[arg(long, default_value = "8G")]
    memory_max: String,
}

#[derive(Debug, clap::Args)]
pub struct Bench {
    /// Root for benchmark manifests, sample homes, archives and the lock.
    /// Defaults to POLYLOGUE_BENCH_HOME, then XDG_DATA_HOME/polylogue-bench.
    #[arg(long, global = true)]
    root: Option<PathBuf>,
    #[command(subcommand)]
    command: BenchCommand,
}

#[derive(Debug, clap::Subcommand)]
pub enum BenchCommand {
    /// Draw a sample of the inputs the engine discovers and write its
    /// manifest; it is kept and reused until drawn again.
    Sample {
        /// The sample's name, for keeping several (default: `sample`).
        #[arg(long)]
        #[arg(value_parser = sample_name)]
        name: Option<String>,
        /// Target size of the sample, in MB.
        #[arg(long, default_value_t = 3000)]
        mb: u64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// The `first_build` whose discovery lists the inputs, instead of
        /// this checkout's.
        #[arg(long)]
        binary: Option<PathBuf>,
    },
    /// Measure this checkout's first build over the sample and record it.
    Run {
        #[command(flatten)]
        conditions: Conditions,
        /// A `first_build` built elsewhere, instead of this checkout's.
        #[arg(long, requires = "build")]
        binary: Option<PathBuf>,
        /// The name the build is recorded under (default: the checkout's
        /// commit).
        #[arg(long)]
        build: Option<String>,
    },
    /// Two builds' first builds over one sample in turn (A B A B), compared
    /// per figure.
    Pair {
        #[command(flatten)]
        conditions: Conditions,
        /// `LABEL=PATH` of the first `first_build`, the base.
        #[arg(long)]
        a: String,
        /// `LABEL=PATH` of the second, compared against the first.
        #[arg(long)]
        b: String,
    },
    /// Cold and warm times of the last pages of a sample run's largest
    /// sessions and of any named ones, through the daemon and CLI.
    Tails {
        #[arg(long)]
        #[arg(value_parser = sample_name)]
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
    /// Cold/warm CLI and bounded session-page measurements of the real binary on synthetic inputs.
    Reads {
        /// Warm repetitions after each operation's first request.
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..))]
        repetitions: u32,
        /// Traverse every page rather than bounded first/middle/tail windows.
        #[arg(long)]
        full_walk: bool,
    },
    /// Recorded results (devkit's shared commands).
    #[command(flatten)]
    Shared(devkit::cli::Bench),
}

pub fn execute(options: Bench) -> ExitCode {
    let root = match std::path::absolute(bench_root(options.root)) {
        Ok(root) => root,
        Err(error) => {
            eprintln!("bench root: {error}");
            return ExitCode::FAILURE;
        }
    };
    let result = match options.command {
        BenchCommand::Sample {
            name,
            mb,
            seed,
            binary,
        } => mb
            .checked_mul(1_000_000)
            .ok_or_else(|| format!("--mb {mb} is too large"))
            .and_then(|bytes| sample(&Drawn::new(&root, name), bytes, seed, binary)),
        BenchCommand::Run {
            conditions,
            binary,
            build,
        } => (|| {
            let binary = match binary {
                Some(binary) => binary,
                None => first_build_binary()?,
            };
            let build = match build {
                Some(build) => sample_name(&build)?,
                None => head()?,
            };
            measure_builds(&root, &conditions, &[(build, binary)])
        })(),
        BenchCommand::Pair { conditions, a, b } => (|| {
            let builds = [labelled(&a)?, labelled(&b)?];
            if builds[0].0 == builds[1].0 {
                return Err("the two builds need different labels".into());
            }
            measure_builds(&root, &conditions, &builds)
        })(),
        BenchCommand::Tails {
            name,
            largest,
            session,
            binary,
        } => tails::run(&Drawn::new(&root, name), largest, &session, binary),
        BenchCommand::Reads {
            repetitions,
            full_walk,
        } => return reads::execute(&root, repetitions, full_walk),
        BenchCommand::Shared(b) => return devkit::cli::bench("polylogue", b),
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

/// One file `first_build --list-inputs` printed.
#[derive(Debug, Clone, PartialEq)]
struct Listed {
    path: PathBuf,
    origin: String,
    layout: PathBuf,
    bytes: u64,
    database: bool,
}

fn listed(line: &str) -> Result<Listed, String> {
    let value: Value = serde_json::from_str(line).map_err(|e| format!("{line}: {e}"))?;
    let text = |key: &str| {
        value[key]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| format!("listed input without {key}: {line}"))
    };
    Ok(Listed {
        path: text("path")?.into(),
        origin: text("origin")?,
        layout: text("layout")?.into(),
        bytes: value["bytes"]
            .as_u64()
            .ok_or_else(|| format!("listed input without bytes: {line}"))?,
        database: value["database"] == true,
    })
}

/// The inputs the engine discovers under this environment's home and the
/// tools' own root variables, read through `binary`'s discovery.
fn discovered(binary: &Path) -> Result<Vec<Listed>, String> {
    let output = Command::new(binary)
        .arg("--list-inputs")
        .arg("--home")
        .arg(home())
        .output()
        .map_err(|e| format!("{}: {e}", binary.display()))?;
    if !output.status.success() {
        return Err(format!(
            "first_build --list-inputs failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(listed)
        .collect()
}

/// The group a file is drawn with: a Claude Code session's file, its
/// subagents, workflows and tool results share the session's directory.
/// Every other file is drawn alone.
fn session_group(layout: &Path) -> Option<String> {
    let parts: Vec<&str> = layout.iter().filter_map(|p| p.to_str()).collect();
    match parts.as_slice() {
        [".claude", "projects", project, session, rest @ ..] => {
            let session = match rest {
                [] => session.strip_suffix(".jsonl")?,
                _ => session,
            };
            Some(format!(".claude/projects/{project}/{session}"))
        }
        _ => None,
    }
}

fn year(path: &Path) -> Option<u64> {
    fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        // Seconds to calendar years, close enough for an era bucket.
        .map(|d| 1970 + d.as_secs() / 31_556_952)
}

/// The draw's candidates: each listed file with its group, and the stratum
/// of its group (the origin of the group's own file, nearest the root, and
/// the year its newest file was written).
fn candidates(listed: &[Listed]) -> Vec<Candidate> {
    let mut groups: BTreeMap<String, Vec<&Listed>> = BTreeMap::new();
    let mut alone = Vec::new();
    for file in listed {
        match session_group(&file.layout) {
            Some(group) => groups.entry(group).or_default().push(file),
            None => alone.push((None, vec![file])),
        }
    }
    groups
        .into_iter()
        .map(|(group, files)| (Some(group), files))
        .chain(alone)
        .flat_map(|(group, mut files)| {
            files.sort_by(|a, b| a.layout.cmp(&b.layout));
            let year = files.iter().filter_map(|f| year(&f.path)).max();
            // The session's own file is the one nearest the root.
            let origin = &files
                .iter()
                .min_by_key(|f| f.layout.components().count())
                .expect("a group has a file")
                .origin;
            let stratum = match year {
                Some(year) => format!("{origin}@{year}"),
                None => origin.clone(),
            };
            files
                .into_iter()
                .map(|f| Candidate {
                    path: f.path.clone(),
                    rel: f.layout.clone(),
                    stratum: stratum.clone(),
                    size: f.bytes,
                    group: group.clone(),
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The benchmark root's lock, held for a whole command: runs and draws that
/// share a root exclude one another.
fn lock(root: &Path) -> Result<fs::File, String> {
    fs::create_dir_all(root).map_err(|e| format!("{}: {e}", root.display()))?;
    let path = root.join("bench.lock");
    let file = fs::File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    file.lock()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(file)
}

fn sample(drawn: &Drawn, bytes: u64, seed: u64, binary: Option<PathBuf>) -> Result<(), String> {
    let binary = match binary {
        Some(binary) => binary,
        None => first_build_binary()?,
    };
    let listed = discovered(&binary)?;
    if listed.is_empty() {
        return Err("discovery found no inputs under this home and its tools' directories".into());
    }
    let databases: Vec<&Path> = listed
        .iter()
        .filter(|f| f.database)
        .map(|f| f.layout.as_path())
        .collect();
    let all = candidates(&listed);
    let chosen = bench::select(&all, bytes, seed);
    let _lock = lock(&drawn.root)?;
    let home = drawn.home();
    if home.exists() {
        fs::remove_dir_all(&home).map_err(|e| format!("{}: {e}", home.display()))?;
    }
    // Hash copies, so a session that keeps growing cannot change the sample
    // between drawing it and measuring a build.
    let chosen = copy_sample(&home, &chosen, &databases)?;
    let manifest = Manifest::build("polylogue", &all, &chosen, bytes, seed)
        .map_err(|e| format!("hash the sample: {e}"))?;
    let path = drawn.manifest();
    manifest
        .write(&path)
        .map_err(|e| format!("write {}: {e}", path.display()))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "manifest": path,
            "home": home,
            "id": manifest.id(),
            "files": manifest.entries.len(),
            "bytes": manifest.bytes(),
            "discovered_files": listed.len(),
            "representativeness": manifest.representativeness(),
        }))
        .unwrap_or_default()
    );
    Ok(())
}

/// Copies the selected files into `home` at their layouts before the
/// manifest hashes them; the copies are what every measured build reads.
fn copy_sample(
    home: &Path,
    chosen: &[Candidate],
    databases: &[&Path],
) -> Result<Vec<Candidate>, String> {
    chosen
        .iter()
        .map(|file| {
            let path = home.join(&file.rel);
            fs::create_dir_all(path.parent().unwrap_or(home)).map_err(|e| e.to_string())?;
            copy_input(&file.path, &path, databases.contains(&file.rel.as_path()))?;
            let mut file = file.clone();
            file.size = fs::metadata(&path).map_err(|e| e.to_string())?.len();
            file.path = path;
            Ok(file)
        })
        .collect()
}

/// A database copy must include committed WAL rows from one read transaction.
fn copy_input(source: &Path, destination: &Path, database: bool) -> Result<(), String> {
    let copied = if database {
        match fs::remove_file(destination) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.to_string()),
        }
        let destination_sql = destination.to_string_lossy().replace('\'', "''");
        Command::new("sqlite3")
            .arg("-readonly")
            .arg(source)
            .arg(format!("VACUUM INTO '{destination_sql}'"))
            .output()
    } else {
        Command::new("cp")
            .args(["--reflink=auto", "--preserve=timestamps"])
            .arg(source)
            .arg(destination)
            .output()
    }
    .map_err(|error| error.to_string())?;
    if !copied.status.success() {
        return Err(format!(
            "copy {}: {}",
            source.display(),
            String::from_utf8_lossy(&copied.stderr)
        ));
    }
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
    std::path::absolute(target.join("release/examples/first_build")).map_err(|e| e.to_string())
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
    let stages = result["stages"]
        .as_object()
        .ok_or("first_build omitted its per-stage seconds")?;
    if let Some((stage, _)) = stages
        .iter()
        .find(|(_, seconds)| !seconds.as_f64().is_some_and(f64::is_finite))
    {
        return Err(format!("first_build has no seconds for stage {stage}"));
    }
    let messages = result["messages_written"]
        .as_u64()
        .ok_or("first_build omitted its message write count")?;
    if messages_required && messages == 0 {
        return Err("first_build wrote no messages".into());
    }
    Ok(result)
}

/// The figures one repetition gives, from the unit and from `first_build`'s
/// own result. Per-input figures divide by the sample's bytes. Stage times
/// are thread-seconds for pool stages.
fn figures(unit: &[Figure], result: &Value, input_bytes: u64) -> Vec<Figure> {
    let per_input = |bytes: Option<f64>| bytes.map(|b| b / input_bytes.max(1) as f64);
    let unit_value = |name: &str| unit.iter().find(|f| f.name == name).and_then(|f| f.value);
    let number = |key: &str| result[key].as_f64();
    let mb = |key: &str| number(key).map(|mb| mb * 1e6);
    let mut figures = vec![
        Figure::work("cpu_seconds", unit_value("cpu_seconds")),
        // Linux reports process CPU in ticks of 1/100 s.
        Figure::work(
            "process_cpu_seconds",
            number("cpu_ticks").map(|t| t / 100.0),
        ),
        Figure::work(
            "written_per_input_byte",
            per_input(unit_value("written_bytes")),
        ),
        Figure::work(
            "process_written_per_input_byte",
            per_input(mb("disk_written_mb")),
        ),
        Figure::work("read_per_input_byte", per_input(unit_value("read_bytes"))),
        Figure::work("memory_peak_bytes", unit_value("memory_peak_bytes")),
        Figure::work("peak_rss_bytes", mb("peak_rss_mb")),
        Figure::work("index_per_input_byte", per_input(mb("index_mb"))),
        Figure::work("stored_per_input_byte", per_input(mb("stored_mb"))),
        Figure::work("messages_written", number("messages_written")),
        Figure::time("wall_seconds", unit_value("wall_seconds")),
        Figure::time("ingest_seconds", number("ingest_seconds")),
        Figure::time("checkpoint_seconds", number("checkpoint_seconds")),
    ];
    if let Some(stages) = result["stages"].as_object() {
        figures.extend(
            stages
                .iter()
                .map(|(stage, seconds)| Figure::time(format!("stage.{stage}"), seconds.as_f64())),
        );
    }
    figures
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

/// The sample's manifest, with its home checked against it.
fn drawn_sample(drawn: &Drawn) -> Result<Manifest, String> {
    let path = drawn.manifest();
    let manifest = Manifest::read(&path).map_err(|e| {
        format!(
            "{}: {e} (run `xtask bench sample --name {}` first)",
            path.display(),
            drawn.name
        )
    })?;
    let home = drawn.home();
    for entry in &manifest.entries {
        let at = home.join(&entry.rel);
        if fs::metadata(&at).map(|m| m.len()).ok() != Some(entry.size) {
            return Err(format!(
                "{} does not match the manifest; draw the sample again",
                at.display()
            ));
        }
    }
    Ok(manifest)
}

/// Reads every sample file once, so each repetition starts with the same
/// page cache rather than the first paying for cold reads.
fn warm(home: &Path, manifest: &Manifest) -> Result<(), String> {
    for entry in &manifest.entries {
        let path = home.join(&entry.rel);
        let mut file = fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        std::io::copy(&mut file, &mut std::io::sink())
            .map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}

/// The checkout the runs are recorded against.
fn checkout() -> Result<PathBuf, String> {
    env::current_dir().map_err(|e| format!("the current directory: {e}"))
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

/// Splits `LABEL=PATH`.
fn labelled(spec: &str) -> Result<(String, PathBuf), String> {
    let (label, path) = spec
        .split_once('=')
        .filter(|(label, path)| !label.is_empty() && !path.is_empty())
        .ok_or_else(|| format!("{spec:?} is not LABEL=PATH"))?;
    let path = std::path::absolute(path).map_err(|e| e.to_string())?;
    Ok((sample_name(label)?, path))
}

/// Measures each build's first build over the sample `repetitions` times,
/// in turn, each in a fresh archive, and records the result. A single build
/// keeps its last archive in `<sample>/out` for `bench tails`; builds of a
/// pair keep theirs in `<sample>/out-<label>`.
fn measure_builds(
    root: &Path,
    conditions: &Conditions,
    builds: &[(String, PathBuf)],
) -> Result<(), String> {
    let drawn = Drawn::new(root, conditions.name.clone());
    let _lock = lock(root)?;
    let manifest = drawn_sample(&drawn)?;
    let home = drawn.home();
    let outs: Vec<PathBuf> = builds
        .iter()
        .map(|(label, _)| {
            let name = if builds.len() == 1 {
                "out".to_owned()
            } else {
                format!("out-{label}")
            };
            match conditions.storage {
                Storage::Disk => drawn.dir().join(name),
                Storage::Memory => {
                    PathBuf::from("/dev/shm").join(format!("polylogue-bench-{}-{name}", drawn.name))
                }
            }
        })
        .collect();
    // An archive in memory takes about its input's size again for each
    // build; filling /dev/shm would push the desktop into swap, so unknown
    // free space refuses too.
    if conditions.storage == Storage::Memory {
        let needed = manifest
            .bytes()
            .saturating_mul(3)
            .saturating_mul(builds.len() as u64);
        match available(Path::new("/dev/shm")) {
            Some(free) if free >= needed => {}
            _ => {
                return Err(format!(
                    "/dev/shm does not report at least three times the input ({:.0} MB) per build free; use --storage disk",
                    manifest.bytes() as f64 / 1e6
                ));
            }
        }
    }
    let arms: Vec<Arm> = builds
        .iter()
        .map(|(label, binary)| {
            Ok(Arm {
                label: label.clone(),
                build: json!({
                    "binary": binary,
                    "blake3": bench::hash_file(binary).map_err(|e| format!("{}: {e}", binary.display()))?,
                }),
            })
        })
        .collect::<Result<_, String>>()?;
    warm(&home, &manifest)?;
    let limit = Limit::memory(&conditions.memory_max);
    let input_bytes = manifest.bytes();
    eprintln!(
        "sample {} {} ({} files, {:.0} MB), {} repetitions per build under MemoryMax={}",
        drawn.name,
        manifest.id(),
        manifest.entries.len(),
        input_bytes as f64 / 1e6,
        conditions.repetitions,
        conditions.memory_max
    );
    let repetitions = bench::interleave(builds.len(), conditions.repetitions, |arm, round| {
        let (label, binary) = &builds[arm];
        let out = &outs[arm];
        if out.exists() {
            fs::remove_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
        }
        let process = Process::new(binary)
            .arg("--home")
            .arg(&home)
            .arg("--dir")
            .arg(out);
        let measured = bench::measure(&process, &limit)?;
        if !measured.succeeded() {
            return Err(format!(
                "{label}, round {}: first_build ended with {} ({}): {}",
                round + 1,
                measured.outcome,
                measured.accounting.to_json(),
                measured.last_error_line()
            ));
        }
        let result = first_build_result(&String::from_utf8_lossy(&measured.stdout), true)?;
        let figures = figures(&Figure::of_unit(&measured), &result, input_bytes);
        eprintln!(
            "{label}, round {}: {:.1} s, {:.1} CPU-s",
            round + 1,
            measured.accounting.wall_seconds.unwrap_or(f64::NAN),
            measured.accounting.cpu_seconds.unwrap_or(f64::NAN)
        );
        Ok(Repetition {
            arm,
            round,
            figures,
            record: json!({"measured": measured.to_json(), "first_build": result}),
        })
    })?;
    if conditions.storage == Storage::Memory {
        for out in &outs {
            let _ = fs::remove_dir_all(out);
        }
    }
    let result = bench::report(
        json!({"name": drawn.name, "id": manifest.id(), "files": manifest.entries.len(),
               "bytes": input_bytes}),
        &limit,
        &arms,
        &repetitions,
    );
    let store = Store::open().map_err(|e| e.to_string())?;
    let id = bench::record(
        &store,
        &checkout()?,
        "polylogue",
        "first_build",
        &manifest.id(),
        &outs[0],
        &result,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"result": id, "headline": result["headline"]}))
            .unwrap_or_default()
    );
    Ok(())
}

#[cfg(test)]
mod tests;
