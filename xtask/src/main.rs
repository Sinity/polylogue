use serde_json::Value;
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::{self, Command, ExitCode},
};

/// Packages rustfmt checks. The shared crates keep their upstream layout.
const FORMATTED: &[&str] = &[
    "polylogue-model",
    "polylogue-query",
    "polylogue-sources",
    "polylogue-engine",
    "polylogue-store",
    "polylogue",
    "xtask",
];

/// Internal dependencies each workspace package may use. Only the store may
/// depend on rusqlite.
fn allowed_dependencies(package: &str) -> Option<&'static [&'static str]> {
    Some(match package {
        "polylogue-model" | "hashed-blobs" | "credential-output" | "xtask" => &[],
        "polylogue-sources" | "polylogue-query" => &["polylogue-model"],
        "polylogue-store" => &["polylogue-model", "hashed-blobs"],
        "polylogue-engine" => &[
            "polylogue-model",
            "polylogue-sources",
            "polylogue-query",
            "polylogue-store",
        ],
        "polylogue" => &[
            "polylogue-model",
            "polylogue-store",
            "polylogue-sources",
            "polylogue-query",
            "polylogue-engine",
            "credential-output",
        ],
        _ => return None,
    })
}

fn metadata(root: &Path) -> Result<Value, String> {
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(root)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    serde_json::from_slice(&output.stdout).map_err(|e| e.to_string())
}

fn boundaries(metadata: &Value) -> Result<(), String> {
    let packages = metadata["packages"]
        .as_array()
        .ok_or("metadata packages missing")?;
    let workspace: Vec<&str> = packages.iter().filter_map(|p| p["name"].as_str()).collect();
    for package in packages {
        let name = package["name"].as_str().ok_or("package name missing")?;
        let allowed = allowed_dependencies(name)
            .ok_or_else(|| format!("unassigned workspace package {name}"))?;
        for dependency in package["dependencies"]
            .as_array()
            .ok_or("dependencies missing")?
        {
            let dependency = dependency["name"]
                .as_str()
                .ok_or("dependency name missing")?;
            if (workspace.contains(&dependency) && !allowed.contains(&dependency))
                || (dependency == "rusqlite" && name != "polylogue-store")
            {
                return Err(format!("forbidden dependency: {name} -> {dependency}"));
            }
        }
    }
    Ok(())
}

fn status(command: &mut Command) -> Result<(), ExitCode> {
    match command.status() {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(ExitCode::from(status.code().unwrap_or(1) as u8)),
        Err(error) => {
            eprintln!("{command:?}: {error}");
            Err(ExitCode::FAILURE)
        }
    }
}

fn cargo(args: &[&str]) -> Result<(), ExitCode> {
    status(Command::new("cargo").args(args))
}

/// A per-run temporary directory on tmpfs, removed when dropped. Tests write
/// fsync'd archives; on the btrfs work disk those syncs dominate test time.
/// Under an AgentCTL tmpfs scratch it nests inside that, so a killed run is
/// still cleaned up by the job.
struct RunTemp(PathBuf);
impl RunTemp {
    fn new() -> Result<Self, ExitCode> {
        let inherited = env::temp_dir();
        let base = if inherited.starts_with("/dev/shm") || !Path::new("/dev/shm").is_dir() {
            inherited
        } else {
            PathBuf::from("/dev/shm")
        };
        let path = base.join(format!("polylogue-test-{}", process::id()));
        fs::create_dir(&path).map_err(|error| {
            eprintln!("{}: {error}", path.display());
            ExitCode::FAILURE
        })?;
        Ok(Self(path))
    }
}
impl Drop for RunTemp {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("{}: {error}", self.0.display());
        }
    }
}

/// Runs `cargo nextest` with temporary files on tmpfs. Tests that check
/// durability on a real filesystem find one in `TEST_DURABLE_TMPDIR`.
fn nextest(metadata: &Value, filter: &[&str]) -> Result<(), ExitCode> {
    let target = metadata["target_directory"].as_str().ok_or_else(|| {
        eprintln!("metadata target_directory missing");
        ExitCode::FAILURE
    })?;
    let durable = Path::new(target).join("tmp");
    fs::create_dir_all(&durable).map_err(|error| {
        eprintln!("{}: {error}", durable.display());
        ExitCode::FAILURE
    })?;
    let temp = RunTemp::new()?;
    status(
        Command::new("cargo")
            .args([
                "nextest",
                "run",
                "--locked",
                "--no-fail-fast",
                "--no-tests=fail",
            ])
            .args(filter)
            .env("TMPDIR", &temp.0)
            .env("TEST_DURABLE_TMPDIR", &durable),
    )
}

fn fmt_check() -> Result<(), ExitCode> {
    let mut args = vec!["fmt"];
    for package in FORMATTED {
        args.extend(["-p", package]);
    }
    args.extend(["--", "--check"]);
    cargo(&args)
}

fn run(command: &str) -> Result<(), ExitCode> {
    let metadata = metadata(Path::new(".")).and_then(|metadata| {
        boundaries(&metadata)?;
        Ok(metadata)
    });
    let metadata = metadata.map_err(|error| {
        eprintln!("{error}");
        ExitCode::FAILURE
    })?;
    match command {
        "boundaries" => Ok(()),
        "fast" => {
            fmt_check()?;
            cargo(&["check", "--locked", "--workspace", "--all-targets"])
        }
        "check" => {
            fmt_check()?;
            nextest(&metadata, &["--workspace"])
        }
        "scenarios" => nextest(&metadata, &["-p", "polylogue", "-E", "kind(test)"]),
        _ => unreachable!(),
    }
}

fn usage() -> ExitCode {
    eprintln!("usage: cargo run -p xtask -- <fast|check|scenarios|boundaries>");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<_> = env::args().skip(1).collect();
    let [command] = args.as_slice() else {
        return usage();
    };
    if !matches!(
        command.as_str(),
        "fast" | "check" | "scenarios" | "boundaries"
    ) {
        return usage();
    }
    match run(command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(code) => code,
    }
}

#[cfg(test)]
mod tests;
