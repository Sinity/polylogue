//! `cargo xtask run`: the checkout's daemon, standalone, in a sandbox on a
//! scratch root, until the command is stopped.
use devkit::process::{Process, development_lock, source_hash_excluding};
use polylogue_testkit::Sandbox;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::time::Duration;

#[derive(Debug, clap::Args)]
pub struct Development {
    /// The scratch root: the sandbox's home, configuration and archive.
    /// Created when missing; an archive already there is reopened. Without
    /// it, a new directory under `$TMPDIR` that is removed at the stop.
    #[arg(long)]
    root: Option<PathBuf>,
    /// Serve with this binary (an installed one, say) instead of building
    /// the checkout's.
    #[arg(long, conflicts_with = "watch")]
    binary: Option<PathBuf>,
    /// Rebuild the daemon when the checkout's sources change, and restart it
    /// on the new build; a failed build keeps the running one.
    #[arg(long)]
    watch: bool,
}

/// How often `--watch` hashes the sources.
const WATCH_INTERVAL: Duration = Duration::from_secs(1);

pub fn execute(options: Development) -> Result<(), String> {
    let checkout = std::fs::canonicalize(concat!(env!("CARGO_MANIFEST_DIR"), "/.."))
        .map_err(|e| e.to_string())?;
    // One development daemon per checkout builds and runs its binary.
    let _lock = match options.binary {
        Some(_) => None,
        None => Some(development_lock(&checkout)?),
    };
    let scratch;
    let root = match &options.root {
        Some(root) => root.clone(),
        None => {
            scratch = tempfile::Builder::new()
                .prefix("polylogue-run-")
                .tempdir()
                .map_err(|e| format!("scratch root: {e}"))?;
            scratch.path().to_owned()
        }
    };
    let sandbox = Sandbox::new(&root).map_err(|e| format!("sandbox: {e}"))?;
    let archive = sandbox.root().join("archive");
    sandbox
        .create_directories()
        .and_then(|()| sandbox.write_deployment(&archive))
        .map_err(|e| format!("sandbox {}: {e}", sandbox.root().display()))?;
    let binary = match options.binary {
        Some(binary) => std::path::absolute(binary).map_err(|e| e.to_string())?,
        None => build(&checkout)?,
    };
    let cli = sandbox.root().join("polylogue");
    write_cli(&cli, &sandbox, &binary)?;
    let mut daemon = start(&sandbox, &binary)?;
    println!("archive: {}", archive.display());
    println!(
        "home: {} (the daemon finds sessions written here)",
        sandbox.home().display()
    );
    println!("cli: {}", cli.display());
    std::io::stdout().flush().map_err(|e| e.to_string())?;
    if !options.watch {
        return stopped(daemon.wait()?);
    }
    // Paths compare as the walk from the canonical checkout names them.
    let excluded = [
        std::fs::canonicalize(sandbox.root()).map_err(|e| e.to_string())?,
        checkout.join("extension"),
    ];
    let mut observed = source_hash_excluding(&checkout, &excluded)?;
    loop {
        if let Some(status) = daemon.poll()? {
            return stopped(status);
        }
        let current = source_hash_excluding(&checkout, &excluded)?;
        if current != observed {
            observed = current;
            match build(&checkout) {
                Ok(binary) => {
                    daemon.stop()?;
                    write_cli(&cli, &sandbox, &binary)?;
                    daemon = start(&sandbox, &binary)?;
                }
                Err(error) => {
                    eprintln!("the rebuild failed; the daemon keeps its build: {error}");
                }
            }
        }
        std::thread::sleep(WATCH_INTERVAL);
    }
}

/// Builds the checkout's `polylogue` and returns the executable Cargo wrote.
fn build(checkout: &Path) -> Result<PathBuf, String> {
    let output = devkit::cargo_command()
        .current_dir(checkout)
        .args([
            "build",
            "--locked",
            "-p",
            "polylogue",
            "--bin",
            "polylogue",
            "--message-format",
            "json-render-diagnostics",
        ])
        .stderr(Stdio::inherit())
        .output()
        .map_err(|e| format!("cargo build: {e}"))?;
    if !output.status.success() {
        return Err(format!("cargo build: {}", output.status));
    }
    output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter_map(|line| serde_json::from_slice::<serde_json::Value>(line).ok())
        .filter(|message| {
            message["reason"] == "compiler-artifact" && message["target"]["name"] == "polylogue"
        })
        .find_map(|message| message["executable"].as_str().map(PathBuf::from))
        .ok_or_else(|| "cargo build named no polylogue executable".to_owned())
}

/// Starts the daemon in the sandbox and returns once it serves its socket
/// and has started or declined its web reader, which it reports next.
fn start(sandbox: &Sandbox, binary: &Path) -> Result<Process, String> {
    let (reader, writer) = std::io::pipe().map_err(|e| e.to_string())?;
    let mut command = Command::new(binary);
    sandbox
        .apply(&mut command)
        .current_dir(sandbox.root())
        .arg("daemon")
        .stdin(Stdio::null())
        .stderr(writer);
    polylogue_testkit::die_with_parent(&mut command);
    let mut daemon = Process::start(&mut command)?;
    // The daemon holds the only write end; its exit ends the lines below.
    drop(command);
    let mut lines = BufReader::new(reader).lines();
    let mut serving = false;
    for line in lines.by_ref() {
        let line = line.map_err(|e| format!("daemon output: {e}"))?;
        eprintln!("{line}");
        serving |= line.contains("polylogue: serving ");
        if serving && line.contains("polylogue: web reader ") {
            println!("daemon: {}", daemon.id());
            if line.contains("polylogue: web reader on ") {
                println!("web reader: http://{}/", sandbox.listen());
            }
            std::thread::spawn(move || {
                for line in lines.map_while(Result::ok) {
                    eprintln!("{line}");
                }
            });
            return Ok(daemon);
        }
    }
    Err(format!(
        "the daemon stopped before serving: {}",
        daemon.wait()?
    ))
}

/// The daemon ends only when stopped or failed. Stopping the command stops
/// the daemon with its signal, and with SIGKILL when it outlasts its grace.
fn stopped(status: ExitStatus) -> Result<(), String> {
    use std::os::unix::process::ExitStatusExt;
    println!("daemon stopped: {status}");
    let stop = [libc::SIGINT, libc::SIGTERM, libc::SIGKILL];
    if status.success() || status.signal().is_some_and(|signal| stop.contains(&signal)) {
        Ok(())
    } else {
        Err(format!("the daemon failed: {status}"))
    }
}

/// A script that runs `binary` as a CLI of this sandbox's daemon, with the
/// sandbox's environment and nothing of its caller's.
fn write_cli(path: &Path, sandbox: &Sandbox, binary: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let mut script = b"#!/bin/sh\nexec env -i".to_vec();
    for (name, value) in sandbox.environment() {
        script.push(b' ');
        script.extend(quoted(&[name.as_bytes(), b"=", value.as_bytes()].concat()));
    }
    script.push(b' ');
    script.extend(quoted(binary.as_os_str().as_bytes()));
    script.extend_from_slice(b" \"$@\"\n");
    std::fs::write(path, script)
        .and_then(|()| std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)))
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// `text` as one single-quoted shell word.
fn quoted(text: &[u8]) -> Vec<u8> {
    let mut word = vec![b'\''];
    for &byte in text {
        if byte == b'\'' {
            word.extend_from_slice(b"'\\''");
        } else {
            word.push(byte);
        }
    }
    word.push(b'\'');
    word
}
