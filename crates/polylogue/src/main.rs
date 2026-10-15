use polylogue_model::types::{ErrorKind, Failure};
use serde_json::json;
use std::{
    env, fs,
    io::{self, Write},
    path::PathBuf,
    process::ExitCode,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

fn error(kind: ErrorKind, message: impl Into<String>) -> Failure {
    Failure::new(kind, message)
}
fn io_error(error: impl ToString) -> Failure {
    Failure::new(ErrorKind::Io, error.to_string())
}
fn schema() -> serde_json::Value {
    json!({"$schema":"https://json-schema.org/draft/2020-12/schema","title":"Polylogue foundation configuration","type":"object","properties":{"archive":{"type":"string","description":"CLI --archive; default XDG_DATA_HOME/polylogue"}},"additionalProperties":false})
}

fn run() -> Result<i32, Failure> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let mut archive = None;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--archive" {
            if i + 1 == args.len() || args[i + 1].is_empty() {
                return Err(error(
                    ErrorKind::InvalidQuery,
                    "--archive needs a directory",
                ));
            }
            archive = Some(PathBuf::from(args.remove(i + 1)));
            args.remove(i);
        } else {
            i += 1;
        }
    }
    let mut out = io::stdout().lock();
    if args == ["--help"] || args == ["-h"] {
        writeln!(out,"Polylogue foundation\nUsage: polylogue [--archive DIR] <daemon|status|config schema>\nNo session sources, queries, export or MCP are registered.").map_err(io_error)?;
        return Ok(0);
    }
    if args == ["--version"] {
        writeln!(out, "polylogue {}", env!("CARGO_PKG_VERSION")).map_err(io_error)?;
        return Ok(0);
    }
    if args == ["config", "schema"] {
        serde_json::to_writer(&mut out, &schema()).map_err(io_error)?;
        writeln!(out).map_err(io_error)?;
        return Ok(0);
    }
    let archive = archive
        .or_else(|| {
            env::var_os("XDG_DATA_HOME")
                .filter(|value| !value.is_empty())
                .map(|p| PathBuf::from(p).join("polylogue"))
        })
        .or_else(|| {
            env::var_os("HOME")
                .filter(|value| !value.is_empty())
                .map(|p| PathBuf::from(p).join(".local/share/polylogue"))
        })
        .ok_or_else(|| {
            error(
                ErrorKind::InvalidQuery,
                "no archive selected; pass --archive DIR",
            )
        })?;
    let archive = if archive.is_absolute() {
        archive
    } else {
        env::current_dir().map_err(io_error)?.join(archive)
    };
    if args != ["daemon"] && args != ["status"] && !args.is_empty() {
        return Err(error(
            ErrorKind::NotYetAvailable,
            format!("operation is not registered; archive {}", archive.display()),
        ));
    }
    let runtime = env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| {
            error(
                ErrorKind::InvalidQuery,
                format!("XDG_RUNTIME_DIR is required; archive {}", archive.display()),
            )
        })?;
    if !runtime.is_absolute() {
        return Err(error(
            ErrorKind::InvalidQuery,
            format!(
                "XDG_RUNTIME_DIR must be absolute; archive {}",
                archive.display()
            ),
        ));
    }
    if args == ["daemon"] {
        let mut running = polylogue::Running::start(&archive, &runtime)
            .map_err(|e| error(ErrorKind::Io, format!("archive {}: {e}", archive.display())))?;
        let stop = Arc::new(AtomicBool::new(false));
        let signalled = Arc::clone(&stop);
        ctrlc::set_handler(move || signalled.store(true, Ordering::Release)).map_err(io_error)?;
        eprintln!(
            "foundation listening: {} (no sources registered)",
            running.socket().display()
        );
        while !stop.load(Ordering::Acquire) {
            thread::sleep(Duration::from_millis(10));
        }
        running.stop();
        return Ok(0);
    }
    let resolved = fs::canonicalize(&archive).map_err(|e| {
        error(
            if e.kind() == io::ErrorKind::NotFound {
                ErrorKind::NotRunning
            } else {
                ErrorKind::Io
            },
            format!(
                "archive {}: daemon is not running; start polylogue daemon ({e})",
                archive.display()
            ),
        )
    })?;
    let socket = polylogue::socket_path(&resolved, &runtime);
    let value = polylogue::request(&socket).map_err(|e| {
        error(
            if matches!(
                e.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
            ) {
                ErrorKind::NotRunning
            } else {
                ErrorKind::Io
            },
            format!("archive {}: {e}", archive.display()),
        )
    })?;
    serde_json::to_writer(&mut out, &value).map_err(io_error)?;
    writeln!(out).map_err(io_error)?;
    Ok(if value["outcome"] == "ok" {
        0
    } else {
        ErrorKind::Io.exit_code()
    })
}
fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code as u8),
        Err(failure) => {
            eprintln!("{}", failure.message);
            ExitCode::from(failure.kind.exit_code() as u8)
        }
    }
}
