mod import_cli;
use polylogue_model::types::{ErrorKind, Failure, ResultOutcome};
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
        writeln!(out,"Polylogue foundation\nUsage: polylogue [--archive DIR] <daemon|status|import PATH --format codex-rollout|progress INPUT|retry INPUT|config schema>\nImport retains one regular file. Canonical queries, search, export and MCP remain unavailable.").map_err(io_error)?;
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
    let command = if args.first().map(String::as_str) == Some("import") {
        Some(import_cli::request(&args)?)
    } else if args.first().map(String::as_str) == Some("progress") && args.len() == 2 {
        Some(json!({"command":"progress","input":args[1]}))
    } else if args.first().map(String::as_str) == Some("retry") && args.len() == 2 {
        Some(json!({"command":"retry","input":args[1]}))
    } else {
        None
    };
    if args != ["daemon"] && args != ["status"] && !args.is_empty() && command.is_none() {
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
        let stop = Arc::new(AtomicBool::new(false));
        let signalled = Arc::clone(&stop);
        ctrlc::set_handler(move || signalled.store(true, Ordering::Release)).map_err(io_error)?;
        let mut running = polylogue::Running::start_cancellable(&archive, &runtime, &stop)
            .map_err(|e| error(ErrorKind::Io, format!("archive {}: {e}", archive.display())))?;
        eprintln!(
            "listening: {} (immutable Codex import; canonical finalization incomplete)",
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
    if let Some(command) = command.as_ref().filter(|v| v["command"] == "import") {
        eprintln!(
            "import request {} --seen-at {}",
            command["id"].as_str().unwrap_or(""),
            command["seen_at"]
        );
    }
    let mut value = match &command {
        Some(command) => polylogue::request_value(&socket, command),
        None => polylogue::request(&socket),
    }
    .map_err(|e| {
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
    if let Some(command) = command.filter(|v| v["command"] == "import") {
        value["request_id"] = command["id"].clone();
        value["seen_at"] = command["seen_at"].clone();
    }
    serde_json::to_writer(&mut out, &value).map_err(io_error)?;
    writeln!(out).map_err(io_error)?;
    reply_exit_code(&value)
}
// The transport envelope carries the full Failure beneath `error`.
// Decode it before choosing an exit code so remote and local failures agree.
fn reply_exit_code(value: &serde_json::Value) -> Result<i32, Failure> {
    if value["outcome"] == "error" {
        let failure: Failure = serde_json::from_value(value["error"].clone()).map_err(io_error)?;
        Ok(failure.outcome().exit_code())
    } else {
        let outcome: ResultOutcome = serde_json::from_value(value.clone()).map_err(io_error)?;
        Ok(outcome.exit_code())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_outcomes_preserve_typed_exit_codes() {
        for kind in [
            ErrorKind::InvalidQuery,
            ErrorKind::NotYetAvailable,
            ErrorKind::NotFound,
            ErrorKind::Ambiguous,
            ErrorKind::Conflict,
            ErrorKind::Refused {
                reason: "unsupported".into(),
            },
            ErrorKind::Timeout,
            ErrorKind::NotRunning,
            ErrorKind::Io,
        ] {
            let expected = kind.exit_code();
            let reply = json!({"outcome":"error","error":Failure::new(kind,"failed")});
            assert_eq!(reply_exit_code(&reply).unwrap(), expected);
        }
        for (reply, code) in [
            (json!({"outcome":"ok"}), 0),
            (json!({"outcome":"empty"}), 1),
            (json!({"outcome":"degraded","missing":["input"]}), 5),
        ] {
            assert_eq!(reply_exit_code(&reply).unwrap(), code);
        }
        for reply in [
            json!({}),
            json!({"outcome":"error"}),
            json!({"outcome":"unknown"}),
        ] {
            assert_eq!(reply_exit_code(&reply).unwrap_err().kind, ErrorKind::Io);
        }
    }
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
