mod import_cli;
mod source_wait;
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
        writeln!(out,"Polylogue foundation\nUsage: polylogue [--archive DIR] <daemon|status|recover --to NEW_DEST|import PATH --format NAME [--context FILE] [--wait [--wait-timeout SECONDS]]|progress INPUT|retry INPUT|message:REF read --format terminal|SID read [--format json|terminal]|SID export --format json --to DIR|[QUERY] [list] [--json]|config schema>\nRecovery copies a supported reused WAL tail from a stopped archive and returns degraded exit 5 with persistent historical uncertainty. Import retains one regular file. Selected Message terminal reads remain degraded. Exact native Session JSON/terminal reads can prove the supported provider-title shape; other shapes retain diagnostics. Session listing and prose words/phrases/prefixes share a degraded query capability (origin, kind, id, sort by last_active desc, limit). Exact native Session JSON directory export shares the pinned read; general exports and MCP remain unavailable.").map_err(io_error)?;
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
    if args.first().map(String::as_str) == Some("recover") {
        if args.len() != 3 || args[1] != "--to" || args[2].is_empty() {
            return Err(error(
                ErrorKind::InvalidQuery,
                "recover requires exactly --to DEST",
            ));
        }
        let stop = Arc::new(AtomicBool::new(false));
        let signalled = Arc::clone(&stop);
        ctrlc::set_handler(move || signalled.store(true, Ordering::Release)).map_err(io_error)?;
        let report = polylogue::Running::recover_copy(&archive, &PathBuf::from(&args[2]), &stop)
            .map_err(io_error)?;
        serde_json::to_writer(&mut out,&json!({"outcome":"degraded","destination":report.destination,"recovery":report.recovery,"receipt":report.receipt,"notes":[polylogue_model::RecoveryQualification::NOTE],"missing":[polylogue_model::RecoveryQualification::NOTE]})).map_err(io_error)?;
        writeln!(out).map_err(io_error)?;
        return Ok(5);
    }
    let terminal_read = args.first().is_some_and(|s| s.starts_with("message:"))
        && (args.len() == 2 && args[1] == "read"
            || args.len() == 4 && args[1..] == ["read", "--format", "terminal"]);
    let session_terminal = args.first().is_some_and(|s| !s.starts_with("message:"))
        && (args.len() == 2 && args[1] == "read"
            || args.len() == 4 && args[1..] == ["read", "--format", "terminal"]);
    let session_json = args.len() == 4
        && args[1..] == ["read", "--format", "json"]
        && !args[0].starts_with("message:");
    let session_export = args.get(1).is_some_and(|arg| arg == "export");
    let export_to = if session_export {
        let mut destination = None;
        let mut format = false;
        let mut i = 2;
        while i < args.len() {
            match args[i].as_str() {
                "--format" if !format && args.get(i + 1).is_some_and(|s| s == "json") => {
                    format = true
                }
                "--to"
                    if destination.is_none()
                        && args
                            .get(i + 1)
                            .is_some_and(|s| !s.is_empty() && !s.starts_with("--")) =>
                {
                    destination = Some(PathBuf::from(&args[i + 1]))
                }
                _ => {
                    return Err(error(
                        ErrorKind::InvalidQuery,
                        "export supports one exact native Session with --format json --to DIR only",
                    ));
                }
            }
            i += 2;
        }
        Some(destination.filter(|_| format).ok_or_else(|| {
            error(
                ErrorKind::InvalidQuery,
                "export requires --format json --to DIR",
            )
        })?)
    } else {
        None
    };
    let command = if session_export || session_json || session_terminal {
        Some(json!({"command":"session-read","reference":args[0],"format":"json"}))
    } else if terminal_read {
        Some(json!({"command":"read","reference":args[0],"format":"terminal"}))
    } else if args.first().map(String::as_str) == Some("import") {
        Some(import_cli::request(&args)?)
    } else if args.first().map(String::as_str) == Some("progress") && args.len() == 2 {
        Some(json!({"command":"progress","input":args[1]}))
    } else if args.first().map(String::as_str) == Some("retry") && args.len() == 2 {
        Some(json!({"command":"retry","input":args[1]}))
    } else {
        None
    };
    let mut query_args = args.clone();
    let json_output = query_args.iter().any(|a| a == "--json");
    query_args.retain(|a| a != "--json");
    if query_args.first().is_none_or(|a| a != "find")
        && query_args.last().is_some_and(|a| a == "list")
    {
        query_args.pop();
    }
    let query = if args != ["daemon"] && args != ["status"] && command.is_none() {
        let text = query_args.join(" ");
        polylogue_query::bind(&text)?;
        Some(text)
    } else {
        None
    };
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
            "listening: {} (immutable source imports; canonical finalization incomplete)",
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
    if let Some(destination) = export_to {
        let stop = Arc::new(AtomicBool::new(false));
        let signalled = Arc::clone(&stop);
        ctrlc::set_handler(move || signalled.store(true, Ordering::Release)).map_err(io_error)?;
        let reply = polylogue::session_export::request(&socket, &args[0], &destination, &stop)
            .map_err(io_error)?;
        if reply["outcome"] == "error" {
            return Err(
                serde_json::from_value::<Failure>(reply["error"].clone()).map_err(io_error)?
            );
        }
        if reply["outcome"] == "degraded" {
            eprintln!(
                "Export coverage is incomplete; detached documents and manifest retain the result diagnostics."
            );
        }
        return reply_exit_code(&reply);
    }
    if let Some(query) = query {
        let value = polylogue::session_query::request(&socket, &query).map_err(|e| {
            error(
                match e.kind() {
                    io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused => {
                        ErrorKind::NotRunning
                    }
                    io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock => ErrorKind::Timeout,
                    _ => ErrorKind::Io,
                },
                format!("archive {}: {e}", archive.display()),
            )
        })?;
        if json_output || value["outcome"] == "error" {
            serde_json::to_writer(&mut out, &value).map_err(io_error)?;
            writeln!(out).map_err(io_error)?;
        } else {
            for row in value["rows"]
                .as_array()
                .ok_or_else(|| io_error("query rows missing"))?
            {
                writeln!(
                    out,
                    "{}  {}",
                    terminal_text(row["id"].as_str().unwrap_or("")),
                    terminal_text(row["title"].as_str().unwrap_or(""))
                )
                .map_err(io_error)?;
                if let Some(hits) = row["hits"].as_array() {
                    for hit in hits {
                        writeln!(
                            out,
                            "  message:{}",
                            terminal_text(hit["message"].as_str().unwrap_or(""))
                        )
                        .map_err(io_error)?;
                    }
                }
            }
            for note in value["notes"].as_array().into_iter().flatten() {
                eprintln!("{}", terminal_text(note.as_str().unwrap_or("")));
            }
        }
        return reply_exit_code(&value);
    }
    if terminal_read || session_json || session_terminal {
        let reply = if session_json {
            polylogue::session_read::request(&socket, &args[0], &mut out)
        } else if session_terminal {
            polylogue::session_read::request_terminal(&socket, &args[0], &mut out)
        } else {
            polylogue::reference_read::request(&socket, &args[0], &mut out)
        }
        .map_err(io_error)?;
        if reply["outcome"] == "error" {
            return Err(
                serde_json::from_value::<Failure>(reply["error"].clone()).map_err(io_error)?
            );
        }
        if reply["outcome"] == "degraded" {
            if session_json || session_terminal {
                eprintln!(
                    "Read coverage is incomplete; result diagnostics name the unsupported or unfinished dependencies."
                );
            } else {
                eprintln!(
                    "Read coverage is incomplete: canonical history and source provenance remain pending."
                );
            }
        }
        return reply_exit_code(&reply);
    }
    if let Some(command) = command.as_ref().filter(|v| v["command"] == "import") {
        eprintln!(
            "import request {} --seen-at {}",
            command["id"].as_str().unwrap_or(""),
            command["seen_at"]
        );
    }
    let mut command = command;
    let (wait, wait_seconds) =
        if let Some(request) = command.as_mut().filter(|v| v["command"] == "import") {
            let object = request
                .as_object_mut()
                .ok_or_else(|| error(ErrorKind::InvalidQuery, "invalid import request"))?;
            (
                object
                    .remove("wait")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                object
                    .remove("wait_seconds")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(30),
            )
        } else {
            (false, 30)
        };
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
    if wait && value["outcome"] == "ok" {
        value = source_wait::wait(&socket, value, std::time::Duration::from_secs(wait_seconds));
    }
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

fn terminal_text(text: &str) -> String {
    text.chars()
        .flat_map(|c| {
            if c.is_control() {
                c.escape_default().collect::<Vec<_>>()
            } else {
                vec![c]
            }
        })
        .collect()
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
