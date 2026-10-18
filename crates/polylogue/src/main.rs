mod authored_cli;
mod host_identity;
mod import_cli;
mod mcp;
mod observed_cli;
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
fn model_read_io_error(error: io::Error) -> Failure {
    let kind = match error.kind() {
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock => ErrorKind::Timeout,
        _ => ErrorKind::Io,
    };
    let mut failure = Failure::new(kind, format!("read: {error}"));
    if failure.kind == ErrorKind::Timeout {
        let suggestion = "Read a single message or block, or use a smaller budget.";
        failure.message.push_str(". ");
        failure.message.push_str(suggestion);
        failure.suggestion = Some(suggestion.into());
    }
    failure
}
const HELP: &str = "\
Polylogue: a local archive of AI coding sessions.

Usage: polylogue [--archive DIR] COMMAND

Archive and daemon:
  daemon                              Run the archive daemon (Ctrl-C stops it)
  daemon --config FILE [--until-idle] [--json]
                                      Also collect from one configured source
  import PATH --format FORMAT [--context FILE] [--wait [--wait-timeout SECONDS]]
                                      Store one file; FORMAT is claude-code-jsonl,
                                      codex-rollout or codex-history-sqlite
  progress INPUT                      Show parsing progress of a stored file
  retry INPUT                         Retry a failed parse
  input cat INPUT                     Print a stored file, credentials masked
  status                              Show daemon and source status
  mcp                                 Serve MCP (status, search, read, note) on stdio
  config schema                       Print the JSON schema of the --config file

Finding and reading:
  [QUERY] [list] [--json]             List sessions, e.g. 'origin:claude-code | limit 20'
  find QUERY [--json]                 Search message text (words, \"phrases\", prefix*)
  SESSION read [--format json|terminal]
                                      Read a whole session
  message:ID read                     Read one message in the terminal
  REF read --budget TOKENS [--continuation TOKEN] [--view VIEW]
                                      Read in pages sized for a token budget;
                                      VIEW is dialogue, transcript (with --tools full
                                      --thinking full --harness full [--branches]),
                                      annotations, or for notes history, withdrawals,
                                      current, value
  SESSION export --format json --to DIR
                                      Export one session into DIR

Notes:
  REF note TEXT [--kind note|correction] [--corrects REF] [--targets FILE]
  note:ID revise TEXT [--targets FILE]
  note:ID withdraw --reason TEXT
                                      All note commands accept --request-id UUIDv7;
                                      reuse it to retry after a lost reply.
                                      --targets FILE holds a JSON array of
                                      {ref, seen_hash?}; seen_hash is the
                                      target_hash.seen_hash from a read

Recovery (archive must be stopped):
  recover --to NEW_DIR                Copy an archive whose last WAL segment is damaged
  daemon --prepare-copy restore --request-id UUIDv7 --retired-writers
                                      Prepare a stopped local copy to take over

The archive defaults to $XDG_DATA_HOME/polylogue or ~/.local/share/polylogue.
Commands talk to the daemon over a socket under $XDG_RUNTIME_DIR.
Exit codes: 0 ok, 1 empty, 2 invalid request, 3 refused, 4 not found or I/O
error, 5 partial result, 6 daemon not running, 7 timed out.
";

fn schema() -> serde_json::Value {
    json!({"$schema":"https://json-schema.org/draft/2020-12/schema","title":"Polylogue configuration","type":"object","properties":{"archive":{"type":"string","description":"Archive directory, as --archive; default $XDG_DATA_HOME/polylogue"}},"additionalProperties":false,"$defs":{"observedStartup":{"type":"object","oneOf":[{"required":["claude_main"],"properties":{"claude_main":{"type":"object"}}},{"required":["claude_projects"],"properties":{"claude_projects":{"type":"object"}}},{"required":["codex_home"],"properties":{"codex_home":{"type":"object"}}}],"additionalProperties":false,"properties":{"codex_home":{"type":["object","null"],"required":["home","machine","host"],"additionalProperties":false,"properties":{"home":{"type":"string","description":"Absolute path of the Codex home"},"machine":{"type":"string","minLength":1},"host":{"type":"string","minLength":1},"rollouts":{"type":"string","default":"sessions"},"catalog":{"type":"string","default":"state_5.sqlite"},"history":{"type":"string","default":"history.jsonl"},"paginated_history":{"type":"string","description":"Paginated history database inside the Codex home; collected only when set"},"paginated_interval_seconds":{"type":"integer","minimum":1,"maximum":86400,"default":300}}},"claude_main":{"type":["object","null"],"required":["path","machine","host"],"additionalProperties":false,"properties":{"path":{"type":"string","description":"Absolute path of one Claude Code session JSONL file","pattern":"^/"},"machine":{"type":"string","minLength":1},"host":{"type":"string","minLength":1},"acquisition_quantum_bytes":{"type":"integer","minimum":65536,"maximum":8388608,"default":8388608}}},"claude_projects":{"type":["object","null"],"required":["root","machine","host"],"additionalProperties":false,"properties":{"root":{"type":"string","pattern":"^/","description":"Absolute path of a Claude Code projects directory"},"machine":{"type":"string","minLength":1},"host":{"type":"string","minLength":1},"acquisition_quantum_bytes":{"type":"integer","minimum":65536,"maximum":8388608,"default":8388608}}}}}}})
}

fn stop_on_ctrl_c() -> Result<Arc<AtomicBool>, Failure> {
    let stop = Arc::new(AtomicBool::new(false));
    let signalled = Arc::clone(&stop);
    ctrlc::set_handler(move || signalled.store(true, Ordering::Release)).map_err(io_error)?;
    Ok(stop)
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
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        out.write_all(HELP.as_bytes()).map_err(io_error)?;
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
    if args.first().map(String::as_str) == Some("mcp") {
        if args != ["mcp"] {
            return Err(error(ErrorKind::InvalidQuery, "mcp accepts no arguments"));
        }
        drop(out);
        mcp::run(archive).map_err(io_error)?;
        return Ok(0);
    }
    if args.first().map(String::as_str) == Some("recover") {
        if args.len() != 3 || args[1] != "--to" || args[2].is_empty() {
            return Err(error(
                ErrorKind::InvalidQuery,
                "usage: recover --to NEW_DIR",
            ));
        }
        let stop = stop_on_ctrl_c()?;
        let report = polylogue::Running::recover_copy(&archive, &PathBuf::from(&args[2]), &stop)
            .map_err(io_error)?;
        serde_json::to_writer(&mut out,&json!({"outcome":"degraded","destination":report.destination,"recovery":report.recovery,"receipt":report.receipt,"notes":[polylogue_model::RecoveryRecord::NOTE],"missing":[polylogue_model::RecoveryRecord::NOTE]})).map_err(io_error)?;
        writeln!(out).map_err(io_error)?;
        return Ok(5);
    }
    if args.first().is_some_and(|s| s == "daemon") && args.iter().any(|s| s == "--prepare-copy") {
        let mut mode = None;
        let mut request = None;
        let mut retired = false;
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--prepare-copy" if mode.is_none() && args.get(i + 1).is_some() => {
                    mode = Some(args[i + 1].as_str());
                    i += 2;
                }
                "--request-id" if request.is_none() && args.get(i + 1).is_some() => {
                    request = Some(args[i + 1].as_str());
                    i += 2;
                }
                "--retired-writers" if !retired => {
                    retired = true;
                    i += 1;
                }
                _ => {
                    return Err(error(
                        ErrorKind::InvalidQuery,
                        "usage: daemon --prepare-copy restore --request-id UUIDv7 --retired-writers",
                    ));
                }
            }
        }
        let stop = stop_on_ctrl_c()?;
        let preparation = polylogue_store::copy_preparation::Preparation::begin(
            &archive,
            mode.unwrap_or(""),
            "local",
            request.unwrap_or(""),
            retired,
            &stop,
        )
        .map_err(io_error)?;
        serde_json::to_writer(&mut out, &preparation.progress()).map_err(io_error)?;
        writeln!(out).map_err(io_error)?;
        out.flush().map_err(io_error)?;
        let report = preparation.finish(&stop).map_err(io_error)?;
        serde_json::to_writer(&mut out, &report).map_err(io_error)?;
        writeln!(out).map_err(io_error)?;
        return Ok(5);
    }
    let authored = authored_cli::request(&args)?;
    let model_read = polylogue::model_read::cli(&args)
        .map_err(|v| error(ErrorKind::InvalidQuery, v.to_string()))?
        .or_else(|| {
            authored
                .as_ref()
                .filter(|v| v["command"] == "model-read")
                .cloned()
        });
    if args.first().is_some_and(|s| s == "daemon") && args.len() > 1 {
        return observed_cli::run(&args, &archive, &mut out);
    }
    let input_cat = args.len() == 3 && args[0] == "input" && args[1] == "cat";
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
                        "usage: SESSION export --format json --to DIR",
                    ));
                }
            }
            i += 2;
        }
        Some(destination.filter(|_| format).ok_or_else(|| {
            error(
                ErrorKind::InvalidQuery,
                "usage: SESSION export --format json --to DIR",
            )
        })?)
    } else {
        None
    };
    let command = if let Some(request) = &authored {
        Some(request.clone())
    } else if input_cat {
        Some(json!({"command":"input-cat","input":args[2]}))
    } else if let Some(request) = &model_read {
        Some(request.clone())
    } else if session_export || session_json || session_terminal {
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
        if polylogue_query::bind_tools(&text)?.is_none() {
            polylogue_query::bind(&text)?;
        }
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
        let stop = stop_on_ctrl_c()?;
        let mut running = polylogue::Running::start_cancellable(&archive, &runtime, &stop)
            .map_err(|e| error(ErrorKind::Io, format!("archive {}: {e}", archive.display())))?;
        eprintln!("listening: {}", running.socket().display());
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
    if input_cat {
        polylogue::input_cat::request(&socket, &args[2], &mut out)?;
        return Ok(0);
    }
    if let Some(destination) = export_to {
        // Preserve default SIGINT termination while IPC is blocked. Process
        // exit closes the read socket; unfinished staging may remain private.
        let stop = AtomicBool::new(false);
        let reply = polylogue::session_export::request(&socket, &args[0], &destination, &stop)
            .map_err(io_error)?;
        if reply["outcome"] == "error" {
            return Err(
                serde_json::from_value::<Failure>(reply["error"].clone()).map_err(io_error)?
            );
        }
        if reply["outcome"] == "degraded" {
            eprintln!("Export is incomplete; manifest.json names what is missing.");
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
    if let Some(request) = &model_read {
        let value =
            polylogue::model_read::request(&socket, request).map_err(model_read_io_error)?;
        serde_json::to_writer(&mut out, &value).map_err(io_error)?;
        writeln!(out).map_err(io_error)?;
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
                eprintln!("Read is incomplete; the result names what is missing.");
            } else {
                eprintln!(
                    "Read is incomplete: message history and source locations are not shown yet."
                );
            }
        }
        return reply_exit_code(&reply);
    }
    if let Some(command) = command.as_ref().filter(|v| v["command"] == "import") {
        eprintln!(
            "to retry: --request-id {} --seen-at {}",
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
    let response = match &command {
        Some(command) => polylogue::request_value(&socket, command),
        None => polylogue::request(&socket),
    };
    let mut value = match response {
        Ok(value) => value,
        Err(e) if command.as_ref().is_some_and(|v| v["command"] == "note") => {
            authored_cli::uncertain_reply(command.as_ref().unwrap(), e)
        }
        Err(e) => {
            return Err(error(
                if matches!(
                    e.kind(),
                    io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
                ) {
                    ErrorKind::NotRunning
                } else {
                    ErrorKind::Io
                },
                format!("archive {}: {e}", archive.display()),
            ));
        }
    };
    if let Some(request) = command.as_ref().filter(|v| v["command"] == "note") {
        value = polylogue::authored::bind_write_reply(request, value);
    }
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
    fn model_transport_deadlines_preserve_timeout_exit_seven() {
        for kind in [io::ErrorKind::TimedOut, io::ErrorKind::WouldBlock] {
            let failure = model_read_io_error(io::Error::new(kind, "synthetic deadline"));
            assert_eq!(failure.kind, ErrorKind::Timeout);
            assert_eq!(failure.outcome().exit_code(), 7);
            assert!(failure.suggestion.is_some());
            assert!(!failure.retryable);
            assert!(failure.message.contains("smaller budget"));
        }
        assert_eq!(
            model_read_io_error(io::Error::new(
                io::ErrorKind::InvalidData,
                "synthetic frame"
            ))
            .kind,
            ErrorKind::Io
        );
    }
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
