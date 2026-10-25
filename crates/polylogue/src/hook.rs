//! The hook command a harness calls on its events (`polylogue hook
//! <Event>`, the event's JSON on stdin).
//!
//! Every event is appended as one line to the hook spool file of the calling
//! harness process, with `O_APPEND` and no fsync; the daemon
//! reads those files like any other growing file. Some events also ask
//! the daemon for context:
//!
//! - `SessionStart`: the session-start brief (injected by default; the
//!   `briefing.session_start` setting turns it off). After compaction it
//!   carries every prompt the operator typed in the session.
//! - `UserPromptSubmit`: a session ID pasted into the prompt brings that
//!   session's takeover context.
//!
//! Other events are only recorded. Hooks fail open: when the daemon is
//! down or slow, the harness gets no context and carries on.

use crate::briefing::{Ask, StartSource};
use crate::daemon::{Request, Response};
use polylogue_model::DeliveryOutcome;
use polylogue_model::SessionId;
use polylogue_sources::origin::by_name;
use polylogue_sources::{ORIGINS, Support};
use polylogue_store::Archive;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

/// How long a hook that brings context waits for the daemon.
const CONTEXT_BUDGET: Duration = Duration::from_secs(3);

/// The events Polylogue registers for in Claude Code's settings.
pub const EVENTS: &[&str] = &[
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "Notification",
    "Stop",
    "SubagentStop",
    "PreCompact",
    "SessionEnd",
];

/// Where hook events go: the hook spool the daemon's loop reads.
pub fn hook_spool_dir() -> PathBuf {
    polylogue_engine::Config::from_env().spool_dir(polylogue_sources::hook_spool::ORIGIN)
}

/// Appends one event line to the spool file of the harness process that
/// called the hook (this process's parent).
pub fn append_to_spool(dir: &Path, event: &str, payload: &Value) -> std::io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::create_dir_all(dir)?;
    let harness = std::os::unix::process::parent_id();
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let mut line = serde_json::to_vec(&json!({
        "at_ns": ns.to_string(),
        "event": event,
        "harness_pid": harness,
        "payload": payload,
    }))?;
    line.push(b'\n');
    // One write call per line: with O_APPEND, lines from concurrent hooks
    // of the same harness never interleave inside a line.
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open(dir.join(format!("{harness}.ndjson")))?;
    f.write_all(&line)
}

/// Sends one request with a time budget for the whole exchange.
fn ask_daemon(request: &Request, budget: Duration) -> Option<Response> {
    let socket = Archive::at(&Archive::default_root()).socket();
    let mut stream = UnixStream::connect(socket).ok()?;
    stream.set_read_timeout(Some(budget)).ok()?;
    stream.set_write_timeout(Some(budget)).ok()?;
    let mut line = serde_json::to_string(request).ok()?;
    line.push('\n');
    stream.write_all(line.as_bytes()).ok()?;
    let mut reply = String::new();
    BufReader::new(stream).read_line(&mut reply).ok()?;
    serde_json::from_str(&reply).ok()
}

fn session_of(payload: &Value) -> Option<SessionId> {
    let id = payload["session_id"].as_str()?;
    SessionId::parse(&format!("claude-code:{id}")).ok()
}

/// Session IDs pasted into a prompt: `origin:native` forms Polylogue
/// prints, and bare UUIDs, which name a session of each origin that takes
/// them.
pub fn pasted_session_ids(prompt: &str) -> Vec<SessionId> {
    let mut out: Vec<SessionId> = Vec::new();
    let mut push = |id: SessionId| {
        if !out.contains(&id) {
            out.push(id);
        }
    };
    for word in prompt.split(|c: char| c.is_whitespace() || "`'\"()[]<>,;".contains(c)) {
        let word = word.trim_end_matches(['.', ':']);
        if let Some((origin, native)) = word.split_once(':')
            && by_name(origin).is_some_and(|o| o.support == Support::Parsed)
            && let Ok(id) = SessionId::parse(word)
            && !native.is_empty()
        {
            push(id);
        } else if is_uuid(word) {
            for origin in ORIGINS.iter().filter(|o| o.bare_uuid_ids) {
                if let Ok(id) = SessionId::parse(&format!("{}:{word}", origin.name)) {
                    push(id);
                }
            }
        }
    }
    out
}

fn is_uuid(w: &str) -> bool {
    let b = w.as_bytes();
    b.len() == 36
        && b.iter().enumerate().all(|(i, c)| match i {
            8 | 13 | 18 | 23 => *c == b'-',
            _ => c.is_ascii_hexdigit(),
        })
}

/// Asks for context and returns its text, recording it as delivered once
/// it is handed to the harness by the caller.
fn context(ask: Ask) -> Option<(String, String)> {
    let request_id = polylogue_store::user_writes::new_request_id();
    let reply = ask_daemon(
        &Request::Context {
            request_id,
            ask,
            sender: Some(polylogue_model::DeliverySender::Hook.as_str().into()),
        },
        CONTEXT_BUDGET,
    )?;
    let Response::Ok { data } = reply else {
        return None;
    };
    if data["prepared"]["empty"].as_bool() != Some(false) {
        return None;
    }
    let text = data["prepared"]["text"].as_str()?.to_owned();
    let delivery = data["delivery"].as_str()?.to_owned();
    Some((delivery, text))
}

fn delivered(delivery: &str) {
    let _ = ask_daemon(
        &Request::DeliveryEvent {
            delivery: delivery.to_owned(),
            what: DeliveryOutcome::Delivered,
            detail: Some("written to the harness hook output".into()),
        },
        CONTEXT_BUDGET,
    );
}

/// Claude Code's hook output carrying context for the model.
fn emit(event: &str, text: &str) -> bool {
    let out = json!({
        "hookSpecificOutput": {
            "hookEventName": event,
            "additionalContext": text,
        }
    });
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "{out}").is_ok() && stdout.flush().is_ok()
}

/// What the hook does for one event, given its payload. Returns context to
/// emit, if any.
pub fn respond(event: &str, payload: &Value) -> Option<(String, String)> {
    match event {
        "SessionStart" => {
            let cwd = payload["cwd"].as_str().map(PathBuf::from)?;
            context(Ask::SessionStart {
                cwd,
                session: session_of(payload),
                source: StartSource::parse(payload["source"].as_str().unwrap_or("startup")),
            })
        }
        "UserPromptSubmit" => {
            let prompt = payload["prompt"].as_str()?;
            let me = session_of(payload);
            let mut texts = Vec::new();
            let mut deliveries = Vec::new();
            for id in pasted_session_ids(prompt) {
                if Some(&id) == me.as_ref() {
                    continue;
                }
                if let Some((d, t)) = context(Ask::Takeover {
                    session: id,
                    for_session: me.clone(),
                }) {
                    deliveries.push(d);
                    texts.push(t);
                }
            }
            if texts.is_empty() {
                return None;
            }
            Some((deliveries.join(","), texts.join("\n\n")))
        }
        _ => None,
    }
}

pub fn run(event: &str) -> ExitCode {
    let mut input = String::new();
    let _ = std::io::stdin().read_to_string(&mut input);
    let payload: Value = serde_json::from_str(&input).unwrap_or(Value::Null);
    if let Err(e) = append_to_spool(&hook_spool_dir(), event, &payload) {
        // The harness must not fail because the spool cannot be written.
        eprintln!("polylogue hook: could not record {event}: {e}");
    }
    if let Some((deliveries, text)) = respond(event, &payload)
        && emit(event, &text)
    {
        for d in deliveries.split(',') {
            delivered(d);
        }
    }
    ExitCode::SUCCESS
}

mod settings;
pub use settings::{install, settings_path};

#[cfg(test)]
mod tests;
