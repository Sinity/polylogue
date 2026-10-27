//! The hook command a harness calls on its events (`polylogue hook
//! <Event> [--harness NAME]`, the event's JSON on stdin), for Claude Code
//! (the default), Codex and Hermes.
//!
//! Every event is appended as one line to the hook spool file of the calling
//! harness process, with `O_APPEND` and no fsync, naming the harness's
//! origin and session; the daemon reads those files like any other growing
//! file. Some events also ask the daemon for context:
//!
//! - the session's start (`SessionStart`; Hermes: `pre_llm_call` on its
//!   first turn): the session-start brief, injected by default; the
//!   `briefing.session_start` setting turns it off. After compaction it
//!   carries every prompt the operator typed in the session.
//! - a prompt (`UserPromptSubmit`; Hermes: `pre_llm_call`): a session ID
//!   pasted into it brings that session's takeover context.
//!
//! Claude Code and Codex take context as `hookSpecificOutput.additionalContext`,
//! Hermes as `{"context": ...}`. Other events are only recorded. Hooks fail
//! open: when the daemon is down or slow, the harness gets no context and
//! carries on.

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

/// How long a hook that brings context waits for the daemon: a brief
/// during a first build over a large archive takes a few seconds. The
/// harnesses' hook timeouts are longer.
const CONTEXT_BUDGET: Duration = Duration::from_secs(8);

/// A harness that calls the hook command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Harness {
    ClaudeCode,
    Codex,
    Hermes,
}

impl Harness {
    pub const NAMES: &[&str] = &["claude-code", "codex", "hermes"];

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "claude-code" => Some(Self::ClaudeCode),
            "codex" => Some(Self::Codex),
            "hermes" => Some(Self::Hermes),
            _ => None,
        }
    }

    /// The origin whose sessions its hook events belong to.
    pub fn origin(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude-code",
            Self::Codex => "codex",
            Self::Hermes => polylogue_sources::hermes::ORIGIN,
        }
    }
}

/// The events Polylogue registers for in Claude Code's and Codex's
/// settings.
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
/// called the hook (this process's parent), naming the session the event
/// belongs to when the payload names one.
pub fn append_to_spool(
    dir: &Path,
    harness: Harness,
    event: &str,
    payload: &Value,
) -> std::io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::create_dir_all(dir)?;
    let process = std::os::unix::process::parent_id();
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let mut payload = payload.clone();
    // Hermes sends the whole conversation with every turn; its transcript
    // is read from its own database.
    if let Some(extra) = payload.get_mut("extra").and_then(Value::as_object_mut) {
        extra.remove("conversation_history");
    }
    let mut line = json!({
        "at_ns": ns.to_string(),
        "event": event,
        "harness_pid": process,
        "provider": harness.origin(),
        "payload": payload,
    });
    if let Some(session) = session_of(harness, &line["payload"]) {
        line["session_id"] = json!(session.native().to_string());
    }
    let mut line = serde_json::to_vec(&line)?;
    line.push(b'\n');
    // One write call per line: with O_APPEND, lines from concurrent hooks
    // of the same harness never interleave inside a line.
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open(dir.join(format!("{process}.ndjson")))?;
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

/// The session a hook payload names, as Polylogue names it.
fn session_of(harness: Harness, payload: &Value) -> Option<SessionId> {
    let id = payload["session_id"].as_str().filter(|id| !id.is_empty())?;
    match harness {
        Harness::Hermes => {
            let home = std::env::var_os("HOME")?;
            let state_db = Path::new(&home).join(".hermes/state.db");
            polylogue_sources::hermes::session_at(
                &polylogue_store::user_writes::machine_id(),
                &state_db,
                id,
            )
        }
        _ => SessionId::parse(&format!("{}:{id}", harness.origin())).ok(),
    }
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
    let Response::Ok { data, .. } = reply else {
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

/// The hook output carrying context for the model.
fn emit(harness: Harness, event: &str, text: &str) -> bool {
    let out = match harness {
        Harness::Hermes => json!({ "context": text }),
        Harness::ClaudeCode | Harness::Codex => json!({
            "hookSpecificOutput": {
                "hookEventName": event,
                "additionalContext": text,
            }
        }),
    };
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "{out}").is_ok() && stdout.flush().is_ok()
}

/// The brief for a session starting in the payload's directory.
fn start(harness: Harness, payload: &Value, source: StartSource) -> Option<(String, String)> {
    let cwd = payload["cwd"].as_str().map(PathBuf::from)?;
    context(Ask::SessionStart {
        cwd,
        session: session_of(harness, payload),
        source,
    })
}

/// Takeover context for each session pasted into a prompt.
fn takeovers(harness: Harness, payload: &Value, prompt: &str) -> Option<(String, String)> {
    let me = session_of(harness, payload);
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

/// What the hook does for one event, given its payload. Returns the
/// deliveries and the context to emit, if any.
pub fn respond(harness: Harness, event: &str, payload: &Value) -> Option<(String, String)> {
    match (harness, event) {
        (Harness::ClaudeCode | Harness::Codex, "SessionStart") => start(
            harness,
            payload,
            StartSource::parse(payload["source"].as_str().unwrap_or("startup")),
        ),
        (Harness::ClaudeCode | Harness::Codex, "UserPromptSubmit") => {
            takeovers(harness, payload, payload["prompt"].as_str()?)
        }
        // Hermes injects context only into a turn's prompt: the first
        // turn's carries the brief, and any turn's the takeovers it asks for.
        (Harness::Hermes, "pre_llm_call") => {
            let extra = &payload["extra"];
            let prompt = extra["user_message"].as_str().unwrap_or("");
            let brief = (extra["is_first_turn"].as_bool() == Some(true))
                .then(|| start(harness, payload, StartSource::Startup))
                .flatten();
            let takeover = takeovers(harness, payload, prompt);
            match (brief, takeover) {
                (Some((d1, t1)), Some((d2, t2))) => {
                    Some((format!("{d1},{d2}"), format!("{t1}\n\n{t2}")))
                }
                (one, other) => one.or(other),
            }
        }
        _ => None,
    }
}

pub fn run(harness: Harness, event: &str) -> ExitCode {
    let mut input = String::new();
    let _ = std::io::stdin().read_to_string(&mut input);
    let payload: Value = serde_json::from_str(&input).unwrap_or(Value::Null);
    if let Err(e) = append_to_spool(&hook_spool_dir(), harness, event, &payload) {
        // The harness must not fail because the spool cannot be written.
        eprintln!("polylogue hook: could not record {event}: {e}");
    }
    if let Some((deliveries, text)) = respond(harness, event, &payload)
        && emit(harness, event, &text)
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
