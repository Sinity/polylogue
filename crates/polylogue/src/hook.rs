//! The hook command a harness calls on its events (`polylogue hook
//! <Event>`, the event's JSON on stdin).
//!
//! Every event is appended as one line to a carrier file for the calling
//! harness process, with `O_APPEND` and no fsync; the daemon
//! reads carriers through the ordinary append route. Some events also ask
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
use polylogue_model::SessionId;
use polylogue_sources::origin::by_name;
use polylogue_sources::{ORIGINS, Support};
use polylogue_store::{Archive, DeliveryOutcome};
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

/// Where carrier files go: the hook spool the daemon's loop reads.
pub fn carrier_dir() -> PathBuf {
    polylogue_engine::Config::from_env().spool_dir(polylogue_sources::hook_spool::ORIGIN)
}

/// Appends one event line to the carrier of the harness process that
/// called the hook (this process's parent).
pub fn append_carrier(dir: &Path, event: &str, payload: &Value) -> std::io::Result<()> {
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
            sender: Some("hook".into()),
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
    if let Err(e) = append_carrier(&carrier_dir(), event, &payload) {
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

/// The Claude Code settings file: `settings.json` in Claude Code's own
/// directory (`$CLAUDE_CONFIG_DIR`, else `~/.claude`).
pub fn settings_path() -> PathBuf {
    polylogue_engine::Config::from_env()
        .root(polylogue_sources::claude_code::ROOT)
        .join("settings.json")
}

fn ours(entry: &Value) -> bool {
    entry["hooks"].as_array().is_some_and(|hooks| {
        hooks.iter().any(|h| {
            h["command"]
                .as_str()
                .is_some_and(|c| c.starts_with("polylogue hook "))
        })
    })
}

/// Merges Polylogue's hook entries into a settings document, keeping every
/// other setting and hook. Installing twice changes nothing.
pub fn merge_hooks(mut settings: Value, install: bool) -> Value {
    if !settings.is_object() {
        settings = json!({});
    }
    let hooks = settings
        .as_object_mut()
        .expect("settings is an object")
        .entry("hooks")
        .or_insert_with(|| json!({}));
    if !hooks.is_object() {
        *hooks = json!({});
    }
    let hooks = hooks.as_object_mut().expect("hooks is an object");
    for event in EVENTS {
        let list = hooks.entry(*event).or_insert_with(|| json!([]));
        if !list.is_array() {
            *list = json!([]);
        }
        let list = list.as_array_mut().expect("event hooks are a list");
        list.retain(|e| !ours(e));
        if install {
            list.push(json!({
                "hooks": [{
                    "type": "command",
                    "command": format!("polylogue hook {event}"),
                    "timeout": if *event == "SessionStart" || *event == "UserPromptSubmit" { 10 } else { 2 },
                }]
            }));
        }
    }
    hooks.retain(|_, v| v.as_array().is_none_or(|a| !a.is_empty()));
    settings
}

/// Installs or removes the hooks in a settings file, atomically. A file
/// that is a symlink (managed by Home Manager or similar) is refused: the
/// hooks belong in the file's declared source.
pub fn install(path: &Path, install: bool) -> std::io::Result<()> {
    if path.is_symlink() {
        return Err(std::io::Error::other(format!(
            "{} is a symlink to a managed file; add the hooks in its source instead",
            path.display()
        )));
    }
    let current = match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text).map_err(std::io::Error::other)?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => json!({}),
        Err(e) => return Err(e),
    };
    let merged = merge_hooks(current, install);
    let mut text = serde_json::to_string_pretty(&merged).map_err(std::io::Error::other)?;
    text.push('\n');
    let dir = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir)?;
    use std::os::unix::fs::PermissionsExt;
    let permissions = match std::fs::metadata(path) {
        Ok(meta) => meta.permissions(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            std::fs::Permissions::from_mode(0o600)
        }
        Err(e) => return Err(e),
    };
    atomicwrites::AtomicFile::new(path, atomicwrites::AllowOverwrite)
        .write(|file| {
            file.set_permissions(permissions)?;
            file.write_all(text.as_bytes())
        })
        .map_err(Into::into)
}

#[cfg(test)]
mod tests;
