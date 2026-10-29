//! The hook command a harness calls on its events (`polylogue hook
//! <Event> [--harness NAME]`, the event's JSON on stdin), for Claude Code
//! (the default), Codex, Gemini CLI and Hermes.
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
//!   or supported web link pasted into it brings that session's takeover context.
//! - a shell command about to run (`PreToolUse`, Claude Code and Codex):
//!   advice, only when the command is a verification that failed here repeatedly
//!   since it last passed. Read-only preparation has [`ADVICE_BUDGET`];
//!   recording printed advice has [`ADVICE_DELIVERY_BUDGET`].
//!
//! Claude Code, Codex and Gemini CLI take context as `hookSpecificOutput.additionalContext`,
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
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

/// How long a hook that brings context waits for the daemon: a brief
/// during a first build over a large archive takes a few seconds. The
/// harnesses' hook timeouts are longer.
const CONTEXT_BUDGET: Duration = Duration::from_secs(8);

/// How long a hook waits for read-only advice preparation.
const ADVICE_BUDGET: Duration = Duration::from_millis(20);
/// Recording printed advice has an independent short budget.
const ADVICE_DELIVERY_BUDGET: Duration = Duration::from_millis(20);

/// A harness that calls the hook command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Harness {
    ClaudeCode,
    Codex,
    Hermes,
    GeminiCli,
}

impl Harness {
    pub const NAMES: &[&str] = &["claude-code", "codex", "hermes", "gemini-cli"];

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "claude-code" => Some(Self::ClaudeCode),
            "codex" => Some(Self::Codex),
            "hermes" => Some(Self::Hermes),
            "gemini-cli" => Some(Self::GeminiCli),
            _ => None,
        }
    }

    /// The origin whose sessions its hook events belong to.
    pub fn origin(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude-code",
            Self::Codex => "codex",
            Self::GeminiCli => "gemini-cli",
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
    native: &Value,
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
        "event_id": uuid::Uuid::now_v7().to_string(),
        "at_ns": ns.to_string(),
        "event": event,
        "harness_pid": process,
        "provider": harness.origin(),
        "payload": payload,
    });
    if let Some(native) = native.as_object() {
        for (key, value) in native {
            line[key] = value.clone();
        }
    }
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
    exchange(
        &Archive::at(&Archive::default_root()).socket(),
        request,
        budget,
    )
}

fn exchange(socket: &Path, request: &Request, budget: Duration) -> Option<Response> {
    crate::client::send_with_budget(socket, request, budget).ok()
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
        if let Some(id) = session_link(word) {
            push(id);
        } else if let Some((origin, native)) = word.split_once(':')
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

/// Resolve only conversation pages declared by the supported web origins.
fn session_link(word: &str) -> Option<SessionId> {
    let url = word.split('#').next()?.parse::<axum::http::Uri>().ok()?;
    if url.scheme_str() != Some("https") {
        return None;
    }
    let host = url.authority()?.as_str();
    let mut parts: Vec<_> = url
        .path()
        .trim_end_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();
    let origin = ORIGINS.iter().find(|o| {
        o.web
            .as_ref()
            .is_some_and(|web| web.hosts.iter().any(|h| h.eq_ignore_ascii_case(host)))
    })?;
    let web = origin.web.as_ref()?;
    if parts.len() == 4
        && web.prefixes.iter().any(|prefix| {
            parts[0] == prefix.path
                && (!prefix.numeric || parts[1].bytes().all(|b| b.is_ascii_digit()))
        })
    {
        parts.drain(..2);
    }
    if parts.len() != 2 || parts[0] != web.path {
        return None;
    }
    polylogue_sources::browser_capture::session_for(origin.name, parts[1])
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

fn delivered(delivery: &str, budget: Duration) {
    let _ = ask_daemon(
        &Request::DeliveryEvent {
            delivery: delivery.to_owned(),
            what: DeliveryOutcome::Delivered,
            detail: Some("written to the harness hook output".into()),
        },
        budget,
    );
}

/// The hook output carrying context for the model.
fn output_for(harness: Harness, event: &str, text: &str) -> Value {
    match harness {
        Harness::Hermes => json!({ "context": text }),
        Harness::ClaudeCode | Harness::Codex | Harness::GeminiCli => json!({
            "hookSpecificOutput": {
                "hookEventName": event,
                "additionalContext": text,
            }
        }),
    }
}

fn emit(harness: Harness, event: &str, text: &str) -> bool {
    write_context(&mut std::io::stdout().lock(), harness, event, text)
}

fn write_context(stdout: &mut impl Write, harness: Harness, event: &str, text: &str) -> bool {
    let text = polylogue_render::terminal::escape(text);
    let out = output_for(harness, event, &text);
    writeln!(stdout, "{}", polylogue_render::terminal::json(&out, false)).is_ok()
        && stdout.flush().is_ok()
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

/// The shell command a pre-tool payload is about to run: Claude Code's
/// `tool_input.command`, or a command given as its words.
fn shell_command(payload: &Value) -> Option<String> {
    match &payload["tool_input"]["command"] {
        Value::String(command) => Some(command.clone()),
        Value::Array(words) => {
            let words: Option<Vec<&str>> = words.iter().map(Value::as_str).collect();
            Some(words?.join(" "))
        }
        _ => None,
    }
    .filter(|command| !command.trim().is_empty())
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

/// Session-start and takeover context for an event, with its deliveries.
pub fn respond(harness: Harness, event: &str, payload: &Value) -> Option<(String, String)> {
    match (harness, event) {
        (Harness::ClaudeCode | Harness::Codex | Harness::GeminiCli, "SessionStart") => start(
            harness,
            payload,
            StartSource::parse(payload["source"].as_str().unwrap_or("startup")),
        ),
        (Harness::ClaudeCode | Harness::Codex, "UserPromptSubmit")
        | (Harness::GeminiCli, "BeforeAgent") => {
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

/// Prepares read-only advice, prints it, then records the delivered text.
fn tool_advice(
    harness: Harness,
    payload: &Value,
    mut send: impl FnMut(&Request, Duration) -> Option<Response>,
    output: impl FnOnce(Harness, &str, &str) -> bool,
) {
    if !matches!(harness, Harness::ClaudeCode | Harness::Codex) {
        return;
    }
    let Some(cwd) = payload["cwd"].as_str().map(PathBuf::from) else {
        return;
    };
    let Some(command) = shell_command(payload) else {
        return;
    };
    let session = session_of(harness, payload);
    let request = Request::Context {
        request_id: polylogue_store::user_writes::new_request_id(),
        ask: Ask::ToolUse {
            cwd: cwd.clone(),
            session: session.clone(),
            command,
        },
        sender: None,
    };
    let Some(Response::Ok { data, .. }) = send(&request, ADVICE_BUDGET) else {
        return;
    };
    if data["prepared"]["empty"].as_bool() != Some(false) {
        return;
    }
    let (Some(id), Some(text)) = (data["delivery"].as_str(), data["prepared"]["text"].as_str())
    else {
        return;
    };
    if output(harness, "PreToolUse", text) {
        let _ = send(
            &Request::AdviceDelivered {
                request_id: id.to_owned(),
                session,
                cwd,
                text: text.to_owned(),
            },
            ADVICE_DELIVERY_BUDGET,
        );
    }
}

pub fn run(harness: Harness, event: &str) -> ExitCode {
    let mut input = String::new();
    let _ = std::io::stdin().read_to_string(&mut input);
    let payload: Value = serde_json::from_str(&input).unwrap_or(Value::Null);
    let native = native::snapshot(event, &payload, &native::Environment::load());
    if let Err(e) = append_to_spool(&hook_spool_dir(), harness, event, &payload, &native) {
        // The harness must not fail because the spool cannot be written.
        eprintln!(
            "{}",
            polylogue_render::terminal::escape(&format!(
                "polylogue hook: could not record {event}: {e}"
            ))
        );
    }
    if event == "PreToolUse" {
        tool_advice(harness, &payload, ask_daemon, emit);
    } else if let Some((deliveries, text)) = respond(harness, event, &payload)
        && emit(harness, event, &text)
    {
        for d in deliveries.split(',') {
            delivered(d, CONTEXT_BUDGET);
        }
    }
    ExitCode::SUCCESS
}

mod native;
mod settings;
pub use settings::{install, settings_path};

#[cfg(test)]
mod tests;
