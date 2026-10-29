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
use crate::config::Startup;
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

/// Terminal adapter facts resolved from the binary's captured environment.
#[derive(Debug, Clone)]
pub struct Inputs {
    pub kitty_pid: Option<u32>,
    pub kitty_window: Option<String>,
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
    startup: &Startup,
) -> std::io::Result<String> {
    append_record(
        dir,
        harness,
        event,
        payload,
        native,
        session_of(harness, payload, startup),
        startup,
    )
}

fn append_record(
    dir: &Path,
    harness: Harness,
    event: &str,
    payload: &Value,
    native: &Value,
    session: Option<SessionId>,
    startup: &Startup,
) -> std::io::Result<String> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::create_dir_all(dir)?;
    let caller = crate::protocol::caller::evidence();
    let process = caller
        .harness_pid
        .filter(|_| {
            caller
                .harness
                .as_ref()
                .is_some_and(|origin| origin.as_str() == harness.origin())
        })
        .unwrap_or_else(std::os::unix::process::parent_id);
    let at = startup.clock.now();
    let ns = at.as_nanosecond();
    let event_id = request_id(at)?;
    let mut payload = payload.clone();
    // Hermes sends the whole conversation with every turn; its transcript
    // is read from its own database.
    if let Some(extra) = payload.get_mut("extra").and_then(Value::as_object_mut) {
        extra.remove("conversation_history");
    }
    let mut line = json!({
        "event_id": event_id,
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
    if let Some(session) = session {
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
    f.write_all(&line)?;
    Ok(event_id)
}

/// A receipt records a successful native output, separate from transcript attribution.
struct EmissionRecorder<'a> {
    dir: &'a Path,
    harness: Harness,
    event: &'a str,
    invocation: Option<&'a str>,
    session: Option<SessionId>,
    native: &'a Value,
    startup: &'a Startup,
}
impl EmissionRecorder<'_> {
    fn emit(
        &self,
        deliveries: &[String],
        text: &str,
        output: impl FnOnce(Harness, &str, &str) -> bool,
    ) -> bool {
        if !output(self.harness, self.event, text) {
            return false;
        }
        if let (Some(invocation), Some(session)) = (self.invocation, self.session.as_ref()) {
            let emission = polylogue_model::HookContextEmission {
                invocation: invocation.into(),
                session: session.clone(),
                hook_event: self.event.into(),
                deliveries: deliveries
                    .iter()
                    .map(|id| polylogue_model::Reference::Delivery(id.clone()).to_string())
                    .collect(),
            };
            let record = serde_json::to_value(emission)
                .map_err(std::io::Error::other)
                .and_then(|payload| {
                    append_record(
                        self.dir,
                        self.harness,
                        "HookContextEmitted",
                        &payload,
                        self.native,
                        Some(session.clone()),
                        self.startup,
                    )
                });
            if let Err(error) = record {
                eprintln!(
                    "{}",
                    polylogue_render::terminal::escape(&format!(
                        "polylogue hook: could not record emitted context: {error}"
                    ))
                );
            }
        }
        true
    }
}

fn request_id(at: jiff::Timestamp) -> std::io::Result<String> {
    let ns = at.as_nanosecond();
    let seconds = u64::try_from(ns.div_euclid(1_000_000_000)).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "hook clock precedes the UUID epoch",
        )
    })?;
    let nanos = u32::try_from(ns.rem_euclid(1_000_000_000)).expect("subseconds fit u32");
    Ok(uuid::Uuid::new_v7(uuid::Timestamp::from_unix(uuid::NoContext, seconds, nanos)).to_string())
}

/// Sends one request with a time budget for the whole exchange.
fn ask_daemon(startup: &Startup, request: &Request, budget: Duration) -> Option<Response> {
    exchange(
        &Archive::at(&startup.config.archive).socket(),
        request,
        budget,
    )
}

fn exchange(socket: &Path, request: &Request, budget: Duration) -> Option<Response> {
    crate::client::send_with_budget(socket, request, budget).ok()
}

/// The session a hook payload names, as Polylogue names it.
fn session_of(harness: Harness, payload: &Value, startup: &Startup) -> Option<SessionId> {
    let id = payload["session_id"].as_str().filter(|id| !id.is_empty())?;
    match harness {
        Harness::Hermes => {
            let origin = by_name(harness.origin())?;
            let location = origin
                .locations
                .iter()
                .find(|location| **location == "state.db")?;
            let state_db = startup.engine.root(origin.root).join(location);
            polylogue_sources::hermes::session_at(&startup.config.identity.machine, &state_db, id)
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
fn context(startup: &Startup, ask: Ask) -> Option<(String, String)> {
    let request_id = request_id(startup.clock.now()).ok()?;
    let reply = ask_daemon(
        startup,
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

fn delivered(startup: &Startup, delivery: &str, budget: Duration) {
    let _ = ask_daemon(
        startup,
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
fn start(
    harness: Harness,
    payload: &Value,
    source: StartSource,
    startup: &Startup,
) -> Option<(String, String)> {
    let cwd = payload["cwd"].as_str().map(PathBuf::from)?;
    context(
        startup,
        Ask::SessionStart {
            cwd,
            session: session_of(harness, payload, startup),
            source,
        },
    )
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
fn takeovers(
    harness: Harness,
    payload: &Value,
    prompt: &str,
    startup: &Startup,
) -> Option<(String, String)> {
    let me = session_of(harness, payload, startup);
    let mut texts = Vec::new();
    let mut deliveries = Vec::new();
    for id in pasted_session_ids(prompt) {
        if Some(&id) == me.as_ref() {
            continue;
        }
        if let Some((d, t)) = context(
            startup,
            Ask::Takeover {
                session: id,
                for_session: me.clone(),
            },
        ) {
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
pub fn respond(
    harness: Harness,
    event: &str,
    payload: &Value,
    startup: &Startup,
) -> Option<(String, String)> {
    match (harness, event) {
        (Harness::ClaudeCode | Harness::Codex | Harness::GeminiCli, "SessionStart") => start(
            harness,
            payload,
            StartSource::parse(payload["source"].as_str().unwrap_or("startup")),
            startup,
        ),
        (Harness::ClaudeCode | Harness::Codex, "UserPromptSubmit")
        | (Harness::GeminiCli, "BeforeAgent") => {
            takeovers(harness, payload, payload["prompt"].as_str()?, startup)
        }
        // Hermes injects context only into a turn's prompt: the first
        // turn's carries the brief, and any turn's the takeovers it asks for.
        (Harness::Hermes, "pre_llm_call") => {
            let extra = &payload["extra"];
            let prompt = extra["user_message"].as_str().unwrap_or("");
            let brief = (extra["is_first_turn"].as_bool() == Some(true))
                .then(|| start(harness, payload, StartSource::Startup, startup))
                .flatten();
            let takeover = takeovers(harness, payload, prompt, startup);
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
    startup: &Startup,
    mut send: impl FnMut(&Request, Duration) -> Option<Response>,
    output: impl FnOnce(Harness, &str, &str, &str) -> bool,
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
    let session = session_of(harness, payload, startup);
    let Ok(request_id) = request_id(startup.clock.now()) else {
        return;
    };
    let request = Request::Context {
        request_id,
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
    if output(harness, "PreToolUse", text, id) {
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

pub fn run(harness: Harness, event: &str, startup: &Startup) -> ExitCode {
    let mut input = String::new();
    let _ = std::io::stdin().read_to_string(&mut input);
    let payload: Value = serde_json::from_str(&input).unwrap_or(Value::Null);
    let native = native::snapshot(
        event,
        &payload,
        &native::Environment::observe(&startup.config.identity.host, &startup.hook),
    );
    let spool = startup
        .engine
        .spool_dir(polylogue_sources::hook_spool::ORIGIN);
    let invocation = match append_to_spool(&spool, harness, event, &payload, &native, startup) {
        Ok(invocation) => Some(invocation),
        Err(error) => {
            eprintln!(
                "{}",
                polylogue_render::terminal::escape(&format!(
                    "polylogue hook: could not record {event}: {error}"
                ))
            );
            None
        }
    };
    let recorder = EmissionRecorder {
        dir: &spool,
        harness,
        event,
        invocation: invocation.as_deref(),
        session: session_of(harness, &payload, startup),
        native: &native,
        startup,
    };
    if event == "PreToolUse" {
        tool_advice(
            harness,
            &payload,
            startup,
            |request, budget| ask_daemon(startup, request, budget),
            |_, _, text, id| recorder.emit(&[id.to_owned()], text, emit),
        );
    } else if let Some((deliveries, text)) = respond(harness, event, &payload, startup)
        && recorder.emit(
            &deliveries.split(',').map(str::to_owned).collect::<Vec<_>>(),
            &text,
            emit,
        )
    {
        for d in deliveries.split(',') {
            delivered(startup, d, CONTEXT_BUDGET);
        }
    }
    ExitCode::SUCCESS
}

mod native;
mod settings;
pub use settings::{install, settings_path};

#[cfg(test)]
mod tests;
