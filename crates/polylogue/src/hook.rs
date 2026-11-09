//! The hook command a harness calls on its events (`polylogue hook
//! <Event> [--harness NAME]`, the event's JSON on stdin), for Claude Code
//! (the default), Codex, Gemini CLI and Hermes.
//!
//! Every event is appended as one line to the hook spool file of the calling
//! harness process, with `O_APPEND` and synced to disk before the hook
//! returns, naming the harness's origin and session and the harness
//! process's identity (pid, start time and boot), which the live fold uses
//! as the session's root process; the daemon reads those files like any
//! other growing file. Some events also ask the daemon for context:
//!
//! - the session's start (`SessionStart`; Hermes: `pre_llm_call` on its
//!   first turn): the session-start brief. Removing the hooks is the only
//!   off switch. After compaction it carries every prompt the operator
//!   typed in the session.
//! - a prompt (`UserPromptSubmit`; Hermes: `pre_llm_call`): a session ID
//!   or supported web link pasted into it brings that session's takeover context.
//!
//! Claude Code, Codex and Gemini CLI take context as `hookSpecificOutput.additionalContext`,
//! Hermes as `{"context": ...}`. Other events are only recorded. Hooks fail
//! open: a timeout emits an unreferenced failure notice and the harness
//! carries on. The original preparation is acknowledged only with positive proof.

use crate::briefing::{Ask, StartSource};
use crate::config::Startup;
use crate::daemon::Request;
use command_surface::Answer;
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

/// How long a hook that brings context waits for the daemon. A brief's
/// reads are bounded by what it reports and its sections stop before this
/// deadline, so the hook gets the sections ready in time rather than
/// nothing; session start never waits longer than this.
const CONTEXT_BUDGET: Duration = Duration::from_millis(1_500);
/// How long the hook waits to hand a delivery's outcome to the daemon,
/// which queues it and answers at once.
const DELIVERED_BUDGET: Duration = Duration::from_millis(20);

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
        "harness_start": crate::live_evidence::identity(process).map(|i| i.start),
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
    // Hook-only facts exist nowhere else, so the line is on disk before the
    // harness carries on.
    f.sync_data()?;
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
fn ask_daemon(startup: &Startup, request: &Request, budget: Duration) -> Option<Answer<Value>> {
    exchange(
        &Archive::at(&startup.config.archive).socket(),
        request,
        budget,
    )
}

fn exchange(socket: &Path, request: &Request, budget: Duration) -> Option<Answer<Value>> {
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
fn context(startup: &Startup, ask: Ask, budget_tokens: Option<u32>) -> Option<(String, String)> {
    let advisory = matches!(&ask, Ask::Advisory { .. });
    let allowance = if advisory {
        crate::advisory::BUDGET
    } else {
        CONTEXT_BUDGET
    };
    let request_id = request_id(startup.clock.now()).ok()?;
    let (operation, arguments, cursor) = match ask {
        Ask::SessionStart {
            cwd,
            session,
            source,
        } => (
            "brief",
            json!({ "cwd": cwd, "session": session, "source": source }),
            None,
        ),
        Ask::Advisory {
            prompt,
            cwd,
            session,
        } => (
            "advisory",
            json!({"prompt":prompt,"cwd":cwd,"session":session}),
            None,
        ),
        Ask::Takeover {
            session,
            for_session,
        } => (
            "takeover",
            json!({ "session": session, "for_session": for_session }),
            None,
        ),
        Ask::Recall {
            text,
            cwd,
            for_session,
            cursor,
        } => (
            "recall",
            json!({ "query": text, "cwd": cwd, "for_session": for_session }),
            cursor,
        ),
    };
    let request = command_surface::Request {
        operation: operation.into(),
        arguments,
        request_id: request_id.clone(),
        deadline: command_surface::Deadline::Milliseconds(allowance.as_millis() as u64),
        budget_tokens: budget_tokens.map_or(
            command_surface::TokenBudget::Default,
            command_surface::TokenBudget::Tokens,
        ),
        cursor,
        fresh: false,
        format: Default::default(),
    };
    let reply = crate::client::call_with_budget(
        &Archive::at(&startup.config.archive).socket(),
        &request,
        allowance,
    );
    let timed_out = match &reply {
        Err(error) => error.kind() == std::io::ErrorKind::TimedOut,
        Ok(reply) => reply.failure().is_some_and(|error| {
            error.code.as_deref() == Some("deadline_exceeded")
                || error.kind == command_surface::FailureKind::Timeout
        }),
    };
    if timed_out && advisory {
        return None;
    }
    if timed_out {
        // No prepared bytes were emitted. Only the daemon's existing queue
        // may confirm preparation; absent proof leaves acceptance unresolved.
        let _ = ask_daemon(
            startup,
            &Request::DeliveryEvent {
                request_id: self::request_id(startup.clock.now()).ok()?,
                delivery: request_id,
                what: DeliveryOutcome::NotSent,
                detail: Some("timeout".into()),
            },
            DELIVERED_BUDGET,
        );
        return Some((
            String::new(),
            "Polylogue context timed out; continue with the current task and local instructions."
                .into(),
        ));
    }
    let data = crate::protocol::into_value(reply.ok()?).ok()?;
    let answer: crate::protocol::context::Answer = serde_json::from_value(data).ok()?;
    if answer.prepared.empty {
        return None;
    }
    Some((answer.delivery?, answer.prepared.text))
}

fn delivered(startup: &Startup, delivery: &str, budget: Duration) {
    let Ok(request_id) = request_id(startup.clock.now()) else {
        return;
    };
    let _ = ask_daemon(
        startup,
        &Request::DeliveryEvent {
            request_id,
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
        None,
    )
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
    let sessions: Vec<_> = pasted_session_ids(prompt)
        .into_iter()
        .filter(|id| Some(id) != me.as_ref())
        .collect();
    // Separators and every session share the named takeover budget.
    let separators = sessions
        .len()
        .saturating_sub(1)
        .saturating_mul(2)
        .div_ceil(4);
    let available = crate::briefing::DEFAULT_BUDGET
        .saturating_sub(u32::try_from(separators).unwrap_or(u32::MAX));
    let budget = available.checked_div(u32::try_from(sessions.len()).ok()?)?;
    for id in sessions {
        if let Some((d, t)) = context(
            startup,
            Ask::Takeover {
                session: id,
                for_session: me.clone(),
            },
            Some(budget),
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

fn prompt_context(
    harness: Harness,
    payload: &Value,
    prompt: &str,
    startup: &Startup,
) -> Option<(String, String)> {
    if !pasted_session_ids(prompt).is_empty() {
        return takeovers(harness, payload, prompt, startup);
    }
    if crate::advisory::paths(prompt).is_empty() {
        return None;
    }
    let session = session_of(harness, payload, startup)?;
    let cwd = payload.get("cwd").and_then(Value::as_str)?;
    context(
        startup,
        Ask::Advisory {
            prompt: prompt.into(),
            cwd: cwd.into(),
            session,
        },
        None,
    )
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
            prompt_context(harness, payload, payload["prompt"].as_str()?, startup)
        }
        // Hermes injects context only into a turn's prompt: the first
        // turn's carries the brief, and any turn's the takeovers it asks for.
        (Harness::Hermes, "pre_llm_call") => {
            let extra = &payload["extra"];
            let prompt = extra["user_message"].as_str().unwrap_or("");
            let brief = (extra["is_first_turn"].as_bool() == Some(true))
                .then(|| start(harness, payload, StartSource::Startup, startup))
                .flatten();
            let takeover = prompt_context(harness, payload, prompt, startup);
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
    let session = session_of(harness, &payload, startup);
    if harness == Harness::ClaudeCode
        && event == "PermissionRequest"
        && let Some(session) = &session
    {
        permission(startup, session, &payload);
        return ExitCode::SUCCESS;
    }
    let recorder = EmissionRecorder {
        dir: &spool,
        harness,
        event,
        invocation: invocation.as_deref(),
        session: session.clone(),
        native: &native,
        startup,
    };
    // Pings and notices waiting in the mailbox: one `stat`, and a claim
    // only when the flag is up. Never before a tool runs.
    let mail = session
        .as_ref()
        .filter(|_| accepts_mail(harness, event))
        .filter(|s| crate::mailbox::raised(&crate::mailbox::dir(&startup.config.archive), s))
        .and_then(|s| claim(startup, s, event));
    let responded = respond(harness, event, &payload, startup);
    let mut ids: Vec<String> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    if let Some((deliveries, text)) = &responded {
        ids.extend(
            deliveries
                .split(',')
                .filter(|id| !id.is_empty())
                .map(str::to_owned),
        );
        texts.push(text.clone());
    }
    if let Some((claimed, text)) = &mail {
        ids.extend(claimed.iter().cloned());
        texts.push(text.clone());
    }
    if !texts.is_empty() && recorder.emit(&ids, &texts.join("\n\n"), emit) {
        for d in &ids {
            delivered(startup, d, DELIVERED_BUDGET);
        }
    }
    ExitCode::SUCCESS
}

/// The events at which a harness accepts context between steps: after a
/// tool result and at a prompt, never before a tool runs. Declared with
/// the harness's other channels (`polylogue_model::live::channel`).
pub fn accepts_mail(harness: Harness, event: &str) -> bool {
    polylogue_model::live::channel::harness(harness.origin())
        .is_some_and(|declared| declared.mail_events.contains(&event))
}

/// Takes the session's pending pings and notices from the daemon: their
/// delivery IDs and their texts joined.
fn claim(startup: &Startup, session: &SessionId, event: &str) -> Option<(Vec<String>, String)> {
    let reply = ask_daemon(
        startup,
        &Request::Claim {
            session: session.clone(),
            event: event.to_owned(),
        },
        CONTEXT_BUDGET,
    )?;
    let data = crate::protocol::into_value(reply).ok()?;
    let items = data["deliveries"].as_array()?;
    let ids: Vec<String> = items
        .iter()
        .filter_map(|i| i["id"].as_str().map(str::to_owned))
        .collect();
    let texts: Vec<&str> = items.iter().filter_map(|i| i["text"].as_str()).collect();
    (!ids.is_empty()).then(|| (ids, texts.join("\n\n")))
}

/// Claude Code's synchronous permission hook: a decision given ahead for
/// the same tool and input is returned at once; with none, nothing is
/// printed and the harness shows its own prompt.
fn permission(startup: &Startup, session: &SessionId, payload: &Value) {
    let Some(reply) = ask_daemon(
        startup,
        &Request::Permission {
            session: session.clone(),
            tool: payload["tool_name"].as_str().unwrap_or("").to_owned(),
            input: payload["tool_input"].clone(),
        },
        CONTEXT_BUDGET,
    ) else {
        return;
    };
    let Ok(data) = crate::protocol::into_value(reply) else {
        return;
    };
    if let Some(output) = permission_output(&data) {
        let mut stdout = std::io::stdout().lock();
        let _ = writeln!(
            stdout,
            "{}",
            polylogue_render::terminal::json(&output, false)
        );
        let _ = stdout.flush();
    }
}

/// The hook output for a daemon decision, or `None` without one.
pub fn permission_output(data: &Value) -> Option<Value> {
    let behavior = match data["decision"].as_str()? {
        "approve" => "allow",
        "deny" => "deny",
        _ => return None,
    };
    let mut decision = json!({ "behavior": behavior });
    if let Some(text) = data["text"].as_str() {
        decision["message"] = json!(polylogue_render::terminal::escape(text));
    }
    Some(json!({
        "hookSpecificOutput": {
            "hookEventName": "PermissionRequest",
            "decision": decision,
        }
    }))
}

mod native;
mod settings;
pub use settings::{CONFIRMED_FROM, HarnessVersion, install, probe_version, settings_path};

#[cfg(test)]
mod tests;
