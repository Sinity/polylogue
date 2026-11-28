//! The hook command a harness calls on its events (`polylogue internal hook
//! <Event> [--harness NAME]`, the event's JSON on stdin), for Claude Code
//! (the default), Codex, Gemini CLI and Hermes.
//!
//! Every event is appended as one line to the hook spool file of the calling
//! harness process, with `O_APPEND` and synced to disk before the hook
//! returns, naming the harness's origin and session and the harness
//! process's identity (pid, start time and boot), which the live-state update uses
//! as the session's root process; the daemon reads those files like any
//! other growing file. Some events also ask the daemon for context:
//!
//! - the session's start (`SessionStart`; Hermes: `pre_llm_call` on its
//!   first turn): the session-start brief. Removing the hooks is the only
//!   off switch. After compaction it carries every prompt the operator
//!   typed in the session.
//! - a prompt (`UserPromptSubmit`; Hermes: `pre_llm_call`): a session ID
//!   or supported web link pasted into it brings that session's brief.
//!
//! Claude Code, Codex and Gemini CLI take context as `hookSpecificOutput.additionalContext`,
//! Hermes as `{"context": ...}`. Other events are only recorded. Hooks fail
//! open: a timeout emits an unreferenced failure notice and the harness
//! carries on. The original preparation is acknowledged only once it is confirmed.

pub(crate) mod open_calls;

use crate::brief::{Ask, StartSource};
use crate::config::Startup;
use base64::Engine as _;
use polylogue_model::DeliveryOutcome;
use polylogue_model::JsonExt as _;
use polylogue_model::SessionId;
use polylogue_sources::origin::by_name;
use polylogue_sources::{ORIGINS, Support};
use polylogue_store::Archive;
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

/// How long a hook that brings context waits for the daemon. A brief's
/// reads are bounded by what it reports and its sections stop before this
/// deadline, so the hook gets the sections ready in time rather than
/// nothing; session start never waits longer than this.
const CONTEXT_BUDGET: Duration = Duration::from_millis(1_500);
const TIMEOUT_NOTICE: &str =
    "Polylogue context timed out; continue with the current task and local instructions.";
/// How long the hook waits to hand a delivery's outcome to the daemon,
/// which queues it and answers at once.
const DELIVERED_BUDGET: Duration = Duration::from_millis(20);

/// A harness that calls the hook command.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    command_surface::JsonSchema,
)]
#[serde(rename_all = "kebab-case")]
pub enum Harness {
    ClaudeCode,
    Codex,
    Hermes,
    GeminiCli,
}

impl Harness {
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
    observed: &Value,
    startup: &Startup,
) -> std::io::Result<String> {
    append_record(
        dir,
        harness,
        event,
        RecordedPayload::Json(payload),
        observed,
        session_of(harness, payload, startup),
        startup,
    )
}

enum RecordedPayload<'a> {
    Json(&'a Value),
    Unreadable { bytes: &'a [u8], reason: &'a str },
}

fn append_record(
    dir: &Path,
    harness: Harness,
    event: &str,
    payload: RecordedPayload<'_>,
    observed: &Value,
    session: Option<SessionId>,
    startup: &Startup,
) -> std::io::Result<String> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::create_dir_all(dir)?;
    let caller = crate::client::this_process();
    let process = caller
        .harness_pid
        .filter(|_| {
            caller
                .harness
                .as_ref()
                .is_some_and(|caller_harness| caller_harness.as_str() == harness.origin())
        })
        .unwrap_or_else(std::os::unix::process::parent_id);
    let at = startup.client.clock.now();
    let ns = at.as_nanosecond();
    let event_id = request_id(at)?;
    let mut line = json!({
        "event_id": event_id,
        "at_ns": ns.to_string(),
        "event": event,
        "harness_pid": process,
        "harness_start": crate::live_events::identity(process).map(|i| i.start),
        "provider": harness.origin(),
    });
    match payload {
        RecordedPayload::Json(payload) => {
            let mut payload = payload.clone();
            // Hermes sends the whole conversation with every turn; its transcript
            // is read from its own database.
            if let Some(extra) = payload.get_mut("extra").and_then(Value::as_object_mut) {
                extra.remove("conversation_history");
            }
            line["payload"] = payload;
        }
        RecordedPayload::Unreadable { bytes, reason } => {
            line["unreadable_payload"] = json!({
                "bytes_base64": base64::engine::general_purpose::STANDARD.encode(bytes),
                "reason": reason,
            });
        }
    }
    if let Some(observed) = observed.as_object() {
        for (key, value) in observed {
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

/// Records each context output the harness took, separate from transcript attribution.
struct EmissionRecorder<'a> {
    dir: &'a Path,
    harness: Harness,
    event: &'a str,
    invocation: Option<&'a str>,
    session: Option<SessionId>,
    observed: &'a Value,
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
                    .map(|id| polylogue_model::Reference::Delivery(id.clone()))
                    .collect(),
            };
            let record = serde_json::to_value(emission)
                .map_err(std::io::Error::other)
                .and_then(|payload| {
                    append_record(
                        self.dir,
                        self.harness,
                        polylogue_sources::hook_spool::CONTEXT_EMITTED,
                        RecordedPayload::Json(&payload),
                        self.observed,
                        Some(session.clone()),
                        self.startup,
                    )
                });
            if let Err(error) = record {
                spool_error(&format!("emitted context for {session}"), &error);
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

/// Why a hook has no usable answer from the daemon. Each cause prints
/// differently, so a hook that cannot reach the daemon says why.
#[derive(Debug)]
pub(crate) enum Unanswered {
    /// Nothing listens on the socket: the daemon is not running.
    NotRunning(std::io::Error),
    ArchiveUnavailable(PathBuf),
    /// The daemon did not answer within the hook's budget.
    TimedOut(Duration),
    /// The answer arrived but is not what the operation answers.
    Undecodable(String),
    /// The daemon answered with a failure.
    Failed(command_surface::Failure),
    /// Any other failure of the exchange.
    Exchange(std::io::Error),
}

impl std::fmt::Display for Unanswered {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotRunning(error) => write!(f, "the daemon is not running ({error})"),
            Self::ArchiveUnavailable(path) => write!(
                f,
                "no archive at {}; check the archive path",
                path.display()
            ),
            Self::TimedOut(budget) => write!(
                f,
                "the daemon did not answer within {} ms",
                budget.as_millis()
            ),
            Self::Undecodable(error) => {
                write!(f, "the daemon's answer could not be decoded: {error}")
            }
            Self::Failed(failure) => match &failure.code {
                Some(code) => write!(f, "the daemon refused ({code}): {}", failure.message),
                None => write!(f, "the daemon refused: {}", failure.message),
            },
            Self::Exchange(error) => write!(f, "the exchange with the daemon failed: {error}"),
        }
    }
}

impl Unanswered {
    fn of(error: std::io::Error, budget: Duration) -> Self {
        use std::io::ErrorKind;
        match error.kind() {
            ErrorKind::ConnectionRefused | ErrorKind::NotFound => Self::NotRunning(error),
            ErrorKind::TimedOut => Self::TimedOut(budget),
            ErrorKind::InvalidData | ErrorKind::UnexpectedEof => {
                Self::Undecodable(error.to_string())
            }
            _ => Self::Exchange(error),
        }
    }

    fn timed_out(&self) -> bool {
        match self {
            Self::TimedOut(_) => true,
            Self::Failed(failure) => {
                failure.code.as_deref() == Some("deadline_exceeded")
                    || failure.kind == command_surface::FailureKind::Timeout
            }
            _ => false,
        }
    }
}

/// One exchange with the daemon on `socket`, its budget covering all of it.
/// An answer that carries a failure is that failure.
pub(crate) fn exchange(
    socket: &Path,
    request: &command_surface::RunnableRequest,
    budget: Duration,
) -> Result<Value, Unanswered> {
    let answer = crate::client::call_with_budget(socket, request, budget)
        .map_err(|error| Unanswered::of(error, budget))?;
    crate::protocol::into_value(answer).map_err(Unanswered::Failed)
}

/// Sends one request with a time budget for the whole exchange.
fn ask_daemon(
    startup: &Startup,
    operation: &str,
    arguments: Value,
    request_id: String,
    budget: Duration,
) -> Result<Value, Unanswered> {
    let mut request = crate::client::request(operation, arguments, request_id);
    request.deadline = command_surface::Deadline::Milliseconds(
        u64::try_from(budget.as_millis()).unwrap_or(u64::MAX),
    );
    exchange(
        &Archive::at(&startup.config.archive).socket(),
        &request,
        budget,
    )
}

fn report(what: &str, cause: &impl std::fmt::Display) {
    eprintln!(
        "{}",
        polylogue_render::terminal::escape(&format!("polylogue hook: {what}: {cause}"))
    );
}

/// A Hermes caller's session as Polylogue names it: Hermes's own session ID
/// is qualified by the profile of its `state.db` (under `HERMES_HOME`, else
/// `HOME/.hermes`); a session no profile names is left unknown.
pub(crate) fn hermes_session(
    caller: &mut command_surface::CallerEvidence,
    variable: impl Fn(&str) -> Option<std::ffi::OsString>,
) {
    let Some(native) = caller
        .session
        .as_ref()
        .filter(|session| session.harness.as_str() == polylogue_sources::hermes::ORIGIN)
        .map(|session| session.native.clone())
    else {
        return;
    };
    let state = variable("HERMES_HOME")
        .map(|home| Path::new(&home).join("state.db"))
        .or_else(|| variable("HOME").map(|home| Path::new(&home).join(".hermes/state.db")));
    caller.session = state
        .and_then(|state| polylogue_sources::hermes::session_at(&state, &native))
        .and_then(|session| command_surface::AgentSession::parse(&session.to_string()));
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
            polylogue_sources::hermes::session_at(&state_db, id)
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

/// A session-start result uses the same durable input as the hook event.
fn record_start_context(
    startup: &Startup,
    harness: Harness,
    session: Option<SessionId>,
    payload: Value,
) {
    let spool = startup
        .engine
        .spool_dir(polylogue_sources::hook_spool::ORIGIN);
    if let Err(error) = append_record(
        &spool,
        harness,
        "SessionContext",
        RecordedPayload::Json(&payload),
        &json!({}),
        session.clone(),
        startup,
    ) {
        spool_error(
            &format!(
                "session context for {}",
                session.map_or("unknown session".into(), |session| session.to_string())
            ),
            &error,
        );
    }
}

/// Read the stored start outcomes through the same event query callers use.
pub(crate) fn missing_context(
    reader: &polylogue_store::Reader<'_>,
) -> Result<crate::protocol::status::MissingContext, crate::daemon::Failure> {
    use std::ops::ControlFlow::{Break, Continue};

    let mut missing = crate::protocol::status::MissingContext::default();
    let mut last_given: Option<polylogue_model::Timestamp> = None;
    // Event ordering uses milliseconds. First find the latest successful
    // start's full timestamp, then count only failures after it.
    for counting in [false, true] {
        let mut invalid_time = false;
        reader.visit_events_by_kind("hook_session_context", |event| {
            let data = event.data;
            let Some(at) = data
                .str("at_ns")
                .and_then(|s| s.parse().ok())
                .and_then(polylogue_model::Timestamp::from_unix_ns)
            else {
                invalid_time = true;
                return Break(());
            };
            if last_given.is_some_and(|given| at.unix_ms() < given.unix_ms()) {
                return Break(());
            }
            if !counting && data["outcome"] == "given" {
                last_given = Some(last_given.map_or(at, |given| given.max(at)));
            }
            if counting
                && data["outcome"] == "not_given"
                && last_given.is_none_or(|given| at > given)
            {
                let cause = data.str("cause").unwrap_or("unknown");
                *missing.causes.entry(cause.into()).or_default() += 1;
                missing.since = Some(missing.since.map_or(at, |since| since.min(at)));
            }
            Continue(())
        })?;
        if invalid_time {
            return Err(crate::daemon::Failure::internal(
                "stored session context outcome has no time",
            ));
        }
    }
    Ok(missing)
}

fn spool_error(what: &str, error: &std::io::Error) {
    let message = format!("polylogue hook: could not record {what}: {error}");
    eprintln!("{}", polylogue_render::terminal::escape(&message));
    // The spool cannot record its own failed write. Send one native journal
    // entry without waiting for the journal to answer.
    if let Ok(socket) = std::os::unix::net::UnixDatagram::unbound()
        && socket.set_nonblocking(true).is_ok()
    {
        let _ = socket.send_to(
            format!("PRIORITY=3\nSYSLOG_IDENTIFIER=polylogue-hook\nMESSAGE={message}\n").as_bytes(),
            "/run/systemd/journal/socket",
        );
    }
}

/// Asks for context and returns its text, recording it as delivered once
/// it is handed to the harness by the caller. Context that cannot be had is
/// none, and the hook says why on its standard error; a timed-out advisory
/// says nothing.
fn context(
    startup: &Startup,
    ask: Ask,
    budget_tokens: Option<u32>,
    until: Instant,
) -> Option<(String, String)> {
    let advisory = matches!(&ask, Ask::Advisory { .. });
    let start = match &ask {
        Ask::SessionStart {
            session,
            cwd,
            source,
        } => Some((session.clone(), cwd.clone(), *source)),
        _ => None,
    };
    let unavailable = |cause: &str, detail: String| {
        if let Some((session, cwd, source)) = &start {
            let harness = session
                .as_ref()
                .and_then(|s| Harness::parse(s.origin().as_str()));
            let Some(harness) = harness else {
                return;
            };
            record_start_context(
                startup,
                harness,
                session.clone(),
                json!({
                    "outcome": "not_given", "cause": cause, "detail": detail, "cwd": cwd, "source": source,
                }),
            );
        }
    };
    let request_id = match request_id(startup.client.clock.now()) {
        Ok(id) => id,
        Err(error) => {
            unavailable("request", error.to_string());
            report("no context", &error);
            return None;
        }
    };
    let (operation, arguments) = match ask {
        Ask::SessionStart {
            cwd,
            session,
            source,
        } => (
            "rows",
            json!({"steps": [{"verb": "brief", "cwd": cwd, "for": session, "source": source}]}),
        ),
        Ask::Advisory {
            prompt,
            cwd,
            session,
        } => (
            "internal advisory",
            json!({"prompt":prompt,"cwd":cwd,"session":session}),
        ),
        Ask::Session {
            session,
            for_session,
        } => (
            "rows",
            json!({"selection": format!("session:{session}"), "steps": [{"verb": "brief", "for": for_session}]}),
        ),
        Ask::Sessions {
            sessions,
            for_session,
        } => (
            "rows",
            json!({
                "references": sessions.iter().map(|session| format!("session:{session}")).collect::<Vec<_>>(),
                "steps": [{"verb": "brief", "for": for_session}],
            }),
        ),
        Ask::Words {
            text,
            cwd,
            for_session,
            cursor,
        } => (
            "rows",
            json!({"selection": text, "steps": [{"verb": "brief", "cwd": cwd, "for": for_session, "cursor": cursor}]}),
        ),
    };
    let remaining = until.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        unavailable("timeout", TIMEOUT_NOTICE.into());
        return (!advisory).then(|| (String::new(), TIMEOUT_NOTICE.into()));
    }
    let allowance = if advisory {
        remaining.min(crate::advisory::BUDGET)
    } else {
        remaining
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
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    let archive = Archive::at(&startup.config.archive);
    let answer = exchange(&archive.socket(), &request, allowance)
        .map_err(|cause| {
            if matches!(cause, Unanswered::NotRunning(_)) && !archive.archive_db().is_file() {
                Unanswered::ArchiveUnavailable(archive.root.clone())
            } else {
                cause
            }
        })
        .and_then(|data| {
            if advisory {
                serde_json::from_value::<crate::brief::BriefReply>(data)
                    .map_err(|error| Unanswered::Undecodable(error.to_string()))
            } else {
                brief_of(data)
            }
        });
    let answer = match answer {
        Ok(answer) => answer,
        Err(cause) if cause.timed_out() && advisory => return None,
        Err(cause) if cause.timed_out() => {
            unavailable("timeout", cause.to_string());
            // No prepared bytes were emitted. Only the daemon's existing
            // queue may confirm preparation; without it, acceptance stays
            // unresolved.
            let accepted = self::request_id(startup.client.clock.now())
                .map_err(Unanswered::Exchange)
                .and_then(|id| {
                    ask_daemon(
                        startup,
                        "internal delivery-event",
                        json!({
                            "delivery": format!("{request_id}-1"),
                            "outcome": DeliveryOutcome::NotSent,
                            "detail": "timeout",
                        }),
                        id,
                        DELIVERED_BUDGET,
                    )
                });
            if let Err(cause) = accepted {
                report(
                    "could not hand the timed-out delivery outcome to the daemon",
                    &cause,
                );
            }
            return Some((String::new(), TIMEOUT_NOTICE.into()));
        }
        Err(cause) => {
            let reason = match &cause {
                Unanswered::NotRunning(_) => "daemon_unreachable",
                Unanswered::ArchiveUnavailable(_) => "archive_path",
                Unanswered::Failed(_) => "refusal",
                Unanswered::Undecodable(_) => "answer",
                _ => "exchange",
            };
            unavailable(reason, cause.to_string());
            report("no context", &cause);
            return None;
        }
    };
    if answer.prepared.empty {
        unavailable("empty", "the daemon returned no context".into());
        return None;
    }
    // An unreferenced missing-session notice still belongs in hook output.
    Some((answer.delivery.unwrap_or_default(), answer.prepared.text))
}

/// The brief a one-step `rows` line answered, or why there is none.
fn brief_of(data: Value) -> Result<crate::brief::BriefReply, Unanswered> {
    use crate::protocol::line::{LineAnswer, StepAnswer, StepOutcome};
    let line = serde_json::from_value::<LineAnswer>(data)
        .map_err(|error| Unanswered::Undecodable(error.to_string()))?;
    let LineAnswer::Steps(mut answer) = line else {
        return Err(Unanswered::Undecodable(
            "the line answered rows, not its brief step".into(),
        ));
    };
    match answer.steps.pop().map(|step| step.outcome) {
        Some(StepOutcome::Done {
            reply: StepAnswer::Brief(prepared),
            ..
        }) => Ok(*prepared),
        Some(StepOutcome::Failed { error }) => Err(Unanswered::Failed(error)),
        _ => Err(Unanswered::Undecodable(
            "the line has no brief step answer".into(),
        )),
    }
}

fn delivered(startup: &Startup, delivery: &str, budget: Duration) {
    let accepted = request_id(startup.client.clock.now())
        .map_err(Unanswered::Exchange)
        .and_then(|request_id| {
            ask_daemon(
                startup,
                "internal delivery-event",
                json!({
                    "delivery": delivery,
                    "outcome": DeliveryOutcome::Delivered,
                    "detail": "written to the harness hook output",
                }),
                request_id,
                budget,
            )
        });
    if let Err(cause) = accepted {
        report(
            &format!("could not hand delivery outcome {delivery} to the daemon"),
            &cause,
        );
    }
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
    until: Instant,
) -> Option<(String, String)> {
    let Some(cwd) = payload["cwd"].as_str().map(PathBuf::from) else {
        record_start_context(
            startup,
            harness,
            session_of(harness, payload, startup),
            json!({
                "outcome":"not_given", "cause":"directory", "detail":"the hook payload names no working directory",
            }),
        );
        return None;
    };
    context(
        startup,
        Ask::SessionStart {
            cwd,
            session: session_of(harness, payload, startup),
            source,
        },
        None,
        until,
    )
}

/// Context for the sessions pasted into a prompt, which share one budget:
/// one or two each get their session brief, split evenly; three or more get one
/// index of them (docs/agent-context.md).
fn pasted_sessions(
    harness: Harness,
    payload: &Value,
    prompt: &str,
    startup: &Startup,
    until: Instant,
) -> Option<(String, String)> {
    let me = session_of(harness, payload, startup);
    let mut texts = Vec::new();
    let mut deliveries = Vec::new();
    let sessions: Vec<_> = pasted_session_ids(prompt)
        .into_iter()
        .filter(|id| Some(id) != me.as_ref())
        .collect();
    if sessions.len() >= 3 {
        return context(
            startup,
            Ask::Sessions {
                sessions,
                for_session: me,
            },
            None,
            until,
        );
    }
    // Separators and every session share the named sessions' budget.
    let separators = sessions
        .len()
        .saturating_sub(1)
        .saturating_mul(2)
        .div_ceil(4);
    let available =
        crate::brief::DEFAULT_BUDGET.saturating_sub(u32::try_from(separators).unwrap_or(u32::MAX));
    let budget = available.checked_div(u32::try_from(sessions.len()).ok()?)?;
    for id in sessions {
        if let Some((d, t)) = context(
            startup,
            Ask::Session {
                session: id,
                for_session: me.clone(),
            },
            Some(budget),
            until,
        ) {
            let timed_out = t == TIMEOUT_NOTICE;
            deliveries.push(d);
            texts.push(t);
            if timed_out {
                break;
            }
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
    until: Instant,
) -> Option<(String, String)> {
    if !pasted_session_ids(prompt).is_empty() {
        return pasted_sessions(harness, payload, prompt, startup, until);
    }
    if crate::advisory::paths(prompt).is_empty() {
        return None;
    }
    let session = session_of(harness, payload, startup)?;
    let cwd = payload.str("cwd")?;
    context(
        startup,
        Ask::Advisory {
            prompt: prompt.into(),
            cwd: cwd.into(),
            session,
        },
        None,
        until,
    )
}

#[derive(Debug, PartialEq, Eq)]
struct HookContext {
    deliveries: String,
    text: String,
    session_start: bool,
}

/// Session-start and pasted-session context for an event, with its deliveries.
fn respond(
    harness: Harness,
    event: &str,
    payload: &Value,
    startup: &Startup,
    until: Instant,
) -> Option<HookContext> {
    let (reply, session_start) = match (harness, event) {
        (Harness::ClaudeCode | Harness::Codex | Harness::GeminiCli, "SessionStart") => {
            let source = payload["source"].as_str().unwrap_or("startup");
            // A fork carries its parent's context already.
            if source == "fork" {
                return None;
            }
            (
                start(harness, payload, StartSource::parse(source), startup, until),
                true,
            )
        }
        (Harness::ClaudeCode | Harness::Codex, "UserPromptSubmit")
        | (Harness::GeminiCli, "BeforeAgent") => (
            prompt_context(
                harness,
                payload,
                payload["prompt"].as_str()?,
                startup,
                until,
            ),
            false,
        ),
        // Hermes's first turn may carry both its start brief and pasted sessions.
        (Harness::Hermes, "pre_llm_call") => {
            let extra = &payload["extra"];
            let prompt = extra["user_message"].as_str().unwrap_or("");
            let brief = (extra["is_first_turn"].as_bool() == Some(true))
                .then(|| start(harness, payload, StartSource::Startup, startup, until))
                .flatten();
            let pasted = if brief
                .as_ref()
                .is_some_and(|(_, text)| text == TIMEOUT_NOTICE)
            {
                None
            } else {
                prompt_context(harness, payload, prompt, startup, until)
            };
            let session_start = brief
                .as_ref()
                .is_some_and(|(_, text)| text != TIMEOUT_NOTICE);
            let reply = match (brief, pasted) {
                (Some((d1, t1)), Some((d2, t2))) => {
                    Some((format!("{d1},{d2}"), format!("{t1}\n\n{t2}")))
                }
                (one, other) => one.or(other),
            };
            (reply, session_start)
        }
        _ => return None,
    };
    reply.map(|(deliveries, text)| HookContext {
        session_start: session_start && text != TIMEOUT_NOTICE,
        deliveries,
        text,
    })
}

pub fn run(harness: Harness, event: &str, startup: &Startup) -> ExitCode {
    let until = Instant::now() + CONTEXT_BUDGET;
    let mut input = Vec::new();
    let payload = std::io::stdin()
        .read_to_end(&mut input)
        .map_err(|error| format!("could not read the hook payload: {error}"))
        .and_then(|_| {
            serde_json::from_slice(&input)
                .map_err(|error| format!("hook payload is not JSON: {error}"))
        });
    let reason = match payload {
        Ok(payload) => return run_event(harness, event, payload, startup, until, emit),
        Err(reason) => reason,
    };
    let spool = startup
        .engine
        .spool_dir(polylogue_sources::hook_spool::ORIGIN);
    let record = append_record(
        &spool,
        harness,
        event,
        RecordedPayload::Unreadable {
            bytes: &input,
            reason: &reason,
        },
        &Value::Null,
        None,
        startup,
    );
    if let Err(error) = record {
        spool_error(
            &format!("unreadable {event} for {}", harness.origin()),
            &error,
        );
    } else {
        eprintln!(
            "{}",
            polylogue_render::terminal::escape(&format!("polylogue hook: {event}: {reason}"))
        );
    }
    ExitCode::SUCCESS
}

fn run_event(
    harness: Harness,
    event: &str,
    payload: Value,
    startup: &Startup,
    until: Instant,
    output: impl FnOnce(Harness, &str, &str) -> bool,
) -> ExitCode {
    let observed = environment::snapshot(
        event,
        &payload,
        &environment::Environment::observe(&startup.config.identity.host, &startup.hook),
        &startup.config.paths,
        until,
    );
    let spool = startup
        .engine
        .spool_dir(polylogue_sources::hook_spool::ORIGIN);
    let invocation = match append_to_spool(&spool, harness, event, &payload, &observed, startup) {
        Ok(invocation) => invocation,
        Err(error) => {
            spool_error(
                &format!(
                    "{event} for {}:{}",
                    harness.origin(),
                    payload.str("session_id").unwrap_or("unknown")
                ),
                &error,
            );
            return ExitCode::SUCCESS;
        }
    };
    let session = session_of(harness, &payload, startup);
    if harness == Harness::ClaudeCode
        && event == "PermissionRequest"
        && let Some(session) = &session
    {
        permission(startup, session, &payload, until);
        return ExitCode::SUCCESS;
    }
    let recorder = EmissionRecorder {
        dir: &spool,
        harness,
        event,
        invocation: Some(&invocation),
        session: session.clone(),
        observed: &observed,
        startup,
    };
    // Messages and notices waiting in the mailbox: one `stat`, and a claim
    // only when the flag is up. Never before a tool runs.
    let mail = session
        .as_ref()
        .filter(|_| accepts_mail(harness, event))
        .filter(|s| {
            crate::mailbox::raised(
                &crate::mailbox::dir(&startup.config.paths, &startup.config.archive),
                s,
            )
        })
        .and_then(|s| claim(startup, s, until));
    let responded = respond(harness, event, &payload, startup, until);
    let mut ids: Vec<String> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    if let Some(context) = &responded {
        ids.extend(
            context
                .deliveries
                .split(',')
                .filter(|id| !id.is_empty())
                .map(str::to_owned),
        );
        texts.push(context.text.clone());
    }
    if let Some((claimed, text)) = &mail {
        ids.extend(claimed.iter().cloned());
        texts.push(text.clone());
    }
    let emitted = !texts.is_empty() && recorder.emit(&ids, &texts.join("\n\n"), output);
    if responded
        .as_ref()
        .is_some_and(|context| context.session_start)
    {
        record_start_context(
            startup,
            harness,
            session,
            json!({
                "outcome": if emitted { "given" } else { "not_given" },
                "cause": if emitted { Value::Null } else { json!("output") },
                "invocation": invocation, "cwd": payload["cwd"],
            }),
        );
    }
    if emitted {
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

/// Takes the session's pending messages and notices from the daemon: their
/// delivery IDs and their texts joined.
fn claim(startup: &Startup, session: &SessionId, until: Instant) -> Option<(Vec<String>, String)> {
    let remaining = until.checked_duration_since(Instant::now())?;
    let data = match ask_daemon(
        startup,
        "internal mail-claim",
        json!({ "session": session }),
        polylogue_store::user_writes::new_request_id(),
        remaining,
    ) {
        Ok(data) => data,
        Err(cause) => {
            report("no mail claimed", &cause);
            return None;
        }
    };
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
fn permission(startup: &Startup, session: &SessionId, payload: &Value, until: Instant) {
    let Some(remaining) = until.checked_duration_since(Instant::now()) else {
        return;
    };
    let data = match ask_daemon(
        startup,
        "internal permission",
        json!({
            "session": session,
            "tool": payload["tool_name"].as_str().unwrap_or(""),
            "input": payload["tool_input"],
        }),
        polylogue_store::user_writes::new_request_id(),
        remaining,
    ) {
        Ok(data) => data,
        Err(cause) => {
            report("no permission decision", &cause);
            return;
        }
    };
    if let Some(output) = permission_output(&data) {
        let mut stdout = std::io::stdout().lock();
        let written = writeln!(
            stdout,
            "{}",
            polylogue_render::terminal::json(&output, false)
        )
        .and_then(|()| stdout.flush());
        drop(stdout);
        if written.is_ok()
            && let Some(delivery) = data["delivery"].as_str()
        {
            delivered(startup, delivery, DELIVERED_BUDGET);
        }
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

mod environment;
mod settings;
pub use settings::{CONFIRMED_FROM, HarnessVersion, install, probe_version, settings_path};

#[cfg(test)]
mod tests;
