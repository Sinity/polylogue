//! The daemon's state for live-session acts: the mailbox directory, the
//! claim lock, resumed runs in progress, and hosted sessions, the
//! sessions Polylogue started or attached to through a harness's control
//! protocol and holds a connection to. The acts themselves are in
//! `daemon::service::acts`.

use crate::live_evidence::CodexConnection;
use polylogue_model::SessionId;
use polylogue_model::live::LiveFacts;
use polylogue_model::live::channel::Probe;
use serde_json::{Value, json};
use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock, Weak};

#[cfg(test)]
mod tests;

/// A ping to a session that is not running, run through the harness's
/// headless resume.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResumeRun {
    pub session: SessionId,
    pub delivery: String,
    pub text: String,
    pub cwd: Option<String>,
}

/// The harness's headless resume command line for a session, when the
/// harness has one: Claude Code `claude --resume ID -p TEXT`, Codex
/// `codex exec resume ID TEXT`.
pub(crate) fn headless_resume(session: &SessionId, text: &str) -> Option<Vec<String>> {
    let id = session.native().to_string();
    match session.origin().as_str() {
        "claude-code" => Some(vec![
            "claude".into(),
            "--resume".into(),
            id,
            "-p".into(),
            text.into(),
        ]),
        "codex" => Some(vec![
            "codex".into(),
            "exec".into(),
            "resume".into(),
            id,
            text.into(),
        ]),
        _ => None,
    }
}

/// A Codex thread Polylogue started on the shared app-server, with the
/// connection that started it: requests the app-server sends for the
/// thread (approvals, questions) arrive on it, and the turn now running is
/// tracked from its notifications.
pub(crate) struct HostedCodex {
    connection: CodexConnection,
    thread: String,
    turn: Option<String>,
    /// Requests awaiting an answer, oldest first: (request ID, method,
    /// parameters).
    requests: VecDeque<(Value, String, Value)>,
}

impl HostedCodex {
    /// Starts a thread in `cwd` with `text` as its first turn.
    pub(crate) fn start(
        program: &Path,
        socket: &Path,
        cwd: &str,
        text: &str,
    ) -> std::io::Result<Self> {
        let mut connection = CodexConnection::open(program, socket)?;
        let started = connection.call("thread/start", json!({"cwd": cwd}))?;
        let thread = started
            .pointer("/thread/id")
            .and_then(Value::as_str)
            .ok_or_else(|| std::io::Error::other("thread/start answered no thread"))?
            .to_owned();
        let mut hosted = Self {
            connection,
            thread,
            turn: None,
            requests: VecDeque::new(),
        };
        hosted.send(text)?;
        Ok(hosted)
    }

    pub(crate) fn thread(&self) -> &str {
        &self.thread
    }

    /// Takes in what the app-server sent since the last act.
    fn absorb(&mut self) {
        for message in self.connection.take_pending() {
            let method = message.get("method").and_then(Value::as_str).unwrap_or("");
            let params = message.get("params").cloned().unwrap_or(Value::Null);
            if params.get("threadId").and_then(Value::as_str) != Some(self.thread.as_str()) {
                continue;
            }
            match (method, message.get("id")) {
                ("turn/started", None) => {
                    self.turn = params
                        .pointer("/turn/id")
                        .and_then(Value::as_str)
                        .map(str::to_owned);
                }
                ("turn/completed", None) => self.turn = None,
                ("serverRequest/resolved", None) => {
                    let id = params.get("requestId").cloned().unwrap_or(Value::Null);
                    self.requests.retain(|(pending, _, _)| *pending != id);
                }
                (method, Some(id)) => {
                    self.requests
                        .push_back((id.clone(), method.to_owned(), params.clone()));
                }
                _ => {}
            }
        }
    }

    /// Steers the running turn, or starts a turn when none runs.
    pub(crate) fn send(&mut self, text: &str) -> std::io::Result<()> {
        self.absorb();
        let input = json!([{"type": "text", "text": text}]);
        match self.turn.clone() {
            Some(turn) => self.connection.call(
                "turn/steer",
                json!({"threadId": self.thread, "expectedTurnId": turn, "input": input}),
            ),
            None => self
                .connection
                .call("turn/start", json!({"threadId": self.thread, "input": input})),
        }
        .map(drop)
    }

    /// Answers the oldest pending approval request.
    pub(crate) fn answer(&mut self, approve: bool, text: Option<&str>) -> std::io::Result<()> {
        self.absorb();
        let (id, method, params) = self
            .requests
            .pop_front()
            .ok_or_else(|| std::io::Error::other("no request is pending on the hosted thread"))?;
        self.connection.answer(&id, answer_result(&method, &params, approve, text)?)
    }

    pub(crate) fn interrupt(&mut self) -> std::io::Result<()> {
        self.absorb();
        let turn = self
            .turn
            .clone()
            .ok_or_else(|| std::io::Error::other("no turn is running"))?;
        self.connection
            .call(
                "turn/interrupt",
                json!({"threadId": self.thread, "turnId": turn}),
            )
            .map(drop)
    }
}

/// The response to an app-server request: a decision for a command or
/// file change approval, the text (or yes or no) for each question asked.
pub(crate) fn answer_result(
    method: &str,
    params: &Value,
    approve: bool,
    text: Option<&str>,
) -> std::io::Result<Value> {
    match method {
        "item/commandExecution/requestApproval" | "item/fileChange/requestApproval" => {
            Ok(json!({"decision": if approve { "accept" } else { "decline" }}))
        }
        "item/tool/requestUserInput" => {
            let reply = text.unwrap_or(if approve { "yes" } else { "no" });
            let answers: serde_json::Map<String, Value> = params
                .get("questions")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|q| q.get("id").and_then(Value::as_str))
                .map(|id| (id.to_owned(), json!({"answers": [reply]})))
                .collect();
            Ok(json!({ "answers": answers }))
        }
        other => Err(std::io::Error::other(format!(
            "unsupported: Polylogue does not answer {other} requests"
        ))),
    }
}

/// What the daemon keeps for live-session acts.
pub(crate) struct Acts {
    mail: PathBuf,
    claims: Mutex<()>,
    /// Deliveries handed to a hook or a resumed run and not yet recorded:
    /// never handed out again while the daemon runs.
    claimed: Mutex<std::collections::BTreeSet<String>>,
    /// Sessions with a resumed run in progress, each with the runs queued
    /// behind it: one resumed run per session at a time, in order.
    resuming: Mutex<BTreeMap<String, VecDeque<ResumeRun>>>,
    hosted: Mutex<BTreeMap<String, HostedCodex>>,
    permission_hook: OnceLock<bool>,
    /// Codex's control socket and the program that proxies it.
    pub(crate) codex_socket: PathBuf,
    pub(crate) codex_program: PathBuf,
    daemon: OnceLock<Weak<crate::daemon::service::Daemon>>,
}

impl Acts {
    pub(crate) fn new(mail: PathBuf, codex_socket: PathBuf, codex_program: PathBuf) -> Self {
        Self {
            mail,
            claims: Mutex::new(()),
            claimed: Mutex::new(std::collections::BTreeSet::new()),
            resuming: Mutex::new(BTreeMap::new()),
            hosted: Mutex::new(BTreeMap::new()),
            permission_hook: OnceLock::new(),
            codex_socket,
            codex_program,
            daemon: OnceLock::new(),
        }
    }

    /// Binds the daemon the resume runner records outcomes through.
    pub(crate) fn bind(&self, daemon: Weak<crate::daemon::service::Daemon>) {
        let _ = self.daemon.set(daemon);
    }

    pub(crate) fn daemon(&self) -> Option<std::sync::Arc<crate::daemon::service::Daemon>> {
        self.daemon.get().and_then(Weak::upgrade)
    }

    pub(crate) fn mail(&self) -> &Path {
        &self.mail
    }

    pub(crate) fn claim_lock(&self) -> MutexGuard<'_, ()> {
        self.claims
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Marks deliveries handed out; returns those not handed out before.
    /// Callers hold the claim lock.
    pub(crate) fn take<T>(&self, pending: Vec<(String, T)>) -> Vec<(String, T)> {
        let mut claimed = self
            .claimed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        pending
            .into_iter()
            .filter(|(id, _)| claimed.insert(id.clone()))
            .collect()
    }

    /// Whether a delivery was handed out.
    pub(crate) fn handed_out(&self, id: &str) -> bool {
        self.claimed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(id)
    }

    /// The oldest request a hosted session waits on, by the app-server's
    /// request ID.
    pub(crate) fn host_request(&self, session: &SessionId) -> Option<String> {
        self.with_hosted(session, |h| {
            h.absorb();
            Ok(h.requests.front().map(|(id, _, _)| id.to_string()))
        })
        .ok()
        .flatten()
    }

    /// Whether the installed Claude Code asks the synchronous permission
    /// hook, probed once.
    fn permission_hook(&self) -> bool {
        *self.permission_hook.get_or_init(|| {
            let confirmed = crate::hook::HarnessVersion::parse(
                crate::hook::CONFIRMED_FROM
                    .iter()
                    .find(|(event, _)| *event == "PermissionRequest")
                    .map_or("", |(_, from)| from),
            );
            crate::hook::probe_version(crate::hook::Harness::ClaudeCode)
                .zip(confirmed)
                .is_some_and(|(installed, from)| installed >= from)
        })
    }

    pub(crate) fn is_hosted(&self, session: &SessionId) -> bool {
        self.hosted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&session.to_string())
    }

    /// Takes a hosted session in; its thread names the session.
    pub(crate) fn host(&self, hosted: HostedCodex) -> Option<SessionId> {
        let session = SessionId::parse(&format!("codex:{}", hosted.thread())).ok()?;
        self.hosted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(session.to_string(), hosted);
        Some(session)
    }

    fn with_hosted<T>(
        &self,
        session: &SessionId,
        act: impl FnOnce(&mut HostedCodex) -> std::io::Result<T>,
    ) -> Result<T, String> {
        let mut hosted = self
            .hosted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let connection = hosted
            .get_mut(&session.to_string())
            .ok_or_else(|| "the session is not hosted".to_owned())?;
        act(connection).map_err(|e| e.to_string())
    }

    pub(crate) fn host_send(&self, session: &SessionId, text: &str) -> Result<(), String> {
        self.with_hosted(session, |h| h.send(text))
    }

    pub(crate) fn host_answer(
        &self,
        session: &SessionId,
        approve: bool,
        text: Option<&str>,
    ) -> Result<(), String> {
        self.with_hosted(session, |h| h.answer(approve, text))
    }

    pub(crate) fn host_interrupt(&self, session: &SessionId) -> Result<(), String> {
        self.with_hosted(session, HostedCodex::interrupt)
    }

    /// Starts a resumed run, or queues it behind the session's run in
    /// progress; `true` when it started.
    pub(crate) fn resume(&self, run: ResumeRun) -> bool {
        {
            let mut resuming = self
                .resuming
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let key = run.session.to_string();
            if let Some(queue) = resuming.get_mut(&key) {
                queue.push_back(run);
                return false;
            }
            resuming.insert(key, VecDeque::new());
        }
        crate::daemon::service::acts::run_resume(self, run);
        true
    }

    /// The run that follows a finished one, if any is queued.
    pub(crate) fn resume_finished(&self, session: &SessionId) -> Option<ResumeRun> {
        let mut resuming = self
            .resuming
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let key = session.to_string();
        let next = resuming.get_mut(&key).and_then(VecDeque::pop_front);
        if next.is_none() {
            resuming.remove(&key);
        }
        next
    }
}

/// What the act channels know about a session's harness: whether
/// Polylogue hosts it, whether its evidence shows the hook installed,
/// whether the installed harness asks the permission hook, and whether it
/// has a headless resume.
pub(crate) fn probe(acts: &Acts, session: &SessionId, facts: &LiveFacts) -> Probe {
    let harness = session.origin().as_str().to_owned();
    Probe {
        hosted: acts.is_hosted(session),
        hooks: facts.root.is_some() || facts.hook_activity.is_some(),
        permission_hook: harness == "claude-code" && acts.permission_hook(),
        headless_resume: headless_resume(session, "").is_some(),
        harness,
    }
}
