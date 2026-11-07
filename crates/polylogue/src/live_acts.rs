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

    /// Connects again to a thread the shared app-server kept, after a
    /// daemon restart: `thread/resume` subscribes this connection to it.
    pub(crate) fn reattach(program: &Path, socket: &Path, thread: &str) -> std::io::Result<Self> {
        let mut connection = CodexConnection::open(program, socket)?;
        let resumed = connection.call("thread/resume", json!({"threadId": thread}))?;
        let turn = resumed
            .pointer("/thread/turns")
            .and_then(Value::as_array)
            .and_then(|turns| turns.last())
            .filter(|turn| turn["status"] == "inProgress")
            .and_then(|turn| turn["id"].as_str())
            .map(str::to_owned);
        Ok(Self {
            connection,
            thread: thread.to_owned(),
            turn,
            requests: VecDeque::new(),
        })
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
            None => self.connection.call(
                "turn/start",
                json!({"threadId": self.thread, "input": input}),
            ),
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
        self.connection
            .answer(&id, answer_result(&method, &params, approve, text)?)
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

/// A session Polylogue holds a control connection to.
pub(crate) enum Hosted {
    /// A Codex thread on the shared app-server.
    Codex(HostedCodex),
    /// A session a `polylogue host` process holds (Claude Code
    /// stream-json, ACP).
    Stdio(crate::hosting::HostedStdio),
}

impl Hosted {
    /// `true` when written now; `false` when it waits for the running
    /// ACP prompt, its outcome then reported when it is written.
    fn send(&mut self, text: &str, delivery: Option<&str>) -> std::io::Result<bool> {
        match self {
            Self::Codex(h) => h.send(text).map(|()| true),
            Self::Stdio(h) => h.send(text, delivery),
        }
    }

    fn answer(&mut self, approve: bool, text: Option<&str>) -> std::io::Result<()> {
        match self {
            Self::Codex(h) => h.answer(approve, text),
            Self::Stdio(h) => h.answer(approve, text),
        }
    }

    fn interrupt(&mut self) -> std::io::Result<()> {
        match self {
            Self::Codex(h) => h.interrupt(),
            Self::Stdio(h) => h.interrupt(),
        }
    }

    /// Whether the connection still reaches the session: a host that
    /// exited closed it.
    fn open(&self) -> bool {
        match self {
            Self::Codex(_) => true,
            Self::Stdio(h) => h.open(),
        }
    }

    fn request(&mut self) -> Option<String> {
        match self {
            Self::Codex(h) => {
                h.absorb();
                h.requests.front().map(|(id, _, _)| id.to_string())
            }
            Self::Stdio(h) => h.request(),
        }
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
    hosted: Mutex<BTreeMap<String, Hosted>>,
    permission_hook: OnceLock<bool>,
    /// The program that proxies Codex's control socket, and the socket.
    codex: Mutex<(PathBuf, PathBuf)>,
    /// The hosted sessions' records and host sockets.
    pub(crate) hosts: PathBuf,
    /// The hook spool, whose `registry/` hosts write status lines to.
    pub(crate) spool: PathBuf,
    daemon: OnceLock<Weak<crate::daemon::service::Daemon>>,
}

impl Acts {
    pub(crate) fn new(
        mail: PathBuf,
        hosts: PathBuf,
        spool: PathBuf,
        codex_socket: PathBuf,
        codex_program: PathBuf,
    ) -> Self {
        Self {
            hosts,
            spool,
            mail,
            claims: Mutex::new(()),
            claimed: Mutex::new(std::collections::BTreeSet::new()),
            resuming: Mutex::new(BTreeMap::new()),
            hosted: Mutex::new(BTreeMap::new()),
            permission_hook: OnceLock::new(),
            codex: Mutex::new((codex_program, codex_socket)),
            daemon: OnceLock::new(),
        }
    }

    /// The program that proxies Codex's control socket, and the socket.
    pub(crate) fn codex(&self) -> (PathBuf, PathBuf) {
        self.codex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Points the Codex client at a fixture app-server.
    #[cfg(test)]
    pub(crate) fn set_codex(&self, program: PathBuf, socket: PathBuf) {
        *self
            .codex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = (program, socket);
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

    /// Gives a delivery back to the mailbox.
    pub(crate) fn release(&self, id: &str) {
        self.claimed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(id);
    }

    /// Whether a delivery was handed out.
    pub(crate) fn handed_out(&self, id: &str) -> bool {
        self.claimed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(id)
    }

    /// The oldest request a hosted session waits on, by its protocol's
    /// request ID.
    pub(crate) fn host_request(&self, session: &SessionId) -> Option<String> {
        self.with_hosted(session, |h| Ok(h.request()))
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

    /// Whether Polylogue holds an open control connection to the session.
    pub(crate) fn is_hosted(&self, session: &SessionId) -> bool {
        self.hosted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&session.to_string())
            .is_some_and(Hosted::open)
    }

    /// Takes a hosted Codex thread in, recording it for reattachment; its
    /// thread names the session.
    pub(crate) fn host_codex(&self, hosted: HostedCodex) -> std::io::Result<SessionId> {
        let session = SessionId::parse(&format!("codex:{}", hosted.thread()))
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        crate::hosting::write_record(
            &self.hosts,
            &format!("codex-{}", hosted.thread()),
            &crate::hosting::Record {
                session: session.to_string(),
                protocol: polylogue_model::live::channel::HostProtocol::CodexAppServer,
                socket: None,
                protocol_session: Some(hosted.thread().to_owned()),
            },
        )?;
        self.hold(&session, Hosted::Codex(hosted));
        Ok(session)
    }

    /// Takes a hosted session's connection in.
    pub(crate) fn hold(&self, session: &SessionId, hosted: Hosted) {
        if let Hosted::Stdio(stdio) = &hosted {
            // A message that waited for the running prompt is recorded
            // when it is written, or `not_sent` when it is dropped.
            let daemon = self.daemon.get().cloned();
            stdio.on_sent(std::sync::Arc::new(move |delivery: &str, sent| {
                if let Some(daemon) = daemon.as_ref().and_then(Weak::upgrade) {
                    crate::daemon::service::acts::record_sent(&daemon, delivery, sent);
                }
            }));
        }
        self.hosted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(session.to_string(), hosted);
    }

    /// Connects again to every recorded hosted session, after a daemon
    /// start; records whose host or thread is gone are removed. Returns
    /// the sessions reattached.
    pub(crate) fn reattach(&self) -> Vec<SessionId> {
        use polylogue_model::live::channel::HostProtocol;
        let (codex_program, codex_socket) = self.codex();
        let mut attached = Vec::new();
        for (key, record) in crate::hosting::records(&self.hosts) {
            let Ok(session) = SessionId::parse(&record.session) else {
                crate::hosting::remove(&self.hosts, &key);
                continue;
            };
            let hosted = match record.protocol {
                HostProtocol::CodexAppServer => HostedCodex::reattach(
                    &codex_program,
                    &codex_socket,
                    record.protocol_session.as_deref().unwrap_or_default(),
                )
                .map(Hosted::Codex),
                _ => crate::hosting::HostedStdio::reattach(&record).map(Hosted::Stdio),
            };
            match hosted {
                Ok(hosted) => {
                    self.hold(&session, hosted);
                    attached.push(session);
                }
                // Only a host or thread that is gone loses its record; a
                // Codex app-server not up yet keeps it for the next start.
                Err(error) if gone(&error) => crate::hosting::remove(&self.hosts, &key),
                Err(_) => {}
            }
        }
        attached
    }

    /// The sessions Polylogue holds a connection to.
    pub(crate) fn hosted_sessions(&self) -> Vec<SessionId> {
        self.hosted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .keys()
            .filter_map(|k| SessionId::parse(k).ok())
            .collect()
    }

    /// Lets go of a session that ended: its connection and its record.
    pub(crate) fn release_host(&self, session: &SessionId) {
        self.hosted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&session.to_string());
        for (key, record) in crate::hosting::records(&self.hosts) {
            if record.session == session.to_string() {
                crate::hosting::remove(&self.hosts, &key);
            }
        }
    }

    fn with_hosted<T>(
        &self,
        session: &SessionId,
        act: impl FnOnce(&mut Hosted) -> std::io::Result<T>,
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

    /// Sends a message to a hosted session: `true` when written now,
    /// `false` when it waits for the running prompt (its outcome is then
    /// recorded under `delivery` when it is written or dropped).
    pub(crate) fn host_send(
        &self,
        session: &SessionId,
        text: &str,
        delivery: Option<&str>,
    ) -> Result<bool, String> {
        self.with_hosted(session, |h| h.send(text, delivery))
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
        self.with_hosted(session, Hosted::interrupt)
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
        permission_hook: polylogue_model::live::channel::harness(&harness)
            .is_some_and(|h| h.permission_hook.is_some())
            && acts.permission_hook(),
        headless_resume: polylogue_model::live::channel::harness(&harness)
            .is_some_and(|h| h.headless_resume),
        harness,
    }
}

/// Whether a reattachment failed because the host or the thread is gone,
/// not because something was briefly unavailable.
fn gone(error: &std::io::Error) -> bool {
    use std::io::ErrorKind;
    matches!(
        error.kind(),
        ErrorKind::NotFound | ErrorKind::ConnectionRefused
    ) || error.to_string().contains("refused thread/resume")
}
