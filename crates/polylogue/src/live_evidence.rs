//! The daemon's live evidence writers: the process watcher, the harness
//! registry readers and the commit watcher; and the live clock, which puts transitions that come
//! from the passage of time alone on the change feed. Both append lines to the hook spool's `process/` and
//! `registry/` directories, which the daemon then reads like any other
//! growing input, so everything they observe is a stored input that a
//! rebuild folds again.
//!
//! **Process watcher.** The roots are the harness processes that sessions'
//! evidence says are alive. Each root gets a pidfd, whose readiness is its
//! exit; the exit is written as `RootGone` at once. A root whose process is
//! already gone, or belongs to an earlier boot, is written gone when first
//! seen (an earlier boot's roots at that boot's end). Descendants are read
//! from `/proc/<pid>/task/*/children`, every second while the session has
//! an open tool call and every fifteen seconds otherwise; only changes are
//! written. With no roots nothing under `/proc` is read, and `environ` is
//! never opened.
//!
//! **Registry readers.** Claude Code keeps one `sessions/<pid>.json` per
//! process with its own status word; each change of a file is written as
//! one `Status` line carrying the file whole, plus the process identity.
//! Codex runs its sessions on a shared app-server; while its control socket
//! exists, each change of a loaded thread's status is written as one
//! `ThreadStatus` line carrying the status object whole.

use polylogue_model::SessionId;
use polylogue_model::live::ProcessIdentity;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::os::fd::OwnedFd;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant, SystemTime};

#[cfg(test)]
mod tests;

/// Descendants of a session with an open tool call are read this often.
const SAMPLE_OPEN_CALL: Duration = Duration::from_secs(1);
/// Descendants of other sessions are read this often: background shells
/// and servers outlive the calls that started them.
const SAMPLE_QUIET: Duration = Duration::from_secs(15);

/// Where the writers read and write.
#[derive(Debug, Clone)]
pub(crate) struct Paths {
    /// The hook spool directory; lines go to its `process/` and `registry/`.
    pub spool: PathBuf,
    /// Claude Code's per-process registry directory.
    pub claude_sessions: PathBuf,
    /// Codex's app-server control socket, and the program that proxies it.
    pub codex_socket: PathBuf,
    pub codex_program: PathBuf,
}

/// What `/proc/<pid>/stat` says about a process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Stat {
    pub ppid: u32,
    /// An exited process not yet reaped by its parent.
    pub zombie: bool,
    /// Clock ticks after boot (field 22).
    pub start: u64,
    pub comm: String,
}

/// Parses a `/proc/<pid>/stat` line. The command name is in parentheses
/// and may itself hold spaces and parentheses, so fields are counted from
/// the last `)`.
pub(crate) fn parse_stat(line: &str) -> Option<Stat> {
    let open = line.find('(')?;
    let close = line.rfind(')')?;
    let comm = line.get(open + 1..close)?.to_owned();
    let mut rest = line.get(close + 1..)?.split_whitespace();
    // Fields after the name: 3 state, 4 ppid, …, 22 start time.
    let zombie = rest.next()? == "Z";
    let ppid = rest.next()?.parse().ok()?;
    let start = rest.nth(17)?.parse().ok()?;
    Some(Stat {
        ppid,
        start,
        comm,
        zombie,
    })
}

fn stat(proc_root: &Path, pid: u32) -> Option<Stat> {
    parse_stat(&std::fs::read_to_string(proc_root.join(pid.to_string()).join("stat")).ok()?)
}

/// This boot's identity.
pub(crate) fn boot_id() -> Option<String> {
    std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .ok()
        .map(|s| s.trim().to_owned())
}

/// The process identity of `pid` now, on this boot.
pub(crate) fn identity(pid: u32) -> Option<ProcessIdentity> {
    Some(ProcessIdentity {
        boot: boot_id()?,
        pid,
        start: stat(Path::new("/proc"), pid)?.start,
    })
}

/// When this boot started, in milliseconds, from `btime` in `/proc/stat`.
fn boot_time_ms() -> Option<i64> {
    let text = std::fs::read_to_string("/proc/stat").ok()?;
    let seconds: i64 = text
        .lines()
        .find_map(|l| l.strip_prefix("btime "))?
        .trim()
        .parse()
        .ok()?;
    Some(seconds * 1000)
}

/// Every descendant of `root`, by pid, from the kernel's child lists.
pub(crate) fn descendants(proc_root: &Path, root: u32) -> BTreeMap<u32, Stat> {
    let mut out = BTreeMap::new();
    let mut queue = vec![root];
    while let Some(pid) = queue.pop() {
        let Ok(tasks) = std::fs::read_dir(proc_root.join(pid.to_string()).join("task")) else {
            continue;
        };
        for task in tasks.flatten() {
            let Ok(children) = std::fs::read_to_string(task.path().join("children")) else {
                continue;
            };
            for child in children
                .split_whitespace()
                .filter_map(|c| c.parse::<u32>().ok())
            {
                if child != root
                    && !out.contains_key(&child)
                    && let Some(s) = stat(proc_root, child)
                {
                    out.insert(child, s);
                    queue.push(child);
                }
            }
        }
    }
    out
}

/// One line for the spool: `route` is `process` or `registry`.
pub(crate) fn line(
    route: &str,
    event: &str,
    session: &SessionId,
    at: jiff::Timestamp,
    payload: Value,
) -> Value {
    json!({
        "at_ns": at.as_nanosecond().to_string(),
        "event": event,
        "payload": payload,
        "provider": session.origin().as_str(),
        "route": route,
        "session_id": session.native().to_string(),
    })
}

/// Appends lines to `<spool>/<route>/<boot>.ndjson` in one write.
pub(crate) fn append(
    spool: &Path,
    route: &str,
    boot: &str,
    lines: &[Value],
) -> std::io::Result<()> {
    if lines.is_empty() {
        return Ok(());
    }
    let mut bytes = Vec::new();
    for l in lines {
        serde_json::to_writer(&mut bytes, l)?;
        bytes.push(b'\n');
    }
    spool_file(spool, route, boot)?.write_all(&bytes)
}

fn spool_file(spool: &Path, route: &str, boot: &str) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    let dir = spool.join(route);
    std::fs::create_dir_all(&dir)?;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open(dir.join(format!("{boot}.ndjson")))
}

/// Retain only the unwritten suffix when an owned spool write fails.
fn write_pending(bytes: &mut Vec<u8>, writer: &mut impl Write) -> std::io::Result<()> {
    let mut written = 0;
    let result = loop {
        if written == bytes.len() {
            break Ok(());
        }
        match writer.write(&bytes[written..]) {
            Ok(0) => break Err(std::io::ErrorKind::WriteZero.into()),
            Ok(n) => written += n,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => break Err(error),
        }
    };
    bytes.drain(..written);
    result
}

fn identity_payload(identity: &ProcessIdentity) -> Value {
    json!({"pid": identity.pid, "start": identity.start, "boot_id": identity.boot})
}

/// One watched root.
struct Watched {
    root: ProcessIdentity,
    pidfd: Option<OwnedFd>,
    children: BTreeMap<(u32, u64), Stat>,
    next_sample: Instant,
    open_call: bool,
}

/// The process watcher's state between steps.
pub(crate) struct Watcher {
    proc_root: PathBuf,
    boot: String,
    boot_time_ms: Option<i64>,
    watched: BTreeMap<String, Watched>,
    /// Roots already queued gone, until their `RootGone` is committed and
    /// the evidence stops naming them.
    reported_gone: BTreeSet<(String, ProcessIdentity)>,
    /// Exact process observations not yet appended, including a partial line.
    pending: Vec<u8>,
}

impl Watcher {
    pub(crate) fn new(proc_root: PathBuf, boot: String, boot_time_ms: Option<i64>) -> Self {
        Self {
            proc_root,
            boot,
            boot_time_ms,
            watched: BTreeMap::new(),
            reported_gone: BTreeSet::new(),
            pending: Vec::new(),
        }
    }

    /// The process spool has one writer. Failed appends retain the original
    /// bytes, so later samples retry without changing times or line identities.
    fn append_process(&mut self, spool: &Path, lines: &[Value]) -> std::io::Result<()> {
        for line in lines {
            serde_json::to_writer(&mut self.pending, line)?;
            self.pending.push(b'\n');
        }
        if self.pending.is_empty() {
            return Ok(());
        }
        let mut file = spool_file(spool, "process", &self.boot)?;
        write_pending(&mut self.pending, &mut file)
    }

    /// Brings the watched set in line with the roots the evidence names
    /// and returns the lines for roots found gone and for new roots.
    pub(crate) fn update_roots(
        &mut self,
        roots: &[polylogue_store::LiveRoot],
        now: jiff::Timestamp,
    ) -> Vec<Value> {
        let mut out = Vec::new();
        let wanted: BTreeSet<&str> = roots.iter().map(|r| r.session.as_str()).collect();
        self.watched.retain(|s, _| wanted.contains(s.as_str()));
        let named: BTreeSet<(String, ProcessIdentity)> = roots
            .iter()
            .map(|r| (r.session.clone(), r.root.clone()))
            .collect();
        self.reported_gone.retain(|k| named.contains(k));
        for r in roots {
            if self
                .reported_gone
                .contains(&(r.session.clone(), r.root.clone()))
            {
                continue;
            }
            let Ok(session) = SessionId::parse(&r.session) else {
                continue;
            };
            if let Some(w) = self.watched.get_mut(&r.session)
                && w.root == r.root
            {
                w.open_call = r.open_call;
                continue;
            }
            self.watched.remove(&r.session);
            if !self.alive(&r.root) {
                self.reported_gone
                    .insert((r.session.clone(), r.root.clone()));
            }
            if r.root.boot != self.boot {
                // The process belonged to an earlier boot: it ended then.
                let at = self
                    .boot_time_ms
                    .and_then(|ms| jiff::Timestamp::from_millisecond(ms).ok())
                    .unwrap_or(now);
                out.push(line(
                    "process",
                    "RootGone",
                    &session,
                    at,
                    identity_payload(&r.root),
                ));
                continue;
            }
            if !self.alive(&r.root) {
                out.push(line(
                    "process",
                    "RootGone",
                    &session,
                    now,
                    identity_payload(&r.root),
                ));
                continue;
            }
            let pidfd = rustix::process::Pid::from_raw(r.root.pid as i32).and_then(|pid| {
                rustix::process::pidfd_open(pid, rustix::process::PidfdFlags::empty()).ok()
            });
            let mut started = identity_payload(&r.root);
            started["role"] = json!("harness");
            if let Some(s) = stat(&self.proc_root, r.root.pid) {
                started["ppid"] = json!(s.ppid);
                started["comm"] = json!(s.comm);
            }
            out.push(line("process", "Started", &session, now, started));
            self.watched.insert(
                r.session.clone(),
                Watched {
                    root: r.root.clone(),
                    pidfd,
                    children: BTreeMap::new(),
                    next_sample: Instant::now(),
                    open_call: r.open_call,
                },
            );
        }
        out
    }

    /// Whether `root` runs now on this boot: same pid and start time, and
    /// not an exited process waiting to be reaped.
    fn alive(&self, root: &ProcessIdentity) -> bool {
        root.boot == self.boot
            && stat(&self.proc_root, root.pid).is_some_and(|s| s.start == root.start && !s.zombie)
    }

    /// Lines for roots whose pidfd reports an exit, or, without pidfd
    /// support, whose process is gone at a sample.
    pub(crate) fn exited(&mut self, ready: &BTreeSet<String>, now: jiff::Timestamp) -> Vec<Value> {
        let mut out = Vec::new();
        let gone: Vec<String> = self
            .watched
            .iter()
            .filter(|(session, w)| {
                ready.contains(*session) || (w.pidfd.is_none() && !self.alive(&w.root))
            })
            .map(|(s, _)| s.clone())
            .collect();
        for session_key in gone {
            let w = self.watched.remove(&session_key).expect("listed above");
            self.reported_gone
                .insert((session_key.clone(), w.root.clone()));
            let Ok(session) = SessionId::parse(&session_key) else {
                continue;
            };
            for (pid, start) in w.children.keys() {
                out.push(line(
                    "process",
                    "Ended",
                    &session,
                    now,
                    json!({"pid": pid, "start": start, "boot_id": self.boot}),
                ));
            }
            out.push(line(
                "process",
                "RootGone",
                &session,
                now,
                identity_payload(&w.root),
            ));
        }
        out
    }

    /// Reads the descendants of every root whose sample is due and returns
    /// lines for what changed.
    pub(crate) fn sample(&mut self, now: jiff::Timestamp) -> Vec<Value> {
        let mut out = Vec::new();
        let instant = Instant::now();
        for (session_key, w) in &mut self.watched {
            if w.next_sample > instant {
                continue;
            }
            w.next_sample = instant
                + if w.open_call {
                    SAMPLE_OPEN_CALL
                } else {
                    SAMPLE_QUIET
                };
            let Ok(session) = SessionId::parse(session_key) else {
                continue;
            };
            let found: BTreeMap<(u32, u64), Stat> = descendants(&self.proc_root, w.root.pid)
                .into_iter()
                .map(|(pid, s)| ((pid, s.start), s))
                .collect();
            for ((pid, start), s) in &found {
                if !w.children.contains_key(&(*pid, *start)) {
                    let base = self.proc_root.join(pid.to_string());
                    let argv: Vec<String> = std::fs::read(base.join("cmdline"))
                        .map(|b| {
                            b.split(|c| *c == 0)
                                .filter(|a| !a.is_empty())
                                .map(|a| String::from_utf8_lossy(a).into_owned())
                                .collect()
                        })
                        .unwrap_or_default();
                    let cwd = std::fs::read_link(base.join("cwd"))
                        .ok()
                        .map(|p| p.to_string_lossy().into_owned());
                    out.push(line(
                        "process",
                        "Started",
                        &session,
                        now,
                        json!({"pid": pid, "start": start, "boot_id": self.boot, "ppid": s.ppid,
                               "comm": s.comm, "argv": argv, "cwd": cwd, "role": "child"}),
                    ));
                }
            }
            for (pid, start) in w.children.keys() {
                if !found.contains_key(&(*pid, *start)) {
                    out.push(line(
                        "process",
                        "Ended",
                        &session,
                        now,
                        json!({"pid": pid, "start": start, "boot_id": self.boot}),
                    ));
                }
            }
            w.children = found;
        }
        out
    }

    /// The time until the next due sample.
    fn next_wake(&self) -> Duration {
        let now = Instant::now();
        self.watched
            .values()
            .map(|w| w.next_sample.saturating_duration_since(now))
            .min()
            .unwrap_or(SAMPLE_QUIET)
            .min(SAMPLE_QUIET)
    }

    /// Waits up to `timeout` for a root's exit; returns the sessions whose
    /// pidfd is readable.
    pub(crate) fn wait(&self, timeout: Duration) -> BTreeSet<String> {
        use rustix::event::{PollFd, PollFlags, poll};
        let keys: Vec<&String> = self
            .watched
            .iter()
            .filter(|(_, w)| w.pidfd.is_some())
            .map(|(k, _)| k)
            .collect();
        let mut fds: Vec<PollFd<'_>> = self
            .watched
            .values()
            .filter_map(|w| w.pidfd.as_ref())
            .map(|fd| PollFd::new(fd, PollFlags::IN))
            .collect();
        if fds.is_empty() {
            return BTreeSet::new();
        }
        let timeout = rustix::event::Timespec::try_from(timeout).ok();
        if poll(&mut fds, timeout.as_ref()).is_err() {
            return BTreeSet::new();
        }
        fds.iter()
            .zip(keys)
            .filter(|(fd, _)| !fd.revents().is_empty())
            .map(|(_, k)| k.clone())
            .collect()
    }
}

/// The registry reader's state: each file's last modification seen.
pub(crate) struct Registry {
    dir: PathBuf,
    seen: BTreeMap<PathBuf, SystemTime>,
}

impl Registry {
    pub(crate) fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            seen: BTreeMap::new(),
        }
    }

    /// Lines for registry files that changed since the last read.
    pub(crate) fn read(
        &mut self,
        now: jiff::Timestamp,
        identity_of: impl Fn(u32) -> Option<ProcessIdentity>,
    ) -> Vec<Value> {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            self.seen.clear();
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut present = BTreeSet::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            present.insert(path.clone());
            let Ok(modified) = entry.metadata().and_then(|m| m.modified()) else {
                continue;
            };
            if self.seen.get(&path) == Some(&modified) {
                continue;
            }
            self.seen.insert(path.clone(), modified);
            let Some(registry) = std::fs::read(&path)
                .ok()
                .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            else {
                continue;
            };
            let (Some(native), Some(status)) = (
                registry.get("sessionId").and_then(Value::as_str),
                registry.get("status").and_then(Value::as_str),
            ) else {
                continue;
            };
            let Ok(session) = SessionId::parse(&format!("claude-code:{native}")) else {
                continue;
            };
            let mut payload = json!({"status": status, "registry": registry});
            if let Some(identity) = registry
                .get("pid")
                .and_then(Value::as_u64)
                .and_then(|pid| u32::try_from(pid).ok())
                .and_then(&identity_of)
            {
                payload["pid"] = json!(identity.pid);
                payload["start"] = json!(identity.start);
                payload["boot_id"] = json!(identity.boot);
            }
            out.push(line("registry", "Status", &session, now, payload));
        }
        self.seen.retain(|p, _| present.contains(p));
        out
    }
}

/// How long the Codex status reader waits for one answer from the
/// app-server before it drops the connection and connects again later.
const CODEX_ANSWER: Duration = Duration::from_secs(2);

/// A JSON-RPC connection to Codex's shared app-server through `codex
/// app-server proxy`, which carries newline-delimited JSON between its
/// stdio and the daemon's control socket.
pub(crate) struct CodexConnection {
    child: std::process::Child,
    input: std::process::ChildStdin,
    lines: Receiver<Value>,
    next_id: u64,
    /// Notifications and server requests that arrived while an answer was
    /// awaited, in arrival order.
    pending: Vec<Value>,
}

impl CodexConnection {
    /// Starts `program app-server proxy --sock SOCKET` and initializes the
    /// session.
    pub(crate) fn open(program: &Path, socket: &Path) -> std::io::Result<Self> {
        use std::io::BufRead;
        use std::process::{Command, Stdio};
        let mut child = Command::new(program)
            .args(["app-server", "proxy", "--sock"])
            .arg(socket)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let input = child.stdin.take().expect("piped stdin");
        let output = child.stdout.take().expect("piped stdout");
        let (sender, lines) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("polylogue-codex".into())
            .spawn(move || {
                for line in std::io::BufReader::new(output).lines() {
                    let Ok(line) = line else { return };
                    if let Ok(value) = serde_json::from_str::<Value>(&line)
                        && sender.send(value).is_err()
                    {
                        return;
                    }
                }
            })?;
        let mut connection = Self {
            child,
            input,
            lines,
            next_id: 0,
            pending: Vec::new(),
        };
        connection.call(
            "initialize",
            json!({"clientInfo": {"name": "polylogue", "version": env!("CARGO_PKG_VERSION")}}),
        )?;
        connection.notify("initialized", Value::Null)?;
        Ok(connection)
    }

    fn send(&mut self, message: &Value) -> std::io::Result<()> {
        let mut bytes = serde_json::to_vec(message)?;
        bytes.push(b'\n');
        self.input.write_all(&bytes)?;
        self.input.flush()
    }

    pub(crate) fn notify(&mut self, method: &str, params: Value) -> std::io::Result<()> {
        let mut message = json!({"jsonrpc": "2.0", "method": method});
        if !params.is_null() {
            message["params"] = params;
        }
        self.send(&message)
    }

    /// Sends a request and waits for its answer; messages that arrive
    /// meanwhile are kept for [`Self::take_pending`].
    pub(crate) fn call(&mut self, method: &str, params: Value) -> std::io::Result<Value> {
        self.next_id += 1;
        let id = self.next_id;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))?;
        let deadline = Instant::now() + CODEX_ANSWER;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let message = self.lines.recv_timeout(left).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!("codex app-server did not answer {method}"),
                )
            })?;
            if message.get("id").and_then(Value::as_u64) == Some(id)
                && message.get("method").is_none()
            {
                if let Some(error) = message.get("error") {
                    return Err(std::io::Error::other(format!(
                        "codex app-server refused {method}: {error}"
                    )));
                }
                return Ok(message.get("result").cloned().unwrap_or(Value::Null));
            }
            self.pending.push(message);
        }
    }

    /// Answers a request the app-server sent.
    pub(crate) fn answer(&mut self, id: &Value, result: Value) -> std::io::Result<()> {
        self.send(&json!({"jsonrpc": "2.0", "id": id, "result": result}))
    }

    /// The next notification or server request, waiting up to `timeout`;
    /// `Err` once the app-server closed the connection.
    pub(crate) fn next(&mut self, timeout: Duration) -> Result<Option<Value>, ()> {
        if !self.pending.is_empty() {
            return Ok(Some(self.pending.remove(0)));
        }
        match self.lines.recv_timeout(timeout) {
            Ok(message) => Ok(Some(message)),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(()),
        }
    }

    /// The notifications and server requests received so far.
    pub(crate) fn take_pending(&mut self) -> Vec<Value> {
        while let Ok(message) = self.lines.try_recv() {
            self.pending.push(message);
        }
        std::mem::take(&mut self.pending)
    }
}

impl Drop for CodexConnection {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The word a Codex thread status states: the first active flag
/// (`waitingOnApproval`, `waitingOnUserInput`), else its type (`active`,
/// `idle`, `systemError`, `notLoaded`).
pub(crate) fn codex_status_word(status: &Value) -> Option<String> {
    status
        .get("activeFlags")
        .and_then(Value::as_array)
        .and_then(|flags| flags.first())
        .or_else(|| status.get("type"))
        .and_then(Value::as_str)
        .map(str::to_owned)
}

/// The Codex status reader: while Codex's app-server control socket
/// exists, it lists the loaded threads and their statuses and writes one
/// registry line per thread whose status changed, the status object whole.
/// A thread that leaves the loaded list is written `notLoaded`.
pub(crate) struct CodexStatus {
    program: PathBuf,
    socket: PathBuf,
    connection: Option<CodexConnection>,
    seen: BTreeMap<String, Value>,
}

impl CodexStatus {
    pub(crate) fn new(program: PathBuf, socket: PathBuf) -> Self {
        Self {
            program,
            socket,
            connection: None,
            seen: BTreeMap::new(),
        }
    }

    /// Lines for thread statuses that changed since the last read.
    pub(crate) fn read(&mut self, now: jiff::Timestamp) -> Vec<Value> {
        if !self.socket.exists() {
            self.connection = None;
            return self.forget_all(now);
        }
        if self.connection.is_none() {
            self.connection = CodexConnection::open(&self.program, &self.socket).ok();
        }
        let Some(connection) = self.connection.as_mut() else {
            return Vec::new();
        };
        match Self::statuses(connection) {
            Ok(current) => self.changes(current, now),
            Err(_) => {
                // A dead or silent app-server: connect again on the next read.
                self.connection = None;
                Vec::new()
            }
        }
    }

    /// Every loaded thread's status now, by thread ID.
    fn statuses(connection: &mut CodexConnection) -> std::io::Result<BTreeMap<String, Value>> {
        let mut threads = Vec::new();
        let mut cursor = Value::Null;
        loop {
            let page = connection.call("thread/loaded/list", json!({"cursor": cursor}))?;
            threads.extend(
                page.get("data")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .map(str::to_owned),
            );
            match page.get("nextCursor") {
                Some(next) if next.is_string() => cursor = next.clone(),
                _ => break,
            }
        }
        let mut out = BTreeMap::new();
        for thread in threads {
            let read = connection.call(
                "thread/read",
                json!({"threadId": thread, "includeTurns": false}),
            )?;
            if let Some(status) = read.pointer("/thread/status") {
                out.insert(thread, status.clone());
            }
        }
        // Status changes announced meanwhile are already in the reads.
        connection.take_pending();
        Ok(out)
    }

    fn changes(&mut self, current: BTreeMap<String, Value>, now: jiff::Timestamp) -> Vec<Value> {
        let mut out = Vec::new();
        for (thread, status) in &current {
            if self.seen.get(thread) != Some(status)
                && let Some(line) = codex_line(thread, status, now)
            {
                out.push(line);
            }
        }
        for thread in self.seen.keys() {
            if !current.contains_key(thread)
                && let Some(line) = codex_line(thread, &json!({"type": "notLoaded"}), now)
            {
                out.push(line);
            }
        }
        self.seen = current;
        out
    }

    /// Waits up to `timeout` for a `thread/status/changed` notification and
    /// returns its line when the status changed.
    pub(crate) fn next_change(&mut self, timeout: Duration, now: jiff::Timestamp) -> Option<Value> {
        let connection = self.connection.as_mut()?;
        let message = match connection.next(timeout) {
            Ok(message) => message?,
            Err(()) => {
                self.connection = None;
                return None;
            }
        };
        if message.get("method").and_then(Value::as_str) != Some("thread/status/changed") {
            return None;
        }
        let params = message.get("params")?;
        let thread = params.get("threadId")?.as_str()?.to_owned();
        let status = params.get("status")?.clone();
        if self.seen.get(&thread) == Some(&status) {
            return None;
        }
        let line = codex_line(&thread, &status, now);
        self.seen.insert(thread, status);
        line
    }

    fn forget_all(&mut self, now: jiff::Timestamp) -> Vec<Value> {
        self.changes(BTreeMap::new(), now)
    }
}

fn codex_line(thread: &str, status: &Value, now: jiff::Timestamp) -> Option<Value> {
    let session = SessionId::parse(&format!("codex:{thread}")).ok()?;
    let word = codex_status_word(status)?;
    Some(line(
        "registry",
        "ThreadStatus",
        &session,
        now,
        json!({"status": word, "registry": {"threadId": thread, "status": status}}),
    ))
}

/// Runs git in `top` and returns its output when it succeeds.
fn git(top: &Path, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new("git")
        .arg("--no-optional-locks")
        .arg("-C")
        .arg(top)
        .args(args)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8(output.stdout).ok())
        .flatten()
}

/// The reflog of a worktree's HEAD, which Git appends to on every commit:
/// `.git/logs/HEAD`, or for a linked worktree, in the directory its `.git`
/// file names.
fn head_log(top: &Path) -> Option<PathBuf> {
    let dot_git = top.join(".git");
    if dot_git.is_dir() {
        return Some(dot_git.join("logs/HEAD"));
    }
    let text = std::fs::read_to_string(&dot_git).ok()?;
    let dir = PathBuf::from(text.trim().strip_prefix("gitdir:")?.trim());
    Some(
        if dir.is_absolute() {
            dir
        } else {
            top.join(dir)
        }
        .join("logs/HEAD"),
    )
}

/// One watched worktree: its reflog's modification time and HEAD when last
/// read.
struct Worktree {
    modified: Option<SystemTime>,
    head: Option<String>,
}

/// The commit watcher: for the worktrees of running sessions with a
/// footprint, it notices HEAD moving (the reflog changes; one `stat` per
/// worktree per step), lists the new commits with their files, and writes
/// one `Commit` line per session whose footprint a commit contains, naming
/// those files. The fold takes them out of the footprint. No command text
/// is read; the files come from Git.
#[derive(Default)]
pub(crate) struct CommitWatcher {
    worktrees: BTreeMap<PathBuf, Worktree>,
}

impl CommitWatcher {
    pub(crate) fn read(
        &mut self,
        footprints: &[polylogue_store::LiveFootprint],
        now: jiff::Timestamp,
    ) -> Vec<Value> {
        let mut by_worktree: BTreeMap<PathBuf, Vec<&polylogue_store::LiveFootprint>> =
            BTreeMap::new();
        for footprint in footprints {
            if let Some(top) = git_identity::repository_of(Path::new(&footprint.cwd)) {
                by_worktree.entry(top).or_default().push(footprint);
            }
        }
        self.worktrees
            .retain(|top, _| by_worktree.contains_key(top));
        let mut out = Vec::new();
        for (top, sessions) in by_worktree {
            let modified = head_log(&top)
                .and_then(|log| std::fs::metadata(log).ok())
                .and_then(|m| m.modified().ok());
            let Some(seen) = self.worktrees.get_mut(&top) else {
                let head = git(&top, &["rev-parse", "HEAD"]).map(|h| h.trim().to_owned());
                self.worktrees.insert(top, Worktree { modified, head });
                continue;
            };
            if seen.modified == modified {
                continue;
            }
            seen.modified = modified;
            let head = git(&top, &["rev-parse", "HEAD"]).map(|h| h.trim().to_owned());
            if head == seen.head {
                continue;
            }
            let commits = match (&seen.head, &head) {
                (Some(old), Some(new)) => git(
                    &top,
                    &[
                        "log",
                        "--format=%x00%H %ct",
                        "--name-only",
                        &format!("{old}..{new}"),
                    ],
                )
                .map(|text| parse_commits(&text))
                .unwrap_or_default(),
                _ => Vec::new(),
            };
            seen.head = head;
            for (commit, seconds, files) in commits {
                let at = jiff::Timestamp::from_second(seconds).unwrap_or(now);
                let files: BTreeSet<PathBuf> = files.iter().map(|f| top.join(f)).collect();
                for footprint in &sessions {
                    let cwd = Path::new(&footprint.cwd);
                    let contained: Vec<&String> = footprint
                        .files
                        .iter()
                        .filter(|path| {
                            let path = Path::new(path.as_str());
                            files.contains(path) || files.contains(&cwd.join(path))
                        })
                        .collect();
                    if contained.is_empty() {
                        continue;
                    }
                    let Ok(session) = SessionId::parse(&footprint.session) else {
                        continue;
                    };
                    out.push(line(
                        "git",
                        "Commit",
                        &session,
                        at,
                        json!({"commit": commit, "worktree": top, "files": contained}),
                    ));
                }
            }
        }
        out
    }
}

/// `git log --format=%x00%H %ct --name-only` output as (commit, commit
/// time in seconds, files).
pub(crate) fn parse_commits(text: &str) -> Vec<(String, i64, Vec<String>)> {
    text.split('\0')
        .filter_map(|chunk| {
            let mut lines = chunk.lines().map(str::trim).filter(|l| !l.is_empty());
            let (commit, seconds) = lines.next()?.split_once(' ')?;
            Some((
                commit.to_owned(),
                seconds.parse().ok()?,
                lines.map(str::to_owned).collect(),
            ))
        })
        .collect()
}

/// The live clock: a session whose `live_until` has passed changed state
/// with no new evidence, and each such transition is put on the change
/// feed once, so followers see it without polling. A reading at `now`
/// gives every instant after `now`; the clock remembers the latest
/// reading and hands out each remembered instant once it has passed.
#[derive(Default)]
pub(crate) struct LiveClock {
    pending: BTreeMap<String, i64>,
}

impl LiveClock {
    /// Remembers every session's next `until`, read at some instant.
    pub(crate) fn remember(&mut self, untils: &[(String, i64)]) {
        self.pending = untils.iter().cloned().collect();
    }

    /// Takes the sessions whose remembered instant has passed.
    pub(crate) fn take_due(&mut self, now_ms: i64) -> Vec<String> {
        let due: Vec<String> = self
            .pending
            .iter()
            .filter(|(_, until)| **until <= now_ms)
            .map(|(session, _)| session.clone())
            .collect();
        for session in &due {
            self.pending.remove(session);
        }
        due
    }

    /// How long until the earliest remembered instant.
    pub(crate) fn wait(&self, now_ms: i64) -> Option<Duration> {
        self.pending
            .values()
            .min()
            .map(|n| Duration::from_millis(u64::try_from(n - now_ms).unwrap_or(0)))
    }
}

/// How often the readers re-read the archive's live sessions when no
/// commit has changed them.
const REFRESH: Duration = Duration::from_secs(15);

/// The Codex status reader's own thread: while the control socket exists
/// it reads the loaded threads' statuses every second and takes
/// `thread/status/changed` notifications as they come between readings;
/// a failed connection is tried again a second later.
fn run_codex(
    mut codex: CodexStatus,
    spool: PathBuf,
    boot: String,
    clock: polylogue_model::WallClock,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    use std::sync::atomic::Ordering;
    while !stop.load(Ordering::Acquire) {
        let lines = codex.read(clock.now());
        let _ = append(&spool, "registry", &boot, &lines);
        let deadline = Instant::now() + SAMPLE_OPEN_CALL;
        loop {
            if stop.load(Ordering::Acquire) {
                return;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            match codex.next_change(left.min(SAMPLE_OPEN_CALL), clock.now()) {
                Some(line) => {
                    let _ = append(&spool, "registry", &boot, &[line]);
                }
                None if codex.connection.is_none() => {
                    std::thread::sleep(left.min(SAMPLE_OPEN_CALL));
                }
                None => {}
            }
        }
    }
}

/// Runs the writers and the live clock until `stop` fires.
pub(crate) fn run(
    daemon: std::sync::Arc<crate::daemon::service::Daemon>,
    paths: Paths,
    clock: polylogue_model::WallClock,
    stop: Receiver<()>,
) {
    let Some(boot) = boot_id() else {
        return;
    };
    let codex_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let codex = {
        let reader = CodexStatus::new(paths.codex_program.clone(), paths.codex_socket.clone());
        let (spool, boot, clock, stop) = (
            paths.spool.clone(),
            boot.clone(),
            clock.clone(),
            std::sync::Arc::clone(&codex_stop),
        );
        std::thread::Builder::new()
            .name("polylogue-codex-status".into())
            .spawn(move || run_codex(reader, spool, boot, clock, stop))
            .ok()
    };
    let mut watcher = Watcher::new(PathBuf::from("/proc"), boot.clone(), boot_time_ms());
    let mut registry = Registry::new(paths.claude_sessions.clone());
    let mut live_clock = LiveClock::default();
    let mut commit_watcher = CommitWatcher::default();
    let commits = daemon.writes.as_ref().map(|w| w.completion_commits());
    let mut roots = Vec::new();
    let mut footprints = Vec::new();
    let mut refreshed: Option<Instant> = None;
    loop {
        let now = clock.now();
        let now_ms = now.as_millisecond();
        // The archive's live sessions change only with a commit (or, for
        // a stale session, a day later): read them again after a commit.
        let committed = commits.as_ref().is_some_and(|c| {
            let mut any = false;
            while c.try_recv().is_ok() {
                any = true;
            }
            any
        });
        let due = live_clock.take_due(now_ms);
        if committed || refreshed.is_none_or(|at| at.elapsed() >= REFRESH) {
            if let Ok(reader) = daemon.reads.reader() {
                roots = reader.live_roots().unwrap_or_default();
                footprints = reader.live_footprints(now_ms).unwrap_or_default();
                if let Ok(untils) = reader.live_untils(now_ms) {
                    live_clock.remember(&untils);
                }
            }
            refreshed = Some(Instant::now());
            crate::daemon::service::acts::settle_ended(&daemon, now_ms, false);
        }
        if let Some(writes) = daemon.writes.as_ref()
            && !due.is_empty()
        {
            let _ = writes.live_clock(due);
        }
        let mut process_lines = watcher.update_roots(&roots, now);
        process_lines.extend(watcher.sample(now));
        if let Err(error) = watcher.append_process(&paths.spool, &process_lines) {
            eprintln!("polylogue: process spool append failed; retained for retry: {error}");
        }
        let _ = append(
            &paths.spool,
            "registry",
            &boot,
            &registry.read(now, identity),
        );
        let _ = append(
            &paths.spool,
            "git",
            &boot,
            &commit_watcher.read(&footprints, now),
        );
        // Wake for the next sample, a root's exit, the next clock
        // transition, new roots (at most a second later) or the stop signal.
        let wake = watcher
            .next_wake()
            .min(SAMPLE_OPEN_CALL)
            .min(live_clock.wait(now_ms).unwrap_or(SAMPLE_OPEN_CALL));
        let polled = watcher.watched.values().any(|w| w.pidfd.is_some());
        let ready = if polled {
            watcher.wait(wake)
        } else {
            BTreeSet::new()
        };
        let gone = watcher.exited(&ready, clock.now());
        if let Err(error) = watcher.append_process(&paths.spool, &gone) {
            eprintln!("polylogue: process spool append failed; retained for retry: {error}");
        }
        match stop.recv_timeout(if polled { Duration::ZERO } else { wake }) {
            Err(RecvTimeoutError::Timeout) => {}
            Ok(()) | Err(RecvTimeoutError::Disconnected) => {
                codex_stop.store(true, std::sync::atomic::Ordering::Release);
                if let Some(codex) = codex {
                    let _ = codex.join();
                }
                return;
            }
        }
    }
}
