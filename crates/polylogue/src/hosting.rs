//! Hosted sessions on stdio protocols: Claude Code's stream-json
//! (`claude -p --input-format stream-json --output-format stream-json`)
//! and the Agent Client Protocol of Gemini CLI and Hermes.
//!
//! **`polylogue internal host`.** These protocols run over the agent's stdin and
//! stdout, so the process that holds those pipes keeps the agent alive. A
//! small per-session `polylogue internal host` process holds them instead of the
//! daemon, so a daemon restart never ends an agent. The host relays lines
//! between the agent and whichever daemon connection its socket has. What
//! the agent writes while no daemon is connected waits in the host and is
//! sent first when a daemon connects again; requests the agent asked and
//! nobody answered (permission prompts) are sent again to each new
//! connection, after a first line stating the session's status. The host
//! also writes the session's status changes to the registry spool
//! (`HostedStatus`, and `HostedEnded` when the agent exits), naming the
//! agent process it started, which is how the live-state update knows the
//! session's root. It reads no archive and keeps no other state; it exits
//! when the agent exits, removing its socket and its record.
//!
//! **Records.** Each hosted session has a record `hosts/<key>.json` in the
//! archive's runtime directory: the session, harness, protocol and, for
//! stdio protocols, the host's socket `hosts/<key>.sock`; for Codex, the
//! thread on its shared app-server, which keeps the thread itself. At
//! start the daemon reattaches to every recorded session.
//!
//! **The daemon's side** ([`HostedStdio`]) speaks the protocol: a message is
//! a user message (Claude) or a prompt (ACP; queued until the running
//! prompt finishes, since ACP has no steering), an answer replies to the
//! oldest open permission request, and an interrupt is Claude's
//! `interrupt` control request or ACP's `session/cancel`.

use polylogue_model::LockExt as _;
use polylogue_model::SessionId;
use polylogue_model::live::channel::HostProtocol;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(test)]
mod tests;

/// How long the daemon waits on a hosted session's host: for its socket to
/// appear after spawning, for its hello line on connecting, and for the
/// answer to one protocol call.
const HOST_WAIT: Duration = Duration::from_secs(10);

/// The hosted-session directory of the archive at `root`.
pub fn dir(paths: &crate::config::Paths, root: &Path) -> PathBuf {
    paths.runtime_dir(root).join("hosts")
}

/// What the daemon needs to reattach to a hosted session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Record {
    pub session: String,
    pub protocol: HostProtocol,
    /// The `polylogue internal host` socket of a stdio protocol.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub socket: Option<PathBuf>,
    /// The protocol's own session or thread ID, when it differs from the
    /// session's native ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol_session: Option<String>,
}

/// The record file of a host key.
pub(crate) fn record_path(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("{key}.json"))
}

pub(crate) fn write_record(dir: &Path, key: &str, record: &Record) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = record_path(dir, key);
    let partial = dir.join(format!(".{key}.json.partial"));
    std::fs::write(&partial, serde_json::to_vec(record)?)?;
    std::fs::rename(partial, path)
}

/// Every record, with its key.
pub(crate) fn records(dir: &Path) -> Vec<std::io::Result<(String, Record)>> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(error) => return vec![Err(error)],
    };
    let mut out: Vec<_> = entries
        .filter_map(|entry| {
            let path = match entry {
                Ok(entry) => entry.path(),
                Err(error) => return Some(Err(error)),
            };
            let key = path
                .file_name()?
                .to_str()?
                .strip_suffix(".json")
                .filter(|k| !k.starts_with('.'))?
                .to_owned();
            Some(
                (|| {
                    let record = serde_json::from_slice(&std::fs::read(&path)?)?;
                    Ok((key, record))
                })()
                .map_err(|error: std::io::Error| {
                    std::io::Error::new(error.kind(), format!("{}: {error}", path.display()))
                }),
            )
        })
        .collect();
    out.sort_by(|a, b| {
        a.as_ref()
            .ok()
            .map(|(key, _)| key)
            .cmp(&b.as_ref().ok().map(|(key, _)| key))
    });
    out
}

/// Removes a host's record and socket.
pub(crate) fn remove(dir: &Path, key: &str) {
    let _ = std::fs::remove_file(record_path(dir, key));
    let _ = std::fs::remove_file(dir.join(format!("{key}.sock")));
}

/// The command line `polylogue internal host` runs for a harness, from its
/// declaration; `None` for a harness without one, or without the session ID
/// its arguments name.
pub fn agent_command(harness: &str, session_id: Option<&str>) -> Option<Vec<String>> {
    let harness = polylogue_model::live::channel::harness(harness)?;
    harness.command(harness.host_arguments?, session_id, None)
}

/// The status a hosted session's protocol traffic states, kept by the
/// host: the word written to the registry spool (`busy`, `idle`,
/// `blocked`), and the agent's requests nobody has answered yet.
#[derive(Debug, Clone)]
pub(crate) struct Tracker {
    protocol: HostProtocol,
    word: &'static str,
    /// Turns in flight: ACP prompt request IDs, or one placeholder per Claude
    /// user message awaiting its result.
    turns: Vec<Value>,
    /// The agent's unanswered requests: (request ID, line).
    open: Vec<(Value, Vec<u8>)>,
}

impl Tracker {
    pub(crate) fn new(protocol: HostProtocol) -> Self {
        Self {
            protocol,
            word: "idle",
            turns: Vec::new(),
            open: Vec::new(),
        }
    }

    pub(crate) fn word(&self) -> &'static str {
        self.word
    }

    /// Turns in flight, the running one included.
    pub(crate) fn turns(&self) -> usize {
        self.turns.len()
    }

    fn update_word(&mut self) -> Option<&'static str> {
        let word = if !self.open.is_empty() {
            "blocked"
        } else if self.busy() {
            "busy"
        } else {
            "idle"
        };
        (word != self.word).then(|| {
            self.word = word;
            word
        })
    }

    /// A turn runs: an ACP prompt awaits its answer, or a Claude user
    /// message awaits its result.
    fn busy(&self) -> bool {
        !self.turns.is_empty()
    }

    /// A line the daemon sent the agent; the new status word if it changed.
    pub(crate) fn client_sent(&mut self, message: &Value) -> Option<&'static str> {
        match self.protocol {
            HostProtocol::ClaudeStreamJson => match message["type"].as_str() {
                Some("user") => {
                    self.turns.push(Value::Bool(true));
                }
                Some("control_response") => {
                    let id = &message["response"]["request_id"];
                    self.open.retain(|(open, _)| open != id);
                }
                _ => {}
            },
            _ => {
                if message["method"] == "session/prompt" {
                    self.turns.push(message["id"].clone());
                } else if message.get("method").is_none() {
                    let id = &message["id"];
                    self.open.retain(|(open, _)| open != id);
                }
            }
        }
        self.update_word()
    }

    /// A line the agent sent; the new status word if it changed.
    pub(crate) fn agent_sent(&mut self, message: &Value, line: &[u8]) -> Option<&'static str> {
        match self.protocol {
            HostProtocol::ClaudeStreamJson => match message["type"].as_str() {
                Some("control_request") => {
                    self.open
                        .push((message["request_id"].clone(), line.to_vec()));
                }
                Some("result") => {
                    // One turn finished; messages sent during it run next,
                    // and its permission requests are moot.
                    if !self.turns.is_empty() {
                        self.turns.remove(0);
                    }
                    self.open.clear();
                }
                Some("control_cancel_request") => {
                    let id = &message["request_id"];
                    self.open.retain(|(open, _)| open != id);
                }
                _ => {}
            },
            _ => {
                if message.get("method").is_some() {
                    if message.get("id").is_some() {
                        self.open.push((message["id"].clone(), line.to_vec()));
                    }
                } else {
                    let id = &message["id"];
                    let before = self.turns.len();
                    self.turns.retain(|p| p != id);
                    if self.turns.len() < before && self.turns.is_empty() {
                        // The finished prompt's requests are moot.
                        self.open.clear();
                    }
                }
            }
        }
        self.update_word()
    }

    /// The agent's requests nobody has answered, oldest first.
    pub(crate) fn unanswered(&self) -> impl Iterator<Item = &[u8]> {
        self.open.iter().map(|(_, line)| line.as_slice())
    }
}

/// What `polylogue internal host` is asked to run.
#[derive(Debug, Clone)]
pub struct HostConfig {
    pub harness: String,
    pub protocol: HostProtocol,
    pub argv: Vec<String>,
    pub cwd: PathBuf,
    pub socket: PathBuf,
    /// The hook spool, whose `registry/` the status lines go to.
    pub spool: PathBuf,
    /// The session's native ID when it is known before the agent starts.
    pub session_id: Option<String>,
}

struct Relay {
    client: Option<UnixStream>,
    backlog: VecDeque<Vec<u8>>,
    tracker: Tracker,
    session: Option<SessionId>,
    spool: crate::live_events::PendingLines,
    stopping: Arc<AtomicBool>,
}

/// The first line a host sends each new connection.
fn hello(relay: &Relay) -> Vec<u8> {
    let mut line = serde_json::to_vec(&json!({"polylogue_host": {
        "status": relay.tracker.word(),
        "turns": relay.tracker.turns(),
    }}))
    .unwrap_or_default();
    line.push(b'\n');
    line
}

fn status_line(
    relay: &mut Relay,
    config: &HostConfig,
    session: &SessionId,
    identity: Option<&polylogue_model::live::ProcessIdentity>,
    event: &str,
    word: Option<&str>,
) {
    if relay.stopping.load(Ordering::Acquire) && event != "HostedEnded" {
        return;
    }
    let mut payload = json!({});
    if let Some(word) = word {
        let field = if event == "HostedEnded" {
            "detail"
        } else {
            "status"
        };
        payload[field] = json!(word);
    }
    payload["protocol"] = json!(config.protocol.as_str());
    if let Some(identity) = identity {
        payload["pid"] = json!(identity.pid);
        payload["start"] = json!(identity.start);
        payload["boot_id"] = json!(identity.boot);
    }
    let line =
        crate::live_events::line("registry", event, session, jiff::Timestamp::now(), payload);
    let boot = crate::live_events::boot_id().unwrap_or_else(|| "unknown".into());
    let _ = relay.spool.append(&config.spool, &boot, &[line]);
}

/// Writes out the spool's pending lines, adding none.
fn flush_spool(relay: &mut Relay, config: &HostConfig) -> std::io::Result<()> {
    let boot = crate::live_events::boot_id().unwrap_or_else(|| "unknown".into());
    relay.spool.append(&config.spool, &boot, &[])
}

/// Called under the relay lock, as is the stop decision and the final flush.
fn retry_status(relay: &mut Relay, config: &HostConfig) {
    if relay.stopping.load(Ordering::Acquire) {
        return;
    }
    let _ = flush_spool(relay, config);
}

/// The session a protocol line names, once the agent has said it.
fn session_named(config: &HostConfig, message: &Value) -> Option<SessionId> {
    let native = match config.protocol {
        HostProtocol::ClaudeStreamJson => message["session_id"].as_str(),
        _ => message["result"]["sessionId"].as_str(),
    }?;
    SessionId::parse(&format!("{}:{native}", config.harness)).ok()
}

/// Reads the end of standard error without storing an unbounded line or stream.
pub(crate) fn stderr_tail(mut input: impl std::io::Read) -> std::io::Result<String> {
    const BYTES: usize = 16 * 1024;
    let mut tail = VecDeque::with_capacity(BYTES);
    let mut bytes = [0; 4096];
    loop {
        let count = match input.read(&mut bytes) {
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if count == 0 {
            break;
        }
        let excess = (tail.len() + count).saturating_sub(BYTES);
        tail.drain(..excess);
        tail.extend(&bytes[..count]);
    }
    let text = String::from_utf8_lossy(tail.make_contiguous());
    let lines: Vec<_> = text.lines().collect();
    Ok(lines[lines.len().saturating_sub(12)..].join("\n"))
}

/// Describes a harness exit even when it emitted no standard error.
pub(crate) fn exit_reason(
    status: std::process::ExitStatus,
    stderr: std::io::Result<String>,
) -> String {
    match stderr {
        Ok(text) if text.is_empty() => status.to_string(),
        Ok(text) => format!("{status}: {text}"),
        Err(error) => format!("{status}: reading harness standard error failed: {error}"),
    }
}

/// Runs a hosted agent until it exits: the body of `polylogue internal host`.
pub fn serve(config: &HostConfig) -> std::io::Result<()> {
    use std::process::{Command, Stdio};
    let (program, args) = config
        .argv
        .split_first()
        .ok_or_else(|| std::io::Error::other("no agent command"))?;
    if let Some(parent) = config.socket.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let listener = crate::socket::bind(&config.socket)?;
    let mut child = Command::new(program)
        .args(args)
        .current_dir(&config.cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stderr = child.stderr.take().expect("piped stderr");
    let stderr =
        polylogue_model::spawn_named("polylogue-host-stderr", move || stderr_tail(stderr))?;
    let (connected, first_connection) = std::sync::mpsc::sync_channel(1);
    let identity = crate::live_events::identity(child.id());
    let input = Arc::new(Mutex::new(child.stdin.take().expect("piped stdin")));
    let output = child.stdout.take().expect("piped stdout");
    let stopping = Arc::new(AtomicBool::new(false));
    let relay = Arc::new(Mutex::new(Relay {
        client: None,
        stopping: Arc::clone(&stopping),
        backlog: VecDeque::new(),
        tracker: Tracker::new(config.protocol),
        spool: crate::live_events::PendingLines::new("registry", "host", Default::default()),
        session: config
            .session_id
            .as_deref()
            .and_then(|id| SessionId::parse(&format!("{}:{id}", config.harness)).ok()),
    }));
    let accept = {
        let relay = Arc::clone(&relay);
        let input = Arc::clone(&input);
        let stopping = Arc::clone(&stopping);
        let config = config.clone();
        let identity = identity.clone();
        polylogue_model::spawn_named("polylogue-host-accept", move || {
            for stream in listener.incoming() {
                if stopping.load(Ordering::SeqCst) {
                    return;
                }
                let Ok(mut stream) = stream else { continue };
                let Ok(reader) = stream.try_clone() else {
                    continue;
                };
                {
                    let mut relay = relay.locked();
                    let mut first = hello(&relay);
                    for line in relay.tracker.unanswered() {
                        first.extend_from_slice(line);
                        first.push(b'\n');
                    }
                    // An unanswered request that waited here is in both.
                    for line in &relay.backlog {
                        if relay
                            .tracker
                            .unanswered()
                            .any(|open| open == line.as_slice())
                        {
                            continue;
                        }
                        first.extend_from_slice(line);
                        first.push(b'\n');
                    }
                    if stream.write_all(&first).is_err() {
                        continue;
                    }
                    let _ = connected.try_send(());
                    relay.backlog.clear();
                    if let Some(previous) = relay.client.replace(stream) {
                        let _ = previous.shutdown(std::net::Shutdown::Both);
                    }
                }
                let relay = Arc::clone(&relay);
                let input = Arc::clone(&input);
                let config = config.clone();
                let identity = identity.clone();
                let client = polylogue_model::spawn_named("polylogue-host-client", move || {
                    for line in BufReader::new(reader).split(b'\n') {
                        let Ok(line) = line else { return };
                        if let Ok(message) = serde_json::from_slice::<Value>(&line) {
                            let mut relay = relay.locked();
                            if let Some(word) = relay.tracker.client_sent(&message)
                                && let Some(session) = relay.session.clone()
                            {
                                status_line(
                                    &mut relay,
                                    &config,
                                    &session,
                                    identity.as_ref(),
                                    "HostedStatus",
                                    Some(word),
                                );
                            }
                            retry_status(&mut relay, &config);
                        }
                        let mut input = input.locked();
                        if input
                            .write_all(&line)
                            .and_then(|()| input.write_all(b"\n"))
                            .and_then(|()| input.flush())
                            .is_err()
                        {
                            return;
                        }
                    }
                });
                if let Err(error) = client {
                    polylogue_store::log!(error, "start a host client worker: {error}");
                }
            }
        })?
    };
    for line in BufReader::new(output).split(b'\n') {
        let Ok(line) = line else { break };
        let mut relay = relay.locked();
        if let Ok(message) = serde_json::from_slice::<Value>(&line) {
            if relay.session.is_none() {
                relay.session = session_named(config, &message);
            }
            if let Some(word) = relay.tracker.agent_sent(&message, &line)
                && let Some(session) = relay.session.clone()
            {
                status_line(
                    &mut relay,
                    config,
                    &session,
                    identity.as_ref(),
                    "HostedStatus",
                    Some(word),
                );
            }
        }
        retry_status(&mut relay, config);
        let sent = relay.client.as_mut().is_some_and(|client| {
            client
                .write_all(&line)
                .and_then(|()| client.write_all(b"\n"))
                .is_ok()
        });
        if !sent {
            if let Some(client) = relay.client.take() {
                let _ = client.shutdown(std::net::Shutdown::Both);
            }
            // A later daemon acts only on requests, answers and turn ends;
            // the agent's narration stays in its transcript.
            if serde_json::from_slice::<Value>(&line).is_ok_and(|m| protocol_traffic(&m)) {
                relay.backlog.push_back(line);
            }
        }
    }
    let status = child.wait();
    let stderr = stderr
        .join()
        .unwrap_or_else(|_| Err(std::io::Error::other("standard error reader panicked")));
    let reason = match status {
        Ok(status) => exit_reason(status, stderr),
        Err(error) => format!("waiting for the harness failed: {error}"),
    };
    // A harness can exit before the starter connects to the published socket.
    let _ = first_connection.recv_timeout(HOST_WAIT);
    {
        // Status production and this stop decision share the append owner's lock.
        // A client may still be writing a pipe held open by an agent descendant.
        let _relay = relay.locked();
        stopping.store(true, Ordering::Release);
    }
    // Wakes the accept loop so it sees it should stop.
    let _ = UnixStream::connect(&config.socket);
    let mut failure = accept
        .join()
        .err()
        .map(|_| std::io::Error::other("host accept worker panicked"));
    let session = relay.locked().session.clone();
    if let Some(session) = session {
        status_line(
            &mut relay.locked(),
            config,
            &session,
            identity.as_ref(),
            "HostedEnded",
            Some(&reason),
        );
    }
    let spool_result = flush_spool(&mut relay.locked(), config);
    if let Some(dir) = config.socket.parent()
        && let Some(key) = config.socket.file_stem().and_then(|s| s.to_str())
    {
        remove(dir, key);
    }
    if let Err(error) = spool_result {
        polylogue_store::log!(error, "host live event spool drain: {error}");
        failure = Some(std::io::Error::other(match failure.take() {
            Some(previous) => format!("{previous}; host live event spool drain: {error}"),
            None => format!("host live event spool drain: {error}"),
        }));
    }
    // The final line follows the session's event, even at startup.
    if let Some(mut client) = relay.locked().client.take() {
        let _ = writeln!(client, "{}", json!({"polylogue_host": {"ended": reason}}));
        let _ = client.shutdown(std::net::Shutdown::Both);
    }
    failure.map_or(Ok(()), Err)
}

/// Whether a line from the agent is one a daemon acts on: a request, an
/// answer, or a turn's end.
fn protocol_traffic(message: &Value) -> bool {
    message.get("id").is_some()
        || matches!(
            message["type"].as_str(),
            Some("control_request" | "control_response" | "control_cancel_request" | "result")
        )
}

/// Starts `polylogue internal host` for a new session in its own process group, so
/// it outlives the daemon, and waits for its socket. Returns the host's
/// process ID.
pub(crate) fn spawn_host(
    harness: &str,
    socket: &Path,
    cwd: &Path,
    spool: &Path,
    session_id: Option<&str>,
) -> std::io::Result<(u32, UnixStream)> {
    let program = std::env::current_exe()?;
    let mut command = std::process::Command::new(program);
    command
        .args(["internal", "host"])
        .arg(harness)
        .arg("--socket")
        .arg(socket)
        .arg("--cwd")
        .arg(cwd)
        .arg("--spool")
        .arg(spool)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::inherit());
    if let Some(id) = session_id {
        command.arg("--session-id").arg(id);
    }
    start_host(&mut command, socket, Instant::now() + HOST_WAIT)
}

fn start_host(
    command: &mut std::process::Command,
    socket: &Path,
    until: Instant,
) -> std::io::Result<(u32, UnixStream)> {
    use std::os::fd::OwnedFd;
    use std::os::unix::process::CommandExt;
    if let Some(parent) = socket.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let watch = crate::socket::Watch::new(socket)?;
    let (exited, notify_exit) = UnixStream::pair()?;
    let mut child = command.process_group(0).spawn()?;
    let pid = child.id();
    let pidfd = rustix::process::Pid::from_raw(pid as i32).and_then(|pid| {
        rustix::process::pidfd_open(pid, rustix::process::PidfdFlags::empty()).ok()
    });
    let exited: OwnedFd = pidfd.unwrap_or_else(|| exited.into());
    // This is also the normal host reaper. On kernels without pidfds, closing
    // its pipe reports the exit to the same poll without a timed fallback.
    let reaper = std::thread::spawn(move || {
        let result = child.wait();
        drop(notify_exit);
        result
    });
    let result = match wait_for_host_socket(&watch, socket, &exited, until) {
        Ok(HostSocket::Listening(stream)) => Ok((pid, stream)),
        Ok(HostSocket::Exited) => match reaper.join() {
            Ok(Ok(status)) => Err(std::io::Error::other(format!(
                "polylogue internal host exited at start ({status})"
            ))),
            Ok(Err(error)) => Err(error),
            Err(_) => Err(std::io::Error::other("host reaper panicked")),
        },
        Err(error) => Err(error),
    };
    if result.is_err() {
        stop_host(pid);
    }
    result
}

#[derive(Debug)]
enum HostSocket {
    Listening(UnixStream),
    Exited,
}

fn wait_for_host_socket(
    watch: &crate::socket::Watch,
    socket: &Path,
    exited: &std::os::fd::OwnedFd,
    until: Instant,
) -> std::io::Result<HostSocket> {
    use rustix::event::{PollFd, PollFlags, Timespec, poll};
    let mut probe = true;
    loop {
        if Instant::now() >= until {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "polylogue internal host did not open its socket before the deadline",
            ));
        }
        let mut events = [
            PollFd::new(watch, PollFlags::IN),
            PollFd::new(exited, PollFlags::IN),
        ];
        // An already exited child wins over a simultaneously published socket.
        match poll(&mut events, Some(&Timespec::default())) {
            Ok(_) => {}
            Err(rustix::io::Errno::INTR) => continue,
            Err(error) => return Err(error.into()),
        }
        if !events[1].revents().is_empty() {
            return Ok(HostSocket::Exited);
        }
        if !events[0].revents().is_empty() {
            probe |= watch.changed()?;
        }
        if probe {
            match UnixStream::connect(socket) {
                Ok(stream) => return Ok(HostSocket::Listening(stream)),
                Err(error) if crate::cli::is_down(&error) => {}
                Err(error) => return Err(error),
            }
            probe = false;
        }
        let remaining = until.saturating_duration_since(Instant::now());
        let timeout = Timespec::try_from(remaining).map_err(std::io::Error::other)?;
        match poll(&mut events, Some(&timeout)) {
            Ok(_) => {}
            Err(rustix::io::Errno::INTR) => continue,
            Err(error) => return Err(error.into()),
        }
    }
}

/// Ends a host that never became a session, and its agent.
fn stop_host(pid: u32) {
    if let Ok(pid) = i32::try_from(pid) {
        // SAFETY: signals the process group the host leads; no memory is
        // shared.
        #[allow(unsafe_code)]
        unsafe {
            libc::kill(-pid, libc::SIGTERM);
        }
    }
}

/// The callback a queued message's outcome is reported through: its
/// delivery, and whether it was written to the agent.
pub(crate) type Sent = Arc<dyn Fn(&str, Result<(), String>) + Send + Sync>;

/// The callback that learns how many messages wait behind the running turn.
pub(crate) type QueueChanged = Arc<dyn Fn(usize) + Send + Sync>;

/// The daemon's state of one stdio-hosted session, updated by its reader
/// thread as lines arrive.
struct StdioState {
    protocol: HostProtocol,
    writer: UnixStream,
    /// The agent's requests awaiting an answer, oldest first.
    requests: VecDeque<Value>,
    /// ACP: a prompt runs now.
    turn: bool,
    /// ACP: the IDs of prompts this connection sent and not yet answered.
    prompts: Vec<Value>,
    /// ACP: prompts waiting for the running one to finish, each with the
    /// delivery it carries.
    queued: VecDeque<(String, Option<String>)>,
    /// Reports the outcome of a queued message once it is sent or dropped.
    on_sent: Option<Sent>,
    /// Claude Code: user messages written and not yet answered by a result;
    /// all but the first wait for the running turn.
    turns: usize,
    /// Reports the number of messages waiting, when it changes.
    on_queue: Option<QueueChanged>,
    queue_reported: usize,
    /// Answers to the daemon's own requests, by ID.
    answers: Vec<Value>,
    /// The connection to the host closed.
    closed: bool,
    ended: Option<String>,
    first_prompt: bool,
    first_reply: Option<Result<(), String>>,
    protocol_session: String,
    /// A per-connection prefix, so request IDs never repeat across
    /// daemon restarts while an earlier request is still in flight.
    nonce: String,
    next_id: u64,
}

impl StdioState {
    fn write(&mut self, message: &Value) -> std::io::Result<()> {
        let mut bytes = serde_json::to_vec(message)?;
        bytes.push(b'\n');
        self.writer.write_all(&bytes)?;
        self.writer.flush()
    }

    fn id(&mut self) -> Value {
        self.next_id += 1;
        json!(format!("{}-{}", self.nonce, self.next_id))
    }

    fn prompt(&mut self, text: &str) -> std::io::Result<()> {
        let id = self.id();
        let message = json!({"jsonrpc": "2.0", "id": id, "method": "session/prompt", "params": {
            "sessionId": self.protocol_session,
            "prompt": [{"type": "text", "text": text}],
        }});
        self.turn = true;
        self.prompts.push(id);
        self.write(&message)
    }

    /// Messages waiting behind the running turn: ACP prompts held here, or
    /// Claude Code user messages the agent has not started.
    fn waiting(&self) -> usize {
        match self.protocol {
            HostProtocol::ClaudeStreamJson => self.turns.saturating_sub(1),
            _ => self.queued.len(),
        }
    }

    /// Tells the daemon when the number of waiting messages changed.
    fn queue_changed(&mut self) {
        let waiting = self.waiting();
        if waiting != self.queue_reported {
            self.queue_reported = waiting;
            if let Some(on_queue) = &self.on_queue {
                on_queue(waiting);
            }
        }
    }

    fn report(&self, delivery: Option<&str>, sent: Result<(), String>) {
        if let (Some(delivery), Some(on_sent)) = (delivery, &self.on_sent) {
            on_sent(delivery, sent);
        }
    }

    /// Takes in one line from the host.
    fn absorb(&mut self, message: Value) {
        if let Some(host) = message.get("polylogue_host") {
            self.ended = host["ended"].as_str().map(str::to_owned);
            self.closed = self.ended.is_some();
            return;
        }
        if self.first_prompt && self.first_reply.is_none() {
            self.first_reply = match self.protocol {
                HostProtocol::ClaudeStreamJson if message["type"].is_string() => Some(Ok(())),
                HostProtocol::Acp if self.prompts.contains(&message["id"]) => {
                    Some(match message.get("error") {
                        Some(error) => Err(error.to_string()),
                        None => Ok(()),
                    })
                }
                HostProtocol::Acp
                    if matches!(
                        message["method"].as_str(),
                        Some("session/update" | "session/request_permission")
                    ) =>
                {
                    Some(Ok(()))
                }
                _ => None,
            };
        }
        self.absorb_line(message);
        self.queue_changed();
    }

    fn absorb_line(&mut self, message: Value) {
        match self.protocol {
            HostProtocol::ClaudeStreamJson => match message["type"].as_str() {
                Some("control_request") => {
                    if message["request"]["subtype"] == "can_use_tool" {
                        self.requests.push_back(message);
                    }
                }
                Some("control_cancel_request") => {
                    let id = &message["request_id"];
                    self.requests.retain(|r| &r["request_id"] != id);
                }
                // A finished turn's permission requests are moot.
                Some("result") => {
                    self.turns = self.turns.saturating_sub(1);
                    self.requests.clear();
                }
                _ => {}
            },
            _ => {
                let request = message.get("method").is_some() && message.get("id").is_some();
                if request {
                    if message["method"] == "session/request_permission" {
                        self.requests.push_back(message);
                    } else {
                        // Polylogue offers no file system or terminal to
                        // the agent: refuse other client methods at once.
                        let refusal = json!({"jsonrpc": "2.0", "id": message["id"],
                            "error": {"code": -32601, "message": "Polylogue does not provide this method"}});
                        let _ = self.write(&refusal);
                    }
                } else if message.get("method").is_none() {
                    let id = &message["id"];
                    // A prompt sent before a daemon restart is known only
                    // by its answer's stop reason.
                    let prompt =
                        self.prompts.contains(id) || message["result"].get("stopReason").is_some();
                    if prompt {
                        self.prompts.retain(|p| p != id);
                        self.turn = false;
                        // A finished prompt's permission requests are moot.
                        self.requests.clear();
                        if let Some((next, delivery)) = self.queued.pop_front() {
                            let sent = self.prompt(&next).map_err(|e| e.to_string());
                            self.report(delivery.as_deref(), sent);
                        }
                    } else {
                        self.answers.push(message);
                    }
                }
            }
        }
    }
}

/// The daemon's connection to a session a `polylogue internal host` holds.
pub(crate) struct HostedStdio {
    state: Arc<(Mutex<StdioState>, std::sync::Condvar)>,
}

impl HostedStdio {
    fn connect(
        socket: &Path,
        protocol: HostProtocol,
        protocol_session: String,
    ) -> std::io::Result<Self> {
        Self::connected(UnixStream::connect(socket)?, protocol, protocol_session)
    }

    fn connected(
        stream: UnixStream,
        protocol: HostProtocol,
        protocol_session: String,
    ) -> std::io::Result<Self> {
        let mut reader = BufReader::new(stream.try_clone()?);
        reader.get_ref().set_read_timeout(Some(HOST_WAIT))?;
        let mut first = String::new();
        reader.read_line(&mut first)?;
        let hello: Value = serde_json::from_str(&first)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        let status = hello["polylogue_host"]["status"]
            .as_str()
            .ok_or_else(|| std::io::Error::other("the host did not send its initial status"))?;
        let turn = status != "idle";
        reader.get_ref().set_read_timeout(None)?;
        let nonce = format!(
            "{:x}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        );
        let state = Arc::new((
            Mutex::new(StdioState {
                protocol,
                writer: stream,
                requests: VecDeque::new(),
                turn,
                prompts: Vec::new(),
                queued: VecDeque::new(),
                on_sent: None,
                turns: hello["polylogue_host"]["turns"]
                    .as_u64()
                    .map_or(0, |n| usize::try_from(n).unwrap_or(usize::MAX)),
                on_queue: None,
                queue_reported: 0,
                answers: Vec::new(),
                closed: false,
                ended: None,
                first_prompt: false,
                first_reply: None,
                protocol_session,
                nonce,
                next_id: 0,
            }),
            std::sync::Condvar::new(),
        ));
        let shared = Arc::clone(&state);
        polylogue_model::spawn_named("polylogue-hosted", move || {
            for line in reader.split(b'\n') {
                let Ok(line) = line else { break };
                let Ok(message) = serde_json::from_slice::<Value>(&line) else {
                    continue;
                };
                shared.0.locked().absorb(message);
                shared.1.notify_all();
            }
            let mut state = shared.0.locked();
            state.closed = true;
            // These prompts never left this connection's local queue.
            // Already-written prompts keep their reported outcome.
            for (_, delivery) in std::mem::take(&mut state.queued) {
                state.report(
                    delivery.as_deref(),
                    Err("the host connection closed before the queued prompt was sent".into()),
                );
            }
            state.turns = 0;
            state.queue_changed();
            shared.1.notify_all();
        })?;
        Ok(Self { state })
    }

    /// Sends a request and waits for its answer.
    fn call(&self, method: &str, params: Value) -> std::io::Result<Value> {
        let (mutex, ready) = &*self.state;
        let mut state = mutex.locked();
        let id = state.id();
        let written =
            state.write(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        let deadline = Instant::now() + HOST_WAIT;
        loop {
            if let Some(at) = state.answers.iter().position(|a| a["id"] == id) {
                let answer = state.answers.remove(at);
                if let Some(error) = answer.get("error") {
                    return Err(std::io::Error::other(format!(
                        "the agent refused {method}: {error}"
                    )));
                }
                return Ok(answer["result"].clone());
            }
            if state.closed {
                return Err(std::io::Error::other(
                    state
                        .ended
                        .as_deref()
                        .unwrap_or("the host closed the connection"),
                ));
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    written.err().map_or_else(
                        || format!("the agent did not answer {method}"),
                        |error| error.to_string(),
                    ),
                ));
            }
            state = polylogue_model::unpoisoned(ready.wait_timeout(state, left)).0;
        }
    }

    /// Starts a hosted session: its host, then the protocol's session, then
    /// the first turn. Returns the connection and the session.
    pub(crate) fn start(
        harness: &str,
        protocol: HostProtocol,
        dir: &Path,
        cwd: &Path,
        spool: &Path,
        text: &str,
    ) -> std::io::Result<(Self, SessionId)> {
        let key = format!("{harness}-{}", uuid::Uuid::now_v7());
        let socket = dir.join(format!("{key}.sock"));
        let claude_id =
            (protocol == HostProtocol::ClaudeStreamJson).then(|| uuid::Uuid::new_v4().to_string());
        let (host, stream) = spawn_host(harness, &socket, cwd, spool, claude_id.as_deref())?;
        let started = Self::begin(
            harness, protocol, dir, &key, &socket, cwd, claude_id, stream, text,
        );
        if started.is_err() {
            // A host that never became a session would hold its agent
            // with nobody to reattach to it.
            stop_host(host);
            remove(dir, &key);
        }
        started
    }

    /// The protocol's session, its record and its first turn, on a host
    /// that just opened its socket.
    #[allow(clippy::too_many_arguments)]
    fn begin(
        harness: &str,
        protocol: HostProtocol,
        dir: &Path,
        key: &str,
        socket: &Path,
        cwd: &Path,
        claude_id: Option<String>,
        stream: UnixStream,
        text: &str,
    ) -> std::io::Result<(Self, SessionId)> {
        let hosted = Self::connected(stream, protocol, String::new())?;
        let native = match protocol {
            HostProtocol::ClaudeStreamJson => claude_id.unwrap_or_default(),
            _ => {
                hosted.call(
                    "initialize",
                    json!({"protocolVersion": 1, "clientCapabilities": {
                        "fs": {"readTextFile": false, "writeTextFile": false}, "terminal": false}}),
                )?;
                let created = hosted.call("session/new", json!({"cwd": cwd, "mcpServers": []}))?;
                let id = created["sessionId"]
                    .as_str()
                    .ok_or_else(|| std::io::Error::other("session/new named no session"))?
                    .to_owned();
                hosted.state.0.locked().protocol_session.clone_from(&id);
                id
            }
        };
        let session = SessionId::parse(&format!("{harness}:{native}"))
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        write_record(
            dir,
            key,
            &Record {
                session: session.to_string(),
                protocol,
                socket: Some(socket.to_owned()),
                protocol_session: (protocol == HostProtocol::Acp).then_some(native),
            },
        )?;
        hosted.state.0.locked().first_prompt = true;
        let written = hosted.send(text, None);
        let (mutex, ready) = &*hosted.state;
        let mut state = mutex.locked();
        let deadline = Instant::now() + HOST_WAIT;
        loop {
            if let Some(reply) = state.first_reply.take() {
                reply.map_err(std::io::Error::other)?;
                state.first_prompt = false;
                break;
            }
            if state.closed {
                return Err(std::io::Error::other(state.ended.as_deref().unwrap_or(
                    "the host closed before the harness answered its first turn",
                )));
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                written?;
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "the harness did not answer its first turn",
                ));
            }
            state = polylogue_model::unpoisoned(ready.wait_timeout(state, left)).0;
        }
        drop(state);
        Ok((hosted, session))
    }

    /// Connects again to a recorded session's host.
    pub(crate) fn reattach(record: &Record) -> std::io::Result<Self> {
        let socket = record
            .socket
            .as_deref()
            .ok_or_else(|| std::io::Error::other("the record names no host socket"))?;
        Self::connect(
            socket,
            record.protocol,
            record.protocol_session.clone().unwrap_or_default(),
        )
    }

    /// Reports queued messages' outcomes through `on_sent`.
    pub(crate) fn on_sent(&self, on_sent: Sent) {
        self.state.0.locked().on_sent = Some(on_sent);
    }

    /// Reports how many messages wait behind the running turn, now and at
    /// each change.
    pub(crate) fn on_queue(&self, on_queue: QueueChanged) {
        let mut state = self.state.0.locked();
        state.on_queue = Some(on_queue);
        state.queue_reported = 0;
        state.queue_changed();
    }

    /// Sends a message: a user message to Claude (which runs it after the
    /// current turn) or a prompt to an ACP agent; `true` when written now.
    /// A message to an ACP agent whose prompt runs waits for that prompt
    /// (`false`); its outcome is reported through `on_sent` when it is
    /// written or dropped.
    pub(crate) fn send(&self, text: &str, delivery: Option<&str>) -> std::io::Result<bool> {
        let mut state = self.state.0.locked();
        if state.closed {
            return Err(std::io::Error::other("the hosted session's host is gone"));
        }
        let written = match state.protocol {
            HostProtocol::ClaudeStreamJson => state
                .write(&json!({
                    "type": "user",
                    "message": {"role": "user", "content": [{"type": "text", "text": text}]},
                    "parent_tool_use_id": null,
                    "session_id": "",
                }))
                .map(|()| {
                    state.turns += 1;
                    true
                }),
            _ if state.turn => {
                state
                    .queued
                    .push_back((text.to_owned(), delivery.map(str::to_owned)));
                Ok(false)
            }
            _ => state.prompt(text).map(|()| true),
        };
        state.queue_changed();
        written
    }

    /// The oldest request awaiting an answer, by the protocol's ID.
    pub(crate) fn request(&self) -> Option<String> {
        let state = self.state.0.locked();
        let request = state.requests.front()?;
        let id = match state.protocol {
            HostProtocol::ClaudeStreamJson => &request["request_id"],
            _ => &request["id"],
        };
        Some(id.as_str().map_or_else(|| id.to_string(), str::to_owned))
    }

    /// Answers the oldest pending request.
    pub(crate) fn answer(
        &self,
        approve: bool,
        text: Option<&str>,
    ) -> std::io::Result<Option<crate::protocol::acts::AnswerCancellation>> {
        let mut state = self.state.0.locked();
        let request = state
            .requests
            .front()
            .ok_or_else(|| std::io::Error::other("no request is pending on the hosted session"))?;
        let reply = answer_message(state.protocol, request, approve, text);
        let cancellation = (reply["result"]["outcome"]["outcome"] == "cancelled").then(|| {
            crate::protocol::acts::AnswerCancellation {
                options: request["params"]["options"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|option| option["kind"].as_str().map(str::to_owned))
                    .collect(),
            }
        });
        state.write(&reply)?;
        state.requests.pop_front();
        Ok(cancellation)
    }

    pub(crate) fn interrupt(&self) -> std::io::Result<()> {
        let mut state = self.state.0.locked();
        match state.protocol {
            HostProtocol::ClaudeStreamJson => {
                let id = state.id();
                state.write(&json!({"type": "control_request", "request_id": id,
                    "request": {"subtype": "interrupt"}}))
            }
            _ => {
                if !state.turn {
                    return Err(std::io::Error::other("no prompt is running"));
                }
                for (_, delivery) in std::mem::take(&mut state.queued) {
                    state.report(
                        delivery.as_deref(),
                        Err("the running prompt was interrupted before it was sent".into()),
                    );
                }
                state.queue_changed();
                let session = state.protocol_session.clone();
                state.write(&json!({"jsonrpc": "2.0", "method": "session/cancel",
                    "params": {"sessionId": session}}))?;
                // ACP: after a cancel the client answers every pending
                // permission request `cancelled`.
                for request in std::mem::take(&mut state.requests) {
                    state.write(&json!({"jsonrpc": "2.0", "id": request["id"],
                        "result": {"outcome": {"outcome": "cancelled"}}}))?;
                }
                Ok(())
            }
        }
    }

    /// Whether the host is still connected.
    pub(crate) fn open(&self) -> bool {
        !self.state.0.locked().closed
    }
}

impl Drop for HostedStdio {
    /// Closes the connection, which ends its reader; the host keeps the
    /// agent running for the next connection.
    fn drop(&mut self) {
        let _ = self
            .state
            .0
            .locked()
            .writer
            .shutdown(std::net::Shutdown::Both);
    }
}

/// The reply to an agent's permission request: Claude's `can_use_tool`
/// control response, or the ACP option of the matching kind (`allow_once`
/// or `allow_always` to approve, `reject_once` or `reject_always` to deny;
/// with none of that kind, the request is cancelled).
pub(crate) fn answer_message(
    protocol: HostProtocol,
    request: &Value,
    approve: bool,
    text: Option<&str>,
) -> Value {
    match protocol {
        HostProtocol::ClaudeStreamJson => {
            let response = if approve {
                json!({"behavior": "allow", "updatedInput": request["request"]["input"]})
            } else {
                json!({"behavior": "deny", "message": text.unwrap_or("Denied by the operator through Polylogue")})
            };
            json!({"type": "control_response", "response": {"subtype": "success",
                "request_id": request["request_id"], "response": response}})
        }
        _ => {
            let kinds: &[&str] = if approve {
                &["allow_once", "allow_always"]
            } else {
                &["reject_once", "reject_always"]
            };
            let option = request["params"]["options"]
                .as_array()
                .into_iter()
                .flatten()
                .find(|o| kinds.iter().any(|k| o["kind"] == *k))
                .and_then(|o| o.get("optionId").cloned());
            let outcome = match option {
                Some(id) => json!({"outcome": "selected", "optionId": id}),
                None => json!({"outcome": "cancelled"}),
            };
            json!({"jsonrpc": "2.0", "id": request["id"], "result": {"outcome": outcome}})
        }
    }
}
