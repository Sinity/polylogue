//! The daemon's live evidence writers: the process watcher and the harness
//! registry reader. Both append lines to the hook spool's `process/` and
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
//! **Registry reader.** Claude Code keeps one `sessions/<pid>.json` per
//! process with its own status word; each change of a file is written as
//! one `Status` line carrying the file whole, plus the process identity.

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
            for child in children.split_whitespace().filter_map(|c| c.parse::<u32>().ok()) {
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
fn append(spool: &Path, route: &str, boot: &str, lines: &[Value]) -> std::io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    if lines.is_empty() {
        return Ok(());
    }
    let dir = spool.join(route);
    std::fs::create_dir_all(&dir)?;
    let mut bytes = Vec::new();
    for l in lines {
        serde_json::to_writer(&mut bytes, l)?;
        bytes.push(b'\n');
    }
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open(dir.join(format!("{boot}.ndjson")))?
        .write_all(&bytes)
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
    /// Roots already written gone, until their `RootGone` is committed and
    /// the evidence stops naming them.
    reported_gone: BTreeSet<(String, ProcessIdentity)>,
}

impl Watcher {
    pub(crate) fn new(proc_root: PathBuf, boot: String, boot_time_ms: Option<i64>) -> Self {
        Self {
            proc_root,
            boot,
            boot_time_ms,
            watched: BTreeMap::new(),
            reported_gone: BTreeSet::new(),
        }
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
            if self.reported_gone.contains(&(r.session.clone(), r.root.clone())) {
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
                self.reported_gone.insert((r.session.clone(), r.root.clone()));
            }
            if r.root.boot != self.boot {
                // The process belonged to an earlier boot: it ended then.
                let at = self
                    .boot_time_ms
                    .and_then(|ms| jiff::Timestamp::from_millisecond(ms).ok())
                    .unwrap_or(now);
                out.push(line("process", "RootGone", &session, at, identity_payload(&r.root)));
                continue;
            }
            if !self.alive(&r.root) {
                out.push(line("process", "RootGone", &session, now, identity_payload(&r.root)));
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
                ready.contains(*session)
                    || (w.pidfd.is_none() && !self.alive(&w.root))
            })
            .map(|(s, _)| s.clone())
            .collect();
        for session_key in gone {
            let w = self.watched.remove(&session_key).expect("listed above");
            self.reported_gone.insert((session_key.clone(), w.root.clone()));
            let Ok(session) = SessionId::parse(&session_key) else {
                continue;
            };
            for ((pid, start), _) in &w.children {
                out.push(line(
                    "process",
                    "Ended",
                    &session,
                    now,
                    json!({"pid": pid, "start": start, "boot_id": self.boot}),
                ));
            }
            out.push(line("process", "RootGone", &session, now, identity_payload(&w.root)));
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
    pub(crate) fn read(&mut self, now: jiff::Timestamp, identity_of: impl Fn(u32) -> Option<ProcessIdentity>) -> Vec<Value> {
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

/// Runs both writers until `stop` fires.
pub(crate) fn run(
    daemon: std::sync::Arc<crate::daemon::service::Daemon>,
    paths: Paths,
    clock: polylogue_model::WallClock,
    stop: Receiver<()>,
) {
    let Some(boot) = boot_id() else {
        return;
    };
    let mut watcher = Watcher::new(PathBuf::from("/proc"), boot.clone(), boot_time_ms());
    let mut registry = Registry::new(paths.claude_sessions.clone());
    loop {
        let now = clock.now();
        let mut process_lines = Vec::new();
        if let Ok(roots) = daemon.reads.reader().and_then(|r| r.live_roots()) {
            process_lines.extend(watcher.update_roots(&roots, now));
        }
        process_lines.extend(watcher.sample(now));
        let _ = append(&paths.spool, "process", &boot, &process_lines);
        let registry_lines = registry.read(now, identity);
        let _ = append(&paths.spool, "registry", &boot, &registry_lines);
        // Wake for the next sample, a root's exit, new roots (at most a
        // second later) or the stop signal.
        let wake = watcher.next_wake().min(SAMPLE_OPEN_CALL);
        let polled = watcher.watched.values().any(|w| w.pidfd.is_some());
        let ready = if polled {
            watcher.wait(wake)
        } else {
            BTreeSet::new()
        };
        let gone = watcher.exited(&ready, clock.now());
        let _ = append(&paths.spool, "process", &boot, &gone);
        match stop.recv_timeout(if polled { Duration::ZERO } else { wake }) {
            Err(RecvTimeoutError::Timeout) => {}
            Ok(()) | Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}
