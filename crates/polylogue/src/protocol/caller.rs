//! Caller facts gathered from this process, without accepting an author label.
pub mod hooks;
use polylogue_model::authorship::{Authorship, CallerEvidence, Channel};
use polylogue_model::{Origin, SessionId};
use std::os::unix::ffi::OsStringExt;
use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::{Path, PathBuf},
};

const ENVIRONMENT: [&str; 9] = [
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_PID",
    "CLAUDECODE",
    "CODEX_THREAD_ID",
    "CODEX_TOOL_CALL_ID",
    "HERMES_SESSION_ID",
    "GEMINI_CLI",
    "HOME",
    "HERMES_HOME",
];

#[derive(Clone)]
pub struct Accepted {
    pub authorship: Authorship,
    pub cancellation: polylogue_store::read::limit::ReadCancellation,
    /// Process-private outer request bound, captured once by its accepting transport.
    pub(crate) parent: Option<std::sync::Arc<crate::operation::ParentLifetime>>,
    pub working_directory: Option<PathBuf>,
    pub invocation: Option<crate::operation::Invocation>,
    /// The existing native token captured after classifying the actual request.
    pub read_scope: Option<polylogue_store::ReadScopeToken>,
}

impl Accepted {
    /// Coalescing must use the same actor and channel as durable request identity.
    pub fn scope(&self) -> String {
        serde_json::to_string(&self.authorship).expect("computed authorship serializes")
    }
}

pub fn accept(
    registry: &hooks::Registry,
    evidence: &CallerEvidence,
    via: Channel,
    cancellation: polylogue_store::read::limit::ReadCancellation,
) -> Accepted {
    let hook = registry.identity(evidence);
    Accepted {
        authorship: polylogue_model::authorship::compute_authorship(via, evidence, hook.as_ref()),
        cancellation,
        parent: None,
        working_directory: None,
        invocation: None,
        read_scope: None,
    }
}

/// The daemon observes the native peer; request JSON contributes only the
/// per-call tool identifier used to join a matching hook event.
pub fn peer(
    stream: &std::os::unix::net::UnixStream,
    supplied: &CallerEvidence,
    machine: &str,
) -> std::io::Result<(CallerEvidence, Channel)> {
    let credentials = rustix::net::sockopt::socket_peercred(stream)?;
    let root = PathBuf::from(format!("/proc/{}", credentials.pid.as_raw_pid()));
    let stat = std::fs::read_to_string(root.join("stat"))?;
    let parent = command_surface::parent_pid(&stat).unwrap_or_default();
    let terminal = stat
        .rsplit_once(')')
        .and_then(|(_, fields)| fields.split_whitespace().nth(4))
        .and_then(|tty| tty.parse::<i64>().ok())
        .is_some_and(|tty| tty != 0);
    let raw = std::fs::read(root.join("environ"))?;
    let environment = raw
        .split(|byte| *byte == 0)
        .filter_map(|entry| {
            let split = entry.iter().position(|byte| *byte == b'=')?;
            let name = std::str::from_utf8(&entry[..split]).ok()?;
            ENVIRONMENT.contains(&name).then(|| {
                (
                    name.to_owned(),
                    OsString::from_vec(entry[split + 1..].to_vec()),
                )
            })
        })
        .collect();
    let unit = std::fs::read_to_string(root.join("cgroup"))
        .ok()
        .and_then(|value| command_surface::cgroup_unit(&value));
    let mut evidence = select(
        machine,
        &environment,
        &ancestors(parent, &environment),
        terminal,
        unit,
    );
    let arguments = std::fs::read(root.join("cmdline"))?;
    let words = arguments
        .split(|byte| *byte == 0)
        .filter_map(|word| std::str::from_utf8(word).ok())
        .collect::<Vec<_>>();
    let via = channel(&words);
    if via == Channel::Mcp && supplied.tool_call_id.is_some() {
        evidence.tool_call_id = supplied.tool_call_id.clone();
    }
    Ok((evidence, via))
}

/// The accepting daemon observes relative-path context from the native peer.
pub fn working_directory(stream: &std::os::unix::net::UnixStream) -> std::io::Result<PathBuf> {
    let credentials = rustix::net::sockopt::socket_peercred(stream)?;
    std::fs::read_link(format!("/proc/{}/cwd", credentials.pid.as_raw_pid()))
}

#[derive(Clone)]
struct Identity {
    harness: Origin,
    session: Option<SessionId>,
    tool_call_id: Option<String>,
    pid: Option<u32>,
}

static CALLER: std::sync::OnceLock<CallerEvidence> = std::sync::OnceLock::new();

pub fn initialize(environment: &crate::config::EnvSnapshot, machine: &str) {
    let _ = CALLER.set(capture(environment, machine));
}
pub fn evidence() -> CallerEvidence {
    CALLER.get().cloned().unwrap_or_default()
}

fn capture(snapshot: &crate::config::EnvSnapshot, machine: &str) -> CallerEvidence {
    let environment = ENVIRONMENT
        .into_iter()
        .filter_map(|name| {
            snapshot
                .get_os(name)
                .map(|value| (name.to_owned(), value.to_owned()))
        })
        .collect::<BTreeMap<_, _>>();
    let ancestors = ancestors(std::os::unix::process::parent_id(), &environment);
    let unit = std::fs::read_to_string("/proc/self/cgroup")
        .ok()
        .and_then(|value| command_surface::cgroup_unit(&value));
    let terminal = std::fs::OpenOptions::new()
        .read(true)
        .open("/dev/tty")
        .is_ok();
    select(machine, &environment, &ancestors, terminal, unit)
}

fn channel(words: &[&str]) -> Channel {
    let mut words = words.iter().skip(1);
    while let Some(word) = words.next() {
        match *word {
            "--json" => continue,
            "--deadline" | "--timeout" | "--archive" | "--config" | "--listen"
            | "--auth-token-file" | "--request" | "--request-id" => {
                words.next();
                continue;
            }
            word if [
                "--deadline=",
                "--timeout=",
                "--archive=",
                "--config=",
                "--listen=",
                "--auth-token-file=",
                "--request=",
                "--request-id=",
            ]
            .iter()
            .any(|prefix| word.starts_with(prefix)) =>
            {
                continue;
            }
            "mcp" => return Channel::Mcp,
            "hook" => return Channel::Hook,
            "browser-host" => return Channel::Extension,
            word if word.starts_with("chrome-extension://") => return Channel::Extension,
            _ => return Channel::Cli,
        }
    }
    Channel::Cli
}

fn select(
    machine: &str,
    environment: &BTreeMap<String, OsString>,
    ancestors: &[(u32, Origin)],
    terminal: bool,
    unit: Option<String>,
) -> CallerEvidence {
    let text = |name: &str| {
        environment
            .get(name)
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty())
    };
    let mut identities = Vec::new();
    for (origin, session_name, flag, pid_name, tool_name) in [
        (
            "claude-code",
            "CLAUDE_CODE_SESSION_ID",
            "CLAUDECODE",
            Some("CLAUDE_PID"),
            None,
        ),
        (
            "codex",
            "CODEX_THREAD_ID",
            "CODEX_THREAD_ID",
            None,
            Some("CODEX_TOOL_CALL_ID"),
        ),
        (
            "hermes",
            "HERMES_SESSION_ID",
            "HERMES_SESSION_ID",
            None,
            None,
        ),
        ("gemini-cli", "", "GEMINI_CLI", None, None),
    ] {
        if (text(session_name).is_none() && text(flag).is_none())
            || (origin == "gemini-cli" && text(flag) != Some("1"))
        {
            continue;
        }
        let harness = Origin::new(origin).expect("declared origin is valid");
        let session = text(session_name).and_then(|native| {
            if origin == "hermes" {
                let state = environment
                    .get("HERMES_HOME")
                    .map(|home| Path::new(home).join("state.db"))
                    .or_else(|| {
                        environment
                            .get("HOME")
                            .map(|home| Path::new(home).join(".hermes/state.db"))
                    })?;
                polylogue_sources::hermes::session_at(machine, &state, native)
            } else {
                SessionId::parse(&format!("{origin}:{native}")).ok()
            }
        });
        identities.push(Identity {
            harness,
            session,
            tool_call_id: tool_name.and_then(text).map(str::to_owned),
            pid: pid_name.and_then(text).and_then(|value| value.parse().ok()),
        });
    }
    let ambiguous_harness = ancestors.is_empty() && identities.len() > 1;
    let found = if let Some((pid, origin)) = ancestors.first() {
        let mut selected = identities
            .iter()
            .find(|identity| identity.harness == *origin)
            .cloned()
            .unwrap_or_else(|| Identity {
                harness: origin.clone(),
                session: None,
                tool_call_id: None,
                pid: None,
            });
        selected.pid = Some(*pid);
        Some(selected)
    } else if identities.len() == 1 {
        identities.pop()
    } else {
        None
    };
    let mut caller = CallerEvidence {
        controlling_terminal: terminal,
        unit,
        ambiguous_harness,
        ..Default::default()
    };
    if let Some(identity) = found {
        caller.harness = Some(identity.harness);
        caller.session = identity.session;
        caller.tool_call_id = identity.tool_call_id;
        caller.harness_pid = identity.pid;
    }
    caller
}

/// The harness processes from `pid` upward, nearest first, recognised by
/// their executable or by the harness variables they set in the processes
/// below them (`environment` holds the caller's).
fn ancestors(pid: u32, environment: &BTreeMap<String, OsString>) -> Vec<(u32, Origin)> {
    command_surface::harness_ancestors(pid, environment)
        .into_iter()
        .filter_map(|(pid, harness)| Some((pid, Origin::new(harness.as_str()).ok()?)))
        .collect()
}

#[cfg(test)]
mod tests;
