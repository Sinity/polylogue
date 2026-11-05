//! Which channel an act reaches a session through now. One pure function,
//! [`channel`], decides from the act, the session's live state and what
//! the installed harness is known to support; its answer is also the
//! availability shown before acting. A channel is never chosen by trying
//! one and falling back: when the chosen channel fails, the delivery
//! records `not_sent` and the caller may choose again.
//!
//! What each harness offers is declared once, in [`HARNESSES`]: the hook
//! events that accept context, the synchronous permission hook, the
//! headless resume and the control protocol a hosted session runs on. The
//! hook command, the act handlers and the channel table in `docs/agents.md`
//! ([`channel_table`]) all read these declarations.

use super::LiveState;
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

/// Something done to an existing session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Act {
    /// An addressed message; `live` refuses rather than resuming a session
    /// that is not running.
    Ping { live: bool },
    /// A signal rule's notice. A notice never starts or resumes a session.
    Notice,
    /// Approve, deny or choose for a pending request, or ahead of one.
    Answer,
    /// Stop the current turn.
    Interrupt,
    /// The harness's own resume.
    Resume,
}

/// How an act reaches a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Channel {
    /// A control connection Polylogue holds (a hosted session).
    Protocol,
    /// The harness's next context-accepting hook event, through the
    /// mailbox; for an answer, its synchronous permission hook.
    Hook,
    /// A headless run of the harness's own resume command.
    Resume,
}

stored_names!(Channel { Protocol => "protocol", Hook => "hook", Resume => "resume" });

/// The control protocol a hosted session of a harness runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostProtocol {
    /// Codex's shared app-server (JSON-RPC over its control socket): the
    /// app-server keeps the thread, so Polylogue reconnects to it.
    CodexAppServer,
    /// Claude Code's `--print` stream-json input and output with control
    /// requests, held by a `polylogue host` process.
    ClaudeStreamJson,
    /// The Agent Client Protocol over stdio, held by a `polylogue host`
    /// process.
    Acp,
}

stored_names!(HostProtocol {
    CodexAppServer => "codex_app_server",
    ClaudeStreamJson => "claude_stream_json",
    Acp => "acp",
});

impl HostProtocol {
    /// Whether the agent's stdio must be held by a `polylogue host`
    /// process so that a daemon restart does not end it.
    pub fn stdio(self) -> bool {
        !matches!(self, Self::CodexAppServer)
    }
}

/// What one harness offers Polylogue's acts, as probed on its installed
/// versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Harness {
    /// The origin name of its sessions.
    pub name: &'static str,
    /// Hook events at which it accepts context between steps: after a
    /// tool result and at a prompt, never before a tool runs.
    pub mail_events: &'static [&'static str],
    /// The synchronous hook event that may return a permission decision.
    pub permission_hook: Option<&'static str>,
    /// It resumes a session headlessly with a prompt.
    pub headless_resume: bool,
    /// The protocol its hosted sessions run on.
    pub host: Option<HostProtocol>,
}

/// Every harness acts can reach.
pub const HARNESSES: &[Harness] = &[
    Harness {
        name: "claude-code",
        mail_events: &["PostToolUse", "UserPromptSubmit"],
        permission_hook: Some("PermissionRequest"),
        headless_resume: true,
        host: Some(HostProtocol::ClaudeStreamJson),
    },
    Harness {
        name: "codex",
        mail_events: &["PostToolUse", "UserPromptSubmit"],
        permission_hook: None,
        headless_resume: true,
        host: Some(HostProtocol::CodexAppServer),
    },
    Harness {
        name: "gemini-cli",
        mail_events: &["AfterTool", "BeforeAgent"],
        permission_hook: None,
        headless_resume: false,
        host: Some(HostProtocol::Acp),
    },
    Harness {
        name: "hermes",
        mail_events: &["pre_llm_call"],
        permission_hook: None,
        headless_resume: false,
        host: Some(HostProtocol::Acp),
    },
];

/// The declaration of the harness named `name`.
pub fn harness(name: &str) -> Option<&'static Harness> {
    HARNESSES.iter().find(|h| h.name == name)
}

/// What is known about a session's harness: probed per installed version,
/// never assumed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Probe {
    /// The harness's origin name (`claude-code`, `codex`, ...).
    pub harness: String,
    /// Polylogue holds a control connection to the session.
    pub hosted: bool,
    /// The hook command is installed and has context-accepting events.
    pub hooks: bool,
    /// The installed harness asks the synchronous permission hook.
    pub permission_hook: bool,
    /// The harness has a headless resume command.
    pub headless_resume: bool,
}

/// Why an act cannot reach a session now.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Refusal {
    /// `unsupported`, `session_not_live` or `session_running`.
    pub reason: &'static str,
    pub detail: String,
}

fn refuse<T>(reason: &'static str, detail: String) -> Result<T, Refusal> {
    Err(Refusal { reason, detail })
}

/// The channel `act` takes to a session in `state` whose harness `probe`
/// describes; preference: protocol, hook, resume.
pub fn channel(act: Act, state: LiveState, probe: &Probe) -> Result<Channel, Refusal> {
    let harness = &probe.harness;
    let running = |what: &str| {
        if probe.hosted {
            Ok(Channel::Protocol)
        } else if probe.hooks {
            Ok(Channel::Hook)
        } else {
            refuse(
                "unsupported",
                format!("{harness} has no hook installed to take {what}"),
            )
        }
    };
    match act {
        Act::Ping { live } => {
            if state.running() {
                running("a message")
            } else if live {
                refuse(
                    "session_not_live",
                    format!(
                        "the session is {}; a live ping never resumes it",
                        state.as_str()
                    ),
                )
            } else if probe.headless_resume {
                Ok(Channel::Resume)
            } else {
                refuse("unsupported", format!("{harness} has no headless resume"))
            }
        }
        Act::Notice => {
            if state.running() {
                running("a notice")
            } else {
                refuse(
                    "session_not_live",
                    format!(
                        "the session is {}; a notice never starts or resumes a session",
                        state.as_str()
                    ),
                )
            }
        }
        Act::Answer => {
            if !state.running() {
                refuse(
                    "session_not_live",
                    format!(
                        "the session is {}; nothing waits for an answer",
                        state.as_str()
                    ),
                )
            } else if probe.hosted {
                Ok(Channel::Protocol)
            } else if probe.permission_hook {
                Ok(Channel::Hook)
            } else {
                refuse(
                    "unsupported",
                    format!(
                        "{harness} offers no channel for answers to a session Polylogue does not host"
                    ),
                )
            }
        }
        Act::Interrupt => {
            if !state.running() {
                refuse(
                    "session_not_live",
                    format!("the session is {}; no turn runs", state.as_str()),
                )
            } else if probe.hosted {
                Ok(Channel::Protocol)
            } else {
                refuse(
                    "unsupported",
                    format!("{harness} can be interrupted only through a session Polylogue hosts"),
                )
            }
        }
        Act::Resume => {
            if state.running() {
                refuse(
                    "session_running",
                    format!("the session is {}", state.as_str()),
                )
            } else if probe.headless_resume {
                Ok(Channel::Resume)
            } else {
                refuse("unsupported", format!("{harness} has no headless resume"))
            }
        }
    }
}

/// The protocol a new session of `harness` is started and hosted on: what
/// `start`, a role start and a hand-off use.
pub fn host_channel(harness: &str) -> Result<HostProtocol, Refusal> {
    match self::harness(harness) {
        Some(Harness {
            host: Some(protocol),
            ..
        }) => Ok(*protocol),
        Some(_) => refuse(
            "unsupported",
            format!("{harness} has no control protocol Polylogue can host it through"),
        ),
        None => refuse(
            "unsupported",
            format!("{harness} is not a harness Polylogue starts"),
        ),
    }
}

/// The probe of a session Polylogue observes through the harness's
/// declarations, its hook installed: what the channel table's observed
/// rows show.
pub fn declared_probe(harness: &Harness, hosted: bool) -> Probe {
    Probe {
        harness: harness.name.to_owned(),
        hosted,
        hooks: !harness.mail_events.is_empty(),
        permission_hook: harness.permission_hook.is_some(),
        headless_resume: harness.headless_resume,
    }
}

/// The channel table in `docs/agents.md`: per harness, observed (its hook
/// installed) and hosted, the channel each act takes or why it is refused.
/// Generated from [`HARNESSES`] through [`channel`], so the docs cannot
/// differ from what the acts do.
pub fn channel_table() -> String {
    let cell = |answer: Result<Channel, Refusal>| match answer {
        Ok(channel) => channel.as_str().to_owned(),
        Err(refusal) => format!("refused: {}", refusal.reason),
    };
    let mut out = String::from(
        "| Harness | Session | ping, running | ping, ended | notice | answer | interrupt | resume, ended | context events |\n\
         |---|---|---|---|---|---|---|---|---|\n",
    );
    for harness in HARNESSES {
        let mut modes = vec![("observed".to_owned(), false)];
        if let Some(protocol) = harness.host {
            modes.push((format!("hosted ({})", protocol.as_str()), true));
        }
        for (mode, hosted) in modes {
            let probe = declared_probe(harness, hosted);
            let events = harness
                .mail_events
                .iter()
                .map(|e| format!("`{e}`"))
                .chain(harness.permission_hook.map(|e| format!("`{e}` (answers)")))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                harness.name,
                mode,
                cell(channel(
                    Act::Ping { live: false },
                    LiveState::Working,
                    &probe
                )),
                cell(channel(Act::Ping { live: false }, LiveState::Ended, &probe)),
                cell(channel(Act::Notice, LiveState::Working, &probe)),
                cell(channel(Act::Answer, LiveState::Waiting, &probe)),
                cell(channel(Act::Interrupt, LiveState::Working, &probe)),
                cell(channel(Act::Resume, LiveState::Ended, &probe)),
                events,
            ));
        }
    }
    out
}
