//! Which channel an act reaches a session through now. One pure function,
//! [`channel`], decides from the act, the session's live state and what
//! the installed harness is known to support; its answer is also the
//! availability shown before acting. A channel is never chosen by trying
//! one and falling back: when the chosen channel fails, the delivery
//! records `not_sent` and the caller may choose again.

use super::LiveState;
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

/// Something done to a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Act {
    /// An addressed message; `live` refuses rather than resuming a session
    /// that is not running.
    Ping { live: bool },
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

fn refuse(reason: &'static str, detail: String) -> Result<Channel, Refusal> {
    Err(Refusal { reason, detail })
}

/// The channel `act` takes to a session in `state` whose harness `probe`
/// describes; preference: protocol, hook, resume.
pub fn channel(act: Act, state: LiveState, probe: &Probe) -> Result<Channel, Refusal> {
    let harness = &probe.harness;
    match act {
        Act::Ping { live } => {
            if state.running() {
                if probe.hosted {
                    Ok(Channel::Protocol)
                } else if probe.hooks {
                    Ok(Channel::Hook)
                } else {
                    refuse(
                        "unsupported",
                        format!("{harness} has no hook installed to take a message"),
                    )
                }
            } else if live {
                refuse(
                    "session_not_live",
                    format!("the session is {}; a live ping never resumes it", state.as_str()),
                )
            } else if probe.headless_resume {
                Ok(Channel::Resume)
            } else {
                refuse(
                    "unsupported",
                    format!("{harness} has no headless resume"),
                )
            }
        }
        Act::Answer => {
            if !state.running() {
                refuse(
                    "session_not_live",
                    format!("the session is {}; nothing waits for an answer", state.as_str()),
                )
            } else if probe.hosted {
                Ok(Channel::Protocol)
            } else if probe.permission_hook {
                Ok(Channel::Hook)
            } else {
                refuse(
                    "unsupported",
                    format!("{harness} offers no channel for answers to a session Polylogue does not host"),
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
