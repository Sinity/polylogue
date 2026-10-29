//! Honest attribution from gathered caller evidence. I/O gathers this evidence
//! at the transport boundary; the decision is the same on every surface.
use crate::{Author, NativeId, Origin, SessionId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Channel {
    Mcp,
    Cli,
    Web,
    Extension,
    Hook,
    Markup,
    Import,
    Daemon,
}
stored_names!(Channel {Mcp=>"mcp",Cli=>"cli",Web=>"web",Extension=>"extension",Hook=>"hook",Markup=>"markup",Import=>"import",Daemon=>"daemon"});

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CallerEvidence {
    pub harness: Option<Origin>,
    pub session: Option<SessionId>,
    pub tool_call_id: Option<String>,
    pub harness_pid: Option<u32>,
    pub controlling_terminal: bool,
    pub unit: Option<String>,
    #[serde(default)]
    pub ambiguous_harness: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Authorship {
    pub author: Author,
    pub via: Channel,
}

/// One selected hook-spool identity. A tool call refines the matching call;
/// a SessionStart identity (no tool_call_id) follows the matching parent pid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookIdentity {
    pub harness_pid: u32,
    pub tool_call_id: Option<String>,
    pub session: SessionId,
    pub agent_id: Option<String>,
}

pub fn compute_authorship(
    via: Channel,
    caller: &CallerEvidence,
    hook: Option<&HookIdentity>,
) -> Authorship {
    let author = if matches!(via, Channel::Web | Channel::Extension) {
        Author::Operator
    } else {
        let hook = hook.filter(|hook| {
            caller.harness_pid == Some(hook.harness_pid)
                && hook
                    .tool_call_id
                    .as_ref()
                    .is_none_or(|id| caller.tool_call_id.as_ref() == Some(id))
        });
        let session = hook
            .map(|hook| {
                if let Some(agent) = hook.agent_id.as_ref().filter(|agent| !agent.is_empty()) {
                    SessionId::new(
                        hook.session.origin().clone(),
                        NativeId::new(&format!("agent-{agent}"))
                            .expect("prefixed native identity is nonempty"),
                    )
                } else {
                    hook.session.clone()
                }
            })
            .or_else(|| caller.session.clone());
        if via == Channel::Cli && caller.ambiguous_harness && hook.is_none() {
            Author::Unattributed
        } else if let Some(session) = session {
            Author::Agent(Some(session))
        } else if let Some(origin) = &caller.harness {
            Author::AgentOrigin(origin.clone())
        } else if via == Channel::Mcp {
            Author::Agent(None)
        } else if via == Channel::Cli && caller.controlling_terminal {
            Author::Operator
        } else if let Some(unit) = caller.unit.as_ref().filter(|unit| !unit.is_empty()) {
            Author::Script(unit.clone())
        } else {
            Author::Unattributed
        }
    };
    Authorship { author, via }
}
#[cfg(test)]
mod tests;
