//! Honest attribution from gathered caller evidence. I/O gathers this evidence
//! at the transport boundary; the decision is the same on every surface.
use crate::{IdError, NativeId, Origin, SessionId};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

/// Honest attribution retained on every authored record.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Author {
    Operator,
    /// An agent, by the session it wrote from when known.
    Agent(Option<SessionId>),
    /// The harness is known but no session identity was recorded.
    AgentOrigin(Origin),
    /// No evidence identifies a person, agent or named program.
    Unattributed,
    /// A program acting for the operator, by name.
    Script(String),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthorError {
    #[error("author {0:?}: {1}")]
    Id(String, IdError),
    #[error("author {0:?} is not a stored actor")]
    Unknown(String),
}

impl Author {
    pub fn is_agent(&self) -> bool {
        matches!(self, Self::Agent(_) | Self::AgentOrigin(_))
    }

    /// Parse stored attribution without inventing a missing harness or session.
    pub fn parse(text: &str) -> Result<Self, AuthorError> {
        match text.split_once(':') {
            None if text == "operator" => Ok(Self::Operator),
            None if text == "agent" => Ok(Self::Agent(None)),
            None if text == "unattributed" => Ok(Self::Unattributed),
            Some(("agent", origin)) if !origin.contains(':') => Ok(Self::AgentOrigin(
                Origin::new(origin).map_err(|e| AuthorError::Id(text.to_owned(), e))?,
            )),
            Some(("agent", s)) => Ok(Self::Agent(Some(
                SessionId::parse(s).map_err(|e| AuthorError::Id(text.to_owned(), e))?,
            ))),
            Some(("script", name)) if !name.is_empty() => Ok(Self::Script(name.to_owned())),
            _ => Err(AuthorError::Unknown(text.to_owned())),
        }
    }
}

impl fmt::Display for Author {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Operator => f.write_str("operator"),
            Self::Agent(None) => f.write_str("agent"),
            Self::Agent(Some(s)) => write!(f, "agent:{s}"),
            Self::AgentOrigin(origin) => write!(f, "agent:{origin}"),
            Self::Unattributed => f.write_str("unattributed"),
            Self::Script(n) => write!(f, "script:{n}"),
        }
    }
}

impl Serialize for Author {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Author {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let text = String::deserialize(d)?;
        Self::parse(&text).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Channel {
    Mcp,
    Cli,
    Web,
    Extension,
    Desktop,
    Hook,
    Markup,
    Import,
    Daemon,
}
stored_names!(Channel {Mcp=>"mcp",Cli=>"cli",Web=>"web",Extension=>"extension",Desktop=>"desktop",Hook=>"hook",Markup=>"markup",Import=>"import",Daemon=>"daemon"});

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
    let author = if matches!(via, Channel::Web | Channel::Extension | Channel::Desktop) {
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
