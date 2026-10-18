use crate::{Json, MessageId, SessionId, Timestamp};
use serde::{Deserialize, Serialize};

/// Something on a session's timeline that is not a message: a compaction,
/// a mode change, a hook run, a title record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub session: SessionId,
    /// Plain snake_case name: `compaction`, `permission_mode`, `hook_run`.
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<Timestamp>,
    /// The message the event follows, when the source places it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<MessageId>,
    /// The byte range of the record the event came from, in its input. It
    /// places the event among the input's messages and cites its bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<(u64, u64)>,
    /// The record's own fields, kept as written.
    pub data: Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkKind {
    /// A fork: the child shares the parent's messages up to `at`, then
    /// diverges. The child stores only its own tail.
    Fork,
    /// A session spawned by a tool call in the parent (a subagent).
    Spawn,
    /// The same work continued in a new session (after compaction or a
    /// resume).
    Continuation,
    /// The child's file repeats messages the parent originated; they are
    /// stored once, in the parent.
    Copy,
}

/// A relationship between two sessions, from an explicit marker in the
/// source, never from guessing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Link {
    pub kind: LinkKind,
    pub parent: SessionId,
    pub child: SessionId,
    /// The parent's message the child branched from or was spawned by.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<MessageId>,
    /// The tool call in the parent that spawned the child.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    /// The source field that states the link, for display and audit.
    pub evidence: String,
}

impl Link {
    /// One link per kind, parent and child, whichever files state it.
    pub fn key(&self) -> (LinkKind, &SessionId, &SessionId) {
        (self.kind, &self.parent, &self.child)
    }

    /// Fills the parts this statement of the link lacks from another
    /// statement of the same link (architecture 5.5a): a subagent's own file
    /// names its parent, the parent's progress record names the tool call.
    pub fn merge(&mut self, other: &Link) {
        debug_assert_eq!(self.key(), other.key());
        if self.at.is_none() {
            self.at.clone_from(&other.at);
        }
        if self.call_id.is_none() {
            self.call_id.clone_from(&other.call_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_parts_merge_from_either_side() {
        let s = |n: &str| SessionId::parse(n).unwrap();
        let mut from_child = Link {
            kind: LinkKind::Spawn,
            parent: s("claude-code:p"),
            child: s("claude-code:agent-1"),
            at: None,
            call_id: None,
            evidence: "agentId".into(),
        };
        let from_parent = Link {
            call_id: Some("toolu_1".into()),
            evidence: "agent_progress".into(),
            ..from_child.clone()
        };
        from_child.merge(&from_parent);
        assert_eq!(from_child.call_id.as_deref(), Some("toolu_1"));
        assert_eq!(from_child.evidence, "agentId");
    }
}
