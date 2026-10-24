use crate::{Json, MessageId, SessionId, Timestamp};
use serde::{Deserialize, Serialize};

/// A retained observation, optionally associated with a conversation.
/// Without that association, its Input and record range own the fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<SessionId>,
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
    /// Native fields kept as written, or the typed data of a normalized
    /// observation (`observations`). Native events remain authoritative.
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

impl LinkKind {
    pub const ALL: &'static [LinkKind] = &[Self::Fork, Self::Spawn, Self::Continuation, Self::Copy];

    /// The stored `(kind, how)` pair: the link table and the query language
    /// name a link's family (`subagent`, `continuation`) and how it was made.
    pub const fn columns(self) -> (&'static str, &'static str) {
        match self {
            Self::Spawn => ("subagent", "fresh"),
            Self::Fork => ("continuation", "fork"),
            Self::Copy => ("continuation", "resume"),
            Self::Continuation => ("continuation", "after_compaction"),
        }
    }

    /// The kind a stored `(kind, how)` pair names, if this version knows it.
    pub fn from_columns(kind: &str, how: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|&k| {
            let (family, made) = k.columns();
            // Any subagent link is a spawn, however it was made.
            family == kind && (made == how || k == Self::Spawn)
        })
    }
}

const LINK_COUNT: usize = LinkKind::ALL.len();

const fn link_column(second: bool) -> ([&'static str; LINK_COUNT], usize) {
    let mut out = [""; LINK_COUNT];
    let mut i = 0;
    while i < LINK_COUNT {
        let (kind, how) = LinkKind::ALL[i].columns();
        out[i] = if second { how } else { kind };
        i += 1;
    }
    crate::stored_names::distinct(out)
}

static LINK_KINDS: ([&str; LINK_COUNT], usize) = link_column(false);
static LINK_HOWS: ([&str; LINK_COUNT], usize) = link_column(true);

/// Every stored link `kind` (the first of [`LinkKind::columns`]).
pub static LINK_KIND_NAMES: &[&str] = LINK_KINDS.0.split_at(LINK_KINDS.1).0;
/// Every stored link `how` (the second of [`LinkKind::columns`]).
pub static LINK_HOW_NAMES: &[&str] = LINK_HOWS.0.split_at(LINK_HOWS.1).0;

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
mod tests;
