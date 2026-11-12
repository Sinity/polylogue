use crate::{Json, MessageId, SessionId, Timestamp};
use serde::{Deserialize, Serialize};

/// A recorded fact from an input, optionally tied to a conversation.
/// Without a conversation, it belongs to its input and record range.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Event {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<SessionId>,
    /// Plain snake_case name: `compaction`, `permission_mode`, `hook_run`.
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<Timestamp>,
    #[serde(default = "crate::time::unknown_quality")]
    pub at_quality: crate::TimeQuality,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at_precision: Option<crate::Precision>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at_until: Option<Timestamp>,
    /// The message the event follows, when the source places it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<MessageId>,
    /// The byte range of the record the event came from, in its input. It
    /// places the event among the input's messages and cites its bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<(u64, u64)>,
    /// Native fields kept as written, or the typed data of a normalized
    /// observation (`observations`). A native event is used as written.
    pub data: Json,
}

/// Provider context metadata normalized once, while reading the source.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CompactionBoundary {
    pub summary: Option<MessageId>,
    pub pre_tokens: Option<u64>,
    pub post_tokens: Option<u64>,
    pub trigger: Option<String>,
    #[serde(default)]
    pub evicted: Vec<MessageId>,
    #[serde(default)]
    pub kept: Vec<MessageId>,
    pub kept_range: Option<KeptRange>,
    /// The recorded attachment point, separate from the retained segment.
    pub kept_after: Option<MessageId>,
    /// Native replacement records describe kept context, never new messages.
    #[serde(default)]
    pub replacement_history: Json,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KeptRange {
    pub first: MessageId,
    pub last: MessageId,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MicrocompactionBoundary {
    pub tokens_saved: Option<u64>,
    #[serde(default)]
    pub removed_calls: Vec<String>,
    #[serde(default)]
    pub removed_attachments: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LinkKind {
    Spawn,
    Fork,
    Continuation,
    Resume,
    Reference,
}

stored_names!(LinkKind {
    Spawn => "spawn",
    Fork => "fork",
    Continuation => "continuation",
    Resume => "resume",
    Reference => "reference",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceReason {
    Briefing,
    Ping,
    Mention,
}

stored_names!(ReferenceReason {
    Briefing => "briefing",
    Ping => "ping",
    Mention => "mention",
});

pub static LINK_KIND_NAMES: &[&str] = LinkKind::NAMES;
pub static LINK_HOW_NAMES: &[&str] = ReferenceReason::NAMES;

/// A relationship between two sessions, from an explicit marker in the
/// source, never from guessing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Link {
    pub kind: LinkKind,
    /// A recorded copied prefix is independent of the provider's relation kind.
    #[serde(default)]
    pub copies_prefix: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub how: Option<ReferenceReason>,
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

/// One input holds messages from the parent in the child's prefix. This
/// observation supplies no provider relationship between the sessions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct PrefixCopy {
    pub parent: SessionId,
    pub child: SessionId,
    /// Explicit members replace inference from this input's message rows.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub explicit_members: bool,
}

/// Original reading coordinates keep independently observed copy orders separate.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, schemars::JsonSchema,
)]
pub struct PrefixSource {
    pub input: String,
    pub chain: i64,
    pub reading_version: String,
}

/// One copied message, independent of any provider relationship.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct PrefixMember {
    pub parent: SessionId,
    pub child: SessionId,
    pub message: MessageId,
    /// Native parsers leave this absent; admission supplies their own reading.
    pub source: Option<PrefixSource>,
    /// Source record coordinate and position within that record. Export
    /// framing supplies the first coordinate for native parser observations.
    pub position: [u64; 2],
}

impl Link {
    /// Coordinates distinguish conflicting statements of the same relationship.
    /// Source paths and copied-prefix observations remain metadata of a statement.
    pub fn key(
        &self,
    ) -> (
        LinkKind,
        &SessionId,
        &SessionId,
        Option<&MessageId>,
        Option<&str>,
        Option<ReferenceReason>,
    ) {
        (
            self.kind,
            &self.parent,
            &self.child,
            self.at.as_ref(),
            self.call_id.as_deref(),
            self.how,
        )
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        match (self.kind, self.how) {
            (LinkKind::Reference, None) => Err("a reference link requires its reason"),
            (LinkKind::Reference, Some(_)) => Ok(()),
            (_, Some(_)) => Err("a provider link has no reference reason"),
            (_, None) => Ok(()),
        }
    }

    /// The portable identity of this recorded statement, including its coordinates.
    pub fn selection_tag(&self) -> String {
        blake3::hash(
            &serde_json::to_vec(&("polylogue.link.selection.v2", self.key()))
                .expect("typed link identity"),
        )
        .to_hex()
        .to_string()
    }
}

#[cfg(test)]
mod tests;

impl Event {
    pub fn time_evidence(&self) -> crate::TimeEvidence {
        crate::TimeEvidence {
            quality: self.at_quality,
            precision: self.at_precision,
            until: self.at_until,
        }
    }

    pub fn set_time(
        &mut self,
        at: Option<Timestamp>,
        evidence: crate::TimeEvidence,
    ) -> Result<(), crate::TimeEvidenceError> {
        evidence.validate(at)?;
        self.at = at;
        self.at_quality = evidence.quality;
        self.at_precision = evidence.precision;
        self.at_until = evidence.until;
        Ok(())
    }
}
