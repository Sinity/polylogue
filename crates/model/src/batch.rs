//! What a parse step hands the writer: rows, the bytes they carried inline,
//! and the record counts that show every record was dealt with. One type for
//! the parser and the writer, so neither restates the other.

use crate::{BlobHash, Document, Event, Link, Message, Session, SessionId, ToolOutcome};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Rows one parse step produced. Sessions and messages are upserts keyed by
/// their IDs: a later step may send a longer version of the same message
/// (one still being written), and writing it again is harmless.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Output {
    /// Facts about every session the step touched. The writer merges them
    /// into what it already holds ([`Session::merge`]).
    pub sessions: Vec<Session>,
    pub messages: Vec<MessageRow>,
    /// Textual provider records without conversation roles or usage.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub documents: Vec<DocumentRow>,
    pub events: Vec<Event>,
    /// Stable exported identities and their original source claims, indexed by event.
    /// The containing input still owns the local bytes.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub imported_events: BTreeMap<usize, crate::bundle::ImportedEventIdentity>,
    /// Explicit session reading order, indexed by message; byte ranges stay local.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub message_order: BTreeMap<usize, u64>,
    /// Upserted by [`Link::key`]; parts one file lacks are filled from
    /// another ([`Link::merge`]).
    pub links: Vec<Link>,
    /// Outcomes learned after the result was written, for tool calls that
    /// ran in the background.
    pub background: Vec<BackgroundUpdate>,
    /// Bytes the input carried inline (base64 images), for the blob store.
    /// Blocks refer to them by hash.
    pub blobs: Vec<ExtractedBlob>,
    /// Repository lookups (a `polylogue-repo-lookup` input holds one).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub repo_lookups: Vec<RepoLookup>,
}

/// What a repository directory is, as read on this host: where its `.git`
/// is, its `origin` remote and its first commit. Sessions whose working
/// directory lies under `path` belong to it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RepoLookup {
    pub path: String,
    /// The remote's URL with any credentials removed.
    pub remote: Option<String>,
    /// The repository's identity; `None` when it could not be read.
    pub root_commit: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ExtractedBlob {
    pub hash: BlobHash,
    pub size: u64,
    #[serde(skip)]
    pub bytes: Vec<u8>,
}

/// A document as one input holds it, with the record's byte range.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DocumentRow {
    pub offset: u64,
    pub end: u64,
    pub document: Document,
}

/// A message as one input holds it. `offset` is where its first record
/// starts in the input and `end` where its last record ends, so the stored
/// byte range covers every record the message was built from. The same
/// message at the same offset may be sent again, longer (a response still
/// being written); the writer replaces it. At another offset or in another
/// input it is another version; the writer keeps every version and prefers
/// one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct MessageRow {
    pub offset: u64,
    pub end: u64,
    pub message: Message,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BackgroundUpdate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_at: Option<crate::Timestamp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reported_duration_ms: Option<u64>,
    pub session: SessionId,
    pub task_id: String,
    /// The tool call that started the task, when the notice names it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    pub outcome: ToolOutcome,
    /// Where the task's full output was written, when the notice says.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_file: Option<String>,
}

/// Where a message sits in its session. A session reads in this order:
/// its parents' messages first (through its links), then its own by key.
/// The key is the input and byte offset of the record that opened the
/// message, so a response that closes after a later user message still
/// sorts where it opened. `input` is the store's small key for the input,
/// assigned in the order inputs were first seen.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
pub struct OrderKey {
    pub input: i64,
    pub offset: u64,
}

/// How one record ended. Every record ends in exactly one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RecordOutcome {
    /// The record became rows: messages, events, links, session facts.
    Used,
    /// Understood, and set aside rather than turned into rows. The reason is
    /// shown with the count, so a reader knows what was set aside and why.
    Skipped { reason: String },
    /// Not a record the parser can read: malformed JSON, a value of the
    /// wrong shape. The bytes stay stored.
    Unreadable { reason: String },
}

impl RecordOutcome {
    pub fn skipped(reason: impl Into<String>) -> Self {
        Self::Skipped {
            reason: reason.into(),
        }
    }

    pub fn unreadable(reason: impl Into<String>) -> Self {
        Self::Unreadable {
            reason: reason.into(),
        }
    }
}

/// Derived wire observations contain shape and declared discriminator sets,
/// never scalar bodies from the input. The writer gives these their receipt time.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WireShape {
    pub record_kind: String,
    pub path: String,
    pub types: BTreeMap<String, u64>,
    pub expected: Vec<String>,
    pub dynamic_map: bool,
    pub known_kinds: Vec<String>,
    pub unknown_kind_count: u64,
    pub unused_count: u64,
    pub table: Option<String>,
    pub unknown_table_count: u64,
}
impl WireShape {
    pub fn merge(&mut self, other: &Self) {
        self.record_kind.clone_from(&other.record_kind);
        self.path.clone_from(&other.path);
        self.unknown_kind_count += other.unknown_kind_count;
        self.unused_count += other.unused_count;
        self.unknown_table_count += other.unknown_table_count;
        if other.table.is_some() {
            self.table.clone_from(&other.table);
        }
        for (kind, count) in &other.types {
            *self.types.entry(kind.clone()).or_default() += count;
        }
        if self.expected.is_empty() {
            self.expected.clone_from(&other.expected);
        }
        self.dynamic_map |= other.dynamic_map;
        if self.known_kinds.is_empty() {
            self.known_kinds.clone_from(&other.known_kinds);
        }
    }
}

/// Outcomes and derived wire observations of one parse step. Outcome counts
/// sum to its records; the writer refuses a step whose counts do not.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RecordCounts {
    /// Derived observations travel with scratch steps through the writer boundary.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub wire_shapes: BTreeMap<String, WireShape>,
    pub records: u64,
    pub used: BTreeMap<String, u64>,
    /// Keyed by `record kind: reason`.
    pub skipped: BTreeMap<String, u64>,
    pub unreadable: BTreeMap<String, u64>,
    /// Wire leaves neither read nor retained in parsed rows, by `record kind.path`. Not a
    /// failure: it shows where a parser falls behind its format.
    pub unread_fields: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
#[error("{records} records read but {counted} counted")]
pub struct CountMismatch {
    pub records: u64,
    pub counted: u64,
}

impl RecordCounts {
    pub fn add(&mut self, kind: &str, outcome: &RecordOutcome) {
        let bump = |m: &mut BTreeMap<_, u64>, k| *m.entry(k).or_default() += 1;
        match outcome {
            RecordOutcome::Used => bump(&mut self.used, kind.to_owned()),
            RecordOutcome::Skipped { reason } => {
                bump(&mut self.skipped, format!("{kind}: {reason}"))
            }
            RecordOutcome::Unreadable { reason } => {
                bump(&mut self.unreadable, format!("{kind}: {reason}"))
            }
        }
    }

    pub fn unread(&mut self, kind: &str, path: &str) {
        *self
            .unread_fields
            .entry(format!("{kind}.{path}"))
            .or_default() += 1;
    }

    pub fn counted(&self) -> u64 {
        self.used.values().sum::<u64>()
            + self.skipped.values().sum::<u64>()
            + self.unreadable.values().sum::<u64>()
    }

    pub fn check(&self) -> Result<(), CountMismatch> {
        let counted = self.counted();
        if counted == self.records {
            Ok(())
        } else {
            Err(CountMismatch {
                records: self.records,
                counted,
            })
        }
    }

    pub fn merge(&mut self, other: &RecordCounts) {
        self.records += other.records;
        for (path, shape) in &other.wire_shapes {
            self.wire_shapes
                .entry(path.clone())
                .or_default()
                .merge(shape);
        }
        let add = |into: &mut BTreeMap<String, u64>, from: &BTreeMap<String, u64>| {
            for (k, v) in from {
                *into.entry(k.clone()).or_default() += v;
            }
        };
        add(&mut self.used, &other.used);
        add(&mut self.skipped, &other.skipped);
        add(&mut self.unreadable, &other.unreadable);
        add(&mut self.unread_fields, &other.unread_fields);
    }
}

#[cfg(test)]
mod tests;
