//! What a parse step hands the writer: rows, the bytes they carried inline,
//! and the accounting that proves every record was dealt with. One type for
//! the parser and the writer, so neither restates the other.

use crate::{BlobHash, Event, Link, Message, Session, SessionId, ToolOutcome};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Rows one parse step produced. Sessions and messages are upserts keyed by
/// their IDs: a later step may send a longer version of the same message
/// (one still being written), and writing it again is harmless.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Output {
    /// Facts about every session the step touched. The writer merges them
    /// into what it already holds ([`Session::merge`]).
    pub sessions: Vec<Session>,
    pub messages: Vec<MessageRow>,
    pub events: Vec<Event>,
    /// Upserted by [`Link::key`]; parts one file lacks are filled from
    /// another ([`Link::merge`]).
    pub links: Vec<Link>,
    /// Outcomes learned after the result was written, for tool calls that
    /// ran in the background.
    pub background: Vec<BackgroundUpdate>,
    /// Bytes the input carried inline (base64 images), for the blob store.
    /// Blocks refer to them by hash.
    pub blobs: Vec<InlineBlob>,
    /// Repository lookups (a `polylogue-repo-lookup` input holds one).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub repo_lookups: Vec<RepoLookup>,
}

/// What a repository directory is, as read on this host: where its `.git`
/// is, its `origin` remote and its first commit. Sessions whose working
/// directory lies under `path` belong to it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepoLookup {
    pub path: String,
    /// The remote's URL with any credentials removed.
    pub remote: Option<String>,
    /// The repository's identity; `None` when it could not be read.
    pub root_commit: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InlineBlob {
    pub hash: BlobHash,
    pub size: u64,
    #[serde(skip)]
    pub bytes: Vec<u8>,
}

/// A message as one input holds it. `offset` is where its first record
/// starts in the input and `end` where its last record ends, so the stored
/// byte range covers every record the message was built from. The same
/// message at the same offset may be sent again, longer (a response still
/// being written); the writer replaces it. At another offset or in another
/// input it is another version; the writer keeps every version and prefers
/// one (architecture 5.5, 5.5a).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageRow {
    pub offset: u64,
    pub end: u64,
    pub message: Message,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackgroundUpdate {
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
/// inherited messages first (through its lineage link), then its own by key.
/// The key is the input and byte offset of the record that opened the
/// message, so a response that closes after a later user message still
/// sorts where it opened. `input` is the store's small key for the input,
/// assigned in the order inputs were first seen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct OrderKey {
    pub input: i64,
    pub offset: u64,
}

/// How one framed record ended. Every record ends in exactly one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Disposition {
    /// The record became rows: messages, events, links, session facts.
    Materialized,
    /// Understood well enough to name, not turned into rows. The reason is
    /// shown with the count, so a reader knows what was set aside and why.
    Unknown { reason: String },
    /// Not a readable record: malformed JSON, a value of the wrong shape.
    /// The bytes stay stored.
    Refused { reason: String },
}

impl Disposition {
    pub fn unknown(reason: impl Into<String>) -> Self {
        Self::Unknown {
            reason: reason.into(),
        }
    }

    pub fn refused(reason: impl Into<String>) -> Self {
        Self::Refused {
            reason: reason.into(),
        }
    }
}

/// Counts per record kind and disposition for one parse step. The counts
/// must add up to the records framed; the writer refuses a step whose
/// counts do not.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Accounting {
    pub framed: u64,
    pub materialized: BTreeMap<String, u64>,
    /// Keyed by `record kind: reason`.
    pub unknown: BTreeMap<String, u64>,
    pub refused: BTreeMap<String, u64>,
    /// Wire fields neither read nor excluded, by `record kind.path`. Not a
    /// failure: it shows where a parser falls behind its format.
    pub unread_fields: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
#[error("{framed} records framed but {accounted} accounted for")]
pub struct CountMismatch {
    pub framed: u64,
    pub accounted: u64,
}

impl Accounting {
    pub fn add(&mut self, kind: &str, disposition: &Disposition) {
        let bump = |m: &mut BTreeMap<_, u64>, k| *m.entry(k).or_default() += 1;
        match disposition {
            Disposition::Materialized => bump(&mut self.materialized, kind.to_owned()),
            Disposition::Unknown { reason } => bump(&mut self.unknown, format!("{kind}: {reason}")),
            Disposition::Refused { reason } => bump(&mut self.refused, format!("{kind}: {reason}")),
        }
    }

    pub fn unread(&mut self, kind: &str, path: &str) {
        *self
            .unread_fields
            .entry(format!("{kind}.{path}"))
            .or_default() += 1;
    }

    pub fn accounted(&self) -> u64 {
        self.materialized.values().sum::<u64>()
            + self.unknown.values().sum::<u64>()
            + self.refused.values().sum::<u64>()
    }

    pub fn check(&self) -> Result<(), CountMismatch> {
        let accounted = self.accounted();
        if accounted == self.framed {
            Ok(())
        } else {
            Err(CountMismatch {
                framed: self.framed,
                accounted,
            })
        }
    }

    pub fn merge(&mut self, other: &Accounting) {
        self.framed += other.framed;
        let add = |into: &mut BTreeMap<String, u64>, from: &BTreeMap<String, u64>| {
            for (k, v) in from {
                *into.entry(k.clone()).or_default() += v;
            }
        };
        add(&mut self.materialized, &other.materialized);
        add(&mut self.unknown, &other.unknown);
        add(&mut self.refused, &other.refused);
        add(&mut self.unread_fields, &other.unread_fields);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_must_add_up() {
        let mut a = Accounting {
            framed: 3,
            ..Accounting::default()
        };
        a.add("user", &Disposition::Materialized);
        a.add("progress", &Disposition::unknown("tick"));
        assert_eq!(
            a.check(),
            Err(CountMismatch {
                framed: 3,
                accounted: 2
            })
        );
        a.add("?", &Disposition::refused("malformed JSON"));
        assert_eq!(a.check(), Ok(()));
    }

    #[test]
    fn merge_adds_every_map() {
        let mut a = Accounting {
            framed: 1,
            ..Accounting::default()
        };
        a.add("user", &Disposition::Materialized);
        let mut b = a.clone();
        b.unread("user", "x");
        a.merge(&b);
        assert_eq!(a.framed, 2);
        assert_eq!(a.materialized["user"], 2);
        assert_eq!(a.unread_fields["user.x"], 1);
    }

    #[test]
    fn order_key_sorts_by_input_then_offset() {
        let mut keys = [
            OrderKey {
                input: 2,
                offset: 0,
            },
            OrderKey {
                input: 1,
                offset: 90,
            },
            OrderKey {
                input: 1,
                offset: 10,
            },
        ];
        keys.sort();
        assert_eq!(
            keys[0],
            OrderKey {
                input: 1,
                offset: 10
            }
        );
        assert_eq!(
            keys[2],
            OrderKey {
                input: 2,
                offset: 0
            }
        );
    }
}
