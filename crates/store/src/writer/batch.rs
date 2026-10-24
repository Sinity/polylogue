//! What the parse pool hands the writer: a batch of one input's bytes and
//! parsed rows, and how blocks and events are packed for storage.

use super::*;

/// The input a batch belongs to, as `archive.db` names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputKey {
    /// `archive.db` input id: a location's stable id, or the BLAKE3 of an
    /// imported file's bytes.
    pub id: String,
    /// The origin that parses it (`claude-code`).
    pub format: String,
    /// How faithful the route is, for choosing between versions of a
    /// message (architecture 5.5): 0 live file, 1 spool, 2 import.
    pub route_rank: u8,
}

/// One look at a file that found something.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation {
    pub machine: String,
    pub host: String,
    pub path: String,
    pub facts: Option<FileFacts>,
    pub seen_at_ms: i64,
    /// Membership established by this explicit import, independent of byte deduplication.
    pub import: Option<ImportMember>,
}

/// Engine-authored membership of one acquisition. It does not assert that
/// every expected export member was present or successfully acquired.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportMember {
    pub version: u8,
    pub acquisition: String,
    pub member: String,
    /// Interpretation selected for this acquisition, separate from immutable Input metadata.
    #[serde(skip)]
    pub format: Option<String>,
    /// Persisted by the Writer in observation_container, not ignorable JSON metadata.
    #[serde(skip)]
    pub container: Option<ImportContainer>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportContainer {
    pub input: String,
    pub member_index: u64,
}

/// One parse step's rows and where they sit in the input's chain.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseStep {
    /// Logical message positions for array records, separate from raw spans.
    pub order_from: Option<u64>,
    /// Replaces accounting from an earlier parse of this chain.
    pub reset: bool,
    pub(crate) output: Output,
    /// Each record restates whole conversations: a later record's version
    /// of a message replaces an earlier one in the same chain.
    pub snapshot: bool,
    pub accounting: Accounting,
    /// Where the step's bytes start in the chain. 0 means the chain was
    /// parsed from the start, so the step's counts replace earlier ones.
    pub start: u64,
    /// Offset after the last complete record parsed.
    pub parsed_upto: u64,
    /// The parse observed source EOF; an unconsumed suffix waits for growth.
    pub at_end: bool,
    /// Packed message, text and event rows produced together on the parse pool.
    /// Default leaves the complete parsed values available for writer fallback.
    pub(crate) prepared: PreparedRows,
}

mod prepared;
pub(crate) use prepared::PreparedRows;

impl ParseStep {
    /// An unprepared parse over this byte range. A range beginning at zero
    /// replaces accounting; logical exports may set `reset` independently.
    pub fn new(
        output: Output,
        accounting: Accounting,
        range: std::ops::Range<u64>,
        at_end: bool,
    ) -> Self {
        Self {
            order_from: None,
            reset: range.start == 0,
            snapshot: false,
            output,
            accounting,
            start: range.start,
            parsed_upto: range.end,
            at_end,
            prepared: PreparedRows::default(),
        }
    }
}

/// What the writer stores for one input in one step. `piece` holds the new
/// bytes (already durable when they are a file); `parse` holds the rows
/// parsed from them, if the origin is parsed.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    pub input: InputKey,
    pub change: Change,
    /// Present on the first batch of a change; absent on the later chunks
    /// of a large file stored in several batches (`continues`).
    pub observation: Option<Observation>,
    /// This batch extends the revision the previous batch of this input
    /// started, instead of starting a new one.
    pub continues: bool,
    pub piece: Option<Piece>,
    /// Bytes records carried inline (base64 images), prepared like pieces;
    /// blocks name them by hash.
    pub blobs: Vec<Piece>,
    /// The chain's stored length after this batch.
    pub length: u64,
    pub parse: Vec<ParseStep>,
    /// Runs when the writer is done with the batch: after its transaction
    /// commits (the writer keeps a transaction's batches to retry them one
    /// at a time), or when a batch is dropped unsent.
    pub release: InFlight,
}

/// A callback that runs once, when the value holding it is dropped. The
/// parse pool puts one in each batch so a job's bytes stay counted in its
/// bytes-in-flight limit until the writer has committed the batch; parsed
/// rows waiting for the writer are memory too. A clone holds nothing.
#[derive(Default)]
pub struct InFlight(Option<Box<dyn FnOnce() + Send>>);

impl InFlight {
    pub fn new(f: impl FnOnce() + Send + 'static) -> Self {
        Self(Some(Box::new(f)))
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        if let Some(f) = self.0.take() {
            f();
        }
    }
}

impl Clone for InFlight {
    fn clone(&self) -> Self {
        Self(None)
    }
}

impl PartialEq for InFlight {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl std::fmt::Debug for InFlight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() {
            "InFlight"
        } else {
            "InFlight(none)"
        })
    }
}

/// What the writer did since the last flush.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WriteReport {
    pub batches: u64,
    pub messages: u64,
    pub transactions: u64,
    /// Batches the writer could not store, with the reason. The rest of
    /// their transaction was retried one batch at a time.
    pub errors: Vec<String>,
    /// Inputs whose progress must be reloaded from committed checkpoints.
    /// Later batches for these inputs are held out until this report is taken.
    pub failed_inputs: std::collections::BTreeSet<String>,
}
