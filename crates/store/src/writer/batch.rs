//! What the parse pool hands the writer: a batch of one input's bytes and
//! parsed rows, and how blocks and events are packed for storage.

use super::*;

/// The input a batch belongs to, as `archive.db` names it.
#[derive(
    Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct InputKey {
    /// `archive.db` input id: a location's stable id, or the BLAKE3 of an
    /// imported file's bytes.
    pub id: String,
    /// The origin that parses it (`claude-code`).
    pub format: String,
    /// How faithful the arrival is, for choosing between versions of a
    /// message: 0 live file, 1 spool, 2 import.
    pub arrival_rank: u8,
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
    /// Exact membership in original container bytes, independent of imports.
    pub container: Option<InContainer>,
}

/// Engine-authored membership of one import. It does not assert that
/// every expected export member was present or stored.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportMember {
    pub version: u8,
    pub import_id: String,
    pub member: String,
    /// The format chosen for this import, separate from the input's own metadata.
    #[serde(skip)]
    pub format: Option<String>,
}

#[derive(
    Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct InContainer {
    pub input: String,
    pub member_index: u64,
    pub kind: String,
    pub path_bytes: Vec<u8>,
}

/// One parse step's rows and where they sit in the input's chain.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ParseStep {
    /// Message positions in reading order for array records, apart from byte ranges.
    pub order_from: Option<u64>,
    pub reading_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<polylogue_model::batch::ParserCheckpoint>,
    #[serde(default)]
    pub(crate) receipt: Option<crate::wire_shapes::ReadingReceipt>,
    /// Confirmed surviving facts can omit source ranges despite a scratch reset.
    #[serde(default)]
    pub(crate) source_prefix_missing: bool,
    /// The bytes are collected, and their origin has no parser yet.
    #[serde(default)]
    pub no_parser_yet: bool,
    /// Replaces accounting from an earlier parse of this chain.
    pub reset: bool,
    pub(crate) output: Output,
    /// Each record restates whole conversations: a later record's version
    /// of a message replaces an earlier one in the same chain.
    pub repeats_whole_conversations: bool,
    pub counts: RecordCounts,
    /// Where the step's bytes start in the chain. 0 means the chain was
    /// parsed from the start, so the step's counts replace earlier ones.
    pub start: u64,
    /// Offset after the last complete record parsed.
    pub parsed_upto: u64,
    /// The parse observed source EOF; an unconsumed suffix waits for growth.
    pub at_end: bool,
    /// Packed message, text and event rows produced together on the parse pool.
    /// Default leaves the complete parsed values available for writer fallback.
    #[serde(skip)]
    pub(crate) prepared: PreparedRows,
    pub(crate) comparison: Option<crate::tool_databases::Comparison>,
}

mod prepared;
pub(crate) use prepared::PreparedRows;

impl ParseStep {
    /// The comparison hashes of the row-dump records parsed in this step.
    pub fn compare_records(&mut self, bytes: &[u8]) -> crate::Result<()> {
        let mut update = crate::tool_databases::Comparison::records(bytes)?;
        update.from = self.start;
        update.length = self.parsed_upto;
        self.comparison = Some(update);
        Ok(())
    }

    /// Whole row-dump parsers commit their comparison with completion.
    pub fn compare_reader(&mut self, reader: impl std::io::BufRead) -> crate::Result<()> {
        let mut update = crate::tool_databases::Comparison::read(reader)?;
        update.from = self.start;
        update.length = self.parsed_upto;
        self.comparison = Some(update);
        Ok(())
    }

    /// An unprepared parse over this byte range. A range beginning at zero
    /// replaces the record counts; row dumps may set `reset` independently.
    pub fn new(
        output: Output,
        counts: RecordCounts,
        range: std::ops::Range<u64>,
        at_end: bool,
    ) -> Self {
        Self {
            order_from: None,
            reading_version: String::new(),
            checkpoint: None,
            receipt: None,
            source_prefix_missing: false,
            no_parser_yet: false,
            reset: range.start == 0,
            repeats_whole_conversations: false,
            output,
            counts,
            start: range.start,
            parsed_upto: range.end,
            at_end,
            prepared: PreparedRows::default(),
            comparison: None,
        }
    }
}

/// What the writer stores for one input in one step. `new blob` holds the new
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
    pub new_blob: Option<NewBlob>,
    /// Bytes records carried inline (base64 images), prepared like new blobs;
    /// blocks name them by hash.
    pub blobs: Vec<NewBlob>,
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
    pub segment_syncs: u64,
    pub segment_bytes_written: u64,
    pub blobs_appended: u64,
    pub blobs_deduplicated: u64,
    /// Batches the writer could not store, with the reason. The rest of
    /// their transaction was retried one batch at a time.
    pub errors: Vec<String>,
}

impl InContainer {
    pub fn display_path(&self) -> String {
        String::from_utf8(self.path_bytes.clone()).unwrap_or_else(|_| {
            self.path_bytes
                .iter()
                .map(|byte| format!("%{byte:02X}"))
                .collect()
        })
    }
}
