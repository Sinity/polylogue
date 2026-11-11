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
    /// Which route its readings are: the route precedence composes a
    /// message seen through several routes from them.
    pub route: polylogue_model::Route,
}

/// One look at a file that found something.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation {
    pub machine: String,
    pub host: String,
    pub path: String,
    /// Exact acquired location bytes; `path` is only its display label.
    pub path_bytes: Vec<u8>,
    pub facts: Option<FileFacts>,
    pub seen_at_ms: i64,
    /// Membership established by this explicit import, independent of byte deduplication.
    pub import: Option<ImportMember>,
    /// Exact membership in original container bytes, independent of imports.
    pub container: Option<InContainer>,
    /// A tool database's write-ahead log as the read that made this
    /// observation found it; `facts` are the database file's.
    pub database: Option<DatabaseLog>,
}

/// A tool database's write-ahead log beside its file: its facts, or `None`
/// when there was no log file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatabaseLog {
    pub wal: Option<FileFacts>,
}

/// Engine-authored membership of one import. It does not assert that
/// every expected export member was present or stored.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportMember {
    pub version: u8,
    pub import_id: String,
    pub member: String,
    /// Receiver-computed import context, independent of claims in the bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authorship: Option<polylogue_model::authorship::Authorship>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delivery: Option<polylogue_model::batch::ExportDelivery>,
    /// Message positions in reading order for array records, apart from byte ranges.
    pub order_from: Option<u64>,
    pub reading_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<polylogue_model::batch::ParserCheckpoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stopped_attempt: Option<polylogue_model::batch::StoppedAttempt>,
    #[serde(default)]
    pub(crate) receipt: Option<crate::wire_shapes::ReadingReceipt>,
    /// Confirmed surviving facts can omit source ranges despite a scratch reset.
    #[serde(default)]
    pub(crate) source_prefix_missing: bool,
    /// IDs from admitted products, separate from the explicit map in a sealed callback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) confirmed_event_ids: Option<Vec<String>>,
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
    /// How far the step read the stored bytes.
    pub progress: polylogue_model::batch::StepProgress,
    /// Packed message, text and event rows produced together on the parse pool.
    /// Default leaves the complete parsed values available for writer fallback.
    #[serde(skip)]
    pub(crate) prepared: PreparedRows,
    /// Row identities and chosen versions, set by the resolver before the
    /// step reaches the writer of `index.db` rows.
    #[serde(skip)]
    pub(crate) resolution: Option<super::resolve::Resolution>,
    pub(crate) comparison: Option<crate::tool_databases::Comparison>,
}

mod export;
mod prepared;
pub(crate) use prepared::PreparedRows;

impl ParseStep {
    pub fn validate_stopped_attempt(&self) -> crate::Result<()> {
        if self.stopped_attempt.is_some()
            && !matches!(
                self.progress,
                polylogue_model::batch::StepProgress::Failed { .. }
                    | polylogue_model::batch::StepProgress::Refused { .. }
            )
        {
            return Err(crate::StoreError::Invalid(
                "only a stopped parse step has a stopped attempt".into(),
            ));
        }
        Ok(())
    }

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
            delivery: None,
            order_from: None,
            reading_version: String::new(),
            checkpoint: None,
            stopped_attempt: None,
            receipt: None,
            source_prefix_missing: false,
            confirmed_event_ids: None,
            reset: range.start == 0,
            repeats_whole_conversations: false,
            output,
            counts,
            start: range.start,
            parsed_upto: range.end,
            progress: polylogue_model::batch::StepProgress::parsed(at_end),
            prepared: PreparedRows::default(),
            resolution: None,
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
    /// Receiver-minted evidence of the body acquisition, not parsed payload data.
    pub acquisition: Option<crate::primary::AcquisitionReceipt>,
    /// Actual source precondition sampled before provider IO. It has no
    /// authority to assign the resulting native chain or revision.
    pub basis: Option<crate::primary::SourceBasis>,
    /// Restricts this new acquisition to physical custody; it cannot request
    /// source acceptance or promote a previously unresolved receipt.
    pub required_disposition: Option<crate::primary::CaptureDisposition>,
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
    /// Session item units read to bring item indexes up to date: what the
    /// committed changes reached, not the length of their sessions.
    pub item_units_read: u64,
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
