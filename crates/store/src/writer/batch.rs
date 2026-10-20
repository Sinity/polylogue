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
}

/// One parse step's rows and where they sit in the input's chain.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseStep {
    pub output: Output,
    pub snapshot: Option<polylogue_model::Snapshot>,
    pub accounting: Accounting,
    /// Where the step's bytes start in the chain. 0 means the chain was
    /// parsed from the start, so the step's counts replace earlier ones.
    pub start: u64,
    /// Offset after the last complete record parsed.
    pub parsed_upto: u64,
    pub parser_version: u32,
    /// The parse observed source EOF; an unconsumed suffix waits for growth.
    pub at_end: bool,
    /// BLAKE3 of each message's canonical JSON, in `output.messages` order,
    /// computed on the parse pool ([`ParseStep::hash_messages`]). Empty
    /// means the writer computes them.
    pub hashes: Vec<[u8; 32]>,
    /// Block texts compressed for the `text` table, by BLAKE3 of the text,
    /// prepared on the parse pool. A text missing here is compressed by the
    /// writer.
    pub packed: HashMap<[u8; 32], Vec<u8>>,
    /// Each message's blocks as stored (compressed JSON without their
    /// text), in `output.messages` and block order.
    pub bodies: Vec<Vec<Vec<u8>>>,
    /// Each event's id and compressed payload, in `output.events` order.
    pub events: Vec<(String, Vec<u8>)>,
}

impl ParseStep {
    /// Does the per-message work that needs no database off the writer
    /// thread: version hashes and compressed block texts.
    pub fn prepare(&mut self) {
        self.hashes = self
            .output
            .messages
            .iter()
            .map(|r| message_hash(&r.message))
            .collect();
        let mut packer = crate::encoding::TextPacker::new().ok();
        for row in &self.output.messages {
            for block in &row.message.blocks {
                let Some(text) = block_text(block) else {
                    continue;
                };
                let hash = *blake3::hash(text.as_bytes()).as_bytes();
                if let (std::collections::hash_map::Entry::Vacant(slot), Some(p)) =
                    (self.packed.entry(hash), packer.as_mut())
                    && let Ok(bytes) = p.pack(text)
                {
                    slot.insert(bytes);
                }
            }
        }
        let prepared: Result<Vec<Vec<Vec<u8>>>> = self
            .output
            .messages
            .iter()
            .map(|r| r.message.blocks.iter().map(pack_block).collect())
            .collect();
        self.bodies = prepared.unwrap_or_default();
        let events: Result<Vec<(String, Vec<u8>)>> =
            self.output.events.iter().map(pack_event).collect();
        self.events = events.unwrap_or_default();
        self.release_packed();
    }

    /// Drops the parsed copies of what the packed bodies already hold, so a
    /// batch waiting for the writer keeps its bulky JSON once, compressed.
    /// The writer reads only text (for the search index), IDs, outcomes,
    /// commands, paths and attachment facts from parsed blocks.
    fn release_packed(&mut self) {
        let packed_all = self.bodies.len() == self.output.messages.len()
            && self
                .bodies
                .iter()
                .zip(&self.output.messages)
                .all(|(b, r)| b.len() == r.message.blocks.len());
        if packed_all {
            for row in &mut self.output.messages {
                for block in &mut row.message.blocks {
                    match block {
                        Block::ToolCall { input, .. } => *input = Json::Null,
                        Block::ToolResult { data, .. } => *data = None,
                        Block::Other { data, .. } => *data = Json::Null,
                        _ => {}
                    }
                }
            }
        }
        if self.events.len() == self.output.events.len() {
            for e in &mut self.output.events {
                e.data = Json::Null;
            }
        }
    }
}

pub(super) fn message_hash(m: &Message) -> [u8; 32] {
    let json = serde_json::to_vec(m).unwrap_or_default();
    *blake3::hash(&json).as_bytes()
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
}
