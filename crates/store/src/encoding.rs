//! Encoding of database row bodies. Original-input blob encoding belongs to `blobs`.

use crate::{Result, StoreError};
use polylogue_model::{
    Block, Document, DocumentId, Event, Message, MessageId, Timestamp, bundle::MessageProjection,
};
use serde::{Serialize, de::DeserializeOwned};
use std::io::Read;
use std::sync::LazyLock;

// Fast row-body compression; the format remains readable across compression levels.
const LEVEL: i32 = 1;

// Fixed for one index generation. Archive rows use pack_bytes
// below; only RowPacker may introduce dictionary frames.
const ROW_DICTIONARY: &[u8] = include_bytes!("encoding/rows.dict");
static ROW_ENCODER: LazyLock<zstd::dict::EncoderDictionary<'static>> =
    LazyLock::new(|| zstd::dict::EncoderDictionary::copy(ROW_DICTIONARY, LEVEL));
static ROW_DECODER: LazyLock<zstd::dict::DecoderDictionary<'static>> =
    LazyLock::new(|| zstd::dict::DecoderDictionary::copy(ROW_DICTIONARY));

pub(crate) fn pack_bytes(bytes: &[u8]) -> std::io::Result<Vec<u8>> {
    zstd::bulk::compress(bytes, LEVEL)
}

pub(crate) fn unpack_bytes(packed: &[u8]) -> std::io::Result<Vec<u8>> {
    if zstd::zstd_safe::get_dict_id_from_frame(packed).is_none() {
        return zstd::stream::decode_all(packed);
    }
    let mut decoder = zstd::stream::read::Decoder::with_prepared_dictionary(packed, &ROW_DECODER)?;
    let mut bytes = Vec::new();
    decoder.read_to_end(&mut bytes)?;
    Ok(bytes)
}

pub(crate) fn pack_text(text: &str) -> std::io::Result<Vec<u8>> {
    pack_bytes(text.as_bytes())
}

pub(crate) fn unpack_text(packed: &[u8]) -> Result<String> {
    String::from_utf8(unpack_bytes(packed)?).map_err(|e| StoreError::Io(std::io::Error::other(e)))
}

/// Packs JSON without the row dictionary, for test rows; the writer uses
/// [`RowPacker`].
#[cfg(any(test, feature = "fixture"))]
pub(crate) fn pack_json(value: &impl Serialize) -> Result<Vec<u8>> {
    Ok(pack_bytes(&serde_json::to_vec(value)?)?)
}

/// Canonical source time; SQL milliseconds are only query caches.
pub(crate) fn time_bytes(
    at: Option<Timestamp>,
    evidence: polylogue_model::TimeEvidence,
) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&(at, evidence))?)
}

pub(crate) fn read_time(
    row: &rusqlite::Row<'_>,
    column: usize,
) -> rusqlite::Result<(Option<Timestamp>, polylogue_model::TimeEvidence)> {
    let bytes: Vec<u8> = row.get(column)?;
    serde_json::from_slice(&bytes).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            column,
            rusqlite::types::Type::Blob,
            Box::new(error),
        )
    })
}

/// One message's identity and packed rows, prepared together before its parsed
/// payloads are released. The writer borrows these bytes without repacking.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PreparedMessage {
    pub(crate) id: MessageId,
    pub(crate) projection: MessageProjection,
    pub(crate) time: Vec<u8>,
    pub(crate) body: Vec<u8>,
    content: PreparedContent,
}

/// Encoded content rows shared by role-bearing messages and textual documents.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PreparedContent {
    pub(crate) hash: [u8; 32],
    pub(crate) blocks: Vec<Vec<u8>>,
    pub(crate) texts: Vec<Option<[u8; 32]>>,
    /// Each block's structured value, packed for `block.data` ([`block_data`]).
    pub(crate) data: Vec<Option<Vec<u8>>>,
}

impl std::ops::Deref for PreparedMessage {
    type Target = PreparedContent;
    fn deref(&self) -> &Self::Target {
        &self.content
    }
}

#[cfg(test)]
impl std::ops::DerefMut for PreparedMessage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.content
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PreparedDocument {
    pub(crate) id: DocumentId,
    pub(crate) time: Vec<u8>,
    /// The document without its blocks, packed JSON.
    pub(crate) fields: Vec<u8>,
    pub(crate) body: Vec<u8>,
    content: PreparedContent,
}

impl std::ops::Deref for PreparedDocument {
    type Target = PreparedContent;
    fn deref(&self) -> &Self::Target {
        &self.content
    }
}

impl PreparedDocument {
    pub(crate) fn new(document: &Document) -> Result<Self> {
        RowPacker::new()?.pack_document(document)
    }
}

impl PreparedMessage {
    pub(crate) fn new(message: &Message) -> Result<Self> {
        RowPacker::new()?.pack_message(message)
    }

    pub(crate) fn projected(message: &Message, projection: &MessageProjection) -> Result<Self> {
        RowPacker::new()?.pack_projected_message(message, projection)
    }
}

pub(crate) fn read_json<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    let mut reader = serde_json::Deserializer::from_slice(bytes);
    reader.disable_recursion_limit();
    let value = T::deserialize(&mut reader)?;
    reader.end()?;
    Ok(value)
}

pub(crate) fn unpack_json<T: DeserializeOwned>(packed: &[u8]) -> Result<T> {
    read_json(&unpack_bytes(packed)?)
}

/// One independent frame per row; both encoders are reused for the batch.
pub(crate) struct RowPacker {
    plain: zstd::bulk::Compressor<'static>,
    rows: zstd::bulk::Compressor<'static>,
}

impl RowPacker {
    pub(crate) fn new() -> std::io::Result<Self> {
        Ok(Self {
            plain: zstd::bulk::Compressor::new(LEVEL)?,
            rows: zstd::bulk::Compressor::with_prepared_dictionary(&ROW_ENCODER)?,
        })
    }

    pub(crate) fn pack_text(&mut self, text: &str) -> std::io::Result<Vec<u8>> {
        self.plain.compress(text.as_bytes())
    }

    fn pack_row(&mut self, bytes: &[u8]) -> std::io::Result<Vec<u8>> {
        let plain = self.plain.compress(bytes)?;
        let dictionary = self.rows.compress(bytes)?;
        Ok(if dictionary.len() < plain.len() {
            dictionary
        } else {
            plain
        })
    }

    pub(crate) fn pack_json(&mut self, value: &impl Serialize) -> Result<Vec<u8>> {
        Ok(self.pack_row(&serde_json::to_vec(value)?)?)
    }

    pub(crate) fn pack_event(&mut self, event: &Event) -> Result<(String, Vec<u8>)> {
        let (id, data) = event_bytes(event)?;
        Ok((id, self.pack_row(&data)?))
    }

    pub(crate) fn pack_message(&mut self, message: &Message) -> Result<PreparedMessage> {
        let projection = MessageProjection {
            source_version: message.content_hash()?,
            block_count: message.blocks.len() as u64,
            block_ordinals: (0..message.blocks.len() as u64).collect(),
        };
        self.pack_message_value(message, &projection)
    }

    pub(crate) fn pack_projected_message(
        &mut self,
        message: &Message,
        projection: &MessageProjection,
    ) -> Result<PreparedMessage> {
        projection.validate(message.blocks.len())?;
        if projection.block_count == message.blocks.len() as u64
            && message.content_hash()? != projection.source_version
        {
            return Err(polylogue_model::bundle::BundleError::SourceVersionMismatch.into());
        }
        self.pack_message_value(message, projection)
    }

    fn pack_message_value(
        &mut self,
        message: &Message,
        projection: &MessageProjection,
    ) -> Result<PreparedMessage> {
        Ok(PreparedMessage {
            id: message.id.clone(),
            projection: projection.clone(),
            time: time_bytes(message.at, message.time_evidence())?,
            body: self.pack_version(message, &message.blocks)?,
            content: self.pack_content(&message.blocks, projection.source_version.0)?,
        })
    }

    pub(crate) fn pack_document(&mut self, document: &Document) -> Result<PreparedDocument> {
        let fields = Document {
            id: document.id.clone(),
            purpose: document.purpose,
            title: document.title.clone(),
            source_time: document.source_time,
            source_time_quality: document.source_time_quality,
            source_time_precision: document.source_time_precision,
            source_time_until: document.source_time_until,
            authorship: document.authorship,
            blocks: Vec::new(),
            metadata: document.metadata.clone(),
        };
        let hash = {
            let mut sorted = serde_json::to_value(document)?;
            sorted.sort_all_objects();
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"polylogue-document-v1\0");
            hasher.update(&serde_json::to_vec(&sorted)?);
            *hasher.finalize().as_bytes()
        };
        Ok(PreparedDocument {
            id: document.id.clone(),
            time: time_bytes(document.source_time, document.time_evidence())?,
            body: self.pack_version(document, &document.blocks)?,
            fields: self.pack_json(&fields)?,
            content: self.pack_content(&document.blocks, hash)?,
        })
    }

    fn pack_version(&mut self, value: &impl Serialize, blocks: &[Block]) -> Result<Vec<u8>> {
        let mut value = serde_json::to_value(value)?;
        for (stored, block) in value["blocks"]
            .as_array_mut()
            .expect("blocks are an array")
            .iter_mut()
            .zip(blocks)
        {
            if block_text(block).is_some() {
                let field = if matches!(block, Block::ToolResult { .. }) {
                    "output"
                } else {
                    "text"
                };
                stored[field] = serde_json::Value::String(String::new());
            }
        }
        self.pack_json(&value)
    }

    fn pack_content(&mut self, blocks: &[Block], hash: [u8; 32]) -> Result<PreparedContent> {
        let mut packed = Vec::with_capacity(blocks.len());
        let mut data = Vec::with_capacity(blocks.len());
        for block in blocks {
            let bytes = block_bytes(block)?;
            packed.push(self.pack_row(&bytes)?);
            data.push(
                block_data(block)
                    .map(|value| self.pack_json(value))
                    .transpose()?,
            );
        }
        Ok(PreparedContent {
            hash,
            texts: blocks
                .iter()
                .map(|block| {
                    block_text(block).map(|text| *blake3::hash(text.as_bytes()).as_bytes())
                })
                .collect(),
            blocks: packed,
            data,
        })
    }
}

/// One-shot block encoding for compatibility checks.
#[cfg(test)]
pub(crate) fn pack_block(block: &Block) -> Result<Vec<u8>> {
    Ok(pack_bytes(&block_bytes(block)?)?)
}

fn block_bytes(block: &Block) -> Result<Vec<u8>> {
    let body = match block {
        Block::Text { .. } => Block::Text {
            text: String::new(),
        },
        Block::Thinking {
            text: Some(_),
            signature,
            redacted,
        } => Block::Thinking {
            text: Some(String::new()),
            signature: signature.clone(),
            redacted: *redacted,
        },
        Block::ToolResult {
            reported_duration_ms,
            call_id,
            outcome,
            output,
            data,
            attachments,
        } if output.is_some() || data.is_some() => Block::ToolResult {
            reported_duration_ms: *reported_duration_ms,
            call_id: call_id.clone(),
            outcome: outcome.clone(),
            output: output.as_ref().map(|_| String::new()),
            data: None,
            attachments: attachments.clone(),
        },
        // These values are stored whole, so serialization can borrow them.
        _ => return Ok(serde_json::to_vec(block)?),
    };
    Ok(serde_json::to_vec(&body)?)
}

/// An event's id includes its message anchor: identical state recorded at
/// different messages is distinct. Physical ranges do not change the fact's
/// identity when another input supplies it.
#[cfg(test)]
pub(crate) fn pack_event(e: &Event) -> Result<(String, Vec<u8>)> {
    let (id, data) = event_bytes(e)?;
    Ok((id, pack_bytes(&data)?))
}

fn event_bytes(e: &Event) -> Result<(String, Vec<u8>)> {
    let data = serde_json::to_vec(&e.data)?;
    if let Some(id) = reading_id(e) {
        return Ok((id, data));
    }
    let mut h = blake3::Hasher::new();
    if let Some(session) = &e.session {
        h.update(session.to_string().as_bytes());
    } else {
        h.update(b"\0input-event-content-v1\0");
    }
    h.update(b"\0");
    h.update(e.kind.as_bytes());
    h.update(b"\0");
    h.update(
        &e.at
            .map(Timestamp::unix_ms)
            .unwrap_or(i64::MIN)
            .to_le_bytes(),
    );
    if e.session.is_none() {
        let mut sorted = e.data.clone();
        sorted.sort_all_objects();
        h.update(&serde_json::to_vec(&sorted)?);
    } else {
        h.update(&data);
    }
    if let Some(after) = &e.after {
        h.update(b"\0after\0");
        h.update(after.to_string().as_bytes());
    }
    Ok((h.finalize().to_hex().to_string(), data))
}

/// A usage or quota reading names its own identity, which every copy of
/// it shares: it is stored once per origin under that identity.
fn reading_id(e: &Event) -> Option<String> {
    use polylogue_model::reports::{QUOTA_REPORT, USAGE_REPORT};
    if e.kind != USAGE_REPORT && e.kind != QUOTA_REPORT {
        return None;
    }
    let reading = e.data.get("report_id")?.as_str()?;
    let origin = e.session.as_ref()?.origin();
    let mut h = blake3::Hasher::new_derive_key("polylogue reading v1");
    for part in [e.kind.as_str(), origin.as_str(), reading] {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part.as_bytes());
    }
    Some(h.finalize().to_hex().to_string())
}

/// Reads a block back from `block.body`.
pub(crate) fn unpack_block(packed: &[u8]) -> Result<Block> {
    unpack_json(packed)
}

/// The text a block keeps in the `text` table rather than its packed body.
pub(crate) fn block_text(block: &Block) -> Option<&str> {
    match block {
        Block::Text { text } if !text.is_empty() => Some(text),
        Block::Thinking { text: Some(t), .. } if !t.is_empty() => Some(t),
        Block::ToolResult {
            output: Some(o), ..
        } if !o.is_empty() => Some(o),
        _ => None,
    }
}

/// The structured value a block keeps in `block.data` rather than its packed
/// body: a tool result's data. No text view shows it, and it can be most of
/// a large message, so a text read never decodes it.
pub(crate) fn block_data(block: &Block) -> Option<&serde_json::Value> {
    match block {
        Block::ToolResult {
            data: Some(data), ..
        } => Some(data),
        _ => None,
    }
}

/// Puts a block's structured value back from `block.data`.
pub(crate) fn restore_data(block: &mut Block, packed: &[u8]) -> Result<()> {
    if let Block::ToolResult { data, .. } = block {
        *data = Some(unpack_json(packed)?);
    }
    Ok(())
}

/// Puts a block's text back from its `text` row.
pub(crate) fn restore_text(block: &mut Block, text: String) {
    match block {
        Block::Text { text: t } => *t = text,
        Block::Thinking { text: t, .. } => *t = Some(text),
        Block::ToolResult { output, .. } => *output = Some(text),
        _ => {}
    }
}

/// Lowercase hexadecimal.
pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests;
