//! Encoding of database row bodies. Original-input blob encoding belongs to `blobs`.

use crate::{Result, StoreError};
use polylogue_model::{Block, Document, DocumentId, Event, Message, MessageId, Timestamp};
use serde::{Serialize, de::DeserializeOwned};
use std::io::Read;
use std::sync::LazyLock;

// Fast row-body compression; the format remains readable across compression levels.
const LEVEL: i32 = 1;

// Immutable for this derived-index generation. Canonical rows use pack_bytes
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

pub(crate) fn pack_json(value: &impl Serialize) -> Result<Vec<u8>> {
    Ok(pack_bytes(&serde_json::to_vec(value)?)?)
}

/// One message's identity and packed rows, prepared together before its parsed
/// payloads are released. The writer borrows these bytes without repacking.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedMessage {
    pub(crate) id: MessageId,
    content: PreparedContent,
}

/// Encoded content rows shared by role-bearing messages and textual documents.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedContent {
    pub(crate) hash: [u8; 32],
    pub(crate) attrs: Vec<u8>,
    pub(crate) blocks: Vec<Vec<u8>>,
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

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedDocument {
    pub(crate) id: DocumentId,
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
}

pub(crate) fn unpack_json<T: DeserializeOwned>(packed: &[u8]) -> Result<T> {
    Ok(serde_json::from_slice(&unpack_bytes(packed)?)?)
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

    fn pack_json(&mut self, value: &impl Serialize) -> Result<Vec<u8>> {
        Ok(self.pack_row(&serde_json::to_vec(value)?)?)
    }

    fn pack_block(&mut self, block: &Block) -> Result<Vec<u8>> {
        Ok(self.pack_row(&block_bytes(block)?)?)
    }

    pub(crate) fn pack_event(&mut self, event: &Event) -> Result<(String, Vec<u8>)> {
        let (id, data) = event_bytes(event)?;
        Ok((id, self.pack_row(&data)?))
    }

    pub(crate) fn pack_message(&mut self, message: &Message) -> Result<PreparedMessage> {
        let attrs = Message {
            id: message.id.clone(),
            session: message.session.clone(),
            parent: message.parent.clone(),
            role: message.role,
            at: message.at,
            model: message.model.clone(),
            prompt_kind: message.prompt_kind,
            blocks: Vec::new(),
            usage: message.usage.clone(),
            stop_reason: message.stop_reason.clone(),
            reported_cost_usd: message.reported_cost_usd.clone(),
            duration_ms: message.duration_ms,
            production: message.production.clone(),
        };
        Ok(PreparedMessage {
            id: message.id.clone(),
            content: self.pack_content(message, &attrs, &message.blocks, None)?,
        })
    }

    pub(crate) fn pack_document(&mut self, document: &Document) -> Result<PreparedDocument> {
        let attrs = Document {
            id: document.id.clone(),
            purpose: document.purpose,
            title: document.title.clone(),
            source_time: document.source_time,
            authorship: document.authorship,
            blocks: Vec::new(),
            metadata: document.metadata.clone(),
        };
        Ok(PreparedDocument {
            id: document.id.clone(),
            content: self.pack_content(
                document,
                &attrs,
                &document.blocks,
                Some(b"polylogue-document-v1\0"),
            )?,
        })
    }

    fn pack_content(
        &mut self,
        value: &impl Serialize,
        attrs: &impl Serialize,
        blocks: &[Block],
        domain: Option<&[u8]>,
    ) -> Result<PreparedContent> {
        // Canonicalization changes derived identity only. Message hashes retain
        // their existing byte domain; Documents have a separate explicit tag.
        let hash = {
            let mut canonical = serde_json::to_value(value)?;
            canonical.sort_all_objects();
            let mut hasher = blake3::Hasher::new();
            if let Some(domain) = domain {
                hasher.update(domain);
            }
            hasher.update(&serde_json::to_vec(&canonical)?);
            *hasher.finalize().as_bytes()
        };
        Ok(PreparedContent {
            hash,
            attrs: self.pack_json(attrs)?,
            blocks: blocks
                .iter()
                .map(|block| self.pack_block(block))
                .collect::<Result<_>>()?,
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
            call_id,
            outcome,
            output: Some(_),
            data,
            attachments,
        } => Block::ToolResult {
            call_id: call_id.clone(),
            outcome: outcome.clone(),
            output: Some(String::new()),
            data: data.clone(),
            attachments: attachments.clone(),
        },
        // These values are stored whole, so serialization can borrow them.
        _ => return Ok(serde_json::to_vec(block)?),
    };
    Ok(serde_json::to_vec(&body)?)
}

/// An event's id includes its message anchor: identical state recorded at
/// different messages is distinct. Physical ranges do not change the fact's
/// identity when another route supplies it.
pub(crate) fn pack_event(e: &Event) -> Result<(String, Vec<u8>)> {
    let (id, data) = event_bytes(e)?;
    Ok((id, pack_bytes(&data)?))
}

fn event_bytes(e: &Event) -> Result<(String, Vec<u8>)> {
    let data = serde_json::to_vec(&e.data)?;
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
        let mut canonical = e.data.clone();
        canonical.sort_all_objects();
        h.update(&serde_json::to_vec(&canonical)?);
    } else {
        h.update(&data);
    }
    if let Some(after) = &e.after {
        h.update(b"\0after\0");
        h.update(after.to_string().as_bytes());
    }
    Ok((h.finalize().to_hex().to_string(), data))
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

/// Bytes from even-length hexadecimal of either case.
pub(crate) fn unhex(text: &str) -> Result<Vec<u8>> {
    if !text.is_ascii() || !text.len().is_multiple_of(2) {
        return Err(StoreError::Invalid(format!("bad hex {text}")));
    }
    (0..text.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&text[i..i + 2], 16)
                .map_err(|_| StoreError::Invalid(format!("bad hex {text}")))
        })
        .collect()
}

#[cfg(test)]
mod tests;
