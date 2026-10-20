//! Encoding of database row bodies. Original-input blob encoding belongs to `blobs`.

use crate::{Result, StoreError};
use polylogue_model::{Block, Event, Timestamp};
use serde::{Serialize, de::DeserializeOwned};

// Fast row-body compression; the format remains readable across compression levels.
const LEVEL: i32 = 1;

pub(crate) fn pack_bytes(bytes: &[u8]) -> std::io::Result<Vec<u8>> {
    zstd::bulk::compress(bytes, LEVEL)
}

pub(crate) fn unpack_bytes(packed: &[u8]) -> std::io::Result<Vec<u8>> {
    zstd::stream::decode_all(packed)
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

pub(crate) fn unpack_json<T: DeserializeOwned>(packed: &[u8]) -> Result<T> {
    Ok(serde_json::from_slice(&unpack_bytes(packed)?)?)
}

/// Reuses the compressor across a parse batch's texts.
pub(crate) struct TextPacker(zstd::bulk::Compressor<'static>);

impl TextPacker {
    pub(crate) fn new() -> std::io::Result<Self> {
        zstd::bulk::Compressor::new(LEVEL).map(Self)
    }

    pub(crate) fn pack(&mut self, text: &str) -> std::io::Result<Vec<u8>> {
        self.0.compress(text.as_bytes())
    }
}

/// A block as the `block.body` column stores it.
pub(crate) fn pack_block(block: &Block) -> Result<Vec<u8>> {
    let mut body = block.clone();
    strip_text(&mut body);
    pack_json(&body)
}

/// An event's id includes its message anchor: identical state recorded at
/// different messages is distinct. Physical ranges do not change the fact's
/// identity when another route supplies it.
pub(crate) fn pack_event(e: &Event) -> Result<(String, Vec<u8>)> {
    let data = serde_json::to_vec(&e.data)?;
    let mut h = blake3::Hasher::new();
    h.update(e.session.to_string().as_bytes());
    h.update(b"\0");
    h.update(e.kind.as_bytes());
    h.update(b"\0");
    h.update(
        &e.at
            .map(Timestamp::unix_ms)
            .unwrap_or(i64::MIN)
            .to_le_bytes(),
    );
    h.update(&data);
    if let Some(after) = &e.after {
        h.update(b"\0after\0");
        h.update(after.to_string().as_bytes());
    }
    Ok((h.finalize().to_hex().to_string(), pack_bytes(&data)?))
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

fn strip_text(block: &mut Block) {
    match block {
        Block::Text { text }
        | Block::Thinking {
            text: Some(text), ..
        }
        | Block::ToolResult {
            output: Some(text), ..
        } => text.clear(),
        _ => {}
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

#[cfg(test)]
mod tests;
