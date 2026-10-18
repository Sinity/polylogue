//! Comparable authored text uses the existing selected, decoded body reader.
//! Digests are raw UTF-8 body evidence, never display text or continuation hashes.
use polylogue_model::{BlobHash, BlockSlot, authored};
use polylogue_store::{
    Error, ReadHandle,
    contributions::{MessageLookup, MessageView},
};
use serde::Serialize;
use std::time::Instant;
#[derive(Clone, Debug, Serialize)]
pub struct TargetText {
    #[serde(rename = "ref")]
    pub reference: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seen_hash: Option<BlobHash>,
    pub projection: &'static str,
    pub covers: &'static str,
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
}
impl TargetText {
    fn unknown(reference: &str, status: &'static str, reason: &'static str) -> Self {
        Self {
            reference: reference.into(),
            seen_hash: None,
            projection: "decoded-utf8-text-v1",
            covers: "whole_target",
            status,
            reason: Some(reason),
        }
    }
}
pub fn selected_text(
    m: &MessageView,
    slot: Option<&BlockSlot>,
    reference: &str,
    until: Instant,
    cancel: &mut dyn FnMut() -> bool,
) -> Result<TargetText, Error> {
    let block = match slot {
        Some(slot) => m.block(slot.clone())?,
        None => {
            if m.block_at(1)?.is_some() {
                return Ok(TargetText::unknown(
                    reference,
                    "unknown",
                    "composite_message_text_projection_unsupported",
                ));
            }
            m.block_at(0)?
        }
    };
    let Some(block) = block else {
        return Ok(TargetText::unknown(
            reference,
            "waiting",
            "target_body_absent",
        ));
    };
    if !matches!(block.kind.as_str(), "text" | "thinking") || block.encrypted == Some(true) {
        return Ok(TargetText::unknown(
            reference,
            "unknown",
            "structured_or_encrypted_text_projection_unsupported",
        ));
    }
    let mut hash = blake3::Hasher::new();
    let mut offset = 0;
    loop {
        if Instant::now() >= until || cancel() {
            return Err(Error::Cancelled);
        }
        let Some(range) = block.text_range(offset, 4096)? else {
            return Ok(TargetText::unknown(
                reference,
                "unknown",
                "decoded_text_unavailable",
            ));
        };
        if range.next_offset < offset || (!range.eof && range.next_offset == offset) {
            return Err(Error::Invariant(
                "authored text range made no progress".into(),
            ));
        }
        hash.update(&range.bytes);
        offset = range.next_offset;
        if range.eof {
            break;
        }
    }
    // Check current visibility after hashing; a removal is never a hash witness.
    m.header()?;
    Ok(TargetText {
        reference: reference.into(),
        seen_hash: Some(BlobHash::from_bytes(*hash.finalize().as_bytes())),
        projection: "decoded-utf8-text-v1",
        covers: "whole_target",
        status: "comparable",
        reason: None,
    })
}
pub fn current_text(
    reader: &ReadHandle,
    reference: &str,
    until: Instant,
    cancel: &mut dyn FnMut() -> bool,
) -> Result<TargetText, Error> {
    let (id, slot) = authored::target_address(reference).map_err(Error::Authored)?;
    match reader.message_reference(id.as_ref(), 65536)? {
        MessageLookup::Ready(m) => {
            let basis = selected_text(&m, slot.as_ref(), reference, until, cancel)?;
            let current = match reader.message_reference(id.as_ref(), 65536)? {
                MessageLookup::Ready(m) => m,
                _ => return Err(Error::StaleMessage),
            };
            if current.read_certificate_with_cancel(&mut || Instant::now() >= until || cancel())?
                != m.read_certificate_with_cancel(&mut || Instant::now() >= until || cancel())?
            {
                return Err(Error::StaleMessage);
            }
            Ok(basis)
        }
        MessageLookup::Pending => Ok(TargetText::unknown(
            reference,
            "unavailable",
            "selected_message_pending",
        )),
        MessageLookup::Absent => Ok(TargetText::unknown(
            reference,
            "waiting",
            "target_not_collected",
        )),
    }
}
