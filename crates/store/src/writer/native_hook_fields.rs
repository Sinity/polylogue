//! Native parser field claims are checked independently of message content.
use crate::encoding::PreparedMessage;
use crate::{Result, StoreError};
use polylogue_model::{Block, HookFieldClaim, Output};
use std::collections::BTreeMap;

pub(super) fn validate(output: &Output, messages: &[PreparedMessage]) -> Result<()> {
    for (&index, claims) in &output.native_hook_fields {
        let row = output.messages.get(index).ok_or_else(invalid)?;
        let prepared = messages.get(index).ok_or_else(invalid)?;
        let mut by_block = BTreeMap::<u64, &HookFieldClaim>::new();
        let mut by_pointer = BTreeMap::new();
        for claim in claims {
            let selected = prepared
                .projection
                .block_ordinals
                .binary_search(&claim.block)
                .map_err(|_| invalid())?;
            if claim.start != row.offset
                || claim.end != row.end
                || claim.start >= claim.end
                || !matches!(row.message.blocks.get(selected), Some(Block::Text { .. }))
                || !native_pointer(&claim.pointer)
            {
                return Err(invalid());
            }
            if let Some(prior) = by_pointer.insert(claim.pointer.as_str(), claim.block)
                && prior != claim.block
            {
                return Err(invalid());
            }
            if let Some(prior) = by_block.insert(claim.block, claim)
                && prior != claim
            {
                return Err(invalid());
            }
        }
    }
    for (&index, evidence) in &output.declared_injections {
        let row = output.messages.get(index).ok_or_else(invalid)?;
        let prepared = messages.get(index).ok_or_else(invalid)?;
        for claim in evidence {
            claim.validate_for(&row.message.id, &prepared.projection)?;
        }
    }
    Ok(())
}

pub(super) struct Source<'a> {
    pub input: &'a str,
    pub input_pk: i64,
    pub chain: i64,
}

/// All fields in this path remain declared. Acquisition payloads alone do not
/// authenticate their owner; future native proof must enter through that owner.
pub(super) fn write(
    conn: &rusqlite::Connection,
    source: Source<'_>,
    step: &super::ParseStep,
    index: usize,
) -> Result<()> {
    use polylogue_model::injection::NativeHookField;
    use rusqlite::params;
    let row = &step.output.messages[index];
    let prepared = &step.prepared.messages[index];
    let keys = step.prepared.keys.as_deref().ok_or_else(invalid)?;
    let pk = keys
        .get(
            super::resolve::Kind::Message,
            row.message.id.to_string().as_bytes(),
        )?
        .pk;
    conn.execute(
        "DELETE FROM message_injection_field WHERE message_pk=?1 AND input_pk=?2 AND chain=?3 AND source_version<>?4",
        params![pk, source.input_pk, source.chain, prepared.hash.as_slice()],
    )?;
    let insert = |block: u64, field: NativeHookField| -> Result<()> {
        field.validate()?;
        conn.execute(
            "INSERT OR IGNORE INTO message_injection_field(message_pk,source_version,block_ordinal,input_pk,chain,field) VALUES (?1,?2,?3,?4,?5,?6)",
            params![pk,prepared.hash.as_slice(),block.to_string(),source.input_pk,source.chain,serde_json::to_string(&field)?],
        )?;
        Ok(())
    };
    if let Some(claims) = step.output.native_hook_fields.get(&index) {
        for claim in claims {
            insert(
                claim.block,
                NativeHookField {
                    input: source.input.into(),
                    chain: source.chain,
                    start: claim.start,
                    end: claim.end,
                    pointer: claim.pointer.clone(),
                },
            )?;
        }
    }
    if let Some(claims) = step.output.declared_injections.get(&index) {
        for claim in claims {
            // A forged observed member remains a foreign declaration even if
            // an importer omitted its ordinary into_declared conversion.
            let claim = claim.clone().into_declared();
            for field in claim.declared {
                insert(claim.block, field)?;
            }
        }
    }
    Ok(())
}

fn native_pointer(pointer: &str) -> bool {
    let Some(index) = pointer.strip_prefix("/hookAdditionalContext/") else {
        return false;
    };
    index
        .parse::<u64>()
        .is_ok_and(|parsed| parsed.to_string() == index)
}

fn invalid() -> StoreError {
    StoreError::Invalid(
        "native hook field evidence does not match its message block and source record".into(),
    )
}

#[cfg(test)]
mod tests;
