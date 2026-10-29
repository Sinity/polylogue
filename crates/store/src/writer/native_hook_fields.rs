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
