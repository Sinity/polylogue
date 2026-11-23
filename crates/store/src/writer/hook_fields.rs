//! Stated hook fields are checked independently of message content.
use crate::encoding::PreparedMessage;
use crate::{OptionExt as _, Result, ResultExt as _, StoreError};
use polylogue_model::{Block, HookFieldClaim, Output};
use std::collections::BTreeMap;

pub(super) fn validate(output: &Output, messages: &[PreparedMessage]) -> Result<()> {
    for (&index, stated) in &output.hook_fields {
        let row = output
            .messages
            .get(index)
            .or_invalid("stated hook field names a message the step does not hold")?;
        let prepared = messages
            .get(index)
            .or_invalid("stated hook field names a message the step did not prepare")?;
        let mut by_block = BTreeMap::<u64, &HookFieldClaim>::new();
        let mut by_pointer = BTreeMap::new();
        for field in stated {
            let selected = prepared
                .block_selection
                .block_ordinals
                .binary_search(&field.block)
                .or_invalid("stated hook field names a block its message did not select")?;
            if !matches!(row.message.blocks.get(selected), Some(Block::Text { .. })) {
                return Err(StoreError::invalid(
                    "stated hook field names a block that is not text",
                ));
            }
            if field.start != row.offset || field.end != row.end {
                return Err(StoreError::invalid(
                    "stated hook field span differs from its source record",
                ));
            }
            if field.start >= field.end {
                return Err(StoreError::invalid("stated hook field span is empty"));
            }
            if !valid_hook_pointer(&field.pointer) {
                return Err(StoreError::invalid(
                    "stated hook field pointer is not a hook context index",
                ));
            }
            if let Some(prior) = by_pointer.insert(field.pointer.as_str(), field.block)
                && prior != field.block
            {
                return Err(StoreError::invalid(
                    "stated hook field pointer names two blocks",
                ));
            }
            if let Some(prior) = by_block.insert(field.block, field)
                && prior != field
            {
                return Err(StoreError::invalid(
                    "one block carries two different stated hook fields",
                ));
            }
        }
    }
    for (&index, injections) in &output.declared_injections {
        let row = output
            .messages
            .get(index)
            .or_invalid("declared injection names a message the step does not hold")?;
        let prepared = messages
            .get(index)
            .or_invalid("declared injection names a message the step did not prepare")?;
        for injection in injections {
            injection.validate_for(&row.message.id, &prepared.block_selection)?;
        }
    }
    Ok(())
}

fn valid_hook_pointer(pointer: &str) -> bool {
    let Some(index) = pointer.strip_prefix("/hookAdditionalContext/") else {
        return false;
    };
    index
        .parse::<u64>()
        .is_ok_and(|parsed| parsed.to_string() == index)
}

#[cfg(test)]
mod tests;
