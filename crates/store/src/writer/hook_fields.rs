//! Stated hook fields are checked independently of message content.
use crate::encoding::PreparedMessage;
use crate::{Result, StoreError};
use polylogue_model::{Block, HookFieldClaim, Output};
use std::collections::BTreeMap;

pub(super) fn validate(output: &Output, messages: &[PreparedMessage]) -> Result<()> {
    for (&index, stated) in &output.hook_fields {
        let row = output.messages.get(index).ok_or_else(invalid)?;
        let prepared = messages.get(index).ok_or_else(invalid)?;
        let mut by_block = BTreeMap::<u64, &HookFieldClaim>::new();
        let mut by_pointer = BTreeMap::new();
        for field in stated {
            let selected = prepared
                .block_selection
                .block_ordinals
                .binary_search(&field.block)
                .map_err(|_| invalid())?;
            if field.start != row.offset
                || field.end != row.end
                || field.start >= field.end
                || !matches!(row.message.blocks.get(selected), Some(Block::Text { .. }))
                || !valid_hook_pointer(&field.pointer)
            {
                return Err(invalid());
            }
            if let Some(prior) = by_pointer.insert(field.pointer.as_str(), field.block)
                && prior != field.block
            {
                return Err(invalid());
            }
            if let Some(prior) = by_block.insert(field.block, field)
                && prior != field
            {
                return Err(invalid());
            }
        }
    }
    for (&index, injections) in &output.declared_injections {
        let row = output.messages.get(index).ok_or_else(invalid)?;
        let prepared = messages.get(index).ok_or_else(invalid)?;
        for injection in injections {
            injection.validate_for(&row.message.id, &prepared.block_selection)?;
        }
    }
    Ok(())
}

pub(super) struct Source<'a> {
    pub input: &'a str,
    pub input_pk: i64,
    pub chain: i64,
}

/// All fields in this path stay declared. A captured payload does not show who
/// wrote it; any future check of that must come from the payload's writer.
/// A message row `created` by this reading has no earlier stated fields to replace.
#[allow(clippy::too_many_arguments)]
pub(super) fn write(
    conn: &rusqlite::Connection,
    source: Source<'_>,
    keys: &super::resolve::ResolvedKeys,
    step: &super::ParseStep,
    index: usize,
    version_pk: i64,
    created: bool,
) -> Result<()> {
    use polylogue_model::injection::NativeHookField;
    use rusqlite::params;
    let row = &step.output.messages[index];
    let pk = keys
        .get(
            super::resolve::Kind::Message,
            row.message.id.to_string().as_bytes(),
        )?
        .pk;
    if !created {
        conn.prepare_cached(
            "DELETE FROM message_injection_field WHERE message_pk=?1 AND input_pk=?2 AND chain=?3 AND version_pk<>?4",
        )?
        .execute(params![pk, source.input_pk, source.chain, version_pk])?;
    }
    let insert = |block: u64, field: NativeHookField| -> Result<()> {
        field.validate()?;
        conn.prepare_cached(
            "INSERT OR IGNORE INTO message_injection_field(message_pk,version_pk,block_ordinal,input_pk,chain,field) VALUES (?1,?2,?3,?4,?5,?6)",
        )?
        .execute(params![pk,version_pk,block.to_string(),source.input_pk,source.chain,serde_json::to_string(&field)?])?;
        Ok(())
    };
    if let Some(stated) = step.output.hook_fields.get(&index) {
        for stated_field in stated {
            insert(
                stated_field.block,
                NativeHookField {
                    input: source.input.into(),
                    chain: source.chain,
                    start: stated_field.start,
                    end: stated_field.end,
                    pointer: stated_field.pointer.clone(),
                },
            )?;
        }
    }
    if let Some(injections) = step.output.declared_injections.get(&index) {
        for injection in injections {
            // A forged observed member stays a declaration from another source
            // even if an importer skipped its into_declared conversion.
            let declared = injection.clone().into_declared();
            for field in declared.declared {
                insert(declared.block, field)?;
            }
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

fn invalid() -> StoreError {
    StoreError::Invalid(
        "stated hook field does not match its message block and source record".into(),
    )
}

#[cfg(test)]
mod tests;
