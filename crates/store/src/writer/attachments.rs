//! Attachment references only name bytes the archive holds. Source bytes stay
//! unchanged when a capture names a file that has not been fetched.
use super::*;
pub(super) fn normalize(
    batch: &mut Batch,
    archive: &Connection,
    pending: &HashMap<BlobHash, u64>,
) -> Result<()> {
    normalize_using(&mut batch.parse, |hash| match pending.get(&hash) {
        Some(size) => Ok(Some(*size)),
        None => size(archive, hash),
    })
}
pub(super) fn normalize_known(
    step: &mut ParsePart,
    known: &HashMap<BlobHash, Option<u64>>,
) -> Result<()> {
    normalize_using(std::slice::from_mut(step), |hash| {
        known.get(&hash).copied().ok_or_else(|| {
            StoreError::Invalid("a chain attachment has no snapshot identity".into())
        })
    })
}
pub(super) fn remember(
    step: &ParsePart,
    archive: &Connection,
    pending: &HashMap<BlobHash, u64>,
    known: &mut HashMap<BlobHash, Option<u64>>,
) -> Result<()> {
    for block in step
        .output
        .messages
        .iter()
        .flat_map(|row| &row.message.blocks)
        .chain(
            step.output
                .documents
                .iter()
                .flat_map(|row| &row.document.blocks),
        )
    {
        for attachment in block.attachments() {
            if let Some(hash) = attachment.blob
                && let std::collections::hash_map::Entry::Vacant(slot) = known.entry(hash)
            {
                slot.insert(match pending.get(&hash) {
                    Some(size) => Some(*size),
                    None => size(archive, hash)?,
                });
            }
        }
    }
    Ok(())
}
fn size(archive: &Connection, hash: BlobHash) -> Result<Option<u64>> {
    archive
        .query_row(
            &crate::source::Source::of(archive)?.sql("SELECT size FROM @source.blob WHERE hash=?1"),
            [hash.0.as_slice()],
            |row| row.get::<_, i64>(0),
        )
        .optional()?
        .map(|size| {
            u64::try_from(size).map_err(|_| crate::StoreError::stored(0, "negative blob size"))
        })
        .transpose()
}

fn normalize_using(
    steps: &mut [ParsePart],
    mut available: impl FnMut(BlobHash) -> Result<Option<u64>>,
) -> Result<()> {
    for step in steps {
        for (message_index, row) in step.output.messages.iter_mut().enumerate() {
            let Some(mut blocks) = stored_blocks(&row.message.blocks, &mut available)? else {
                continue;
            };
            let prepared = message_index < step.prepared.messages.len();
            if prepared {
                let stored = step.prepared.message_value(message_index)?;
                restore_values(&mut blocks, &stored.blocks)?;
            }
            row.message.blocks = blocks;
            if prepared {
                step.prepared.repack_message(
                    message_index,
                    &mut row.message,
                    step.output
                        .stored_fields
                        .get(&message_index)
                        .map_or(&[][..], Vec::as_slice),
                )?;
            }
        }
        for (document_index, row) in step.output.documents.iter_mut().enumerate() {
            let Some(mut blocks) = stored_blocks(&row.document.blocks, &mut available)? else {
                continue;
            };
            if let Some(prepared) = step.prepared.documents.get(document_index) {
                restore_packed(&mut blocks, prepared)?;
            }
            if let Some(prepared) = step.prepared.documents.get_mut(document_index) {
                row.document = crate::encoding::unpack_json(&prepared.fields)?;
                row.document.blocks = blocks;
                *prepared = PreparedDocument::new(&row.document)?;
            } else {
                row.document.blocks = blocks;
            }
        }
    }
    Ok(())
}

fn stored_blocks(
    blocks: &[Block],
    available: &mut impl FnMut(BlobHash) -> Result<Option<u64>>,
) -> Result<Option<Vec<Block>>> {
    let mut changed = false;
    let mut normalized = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        let attachments = block.attachments();
        for (attachment_index, attachment) in attachments.iter().enumerate() {
            let Some(hash) = attachment.blob else {
                continue;
            };
            let size = available(hash)?;
            if size.is_some() && attachment.size == size {
                continue;
            }
            if !changed {
                normalized = blocks.to_vec();
                changed = true;
            }
            let attachment = &mut normalized[index].attachments_mut()[attachment_index];
            attachment.blob = size.map(|_| hash);
            if let Some(size) = size {
                attachment.size = Some(size);
            }
        }
    }
    Ok(changed.then_some(normalized))
}

/// Puts back the values preparing released from parsed blocks (a call's
/// input, an unknown block, a result's data), from the same blocks as stored.
pub(super) fn restore_values(blocks: &mut [Block], stored: &[Block]) -> Result<()> {
    for (block, stored) in blocks.iter_mut().zip(stored) {
        match block {
            Block::ToolCall { input, .. } => {
                let Block::ToolCall {
                    input: restored, ..
                } = stored
                else {
                    return Err(StoreError::invalid(
                        "prepared tool call has a different block kind",
                    ));
                };
                // The stored block holds the released input. Written text is
                // still on the parsed block and is empty in the stored one.
                *input = restored.clone();
            }
            Block::Other { .. } => *block = stored.clone(),
            Block::ToolResult { data, .. } => {
                if let Block::ToolResult {
                    data: Some(restored),
                    ..
                } = stored
                {
                    *data = Some(restored.clone());
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// [`restore_values`] from a document's packed block rows.
pub(super) fn restore_packed(
    blocks: &mut [Block],
    prepared: &crate::encoding::PackedBlocks,
) -> Result<()> {
    for (index, block) in blocks.iter_mut().enumerate() {
        match block {
            Block::ToolCall { input, .. } => {
                let Block::ToolCall {
                    input: restored, ..
                } = unpack_block(&prepared.blocks[index])?
                else {
                    return Err(StoreError::invalid(
                        "prepared tool call has a different block kind",
                    ));
                };
                *input = restored;
            }
            Block::Other { .. } => {
                *block = unpack_block(&prepared.blocks[index])?;
            }
            Block::ToolResult { .. } => {
                if let Some(data) = &prepared.data[index] {
                    crate::encoding::set_data(block, crate::encoding::unpack_json(data)?);
                }
            }
            _ => {}
        }
    }
    Ok(())
}
