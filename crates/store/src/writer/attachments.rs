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
    step: &mut ParseStep,
    known: &HashMap<BlobHash, Option<u64>>,
) -> Result<()> {
    normalize_using(std::slice::from_mut(step), |hash| {
        known.get(&hash).copied().ok_or_else(|| {
            StoreError::Invalid("a chain attachment has no snapshot identity".into())
        })
    })
}
pub(super) fn remember(
    step: &ParseStep,
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
        for attachment in attachments(block) {
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
            "SELECT size FROM blob WHERE hash=?1",
            [hash.0.as_slice()],
            |row| row.get::<_, i64>(0),
        )
        .optional()?
        .map(|size| {
            u64::try_from(size).map_err(|_| StoreError::Invalid("negative blob size".into()))
        })
        .transpose()
}
fn attachments(block: &Block) -> &[polylogue_model::Attachment] {
    match block {
        Block::File(attachment) | Block::Image(attachment) => std::slice::from_ref(attachment),
        Block::ToolResult { attachments, .. } => attachments,
        _ => &[],
    }
}
fn normalize_using(
    steps: &mut [ParseStep],
    mut available: impl FnMut(BlobHash) -> Result<Option<u64>>,
) -> Result<()> {
    for step in steps {
        for (message_index, row) in step.output.messages.iter_mut().enumerate() {
            let Some(mut blocks) = stored_blocks(&row.message.blocks, &mut available)? else {
                continue;
            };
            if let Some(prepared) = step.prepared.messages.get(message_index) {
                restore_values(&mut blocks, prepared)?;
            }
            row.message.blocks = blocks;
            if let Some(prepared) = step.prepared.messages.get_mut(message_index) {
                *prepared = PreparedMessage::new(&row.message)?;
            }
        }
        for (document_index, row) in step.output.documents.iter_mut().enumerate() {
            let Some(mut blocks) = stored_blocks(&row.document.blocks, &mut available)? else {
                continue;
            };
            if let Some(prepared) = step.prepared.documents.get(document_index) {
                restore_values(&mut blocks, prepared)?;
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
        let attachments = attachments(block);
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
            let attachment = match &mut normalized[index] {
                Block::File(a) | Block::Image(a) => a,
                Block::ToolResult { attachments, .. } => &mut attachments[attachment_index],
                _ => unreachable!("attachment block"),
            };
            attachment.blob = size.map(|_| hash);
            if let Some(size) = size {
                attachment.size = Some(size);
            }
        }
    }
    Ok(changed.then_some(normalized))
}

pub(super) fn restore_values(
    blocks: &mut [Block],
    prepared: &crate::encoding::PreparedContent,
) -> Result<()> {
    for (index, block) in blocks.iter_mut().enumerate() {
        match block {
            Block::ToolCall { .. } | Block::Other { .. } => {
                *block = unpack_block(&prepared.blocks[index])?;
            }
            Block::ToolResult { .. } => {
                if let Some(data) = &prepared.data[index] {
                    crate::encoding::restore_data(block, data)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}
