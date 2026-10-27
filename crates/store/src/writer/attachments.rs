//! Attachment references only name bytes the archive holds. Source bytes stay
//! unchanged when a capture names a file that has not been fetched.
use super::*;
use std::borrow::Cow;

pub(super) fn with_stored_blobs<'a>(
    batch: &'a Batch,
    archive: &Connection,
) -> Result<Cow<'a, Batch>> {
    let mut indexed = Cow::Borrowed(batch);
    for (step_index, step) in batch.parse.iter().enumerate() {
        for (message_index, row) in step.output.messages.iter().enumerate() {
            let Some(mut blocks) = stored_blocks(&row.message.blocks, archive)? else {
                continue;
            };
            if let Some(prepared) = step.prepared.messages.get(message_index) {
                restore_values(&mut blocks, prepared)?;
            }
            let step = &mut indexed.to_mut().parse[step_index];
            let message = &mut step.output.messages[message_index].message;
            message.blocks = blocks;
            if let Some(prepared) = step.prepared.messages.get_mut(message_index) {
                *prepared = PreparedMessage::new(message)?;
            }
        }
        for (document_index, row) in step.output.documents.iter().enumerate() {
            let Some(mut blocks) = stored_blocks(&row.document.blocks, archive)? else {
                continue;
            };
            if let Some(prepared) = step.prepared.documents.get(document_index) {
                restore_values(&mut blocks, prepared)?;
            }
            let step = &mut indexed.to_mut().parse[step_index];
            let document = &mut step.output.documents[document_index].document;
            if let Some(prepared) = step.prepared.documents.get_mut(document_index) {
                *document = crate::encoding::unpack_json(&prepared.fields)?;
                document.blocks = blocks;
                *prepared = PreparedDocument::new(document)?;
            } else {
                document.blocks = blocks;
            }
        }
    }
    Ok(indexed)
}

fn stored_blocks(blocks: &[Block], archive: &Connection) -> Result<Option<Vec<Block>>> {
    let mut changed = false;
    let mut normalized = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        let attachments = match block {
            Block::File(a) | Block::Image(a) => std::slice::from_ref(a),
            Block::ToolResult { attachments, .. } => attachments.as_slice(),
            _ => continue,
        };
        for (attachment_index, attachment) in attachments.iter().enumerate() {
            let Some(hash) = attachment.blob else {
                continue;
            };
            let size: Option<u64> = archive
                .query_row(
                    "SELECT size FROM blob WHERE hash = ?1 AND state = 'present'",
                    [&hash.0[..]],
                    |row| row.get::<_, i64>(0),
                )
                .optional()?
                .map(|size| {
                    u64::try_from(size)
                        .map_err(|_| StoreError::Invalid("negative blob size".into()))
                })
                .transpose()?;
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

fn restore_values(blocks: &mut [Block], prepared: &crate::encoding::PreparedContent) -> Result<()> {
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
