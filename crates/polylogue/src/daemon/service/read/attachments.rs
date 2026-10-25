use polylogue_model::{Block, Message};
use polylogue_render::{AttachmentLabel, View};
use polylogue_store::{
    Reader,
    assets::{AttachmentResolution, AttachmentState},
};

pub(super) fn labels(
    reader: &Reader<'_>,
    message: &Message,
    view: View,
) -> polylogue_store::Result<(Vec<AttachmentLabel>, Vec<AttachmentResolution>)> {
    let mut labels = Vec::new();
    let mut descriptors = Vec::new();
    for (block_index, block) in message.blocks.iter().enumerate() {
        if !view.keeps(message, block) {
            continue;
        }
        let attachments: &[polylogue_model::Attachment] = match block {
            Block::Image(attachment) | Block::File(attachment) => std::slice::from_ref(attachment),
            Block::ToolResult { attachments, .. } => attachments,
            _ => continue,
        };
        for (index, attachment) in attachments.iter().enumerate() {
            if attachment.source_id.is_none() {
                continue;
            }
            let mut resolution = reader.attachment(&message.id, block_index, index)?;
            // The caller has already applied the shared ordered source masker.
            resolution.source = attachment.clone();
            let name = attachment
                .name
                .clone()
                .or_else(|| {
                    resolution
                        .matches
                        .first()
                        .map(|found| found.names.join(", "))
                        .filter(|name| !name.is_empty())
                })
                .or_else(|| attachment.source_id.clone())
                .unwrap_or_else(|| "attachment".into());
            let state = match resolution.state {
                AttachmentState::Available => {
                    format!("{} stored bytes", resolution.matches[0].size)
                }
                AttachmentState::Ambiguous => format!(
                    "{} stored files match, choose one with --asset",
                    resolution.matches.len()
                ),
                AttachmentState::Unavailable => "no stored file matches".into(),
                AttachmentState::Unknown => "byte availability unknown".into(),
            };
            labels.push(AttachmentLabel {
                block: block_index,
                attachment: index,
                text: format!(
                    "[{name}; {state}; polylogue attachment {} {block_index} --attachment {index}]",
                    message.id
                ),
            });
            descriptors.push(resolution);
        }
    }
    Ok((labels, descriptors))
}
