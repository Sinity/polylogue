//! Derived labels are display values, masked independently after source masking.
use crate::{View, mask};
use polylogue_model::{Block, Message, MessageId};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AttachmentRef {
    pub message: MessageId,
    pub block: usize,
    pub attachment: usize,
}

pub struct AttachmentLabel {
    pub block: usize,
    pub attachment: usize,
    pub text: String,
}

pub(crate) struct Annotation {
    pub piece: usize,
    pub from: usize,
    pub until: usize,
    pub reference: AttachmentRef,
}

pub(crate) fn apply(
    message: &mut Message,
    view: View,
    labels: Vec<AttachmentLabel>,
) -> Vec<Annotation> {
    let mut annotations = Vec::new();
    let mut by_block = std::collections::BTreeMap::<usize, Vec<AttachmentLabel>>::new();
    for label in labels {
        by_block.entry(label.block).or_default().push(label);
    }
    let mut piece = 0;
    let prompt = crate::is_prompt(message);
    for (index, block) in message.blocks.iter_mut().enumerate() {
        if !view.keeps_block(prompt, block) {
            continue;
        }
        let mut length = match block {
            Block::ToolResult { output, .. } => {
                output.as_deref().unwrap_or_default().chars().count()
            }
            _ => 0,
        };
        for label in by_block.remove(&index).unwrap_or_default() {
            let text = mask(&label.text);
            let from = match block {
                Block::Image(attachment) | Block::File(attachment) if label.attachment == 0 => {
                    attachment.name = Some(text.clone());
                    0
                }
                Block::ToolResult {
                    output,
                    attachments,
                    ..
                } if label.attachment < attachments.len() => {
                    let output = output.get_or_insert_default();
                    if !output.is_empty() {
                        output.push('\n');
                        length += 1;
                    }
                    let from = length;
                    output.push_str(&text);
                    length += text.chars().count();
                    from
                }
                _ => continue,
            };
            annotations.push(Annotation {
                piece,
                from,
                until: from + text.chars().count(),
                reference: AttachmentRef {
                    message: message.id.clone(),
                    block: index,
                    attachment: label.attachment,
                },
            });
        }
        piece += 1;
    }
    annotations
}
