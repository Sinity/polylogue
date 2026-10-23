//! Mask decoded output strings before an encoder escapes their contents.

use serde_json::Value;

/// Text with credentials masked, as one block.
pub fn mask(text: &str) -> String {
    let mut masker = credential_masking::Masker::new();
    let mut out = Vec::with_capacity(text.len());
    masker
        .push(text, &mut out)
        .expect("writing a Vec cannot fail");
    masker.finish(&mut out).expect("writing a Vec cannot fail");
    String::from_utf8(out).expect("masking preserves UTF-8")
}

/// Preserve JSON types and all values, including empty fields. Credential-shaped
/// keys are masked too. If keys collapse to one label, append a number rather
/// than overwriting a value; ordinary keys retain their original names.
pub fn mask_json(value: &mut Value) {
    mask_json_values(value, mask);
}

fn mask_json_values(value: &mut Value, mut string: impl FnMut(&str) -> String) {
    let mut pending = vec![value];
    while let Some(value) = pending.pop() {
        match value {
            Value::String(text) => *text = string(text),
            Value::Array(values) => pending.extend(values.iter_mut()),
            Value::Object(values) => {
                let mut renamed = Vec::new();
                for (key, value) in std::mem::take(values) {
                    let masked = mask(&key);
                    if masked == key {
                        values.insert(key, value);
                    } else {
                        renamed.push((masked, value));
                    }
                }
                for (label, value) in renamed {
                    let mut key = label.clone();
                    let mut suffix = 2;
                    while values.contains_key(&key) {
                        key = format!("{label} #{suffix}");
                        suffix += 1;
                    }
                    values.insert(key, value);
                }
                pending.extend(values.values_mut());
            }
            _ => {}
        }
    }
}

/// Request-local source-content masking, before views and output rearrangement.
#[derive(Default)]
pub struct MessageMasker {
    stream: credential_masking::Masker,
}

impl MessageMasker {
    pub fn message(&mut self, message: &mut crate::Message) {
        for block in &mut message.blocks {
            self.block(block);
        }
    }

    pub fn block(&mut self, block: &mut crate::Block) {
        use crate::{Block, ErrorReason, ToolOutcome};
        match block {
            Block::Text { text } => {
                self.fragment(text);
            }
            Block::Thinking {
                text, signature, ..
            } => {
                if let Some(text) = text {
                    self.fragment(text);
                }
                if let Some(signature) = signature {
                    *signature = mask(signature);
                }
            }
            Block::ToolCall {
                tool,
                command,
                path,
                input,
                ..
            } => {
                if let Some(tool) = tool {
                    self.fragment(tool);
                }
                let command_selected = command.is_some();
                let alias = if let Some(command) = command {
                    let original = self.fragment(command);
                    Some((original, command.clone()))
                } else if let Some(path) = path.as_mut() {
                    let original = self.fragment(path);
                    Some((original, path.clone()))
                } else {
                    None
                };
                if command_selected && let Some(path) = path {
                    *path = alias
                        .as_ref()
                        .filter(|(original, _)| original == path)
                        .map_or_else(|| mask(path), |(_, masked)| masked.clone());
                }
                if let Some(alias) = alias {
                    mask_alias(input, &alias);
                } else if let serde_json::Value::String(text) = input {
                    self.fragment(text);
                } else {
                    // Structured arguments are independent values. Map key order
                    // is not an ordering of authored source fragments.
                    mask_json(input);
                }
            }
            Block::ToolResult {
                outcome,
                output,
                data,
                attachments,
                ..
            } => {
                if let ToolOutcome::Error {
                    reason: Some(ErrorReason::Denied(reason)),
                } = outcome
                {
                    self.fragment(reason);
                }
                let alias = output.as_mut().map(|output| {
                    let original = self.fragment(output);
                    (original, output.clone())
                });
                if let Some(data) = data {
                    if let Some(alias) = alias {
                        mask_alias(data, &alias);
                    } else {
                        mask_json(data);
                    }
                }
                for attachment in attachments {
                    for value in [
                        &mut attachment.name,
                        &mut attachment.url,
                        &mut attachment.media_type,
                        &mut attachment.source_id,
                    ]
                    .into_iter()
                    .flatten()
                    {
                        *value = mask(value);
                    }
                }
            }
            Block::Image(attachment) | Block::File(attachment) => self.attachment(attachment),
            Block::Other { type_name, data } => {
                *type_name = mask(type_name);
                mask_json(data);
            }
        }
    }

    fn attachment(&mut self, attachment: &mut crate::Attachment) {
        let alias = if let Some(name) = &mut attachment.name {
            let original = self.fragment(name);
            Some((original, name.clone()))
        } else {
            None
        };
        if let Some(url) = &mut attachment.url {
            if let Some((_, masked)) = alias.as_ref().filter(|(original, _)| original == url) {
                *url = masked.clone();
            } else if alias.is_none() {
                self.fragment(url);
            } else {
                *url = mask(url);
            }
        }
        if let Some(kind) = &mut attachment.media_type {
            *kind = mask(kind);
        }
        if let Some(id) = &mut attachment.source_id {
            *id = mask(id);
        }
    }

    /// Returns original bytes only for recognizing duplicate projections in the
    /// same block. No text or masking carry is stored in a public cursor.
    fn fragment(&mut self, text: &mut String) -> String {
        let original = std::mem::take(text);
        let mut bytes = Vec::new();
        self.stream
            .push(&original, &mut bytes)
            .expect("Vec writes cannot fail");
        self.stream
            .block_boundary(&mut bytes)
            .expect("Vec writes cannot fail");
        *text = String::from_utf8(bytes).expect("masking preserves UTF-8");
        original
    }
}

fn mask_alias(value: &mut Value, alias: &(String, String)) {
    // A projected command/output is not a second authored fragment. All other
    // structured values retain complete-value masking and their JSON types.
    mask_json_values(value, |text| {
        if text == alias.0 {
            alias.1.clone()
        } else {
            mask(text)
        }
    });
}

#[cfg(test)]
mod tests;
