use super::*;

/// Packed rows stay associated with their original message and are published
/// together before parsed structured payloads are released.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PreparedRows {
    pub(in crate::writer) messages: Vec<PreparedMessage>,
    pub(in crate::writer) documents: Vec<PreparedDocument>,
    pub(in crate::writer) texts: HashMap<[u8; 32], Vec<u8>>,
    pub(in crate::writer) events: Vec<(String, Vec<u8>)>,
    released: bool,
}

impl PreparedRows {
    pub(in crate::writer) fn validate(&self, output: &Output) -> Result<()> {
        if !self.released
            && (!self.messages.is_empty()
                || !self.documents.is_empty()
                || !self.events.is_empty()
                || !self.texts.is_empty())
        {
            return Err(StoreError::Invalid("prepared rows are incomplete".into()));
        }
        if self.released
            && (self.messages.len() != output.messages.len()
                || self.documents.len() != output.documents.len()
                || self.documents.iter().zip(&output.documents).any(|(p, r)| {
                    p.id != r.document.id || p.blocks.len() != r.document.blocks.len()
                })
                || self.events.len() != output.events.len()
                || self
                    .messages
                    .iter()
                    .zip(&output.messages)
                    .any(|(p, r)| p.id != r.message.id || p.blocks.len() != r.message.blocks.len()))
        {
            return Err(StoreError::Invalid(
                "prepared rows no longer match their parsed records".into(),
            ));
        }
        Ok(())
    }

    pub(in crate::writer) fn keep_messages(&mut self, keep: &[bool]) {
        retain(&mut self.messages, keep);
    }

    pub(in crate::writer) fn keep_documents(&mut self, keep: &[bool]) {
        retain(&mut self.documents, keep);
    }

    pub(in crate::writer) fn keep_events(&mut self, keep: &[bool]) {
        retain(&mut self.events, keep);
    }
}

fn retain<T>(rows: &mut Vec<T>, keep: &[bool]) {
    if rows.len() == keep.len() {
        let mut positions = keep.iter();
        rows.retain(|_| *positions.next().expect("one flag per row"));
    } else {
        rows.clear();
    }
}

impl ParseStep {
    /// Packs complete rows off the writer thread. A second call keeps the
    /// original hashes and payloads, which parsed values no longer contain.
    pub fn prepare(&mut self) {
        if self.prepared.released {
            return;
        }
        let Ok(mut packer) = crate::encoding::RowPacker::new() else {
            return;
        };
        let Ok(messages) = self
            .output
            .messages
            .iter()
            .map(|r| packer.pack_message(&r.message))
            .collect::<Result<Vec<_>>>()
        else {
            return;
        };
        let Ok(documents) = self
            .output
            .documents
            .iter()
            .map(|r| packer.pack_document(&r.document))
            .collect::<Result<Vec<_>>>()
        else {
            return;
        };
        let Ok(events) = self
            .output
            .events
            .iter()
            .map(|event| packer.pack_event(event))
            .collect::<Result<Vec<_>>>()
        else {
            return;
        };
        let mut texts = HashMap::new();
        for blocks in self
            .output
            .messages
            .iter()
            .map(|r| r.message.blocks.as_slice())
            .chain(
                self.output
                    .documents
                    .iter()
                    .map(|r| r.document.blocks.as_slice()),
            )
        {
            for block in blocks {
                let Some(text) = block_text(block) else {
                    continue;
                };
                let hash = *blake3::hash(text.as_bytes()).as_bytes();
                if let std::collections::hash_map::Entry::Vacant(slot) = texts.entry(hash)
                    && let Ok(bytes) = packer.pack_text(text)
                {
                    slot.insert(bytes);
                }
            }
        }
        self.prepared = PreparedRows {
            messages,
            documents,
            texts,
            events,
            released: true,
        };
        for blocks in self
            .output
            .messages
            .iter_mut()
            .map(|r| r.message.blocks.as_mut_slice())
            .chain(
                self.output
                    .documents
                    .iter_mut()
                    .map(|r| r.document.blocks.as_mut_slice()),
            )
        {
            for block in blocks {
                match block {
                    Block::ToolCall { input, .. } => *input = Json::Null,
                    Block::ToolResult { data, .. } => *data = None,
                    Block::Other { data, .. } => *data = Json::Null,
                    _ => {}
                }
            }
        }
        for row in &mut self.output.documents {
            row.document.metadata = Json::Null;
        }
        for event in &mut self.output.events {
            event.data = Json::Null;
        }
    }
}
