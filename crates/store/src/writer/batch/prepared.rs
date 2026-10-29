use super::*;

/// Packed rows stay with their original message and are committed together,
/// before the parsed structured payloads are dropped.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PreparedRows {
    pub(in crate::writer) messages: Vec<PreparedMessage>,
    pub(in crate::writer) documents: Vec<PreparedDocument>,
    pub(in crate::writer) texts: HashMap<[u8; 32], Vec<u8>>,
    pub(in crate::writer) events: Vec<(String, Vec<u8>)>,
    pub(in crate::writer) released: bool,
    pub(in crate::writer) keys: Option<Arc<super::super::resolve::ResolvedKeys>>,
    pub(in crate::writer) chain: Option<i64>,
    pub(in crate::writer) choices: Option<super::super::resolve::versions::Choices>,
}

impl PreparedRows {
    pub(in crate::writer) fn validate(&self, output: &Output) -> Result<()> {
        if !self.released {
            return Err(StoreError::Invalid("reading rows were not prepared".into()));
        }
        for (&index, identity) in &output.imported_events {
            if index >= output.events.len() {
                return Err(StoreError::Invalid("imported identity has no event".into()));
            }
            if output.events[index].session.is_none() && output.events[index].after.is_some() {
                return Err(StoreError::Invalid(
                    "an imported input event cannot follow a message".into(),
                ));
            }
            identity
                .validate()
                .map_err(|error| StoreError::Invalid(error.to_string()))?;
        }
        for (&index, &order) in &output.message_order {
            if index >= output.messages.len() || i64::try_from(order).is_err() {
                return Err(StoreError::Invalid("invalid imported message order".into()));
            }
        }
        if self.messages.len() != output.messages.len()
            || self.documents.len() != output.documents.len()
            || self
                .documents
                .iter()
                .zip(&output.documents)
                .any(|(p, r)| p.id != r.document.id || p.blocks.len() != r.document.blocks.len())
            || self.events.len() != output.events.len()
            || self
                .messages
                .iter()
                .zip(&output.messages)
                .any(|(p, r)| p.id != r.message.id || p.blocks.len() != r.message.blocks.len())
        {
            return Err(StoreError::Invalid(
                "prepared rows no longer match their parsed records".into(),
            ));
        }
        Ok(())
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
            keys: None,
            chain: None,
            choices: None,
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
impl ParseStep {
    pub(in crate::writer) fn message_position(&self, index: usize) -> Result<Option<i64>> {
        let order =
            match self.output.message_order.get(&index) {
                Some(&order) => Some(order),
                None => match self.order_from {
                    Some(from) => Some(from.checked_add(index as u64).ok_or_else(|| {
                        StoreError::Invalid("message reading order overflow".into())
                    })?),
                    None => self.repeats_whole_conversations.then_some(index as u64),
                },
            };
        order
            .map(|order| {
                i64::try_from(order).map_err(|_| {
                    StoreError::Invalid(
                        "message reading order is outside the stored integer range".into(),
                    )
                })
            })
            .transpose()
    }
}

impl ParseStep {
    /// Restores complete domain facts for the primary producer from the worker's codecs.
    pub fn complete_output(&self) -> Result<Output> {
        self.prepared.validate(&self.output)?;
        let mut output = self.output.clone();
        if !self.prepared.released {
            return Ok(output);
        }
        for (row, prepared) in output.messages.iter_mut().zip(&self.prepared.messages) {
            crate::writer::attachments::restore_values(&mut row.message.blocks, prepared)?;
        }
        for (row, prepared) in output.documents.iter_mut().zip(&self.prepared.documents) {
            let mut blocks = std::mem::take(&mut row.document.blocks);
            crate::writer::attachments::restore_values(&mut blocks, prepared)?;
            row.document = crate::encoding::unpack_json(&prepared.fields)?;
            row.document.blocks = blocks;
        }
        for (event, (_, data)) in output.events.iter_mut().zip(&self.prepared.events) {
            event.data = crate::encoding::unpack_json(data)?;
        }
        Ok(output)
    }
}
