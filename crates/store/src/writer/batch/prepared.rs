use super::*;

/// Packed rows stay with their original message and are committed together,
/// before the parsed structured payloads are dropped.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PreparedRows {
    pub(in crate::writer) messages: Vec<PreparedMessage>,
    pub(in crate::writer) documents: Vec<PreparedDocument>,
    pub(in crate::writer) fragments: crate::fragment::PreparedFragments,
    pub(in crate::writer) texts: crate::text_rows::PreparedTexts,
    pub(in crate::writer) events: Vec<crate::encoding::PreparedEvent>,
    /// The groups holding the messages' and events' content.
    pub(in crate::writer) content: crate::content::Groups,
    pub(in crate::writer) released: bool,
}

impl PreparedRows {
    /// Prepares message `index` again from `message`, in a group after the
    /// step's others.
    pub(in crate::writer) fn repack_message(
        &mut self,
        index: usize,
        message: &mut Message,
        stored: &[polylogue_model::StoredField],
    ) -> Result<()> {
        let mut packer = crate::content::Packer::continuing(std::mem::take(&mut self.content));
        let prepared = PreparedMessage::new(message, stored, &mut packer);
        self.content = packer.finish()?;
        self.messages[index] = prepared?;
        Ok(())
    }

    /// A prepared message's content read back, its texts still moved out.
    pub(in crate::writer) fn message_value(&self, index: usize) -> Result<Message> {
        let prepared = self
            .messages
            .get(index)
            .ok_or_else(|| StoreError::Invalid("no such prepared message".into()))?;
        let bytes = self.content.read(prepared.content)?;
        Ok(crate::encoding::read_message_content(&bytes, true)?.0)
    }

    /// A prepared event's data.
    pub(in crate::writer) fn event_data(&self, index: usize) -> Result<Json> {
        let prepared = self
            .events
            .get(index)
            .ok_or_else(|| StoreError::Invalid("no such prepared event".into()))?;
        let bytes = self.content.read(prepared.content)?;
        crate::encoding::read_json(&bytes[prepared.fields_length as usize..])
    }
}

impl PreparedRows {
    pub(in crate::writer) fn validate(&self, output: &Output) -> Result<()> {
        self.fragments.validate(output)?;
        if !self.released {
            return Err(StoreError::Invalid("reading rows were not prepared".into()));
        }
        for (&index, block_selection) in &output.block_selections {
            let row = output
                .messages
                .get(index)
                .ok_or(polylogue_model::bundle::BundleError::InvalidBlockSelection)?;
            block_selection.validate(row.message.blocks.len())?;
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
            identity.validate().map_err(crate::StoreError::invalid)?;
        }
        for (&index, id) in &output.event_ids {
            if index >= output.events.len()
                || id.is_empty()
                || output
                    .imported_events
                    .get(&index)
                    .is_some_and(|identity| &identity.id != id)
            {
                return Err(StoreError::Invalid(
                    "event identity does not match its imported identity".into(),
                ));
            }
        }
        for member in &output.prefix_members {
            if member.source.as_ref().is_some_and(|source| {
                source.input.is_empty() || source.chain < 1 || source.parse_version.is_empty()
            }) {
                return Err(StoreError::Invalid(
                    "copied-prefix member names no source reading".into(),
                ));
            }
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
                .any(|(p, r)| p.id != r.message.id || p.texts.len() != r.message.blocks.len())
        {
            return Err(StoreError::Invalid(
                "prepared rows no longer match their parsed records".into(),
            ));
        }
        for (index, prepared) in self.messages.iter().enumerate() {
            prepared.block_selection.validate(prepared.texts.len())?;
            if prepared.block_selection.source_version.0 != prepared.hash {
                return Err(polylogue_model::bundle::BundleError::SourceVersionMismatch.into());
            }
            match output.block_selections.get(&index) {
                Some(block_selection) if block_selection != &prepared.block_selection => {
                    return Err(polylogue_model::bundle::BundleError::BlockSelectionConflict.into());
                }
                None if prepared.block_selection.block_count != prepared.texts.len() as u64
                    || prepared
                        .block_selection
                        .block_ordinals
                        .iter()
                        .copied()
                        .ne(0..prepared.texts.len() as u64) =>
                {
                    return Err(polylogue_model::bundle::BundleError::InvalidBlockSelection.into());
                }
                _ => {}
            }
        }
        super::super::hook_fields::validate(output, &self.messages)?;
        Ok(())
    }
}

impl ParsePart {
    /// Packs complete rows off the writer thread. A second call keeps the
    /// original hashes and payloads, which parsed values no longer contain.
    pub fn prepare(&mut self) {
        if self.prepared.released {
            return;
        }
        let Ok(mut packer) = crate::encoding::RowPacker::new() else {
            return;
        };
        // Messages and events lay their content into the step's groups in
        // the order the step states them.
        let mut content = crate::content::Packer::default();
        let block_selections = &self.output.block_selections;
        let stored = &self.output.stored_fields;
        let Ok(messages) = self
            .output
            .messages
            .iter_mut()
            .enumerate()
            .map(|(index, r)| {
                let stored = stored.get(&index).map_or(&[][..], Vec::as_slice);
                match block_selections.get(&index) {
                    Some(block_selection) => PreparedMessage::projected(
                        &mut r.message,
                        block_selection,
                        stored,
                        &mut content,
                    ),
                    None => PreparedMessage::new(&mut r.message, stored, &mut content),
                }
            })
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
        let stored = &self.output.stored_event_fields;
        let Ok(events) = self
            .output
            .events
            .iter_mut()
            .enumerate()
            .map(|(index, event)| {
                let stored = stored.get(&index).map_or(&[][..], Vec::as_slice);
                crate::encoding::PreparedEvent::new(event, stored, &mut content)
            })
            .collect::<Result<Vec<_>>>()
        else {
            return;
        };
        let Ok(content) = content.finish() else {
            return;
        };
        let Ok(fragments) = crate::fragment::PreparedFragments::prepare(&self.output) else {
            return;
        };
        let Ok(texts) = crate::text_rows::PreparedTexts::pack(
            self.output
                .messages
                .iter()
                .map(|r| r.message.blocks.as_slice())
                .chain(
                    self.output
                        .documents
                        .iter()
                        .map(|r| r.document.blocks.as_slice()),
                )
                .flatten()
                .filter_map(block_text),
        ) else {
            return;
        };
        self.prepared = PreparedRows {
            messages,
            documents,
            fragments,
            texts,
            events,
            content,
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
        // An event whose data holds a value left in the stored bytes keeps
        // its data: the index has only the value's place, and a producer of
        // complete rows needs the value ([`Self::complete_output`]).
        for (index, event) in self.output.events.iter_mut().enumerate() {
            if !self.output.stored_event_fields.contains_key(&index) {
                event.data = Json::Null;
            }
        }
    }
}
impl ParsePart {
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

impl ParsePart {
    /// Restores complete domain rows from the worker's codecs.
    pub fn complete_output(&self) -> Result<Output> {
        self.prepared.validate(&self.output)?;
        let mut output = self.output.clone();
        for (index, row) in output.messages.iter_mut().enumerate() {
            let stored = self.prepared.message_value(index)?;
            crate::writer::attachments::restore_values(&mut row.message.blocks, &stored.blocks)?;
        }
        for (row, prepared) in output.documents.iter_mut().zip(&self.prepared.documents) {
            let mut blocks = std::mem::take(&mut row.document.blocks);
            crate::writer::attachments::restore_packed(&mut blocks, prepared)?;
            row.document = crate::encoding::unpack_json(&prepared.fields)?;
            row.document.blocks = blocks;
        }
        for (index, event) in output.events.iter_mut().enumerate() {
            if !output.stored_event_fields.contains_key(&index) {
                event.data = self.prepared.event_data(index)?;
            }
        }
        Ok(output)
    }
}
impl ParsePart {
    /// An event about its input, not a session, is that input's record: it
    /// has the record's byte range and follows no message.
    pub(crate) fn check_events(&self, input: &str) -> Result<()> {
        for (index, event) in self.output.events.iter().enumerate() {
            // An identity the source supplies is kept as given.
            if self.output.event_ids.contains_key(&index)
                || self.output.imported_events.contains_key(&index)
            {
                continue;
            }
            if event.session.is_none() && input.is_empty() {
                return Err(StoreError::Invalid(
                    "an input event needs its input identity".into(),
                ));
            }
            if event.session.is_none()
                && (event.after.is_some() || !event.range.is_some_and(|(start, end)| start < end))
            {
                return Err(StoreError::Invalid(
                    "an input event needs its record range and cannot follow a message".into(),
                ));
            }
        }
        Ok(())
    }

    /// Event identities are derived once from the complete reading. Portable
    /// imports keep the IDs they were given.
    pub fn event_identities(&self, input: &str, chain: i64) -> Result<Vec<String>> {
        self.prepared.validate(&self.output)?;
        let mut occurrences = std::collections::BTreeMap::<(u64, u64), u64>::new();
        self.output
            .events
            .iter()
            .zip(&self.prepared.events)
            .enumerate()
            .map(|(index, (event, packed))| {
                let occurrence = if event.session.is_none() {
                    let next = occurrences
                        .entry(event.range.unwrap_or_default())
                        .or_default();
                    let current = *next;
                    *next = next.checked_add(1).ok_or_else(|| {
                        StoreError::Invalid(
                            "event occurrence is outside the stored integer range".into(),
                        )
                    })?;
                    current
                } else {
                    0
                };
                if let Some(id) = self.output.event_ids.get(&index) {
                    return Ok(id.clone());
                }
                if let Some(identity) = self.output.imported_events.get(&index) {
                    return Ok(identity.id.clone());
                }
                super::super::readings::event_id(
                    input,
                    chain,
                    event,
                    &packed.content_id,
                    occurrence,
                )
            })
            .collect()
    }
}
