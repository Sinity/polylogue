//! Current portable records read from the same stored rows as real exports.
use polylogue_model::{
    SessionId,
    bundle::{LinkRecord, MessageRecord, Record, SessionRecord},
};
use polylogue_store::{Reader, Result, StoreError};

/// A physical session and its own messages, events and links in display order.
/// Commit positions are local to each archive, so comparisons leave them zero.
pub fn session_bundle(reader: &Reader<'_>, id: &SessionId) -> Result<Vec<Record>> {
    let session = reader
        .session(id)?
        .ok_or_else(|| StoreError::NotFound(id.to_string()))?;
    let mut records = vec![Record::Session(SessionRecord {
        reference: format!("session:{id}"),
        commit: 0,
        copied_prefixes: reader.copied_prefixes(id)?,
        session,
    })];
    for message in reader.messages(id, false)? {
        let (message, selected) = reader
            .message_with_block_selection(&message.id)?
            .ok_or_else(|| StoreError::NotFound(message.id.to_string()))?;
        records.push(Record::Message(Box::new(MessageRecord {
            reference: format!("message:{}", message.id),
            commit: 0,
            order: reader
                .message_order(&message.id)?
                .ok_or_else(|| StoreError::NotFound(message.id.to_string()))?,
            source_version: selected.source_version,
            block_count: selected.block_count,
            block_ordinals: selected.block_ordinals,
            injections: reader.injection_evidence(&message.id, selected.source_version)?,
            bindings: reader
                .stated_attachment_bindings(&message.id, selected.source_version)?
                .into_iter()
                .map(|record| record.claim().clone())
                .collect(),
            copied_for: reader.copied_for(&message.id)?,
            message,
        })));
    }
    reader.visit_event_records_from_ordinal(id, 0, |mut event| {
        event.commit = 0;
        records.push(Record::Event(event));
        std::ops::ControlFlow::Continue(())
    })?;
    for link in reader.links(id)? {
        records.push(Record::Link(LinkRecord {
            reference: LinkRecord::reference(&link),
            commit: 0,
            link,
        }));
    }
    Ok(records)
}
