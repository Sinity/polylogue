//! Current bundle records for comparing complete session contents.
use polylogue_model::{
    SessionId,
    bundle::{
        ImportedEventIdentity, InputEventRecord, LinkRecord, MessageRecord, Record, SessionRecord,
        SourceClaim,
    },
};
use polylogue_store::{Reader, Result, StoreError, read::VisitStart};
use std::ops::ControlFlow;

/// Compare complete sessions, messages, events and links in reading order.
/// Source observations and sidecars are outside this fixture's comparison;
/// local commits and event coordinates use deterministic synthetic values.
pub fn session_bundle(reader: &Reader<'_>, id: &SessionId) -> Result<Vec<Record>> {
    let session = reader
        .session(id)?
        .ok_or_else(|| StoreError::NotFound(id.to_string()))?;
    let mut records = vec![Record::Session(SessionRecord {
        reference: format!("session:{id}"),
        commit: 0,
        copied_prefixes: Vec::new(),
        session,
    })];
    let selected = reader.select_messages(id, false)?;
    selected.require_complete()?;
    let mut order = 0;
    selected.visit_from_with_record_arrival(VisitStart::default(), |message, blocks, _, _| {
        records.push(Record::Message(Box::new(MessageRecord {
            reference: format!("message:{}", message.id),
            commit: 0,
            order,
            source_version: blocks.source_version,
            block_count: blocks.block_count,
            block_ordinals: blocks.block_ordinals,
            injections: Vec::new(),
            bindings: Vec::new(),
            copied_for: Vec::new(),
            message,
        })));
        order += 1;
        ControlFlow::Continue(())
    })?;
    for (ordinal, event) in reader.events(id)?.into_iter().enumerate() {
        let identity = format!("fixture:{id}:{ordinal}");
        records.push(Record::Event(InputEventRecord {
            reference: format!("event:{identity}"),
            commit: 0,
            identity: ImportedEventIdentity {
                id: identity,
                sources: vec![SourceClaim {
                    input: "fixture".into(),
                    source: "fixture".into(),
                    source_chain: 1,
                    source_start: None,
                    source_end: None,
                }],
            },
            event,
        }));
    }
    for link in reader.links(id)? {
        records.push(Record::Link(LinkRecord {
            reference: LinkRecord::reference(&link),
            commit: 0,
            link,
        }));
    }
    Ok(records)
}
