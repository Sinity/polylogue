//! Hydrate canonical rows lazily after the shared selector has resolved refs.
use super::super::{Response, error, store_error};
use polylogue_model::{
    Reference, SessionId,
    bundle::*,
    read::{ProjectionEvidence, ReadRecordKind, ReadRecordLocator},
};
use polylogue_store::Reader;
use std::ops::ControlFlow;

pub(super) type Visit<'a> =
    dyn FnMut(Record, ProjectionEvidence, ReadRecordLocator) -> Result<bool, Response> + 'a;

fn locator(kind: ReadRecordKind, ordinal: u64) -> ReadRecordLocator {
    ReadRecordLocator { kind, ordinal }
}

fn stale_record() -> Response {
    error("stale_cursor", "continued record is no longer selected")
}

fn admission(commit: Option<u64>, reference: &str) -> Result<u64, Response> {
    commit.ok_or_else(|| {
        error(
            "not_yet_available",
            format!("{reference} has no retained source admission evidence"),
        )
    })
}

pub(super) fn reference(
    reader: &Reader<'_>,
    reference: &Reference,
    inherited: bool,
    start: Option<&ReadRecordLocator>,
    visit: &mut Visit<'_>,
) -> Result<bool, Response> {
    if let Reference::Session(id) = reference {
        return session(reader, id, inherited, start, visit);
    }
    let kind = match reference {
        Reference::Message(_) | Reference::Block { .. } => ReadRecordKind::Message,
        Reference::Document(_) => ReadRecordKind::Document,
        Reference::Event(_) => ReadRecordKind::Event,
        _ => {
            return Err(error(
                "not_built",
                format!("the selected reference {reference} has no canonical read adapter"),
            ));
        }
    };
    let at = locator(kind, 0);
    if start.is_some_and(|start| *start != at) {
        return Err(stale_record());
    }
    let mut visit = |record, evidence| visit(record, evidence, at);
    match reference {
        Reference::Message(id) | Reference::Block { message: id, .. } => {
            let (message, mut projection) = reader
                .message_with_projection(id)
                .map_err(|e| store_error(&e))?
                .ok_or_else(|| error("not_found", format!("no message {id}")))?;
            let reference_text = format!("message:{id}");
            let commit = admission(
                reader.message_admission(id).map_err(|e| store_error(&e))?,
                &reference_text,
            )?;
            let order = reader
                .message_order(id)
                .map_err(|e| store_error(&e))?
                .ok_or_else(|| {
                    error(
                        "not_yet_available",
                        format!("{reference_text} has no physical reading position"),
                    )
                })?;
            let mut message = message;
            if let Reference::Block { n, .. } = reference {
                let Some(position) = projection
                    .block_ordinals
                    .iter()
                    .position(|ordinal| ordinal == n)
                else {
                    return Err(error(
                        "not_yet_available",
                        format!("block:{id}#{n} is not recorded in the selected source version"),
                    ));
                };
                message.blocks = vec![message.blocks.swap_remove(position)];
                projection.block_ordinals = vec![*n];
            }
            visit(
                message_record(reader, message, projection, commit, order)?,
                ProjectionEvidence::default(),
            )
        }
        Reference::Document(id) => {
            let document = reader
                .document(id)
                .map_err(|e| store_error(&e))?
                .ok_or_else(|| error("not_found", format!("no document {id}")))?;
            let reference = format!("document:{id}");
            let commit = admission(
                reader.document_commit(id).map_err(|e| store_error(&e))?,
                &reference,
            )?;
            visit(
                Record::Document(DocumentRecord {
                    reference,
                    commit,
                    document,
                }),
                ProjectionEvidence::default(),
            )
        }
        Reference::Event(id) => {
            let record = reader
                .event_record(id)
                .map_err(|e| store_error(&e))?
                .ok_or_else(|| error("not_found", format!("no event {id}")))?;
            visit(Record::Event(record), ProjectionEvidence::default())
        }
        _ => Err(error(
            "not_built",
            format!("the selected reference {reference} has no canonical read adapter"),
        )),
    }
}

fn session(
    reader: &Reader<'_>,
    id: &SessionId,
    inherited: bool,
    start: Option<&ReadRecordLocator>,
    visit: &mut Visit<'_>,
) -> Result<bool, Response> {
    use ReadRecordKind::{Event, Link, Message, Session};
    if start.is_some_and(|start| {
        !matches!(start.kind, Session | Message | Event | Link)
            || (start.kind == Session && start.ordinal != 0)
    }) {
        return Err(stale_record());
    }
    let reference = format!("session:{id}");
    if start.is_none_or(|start| start.kind == Session) {
        let session = reader
            .session(id)
            .map_err(|e| store_error(&e))?
            .ok_or_else(|| error("not_found", format!("no session {id}")))?;
        let commit = admission(
            reader.session_commit(id).map_err(|e| store_error(&e))?,
            &reference,
        )?;
        if !visit(
            Record::Session(SessionRecord {
                reference,
                commit,
                session,
            }),
            ProjectionEvidence::default(),
            locator(Session, 0),
        )? {
            return Ok(false);
        }
    }
    if start.is_none_or(|start| matches!(start.kind, Session | Message)) {
        let selected = reader
            .select_messages(id, inherited)
            .map_err(|e| store_error(&e))?;
        let mut ordinal = start
            .filter(|start| start.kind == Message)
            .map_or(0, |s| s.ordinal);
        let from = if start.is_some_and(|start| start.kind == Message) {
            selected
                .locate_ordinal(ordinal)
                .map_err(|e| store_error(&e))?
                .ok_or_else(stale_record)?
        } else {
            Default::default()
        };
        let mut result = Ok(true);
        selected
            .visit_from_with_record_evidence(from, |message, projection, commit, order| {
                let reference = format!("message:{}", message.id);
                let next = admission(commit, &reference).and_then(|commit| {
                    let evidence = ProjectionEvidence {
                        inherited: message.session != *id,
                        ..Default::default()
                    };
                    let current = locator(Message, ordinal);
                    ordinal = ordinal.checked_add(1).ok_or_else(stale_record)?;
                    visit(
                        message_record(reader, message, projection, commit, order)?,
                        evidence,
                        current,
                    )
                });
                match next {
                    Ok(true) => ControlFlow::Continue(()),
                    other => {
                        result = other;
                        ControlFlow::Break(())
                    }
                }
            })
            .map_err(|e| store_error(&e))?;
        if !result? {
            return Ok(false);
        }
    }
    if start.is_none_or(|start| start.kind != Link) {
        let mut ordinal = start
            .filter(|start| start.kind == Event)
            .map_or(0, |s| s.ordinal);
        let mut seen = false;
        let mut result = Ok(true);
        reader
            .visit_event_records_from_ordinal(id, ordinal, |event| {
                seen = true;
                let current = locator(Event, ordinal);
                let next = ordinal
                    .checked_add(1)
                    .ok_or_else(stale_record)
                    .and_then(|next| {
                        ordinal = next;
                        visit(Record::Event(event), ProjectionEvidence::default(), current)
                    });
                match next {
                    Ok(true) => ControlFlow::Continue(()),
                    other => {
                        result = other;
                        ControlFlow::Break(())
                    }
                }
            })
            .map_err(|e| store_error(&e))?;
        if !result? {
            return Ok(false);
        }
        if !seen && start.is_some_and(|start| start.kind == Event) {
            return Err(stale_record());
        }
    }
    let mut ordinal = start
        .filter(|start| start.kind == Link)
        .map_or(0, |s| s.ordinal);
    let mut seen = false;
    let mut result = Ok(true);
    reader
        .visit_links_from_ordinal(id, ordinal, |link| {
            seen = true;
            let reference = LinkRecord::reference(&link);
            let next = reader
                .link_commit(&link)
                .map_err(|e| store_error(&e))
                .and_then(|commit| admission(commit, &reference))
                .and_then(|commit| {
                    let current = locator(Link, ordinal);
                    ordinal = ordinal.checked_add(1).ok_or_else(stale_record)?;
                    visit(
                        Record::Link(LinkRecord {
                            reference,
                            commit,
                            link,
                        }),
                        ProjectionEvidence::default(),
                        current,
                    )
                });
            match next {
                Ok(true) => ControlFlow::Continue(()),
                other => {
                    result = other;
                    ControlFlow::Break(())
                }
            }
        })
        .map_err(|e| store_error(&e))?;
    if !result? {
        return Ok(false);
    }
    if !seen && start.is_some_and(|start| start.kind == Link) {
        return Err(stale_record());
    }
    Ok(true)
}

/// Preserve sidecars independently of immutable message bytes and original ordinals.
pub(super) fn message_record(
    reader: &Reader<'_>,
    message: polylogue_model::Message,
    projection: MessageProjection,
    commit: u64,
    order: u64,
) -> Result<Record, Response> {
    use polylogue_model::{Block, attachment_binding::AttachmentNamespace};
    let mut injections = reader
        .injection_evidence(&message.id, projection.source_version)
        .map_err(|e| store_error(&e))?;
    injections.retain(|evidence| {
        projection
            .block_ordinals
            .binary_search(&evidence.block)
            .is_ok()
    });
    let mut bindings = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (ordinal, block) in projection.block_ordinals.iter().zip(&message.blocks) {
        let attachments: &[polylogue_model::Attachment] = match block {
            Block::Image(attachment) | Block::File(attachment) => std::slice::from_ref(attachment),
            Block::ToolResult { attachments, .. } => attachments,
            _ => &[],
        };
        for (index, attachment) in attachments.iter().enumerate() {
            if attachment.blob.is_some() || attachment.source_id.is_none() {
                continue;
            }
            let contexts = reader
                .attachment_binding_contexts(&message.id, *ordinal, index as u64)
                .map_err(|e| store_error(&e))?;
            for context in contexts.contexts {
                if context.claim.source_version != projection.source_version {
                    continue;
                }
                let actual = context.selection.context;
                let mut claim = context.claim;
                // Original claims and supplied prior exports remain declarations.
                claim.exported_from = AttachmentNamespace {
                    input: actual.input,
                    chain: actual.chain,
                    import_id: actual.import_id,
                };
                claim.validate().map_err(|e| {
                    store_error(&polylogue_store::StoreError::Invalid(e.to_string()))
                })?;
                if seen.insert(claim.clone()) {
                    bindings.push(claim);
                }
            }
        }
    }
    Ok(Record::Message(Box::new(MessageRecord {
        reference: format!("message:{}", message.id),
        commit,
        order,
        source_version: projection.source_version,
        block_count: projection.block_count,
        block_ordinals: projection.block_ordinals,
        injections,
        bindings,
        message,
    })))
}
