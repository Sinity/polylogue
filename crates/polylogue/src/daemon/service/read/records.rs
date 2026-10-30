//! Hydrate canonical rows lazily after the shared selector has resolved refs.
use super::super::{Response, error, store_error};
use polylogue_model::{
    Reference, SessionId,
    bundle::*,
    read::{ProjectionEvidence, ReadRecordKind, ReadRecordLocator},
};
use polylogue_store::Reader;
use std::ops::ControlFlow;

pub(super) enum SelectedRecord {
    Canonical(Box<Record>),
    LinkIssue {
        session: SessionId,
        issue: Box<polylogue_model::read::ParentIssue>,
    },
}
impl From<Record> for SelectedRecord {
    fn from(record: Record) -> Self {
        Self::Canonical(Box::new(record))
    }
}

pub(super) type Visit<'a> =
    dyn FnMut(SelectedRecord, ProjectionEvidence, ReadRecordLocator) -> Result<bool, Response> + 'a;

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
        Reference::Request(_)
        | Reference::Assertion(_)
        | Reference::Result { .. }
        | Reference::Delivery(_)
        | Reference::Variant { .. } => ReadRecordKind::Authored,
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
    let mut visit = |record: Record, evidence| visit(record.into(), evidence, at);
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
        Reference::Request(_)
        | Reference::Assertion(_)
        | Reference::Result { .. }
        | Reference::Delivery(_)
        | Reference::Variant { .. } => super::authored::reference(reader, reference, &mut visit),
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
    use ReadRecordKind::{Event, Link, LinkIssue, Message, Session};
    if start.is_some_and(|start| {
        !matches!(start.kind, Session | LinkIssue | Message | Event | Link)
            || (start.kind == Session && start.ordinal != 0)
    }) {
        return Err(stale_record());
    }
    let selected = reader
        .select_messages(id, inherited)
        .map_err(|e| store_error(&e))?;
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
            })
            .into(),
            ProjectionEvidence {
                lineage_complete: if inherited {
                    polylogue_model::read::ValueState::known(selected.parents().complete)
                } else {
                    polylogue_model::read::ValueState::NotApplicable
                },
                ..Default::default()
            },
            locator(Session, 0),
        )? {
            return Ok(false);
        }
    }
    if inherited && start.is_none_or(|start| matches!(start.kind, Session | LinkIssue)) {
        let from = start
            .filter(|start| start.kind == LinkIssue)
            .map_or(0, |start| start.ordinal);
        if start.is_some_and(|start| start.kind == LinkIssue)
            && usize::try_from(from)
                .ok()
                .is_none_or(|from| from >= selected.parents().issues.len())
        {
            return Err(stale_record());
        }
        for (ordinal, issue) in selected
            .parents()
            .issues
            .iter()
            .enumerate()
            .skip(usize::try_from(from).map_err(|_| stale_record())?)
        {
            if !visit(
                SelectedRecord::LinkIssue {
                    session: id.clone(),
                    issue: Box::new(issue.clone()),
                },
                ProjectionEvidence::default(),
                locator(
                    LinkIssue,
                    u64::try_from(ordinal).map_err(|_| stale_record())?,
                ),
            )? {
                return Ok(false);
            }
        }
    } else if start.is_some_and(|start| start.kind == LinkIssue) {
        return Err(stale_record());
    }
    if start.is_none_or(|start| matches!(start.kind, Session | LinkIssue | Message)) {
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
                        message_record(reader, message, projection, commit, order)?.into(),
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
                        visit(
                            Record::Event(event).into(),
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
                        })
                        .into(),
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
    use polylogue_model::attachment_binding::AttachmentNamespace;
    let mut injections = reader
        .injection_evidence(&message.id, projection.source_version)
        .map_err(|e| store_error(&e))?;
    injections.retain(|evidence| {
        projection
            .block_ordinals
            .binary_search(&evidence.block)
            .is_ok()
    });
    let retained = reader
        .attachment_binding_claims(&message.id, projection.source_version)
        .map_err(|e| store_error(&e))?;
    let mut bindings = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for record in retained {
        let mut claim = record.claim().clone();
        if projection
            .block_ordinals
            .binary_search(&claim.block)
            .is_err()
        {
            continue;
        }
        let actual = record.context();
        // Original claims and supplied prior exports remain declarations.
        claim.exported_from = AttachmentNamespace {
            input: actual.input.clone(),
            chain: actual.chain,
            import_id: actual.import_id.clone(),
        };
        claim
            .validate()
            .map_err(|e| store_error(&polylogue_store::StoreError::Invalid(e.to_string())))?;
        if seen.insert(claim.clone()) {
            bindings.push(claim);
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
