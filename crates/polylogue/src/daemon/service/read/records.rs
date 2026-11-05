//! Hydrate canonical rows lazily after the shared selector has resolved refs.
use super::super::Failure;
use polylogue_model::{
    SessionId,
    bundle::*,
    read::{ProjectionEvidence, ReadRecordKind, ReadRecordLocator, ReadTarget},
};
use polylogue_store::{Reader, read::limit::ReadLimit};
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
    dyn FnMut(SelectedRecord, ProjectionEvidence, ReadRecordLocator) -> Result<bool, Failure> + 'a;

fn locator(kind: ReadRecordKind, ordinal: u64) -> ReadRecordLocator {
    ReadRecordLocator { kind, ordinal }
}

fn stale_record() -> Failure {
    Failure::new("stale_cursor", "continued record is no longer selected")
}

fn admission(commit: Option<u64>, reference: &str) -> Result<u64, Failure> {
    commit.ok_or_else(|| {
        Failure::new(
            "not_yet_available",
            format!("{reference} has no retained source admission evidence"),
        )
    })
}

pub(super) fn reference(
    reader: &Reader<'_>,
    target: &ReadTarget,
    inherited: bool,
    start: Option<&ReadRecordLocator>,
    visit: &mut Visit<'_>,
) -> Result<bool, Failure> {
    let single = |kind| {
        let at = locator(kind, 0);
        if start.is_some_and(|start| *start != at) {
            Err(stale_record())
        } else {
            Ok(at)
        }
    };
    match target {
        ReadTarget::Session(id) => session(reader, id, inherited, start, visit),
        ReadTarget::Message(id) => {
            let at = single(ReadRecordKind::Message)?;
            visit(
                message(reader, id, None)?.into(),
                ProjectionEvidence::default(),
                at,
            )
        }
        ReadTarget::Block { message: id, n } => {
            let at = single(ReadRecordKind::Message)?;
            visit(
                message(reader, id, Some(*n))?.into(),
                ProjectionEvidence::default(),
                at,
            )
        }
        ReadTarget::MessageRange { first, last } => {
            if start.is_some_and(|start| start.kind != ReadRecordKind::Message) {
                return Err(stale_record());
            }
            let mut ordinal = start.map_or(0, |start| start.ordinal);
            let mut seen = false;
            let mut result = Ok(true);
            reader
                .visit_message_range(first, last, ordinal, |id| {
                    seen = true;
                    let current = locator(ReadRecordKind::Message, ordinal);
                    let next = ordinal
                        .checked_add(1)
                        .ok_or_else(stale_record)
                        .and_then(|next| {
                            ordinal = next;
                            visit(
                                message(reader, &id, None)?.into(),
                                ProjectionEvidence::default(),
                                current,
                            )
                        });
                    Ok(match next {
                        Ok(true) => ControlFlow::Continue(()),
                        other => {
                            result = other;
                            ControlFlow::Break(())
                        }
                    })
                })
                .map_err(Failure::from)?;
            if !seen && start.is_some() {
                return Err(stale_record());
            }
            result
        }
        ReadTarget::Document(id) => {
            let at = single(ReadRecordKind::Document)?;
            let document = reader
                .document(id)
                .map_err(Failure::from)?
                .ok_or_else(|| Failure::new("not_found", format!("no document {id}")))?;
            let reference = format!("document:{id}");
            let commit = admission(
                reader.document_commit(id).map_err(Failure::from)?,
                &reference,
            )?;
            visit(
                Record::Document(DocumentRecord {
                    reference,
                    commit,
                    document,
                })
                .into(),
                ProjectionEvidence::default(),
                at,
            )
        }
        ReadTarget::Event(id) => {
            let at = single(ReadRecordKind::Event)?;
            let record = reader
                .event_record(id)
                .map_err(Failure::from)?
                .ok_or_else(|| Failure::new("not_found", format!("no event {id}")))?;
            visit(
                Record::Event(record).into(),
                ProjectionEvidence::default(),
                at,
            )
        }
        ReadTarget::Assertion { .. } | ReadTarget::Delivery(_) => {
            let at = single(ReadRecordKind::Authored)?;
            super::authored::reference(reader, &target.reference(), &mut |record, evidence| {
                visit(record.into(), evidence, at)
            })
        }
    }
}

/// One message, or one of its blocks, with its admission and reading order.
fn message(
    reader: &Reader<'_>,
    id: &polylogue_model::MessageId,
    block: Option<u64>,
) -> Result<Record, Failure> {
    let (mut message, mut projection) = reader
        .message_with_projection(id)
        .map_err(Failure::from)?
        .ok_or_else(|| Failure::new("not_found", format!("no message {id}")))?;
    let reference = format!("message:{id}");
    let commit = admission(
        reader.message_admission(id).map_err(Failure::from)?,
        &reference,
    )?;
    let order = reader
        .message_order(id)
        .map_err(Failure::from)?
        .ok_or_else(|| {
            Failure::new(
                "not_yet_available",
                format!("{reference} has no physical reading position"),
            )
        })?;
    if let Some(n) = block {
        let Some(position) = projection
            .block_ordinals
            .iter()
            .position(|ordinal| *ordinal == n)
        else {
            return Err(Failure::new(
                "not_yet_available",
                format!("block:{id}#{n} is not recorded in the selected source version"),
            ));
        };
        message.blocks = vec![message.blocks.swap_remove(position)];
        projection.block_ordinals = vec![n];
    }
    message_record(reader, message, projection, commit, order)
}

fn session(
    reader: &Reader<'_>,
    id: &SessionId,
    inherited: bool,
    start: Option<&ReadRecordLocator>,
    visit: &mut Visit<'_>,
) -> Result<bool, Failure> {
    use ReadRecordKind::{Event, Link, LinkIssue, Message, Session};
    if start.is_some_and(|start| {
        !matches!(start.kind, Session | LinkIssue | Message | Event | Link)
            || (start.kind == Session && start.ordinal != 0)
    }) {
        return Err(stale_record());
    }
    let selected = reader
        .select_messages(id, inherited)
        .map_err(Failure::from)?;
    let reference = format!("session:{id}");
    if start.is_none_or(|start| start.kind == Session) {
        let session = reader
            .session(id)
            .map_err(Failure::from)?
            .ok_or_else(|| Failure::new("not_found", format!("no session {id}")))?;
        let commit = admission(
            reader.session_commit(id).map_err(Failure::from)?,
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
                .map_err(Failure::from)?
                .ok_or_else(stale_record)?
        } else {
            Default::default()
        };
        let mut result = Ok(true);
        ReadLimit::set_stage("read records: messages");
        selected
            .visit_from_with_record_evidence(from, |message, projection, commit, order| {
                ReadLimit::set_stage("read records: messages");
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
            .map_err(Failure::from)?;
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
        ReadLimit::set_stage("read records: events");
        reader
            .visit_event_records_from_ordinal(id, ordinal, |event| {
                ReadLimit::set_stage("read records: events");
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
            .map_err(Failure::from)?;
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
    ReadLimit::set_stage("read records: links");
    reader
        .visit_links_from_ordinal(id, ordinal, |link| {
            seen = true;
            ReadLimit::set_stage("read records: link admission");
            let reference = LinkRecord::reference(&link);
            let next = reader
                .link_commit(&link)
                .map_err(Failure::from)
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
        .map_err(Failure::from)?;
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
) -> Result<Record, Failure> {
    use polylogue_model::attachment_binding::AttachmentNamespace;
    let mut injections = reader
        .injection_evidence(&message.id, projection.source_version)
        .map_err(Failure::from)?;
    injections.retain(|evidence| {
        projection
            .block_ordinals
            .binary_search(&evidence.block)
            .is_ok()
    });
    let retained = reader
        .attachment_binding_claims(&message.id, projection.source_version)
        .map_err(Failure::from)?;
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
            .map_err(|e| Failure::from(polylogue_store::StoreError::Invalid(e.to_string())))?;
        if seen.insert(claim.clone()) {
            bindings.push(claim);
        }
    }
    let covered: std::collections::HashSet<_> = bindings
        .iter()
        .map(|claim| (claim.block, claim.attachment))
        .collect();
    let uncovered: Vec<_> = polylogue_model::attachment_binding::attachment_slots(
        &message.blocks,
        &projection.block_ordinals,
    )
    .into_iter()
    .filter(|slot| !covered.contains(&(slot.block, slot.attachment)))
    .collect();
    for claim in
        reader.current_attachment_claims(&message.id, projection.source_version, &uncovered)?
    {
        claim
            .validate()
            .map_err(|e| Failure::from(polylogue_store::StoreError::Invalid(e.to_string())))?;
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
