//! Hydrate stored rows lazily after the shared selector has resolved refs.
use crate::Error;
use polylogue_model::{
    SessionId,
    bundle::*,
    read::{LayoutFacts, ReadRecordKind, ReadRecordLocator, ReadTarget},
};
use polylogue_store::{Reader, read::limit::ReadLimit};
use std::ops::ControlFlow;

pub(super) enum SelectedRecord {
    Original(Box<Record>),
    LinkIssue {
        session: SessionId,
        issue: Box<polylogue_model::read::ParentIssue>,
    },
}
impl From<Record> for SelectedRecord {
    fn from(record: Record) -> Self {
        Self::Original(Box::new(record))
    }
}

pub(super) type Visit<'a> =
    dyn FnMut(SelectedRecord, LayoutFacts, ReadRecordLocator) -> Result<bool, Error> + 'a;

fn locator(kind: ReadRecordKind, ordinal: u64) -> ReadRecordLocator {
    ReadRecordLocator { kind, ordinal }
}

fn stale_record() -> Error {
    Error::stale_cursor("the record this page continues from no longer matches the reading")
}

fn required_commit(commit: Option<u64>, reference: &str) -> Result<u64, Error> {
    commit.ok_or_else(|| {
        Error::not_yet_available(format!(
            "{reference} has no stored commit position for its source"
        ))
    })
}

pub(super) fn reference(
    reader: &Reader<'_>,
    target: &ReadTarget,
    inherited: bool,
    start: Option<&ReadRecordLocator>,
    visit: &mut Visit<'_>,
) -> Result<bool, Error> {
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
                LayoutFacts::default(),
                at,
            )
        }
        ReadTarget::Block { message: id, n } => {
            let at = single(ReadRecordKind::Message)?;
            visit(
                message(reader, id, Some(*n))?.into(),
                LayoutFacts::default(),
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
                        .ok_or_else(|| Error::internal("a reading counted more than 2^64 records"))
                        .and_then(|next| {
                            ordinal = next;
                            visit(
                                message(reader, &id, None)?.into(),
                                LayoutFacts::default(),
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
                .map_err(Error::from)?;
            if !seen && start.is_some() {
                return Err(stale_record());
            }
            result
        }
        ReadTarget::Document(id) => {
            let at = single(ReadRecordKind::Document)?;
            let document = reader
                .document(id)
                .map_err(Error::from)?
                .ok_or_else(|| Error::not_found(format!("no document {id}")))?;
            let reference = format!("document:{id}");
            let commit =
                required_commit(reader.document_commit(id).map_err(Error::from)?, &reference)?;
            visit(
                Record::Document(DocumentRecord {
                    reference,
                    commit,
                    document,
                })
                .into(),
                LayoutFacts::default(),
                at,
            )
        }
        ReadTarget::Event(id) => {
            let at = single(ReadRecordKind::Event)?;
            let record = reader
                .event_record(id)
                .map_err(Error::from)?
                .ok_or_else(|| Error::not_found(format!("no event {id}")))?;
            visit(Record::Event(record).into(), LayoutFacts::default(), at)
        }
        ReadTarget::Annotation { .. } | ReadTarget::Delivery(_) => {
            let at = single(ReadRecordKind::Authored)?;
            super::authored::reference(reader, &target.reference(), &mut |record, facts| {
                visit(record.into(), facts, at)
            })
        }
    }
}

/// One message, or one of its blocks, with its commit position and reading order.
pub(super) fn message(
    reader: &Reader<'_>,
    id: &polylogue_model::MessageId,
    block: Option<u64>,
) -> Result<Record, Error> {
    let (mut message, mut block_selection) = reader
        .message_with_block_selection(id)
        .map_err(Error::from)?
        .ok_or_else(|| Error::not_found(format!("no message {id}")))?;
    let reference = format!("message:{id}");
    let commit = required_commit(reader.message_commit(id).map_err(Error::from)?, &reference)?;
    let order = reader
        .message_order(id)
        .map_err(Error::from)?
        .ok_or_else(|| Error::not_yet_available(format!("{reference} has no reading position")))?;
    if let Some(n) = block {
        let Some(position) = block_selection
            .block_ordinals
            .iter()
            .position(|ordinal| *ordinal == n)
        else {
            return Err(Error::not_yet_available(format!(
                "block:{id}#{n} is not recorded in the selected source version"
            )));
        };
        message.blocks = vec![message.blocks.swap_remove(position)];
        block_selection.block_ordinals = vec![n];
    }
    message_record(reader, message, block_selection, commit, order)
}

/// A physical session's original metadata, using its own stored identity and
/// commit even when a composed reading reaches it through another session.
/// The stored link records of a session, in stored order.
pub(super) fn link_records(reader: &Reader<'_>, id: &SessionId) -> Result<Vec<Record>, Error> {
    let mut links = Vec::new();
    reader
        .visit_links(id, 0, |link| {
            links.push(link);
            ControlFlow::Continue(())
        })
        .map_err(Error::from)?;
    links
        .into_iter()
        .map(|link| {
            let reference = LinkRecord::reference(&link);
            let commit = required_commit(reader.link_commit(&link)?, &reference)?;
            Ok(Record::Link(LinkRecord {
                reference,
                commit,
                link,
            }))
        })
        .collect()
}

pub(super) fn session_record(reader: &Reader<'_>, id: &SessionId) -> Result<Record, Error> {
    let reference = format!("session:{id}");
    let session = reader
        .session(id)?
        .ok_or_else(|| Error::not_found(format!("no session {id}")))?;
    let commit = required_commit(reader.session_commit(id)?, &reference)?;
    Ok(Record::Session(SessionRecord {
        copied_prefixes: reader.copied_prefixes(id)?,
        reference,
        commit,
        session,
    }))
}

fn session(
    reader: &Reader<'_>,
    id: &SessionId,
    inherited: bool,
    start: Option<&ReadRecordLocator>,
    visit: &mut Visit<'_>,
) -> Result<bool, Error> {
    use ReadRecordKind::{Event, Link, LinkIssue, Message, Session};
    if start.is_some_and(|start| {
        !matches!(start.kind, Session | LinkIssue | Message | Event | Link)
            || (start.kind == Session && start.ordinal != 0)
    }) {
        return Err(stale_record());
    }
    let selected = reader.select_messages(id, inherited).map_err(Error::from)?;
    if start.is_none_or(|start| start.kind == Session)
        && !visit(
            session_record(reader, id)?.into(),
            LayoutFacts {
                lineage_complete: if inherited {
                    polylogue_model::read::ValueState::known(selected.parents().complete())
                } else {
                    polylogue_model::read::ValueState::NotApplicable
                },
                ..Default::default()
            },
            locator(Session, 0),
        )?
    {
        return Ok(false);
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
                LayoutFacts::default(),
                locator(
                    LinkIssue,
                    u64::try_from(ordinal)
                        .map_err(|_| Error::internal("a record count exceeds 2^64"))?,
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
                .map_err(Error::from)?
                .ok_or_else(stale_record)?
        } else {
            Default::default()
        };
        let mut result = Ok(true);
        ReadLimit::set_stage("read records: messages");
        selected
            .visit_with_sources(
                from,
                |message,
                 polylogue_store::read::MessageParse {
                     blocks: block_selection,
                     committed: commit,
                     ordinal: order,
                 }| {
                    ReadLimit::set_stage("read records: messages");
                    let reference = format!("message:{}", message.id);
                    let next = required_commit(commit, &reference).and_then(|commit| {
                        let facts = LayoutFacts {
                            inherited: message.session != *id,
                            ..Default::default()
                        };
                        let current = locator(Message, ordinal);
                        ordinal = ordinal.checked_add(1).ok_or_else(|| {
                            Error::internal("a reading counted more than 2^64 records")
                        })?;
                        visit(
                            message_record(reader, message, block_selection, commit, order)?.into(),
                            facts,
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
                },
            )
            .map_err(Error::from)?;
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
            .visit_event_records(id, ordinal, |event| {
                ReadLimit::set_stage("read records: events");
                seen = true;
                let current = locator(Event, ordinal);
                let next = ordinal
                    .checked_add(1)
                    .ok_or_else(|| Error::internal("a reading counted more than 2^64 records"))
                    .and_then(|next| {
                        ordinal = next;
                        visit(Record::Event(event).into(), LayoutFacts::default(), current)
                    });
                match next {
                    Ok(true) => ControlFlow::Continue(()),
                    other => {
                        result = other;
                        ControlFlow::Break(())
                    }
                }
            })
            .map_err(Error::from)?;
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
        .visit_links(id, ordinal, |link| {
            seen = true;
            ReadLimit::set_stage("read records: link commit");
            let reference = LinkRecord::reference(&link);
            let next = reader
                .link_commit(&link)
                .map_err(Error::from)
                .and_then(|commit| required_commit(commit, &reference))
                .and_then(|commit| {
                    let current = locator(Link, ordinal);
                    ordinal = ordinal.checked_add(1).ok_or_else(|| {
                        Error::internal("a reading counted more than 2^64 records")
                    })?;
                    visit(
                        Record::Link(LinkRecord {
                            reference,
                            commit,
                            link,
                        })
                        .into(),
                        LayoutFacts::default(),
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
        .map_err(Error::from)?;
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
    block_selection: ShownBlocks,
    commit: u64,
    order: u64,
) -> Result<Record, Error> {
    use polylogue_model::attachment_binding::AttachmentNamespace;
    let mut injections = reader
        .injection_evidence(&message.id, block_selection.source_version)
        .map_err(Error::from)?;
    injections.retain(|facts| {
        block_selection
            .block_ordinals
            .binary_search(&facts.block)
            .is_ok()
    });
    let kept = reader
        .stated_attachment_bindings(&message.id, block_selection.source_version)
        .map_err(Error::from)?;
    let mut bindings = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for record in kept {
        let mut binding = record.claim().clone();
        if block_selection
            .block_ordinals
            .binary_search(&binding.block)
            .is_err()
        {
            continue;
        }
        let context = record.context();
        // The claim and the earlier exports the source supplied stay as
        // declared; `exported_from` names the context of this export.
        binding.exported_from = AttachmentNamespace {
            input: context.input.clone(),
            chain: context.chain,
            import_id: context.import_id.clone(),
        };
        binding
            .validate()
            .map_err(|e| Error::from(polylogue_store::StoreError::stored_data(e.to_string())))?;
        if seen.insert(binding.clone()) {
            bindings.push(binding);
        }
    }
    let covered: std::collections::HashSet<_> = bindings
        .iter()
        .map(|binding| (binding.block, binding.attachment))
        .collect();
    let uncovered: Vec<_> = polylogue_model::attachment_binding::attachment_slots(
        &message.blocks,
        &block_selection.block_ordinals,
    )
    .into_iter()
    .filter(|slot| !covered.contains(&(slot.block, slot.attachment)))
    .collect();
    for binding in reader.current_attachment_bindings(
        &message.id,
        block_selection.source_version,
        &uncovered,
    )? {
        binding
            .validate()
            .map_err(|e| Error::from(polylogue_store::StoreError::stored_data(e.to_string())))?;
        if seen.insert(binding.clone()) {
            bindings.push(binding);
        }
    }
    Ok(Record::Message(Box::new(MessageRecord {
        reference: format!("message:{}", message.id),
        commit,
        order,
        source_version: block_selection.source_version,
        block_count: block_selection.block_count,
        block_ordinals: block_selection.block_ordinals,
        injections,
        bindings,
        copied_for: reader.copied_for(&message.id).map_err(Error::from)?,
        message,
    })))
}

/// Move to the adjacent stored record in the target's reading order.
pub(super) fn step(
    reader: &Reader<'_>,
    target: &ReadTarget,
    inherited: bool,
    from: Option<&ReadRecordLocator>,
    direction: polylogue_model::read::ReadDirection,
) -> Result<Option<ReadRecordLocator>, Error> {
    use polylogue_model::read::ReadDirection;
    match direction {
        ReadDirection::Before => adjacent(reader, target, inherited, from),
        ReadDirection::After => {
            let mut next = None;
            let mut skipped = from.is_none();
            reference(reader, target, inherited, from, &mut |_, _, position| {
                if !skipped {
                    skipped = true;
                    return Ok(true);
                }
                next = Some(position);
                Ok(false)
            })?;
            Ok(next)
        }
        ReadDirection::Past => Err(Error::invalid_query(
            "a row reading continues forward or backward only",
        )),
    }
}
fn stale() -> Error {
    Error::stale_cursor("a record in this reading changed since this page was issued")
}
fn adjacent(
    reader: &Reader<'_>,
    target: &ReadTarget,
    inherited: bool,
    current: Option<&ReadRecordLocator>,
) -> Result<Option<ReadRecordLocator>, Error> {
    use ReadRecordKind::{Authored, Document, Event, Link, LinkIssue, Message, Session};
    reader.check()?;
    let mut families = [
        (Session, 0),
        (LinkIssue, 0),
        (Message, 0),
        (Event, 0),
        (Link, 0),
    ];
    let length = match target {
        ReadTarget::Session(id) => {
            let Some(physical) = reader.physical_record_counts(id)? else {
                return if current.is_some() {
                    Err(stale())
                } else {
                    Ok(None)
                };
            };
            let selected = reader.select_messages(id, inherited)?;
            let issues = if inherited {
                u64::try_from(selected.parents().issues.len())
                    .map_err(|_| Error::internal("a parent issue count exceeds 2^64"))?
            } else {
                0
            };
            let messages = selected.len()?;
            let messages = u64::try_from(messages)
                .map_err(|_| Error::internal("a message count exceeds 2^64"))?;
            families = [
                (Session, 1),
                (LinkIssue, issues),
                (Message, messages),
                (Event, physical.events),
                (Link, physical.links),
            ];
            5
        }
        ReadTarget::Message(_) | ReadTarget::Block { .. } => {
            families[0] = (Message, 1);
            1
        }
        ReadTarget::MessageRange { first, last } => {
            let mut count = 0_u64;
            reader.visit_message_range(first, last, 0, |_| {
                count += 1;
                Ok(std::ops::ControlFlow::Continue(()))
            })?;
            families[0] = (Message, count);
            1
        }
        ReadTarget::Document(_) => {
            families[0] = (Document, 1);
            1
        }
        ReadTarget::Event(_) => {
            families[0] = (Event, 1);
            1
        }
        ReadTarget::Annotation { .. } | ReadTarget::Delivery(_) => {
            families[0] = (Authored, 1);
            1
        }
    };
    let families = &families[..length];
    let before = if let Some(current) = current {
        let family = families
            .iter()
            .position(|(kind, _)| *kind == current.kind)
            .ok_or_else(stale)?;
        if current.ordinal >= families[family].1 {
            return Err(stale());
        }
        if current.ordinal != 0 {
            return Ok(Some(ReadRecordLocator {
                kind: current.kind,
                ordinal: current.ordinal - 1,
            }));
        }
        family
    } else {
        families.len()
    };
    Ok(families[..before]
        .iter()
        .rev()
        .find(|(_, count)| *count != 0)
        .map(|(kind, count)| ReadRecordLocator {
            kind: *kind,
            ordinal: count - 1,
        }))
}
