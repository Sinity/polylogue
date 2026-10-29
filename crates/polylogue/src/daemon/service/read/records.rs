//! Hydrate canonical rows lazily after the shared selector has resolved refs.
use super::super::{Response, error, store_error};
use polylogue_model::{
    Reference, SessionId,
    bundle::*,
    read::{ProjectionEvidence, ReadPosition},
};
use polylogue_store::Reader;
use std::ops::ControlFlow;

type Visit<'a> = dyn FnMut(Record, ProjectionEvidence) -> Result<bool, Response> + 'a;

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
    start: Option<&ReadPosition>,
    visit: &mut Visit<'_>,
) -> Result<bool, Response> {
    match reference {
        Reference::Session(id) => session(reader, id, inherited, start, visit),
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
                Record::Message(Box::new(MessageRecord {
                    reference: reference_text,
                    commit,
                    order,
                    source_version: projection.source_version,
                    block_count: projection.block_count,
                    block_ordinals: projection.block_ordinals,
                    message,
                })),
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
    start: Option<&ReadPosition>,
    visit: &mut Visit<'_>,
) -> Result<bool, Response> {
    let reference = format!("session:{id}");
    if start.is_none_or(|start| start.reference == reference) {
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
        )? {
            return Ok(false);
        }
    }
    let selected = reader
        .select_messages(id, inherited)
        .map_err(|e| store_error(&e))?;
    let message_start = start.and_then(|start| start.reference.strip_prefix("message:"));
    let from = if let Some(message) = message_start {
        let message =
            polylogue_model::MessageId::parse(message).map_err(|e| error("bad_request", e))?;
        selected
            .locate(Some(&message))
            .map_err(|e| store_error(&e))?
            .ok_or_else(|| error("stale_cursor", "continued message is no longer selected"))?
    } else {
        Default::default()
    };
    let skip_messages = start.is_some_and(|start| {
        start.reference.starts_with("event:") || start.reference.starts_with("link:")
    });
    let mut result = Ok(true);
    if !skip_messages {
        selected
            .visit_from_with_record_evidence(from, |message, projection, commit, order| {
                let reference = format!("message:{}", message.id);
                let next = admission(commit, &reference).and_then(|commit| {
                    let evidence = ProjectionEvidence {
                        inherited: message.session != *id,
                        ..Default::default()
                    };
                    visit(
                        Record::Message(Box::new(MessageRecord {
                            reference,
                            commit,
                            order,
                            source_version: projection.source_version,
                            block_count: projection.block_count,
                            block_ordinals: projection.block_ordinals,
                            message,
                        })),
                        evidence,
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
    }
    if !result? {
        return Ok(false);
    }
    let mut result = Ok(true);
    reader
        .visit_event_records(id, |event| {
            match visit(Record::Event(event), ProjectionEvidence::default()) {
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
    let mut result = Ok(true);
    reader
        .visit_links(id, |link| {
            let reference = LinkRecord::reference(&link);
            match reader
                .link_commit(&link)
                .map_err(|e| store_error(&e))
                .and_then(|commit| admission(commit, &reference))
                .and_then(|commit| {
                    visit(
                        Record::Link(LinkRecord {
                            reference,
                            commit,
                            link,
                        }),
                        ProjectionEvidence::default(),
                    )
                }) {
                Ok(true) => ControlFlow::Continue(()),
                other => {
                    result = other;
                    ControlFlow::Break(())
                }
            }
        })
        .map_err(|e| store_error(&e))?;
    result
}
