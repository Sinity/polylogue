//! Physical message ranges reuse the pinned Reader's metadata order and lazy
//! message hydration. Portable span records are prepared by the same source.
use super::{Response, Visit, admission, error, store_error};
use polylogue_model::{
    MessageId,
    bundle::*,
    read::{ProjectionEvidence, ReadRecordKind, ReadRecordLocator},
};
use polylogue_store::Reader;
use std::ops::ControlFlow;

pub(super) fn range(
    reader: &Reader<'_>,
    first: &MessageId,
    last: &MessageId,
    start: Option<&ReadRecordLocator>,
    visit: &mut Visit<'_>,
) -> Result<bool, Response> {
    let session = reader
        .message_session(first)
        .map_err(|error| store_error(&error))?
        .ok_or_else(|| error("not_found", format!("no physical message {first}")))?;
    let last_session = reader
        .message_session(last)
        .map_err(|error| store_error(&error))?
        .ok_or_else(|| error("not_found", format!("no physical message {last}")))?;
    if session != last_session {
        return Err(error(
            "bad_request",
            "message range endpoints belong to different physical sessions",
        ));
    }
    let selected = reader
        .select_messages(&session, false)
        .map_err(|error| store_error(&error))?;
    let first_position = selected
        .locate(Some(first))
        .map_err(|error| store_error(&error))?
        .ok_or_else(|| {
            error(
                "not_yet_available",
                "range start has no canonical physical reading position",
            )
        })?;
    let last_position = selected
        .locate(Some(last))
        .map_err(|error| store_error(&error))?
        .ok_or_else(|| {
            error(
                "not_yet_available",
                "range end has no canonical physical reading position",
            )
        })?;
    if first_position.message > last_position.message {
        return Err(error("bad_request", "message range end precedes its start"));
    }
    let mut record_ordinal = 0;
    let from = if let Some(start) = start {
        if start.kind != ReadRecordKind::Message {
            return Err(error(
                "stale_cursor",
                "range cursor names a different record family",
            ));
        }
        let last_relative =
            u64::try_from(last_position.message - first_position.message).map_err(|_| {
                error(
                    "not_yet_available",
                    "range metadata position exceeds its address space",
                )
            })?;
        if start.ordinal > last_relative {
            return Err(error(
                "stale_cursor",
                "continued record is outside the selected range",
            ));
        }
        let first = u64::try_from(first_position.message).map_err(|_| {
            error(
                "not_yet_available",
                "range metadata position exceeds its address space",
            )
        })?;
        let ordinal = first
            .checked_add(start.ordinal)
            .ok_or_else(|| error("stale_cursor", "range metadata position overflows"))?;
        record_ordinal = start.ordinal;
        selected
            .locate_ordinal(ordinal)
            .map_err(|cause| store_error(&cause))?
            .ok_or_else(|| error("stale_cursor", "continued record is no longer selected"))?
    } else {
        first_position
    };
    let mut result = Ok(true);
    let mut reached_end = false;
    selected
        .visit_from_with_record_evidence(from, |message, projection, commit, order| {
            let at_end = message.id == *last;
            let reference = format!("message:{}", message.id);
            let next = admission(commit, &reference).and_then(|commit| {
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
                    ProjectionEvidence::default(),
                    ReadRecordLocator {
                        kind: ReadRecordKind::Message,
                        ordinal: record_ordinal,
                    },
                )
            });
            match next {
                Ok(true) if at_end => {
                    reached_end = true;
                    ControlFlow::Break(())
                }
                Ok(true) => {
                    if let Some(next) = record_ordinal.checked_add(1) {
                        record_ordinal = next;
                        ControlFlow::Continue(())
                    } else {
                        result = Err(error(
                            "not_yet_available",
                            "range record position overflows",
                        ));
                        ControlFlow::Break(())
                    }
                }
                other => {
                    result = other;
                    ControlFlow::Break(())
                }
            }
        })
        .map_err(|error| store_error(&error))?;
    if !result? {
        return Ok(false);
    }
    if !reached_end {
        return Err(error(
            "not_yet_available",
            "selected range end was not available in this snapshot",
        ));
    }
    Ok(true)
}

#[cfg(test)]
#[path = "grains/tests.rs"]
mod tests;
