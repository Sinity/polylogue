//! Resolve transcript bounds once, then apply their accepted positions.
use super::{Scope, bind::session_read, compose::Composed};
use crate::{Error, ReadRequest, ReadStart, operation::Context};
use polylogue_model::read::{ClassKind, EntryKey, SessionBoundary, SessionRead};
use polylogue_model::{MessageId, Reason};
use polylogue_store::Reader;
use query_language::time;

pub(super) fn trim(
    reader: &Reader<'_>,
    request: &mut ReadRequest,
    operation: &Context,
    scope: &Scope,
    composed: &mut Composed,
) -> Result<bool, Error> {
    let from = request.options.from_position();
    let until = request.options.until.as_deref();
    if from.is_none() && until.is_none() {
        return Ok(false);
    }
    let bound = session_read(request);
    let resolve = |text, accepted, ending| match (text, accepted) {
        (None, _) => Ok(None),
        (Some(_), Some(accepted)) => Ok(Some(accepted)),
        (Some(text), None) => resolve(reader, composed, operation, scope, text, ending),
    };
    let from = resolve(
        from,
        bound.and_then(|bound| bound.from_position.clone()),
        false,
    )?;
    let until = resolve(until, bound.and_then(|bound| bound.until.clone()), true)?;
    let start = from
        .as_ref()
        .map(|cut| locate(reader, composed, cut))
        .transpose()?;
    let end = until
        .as_ref()
        .map(|cut| locate(reader, composed, cut))
        .transpose()?;
    let inherited_from = start.is_some_and(|(i, position)| {
        composed.parts[i].inherited && position < composed.parts[i].end
    });
    if let Some((i, position)) = end {
        composed.parts.truncate(i + 1);
        composed.indexes.truncate(i + 1);
        composed.parts[i].end = position;
    }
    if let Some((i, position)) = start {
        for part in composed.parts.iter_mut().take(i) {
            part.start = part.end;
        }
        if let Some(part) = composed.parts.get_mut(i) {
            part.start = position.clamp(part.start, part.end);
        }
    }
    if matches!(request.start, ReadStart::Beginning) {
        request.start = ReadStart::Bound(Box::new(SessionRead::beginning(
            scope.session.clone(),
            scope.logical,
        )));
    }
    if let ReadStart::Bound(bound) = &mut request.start {
        bound.from_position = from;
        bound.until = until;
    }
    Ok(inherited_from)
}

fn locate(
    reader: &Reader<'_>,
    composed: &Composed,
    cut: &SessionBoundary,
) -> Result<(usize, usize), Error> {
    for (i, part) in composed.parts.iter().enumerate() {
        if part.session == cut.owner
            && let Some(seq) = reader.item_place(composed.indexes[i].pk, &cut.entry)?
            && (part.start as u64..part.end as u64).contains(&seq)
        {
            return Ok((i, seq as usize + usize::from(cut.after)));
        }
    }
    Err(Error::stale_cursor(
        "a boundary of the requested range changed",
    ))
}

fn resolve(
    reader: &Reader<'_>,
    composed: &Composed,
    operation: &Context,
    scope: &Scope,
    text: &str,
    ending: bool,
) -> Result<Option<SessionBoundary>, Error> {
    if text == "compaction" || text.starts_with("compaction:") {
        if !ending {
            return Err(Error::invalid_query(
                "a compaction can only end a read range",
            ));
        }
        let count = text.split_once(':').map_or("1", |(_, count)| count);
        let mut remaining = count
            .parse::<usize>()
            .ok()
            .filter(|&value| value > 0)
            .ok_or_else(|| Reason::CompactionCount {
                text: count.to_owned(),
            })?;
        for (i, part) in composed.parts.iter().enumerate().rev() {
            let entries = reader.item_window(
                composed.indexes[i].pk,
                part.start as u64..part.end as u64,
                false,
                remaining,
                |class| class.kind() == ClassKind::Compaction,
            )?;
            if entries.len() == remaining {
                return Ok(entries.into_iter().last().map(|row| SessionBoundary {
                    owner: part.session.clone(),
                    entry: row.entry.key,
                    after: false,
                }));
            }
            remaining -= entries.len();
        }
        return Err(Error::not_found("the reading has no such compaction"));
    }
    let message_text = text.strip_prefix("message:").unwrap_or(text);
    if let Ok(message) = MessageId::parse(message_text) {
        let entry = EntryKey::header(&message);
        for (i, part) in composed.parts.iter().enumerate() {
            if let Some(seq) = reader.item_place(composed.indexes[i].pk, &entry)?
                && (part.start as u64..part.end as u64).contains(&seq)
            {
                return Ok(Some(SessionBoundary {
                    owner: part.session.clone(),
                    entry,
                    after: false,
                }));
            }
        }
        if message_text != text || !time::is_time_expression(text) {
            return Err(Error::not_found(format!(
                "message {message} is not in the reading"
            )));
        }
    }
    let at = if let Some(elapsed) = time::duration(text) {
        let started = reader
            .session(&scope.session)?
            .and_then(|session| session.started)
            .ok_or_else(|| Error::invalid_query("the session has no recorded start time"))?;
        started.unix_ns() + elapsed.as_nanos()
    } else {
        time::resolve(text, &operation.clock)
            .map_err(|error| Reason::BadTime {
                text: text.to_owned(),
                problem: error.to_string(),
            })?
            .value
            .start()
            .as_nanosecond()
    };
    before_time(reader, composed, at)
}

fn before_time(
    reader: &Reader<'_>,
    composed: &Composed,
    at: i128,
) -> Result<Option<SessionBoundary>, Error> {
    for (i, part) in composed.parts.iter().enumerate() {
        let mut start = part.start as u64;
        let end = part.end as u64;
        while start < end {
            let rows =
                reader.item_window(composed.indexes[i].pk, start..end, true, 128, |class| {
                    matches!(
                        class.kind(),
                        ClassKind::Header
                            | ClassKind::Compaction
                            | ClassKind::Microcompaction
                            | ClassKind::Event
                    )
                })?;
            let Some(next) = rows.last().map(|row| row.seq + 1) else {
                break;
            };
            for row in rows {
                if row
                    .entry
                    .at
                    .is_some_and(|milliseconds| i128::from(milliseconds) * 1_000_000 >= at)
                {
                    return Ok(Some(SessionBoundary {
                        owner: part.session.clone(),
                        entry: row.entry.key,
                        after: false,
                    }));
                }
            }
            start = next;
        }
    }
    let Some((i, part)) = composed
        .parts
        .iter()
        .enumerate()
        .rev()
        .find(|(_, part)| part.start < part.end)
    else {
        return Ok(None);
    };
    Ok(composed
        .entry(reader, i, part.end - 1)?
        .map(|last| SessionBoundary {
            owner: part.session.clone(),
            entry: last.key,
            after: true,
        }))
}
