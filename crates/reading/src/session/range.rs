//! Resolve an exclusive end once against a composed transcript.
use super::{Scope, bind::session_read, compose::Composed};
use crate::{Error, ReadRequest, ReadStart, operation::Context};
use polylogue_model::read::{ClassKind, EntryKey, SessionEnd, SessionRead};
use polylogue_model::{MessageId, Reason};
use polylogue_store::Reader;
use query_language::time;

pub(super) fn trim_end(
    reader: &Reader<'_>,
    request: &mut ReadRequest,
    operation: &Context,
    scope: &Scope,
    composed: &mut Composed,
) -> Result<(), Error> {
    let Some(text) = request.options.until.as_deref() else {
        return Ok(());
    };
    let end = match session_read(request).and_then(|bound| bound.until.clone()) {
        Some(end) => end,
        None => match resolve(reader, composed, operation, scope, text)? {
            Some(end) => end,
            None => return Ok(()),
        },
    };
    for (i, part) in composed.parts.iter().enumerate() {
        if part.session != end.owner {
            continue;
        }
        let Some(seq) = reader.item_place(composed.indexes[i].pk, &end.entry)? else {
            continue;
        };
        let seq = seq as usize;
        if !(part.start..part.end).contains(&seq) {
            continue;
        }
        composed.parts.truncate(i + 1);
        composed.indexes.truncate(i + 1);
        composed.parts[i].end = seq + usize::from(end.after);
        if matches!(request.start, ReadStart::Beginning) {
            request.start = ReadStart::Bound(Box::new(SessionRead {
                session: scope.session.clone(),
                from_seen: false,
                after: None,
                logical: scope.logical,
                parts: Vec::new(),
                window_end: None,
                until: None,
            }));
        }
        if let ReadStart::Bound(bound) = &mut request.start {
            bound.until = Some(end);
        }
        return Ok(());
    }
    Err(Error::stale_cursor(
        "the end of the requested range changed",
    ))
}

fn resolve(
    reader: &Reader<'_>,
    composed: &Composed,
    operation: &Context,
    scope: &Scope,
    text: &str,
) -> Result<Option<SessionEnd>, Error> {
    if text == "compaction" || text.starts_with("compaction:") {
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
                return Ok(entries.into_iter().last().map(|row| SessionEnd {
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
                return Ok(Some(SessionEnd {
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
) -> Result<Option<SessionEnd>, Error> {
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
                    return Ok(Some(SessionEnd {
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
        .map(|last| SessionEnd {
            owner: part.session.clone(),
            entry: last.key,
            after: true,
        }))
}
