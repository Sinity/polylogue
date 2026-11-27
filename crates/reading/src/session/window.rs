//! Message-count windows over the composed reading. Use stored entry counts
//! to find the boundary without reading the bodies in the omitted middle.
use super::compose::{Composed, Runs};
use crate::{Error, ReadRequest, ReadStart};
use polylogue_model::{SessionId, read::*};
use polylogue_store::Reader;

pub(super) struct Omission {
    pub count: u64,
    /// The first header in the tail, if the window has a tail.
    pub next: Option<(usize, usize)>,
}

pub(super) fn apply(
    reader: &Reader<'_>,
    composed: &Composed,
    runs: &mut Runs,
    request: &mut ReadRequest,
    session: &SessionId,
) -> Result<Option<Omission>, Error> {
    let (first, last) = (request.options.first, request.options.last);
    if first.is_none() && last.is_none() {
        return Ok(None);
    }
    if matches!(request.start, ReadStart::Beginning) {
        request.start = ReadStart::Bound(Box::new(SessionRead {
            session: session.clone(),
            from_seen: false,
            after: None,
            logical: false,
            parts: Vec::new(),
            window_end: None,
        }));
    }
    if let ReadStart::Bound(bound) = &mut request.start {
        if let Some((owner, key)) = &bound.window_end {
            let mut found = None;
            for (i, &(part, start, end)) in runs.spans.iter().enumerate().rev() {
                if &composed.parts[part].session == owner
                    && let Some(seq) = reader.item_place(composed.indexes[part].pk, key)?
                    && (start..end).contains(&(seq as usize))
                {
                    found = Some((i, seq as usize + 1));
                    break;
                }
            }
            let (i, end) = found.ok_or_else(|| {
                Error::stale_cursor("the end of the requested message window changed")
            })?;
            runs.spans.truncate(i + 1);
            runs.spans[i].2 = end;
        } else if let Some(&(part, start, end)) =
            runs.spans.iter().rev().find(|span| span.1 < span.2)
        {
            let entry = reader
                .item_window(
                    composed.indexes[part].pk,
                    start as u64..end as u64,
                    false,
                    1,
                    |_| true,
                )?
                .pop();
            bound.window_end =
                entry.map(|row| (composed.parts[part].session.clone(), row.entry.key));
        }
    }

    let profile = ShowProfile::of(&request.options);
    let message = |class: EntryClass| class.is_header() && profile.shows(class);
    let counts: Vec<_> = runs
        .spans
        .iter()
        .map(|&(part, start, end)| composed.count(reader, part, (start, end), message))
        .collect::<Result<_, _>>()?;
    let total = counts.iter().sum::<u64>();
    let head = first.unwrap_or(0);
    let tail = last.unwrap_or(0);
    let tail_rank = total.saturating_sub(tail);
    if head >= tail_rank {
        return Ok(None);
    }

    // A rank is the number of visible message headers before the boundary.
    let at = |mut rank: u64| -> Result<(usize, usize), Error> {
        for (&(part, start, end), count) in runs.spans.iter().zip(&counts) {
            if rank >= *count {
                rank -= *count;
                continue;
            }
            let (mut low, mut high) = (start, end);
            while low < high {
                let middle = low + (high - low) / 2;
                if composed.count(reader, part, (start, middle + 1), message)? > rank {
                    high = middle;
                } else {
                    low = middle + 1;
                }
            }
            return Ok((part, low));
        }
        Err(Error::internal(
            "message window rank is outside the reading",
        ))
    };
    let stop = (head > 0).then(|| at(head)).transpose()?;
    let next = (tail > 0).then(|| at(tail_rank)).transpose()?;
    let mut kept = Vec::new();
    for &(part, start, end) in &runs.spans {
        if let Some((stop_part, stop_seq)) = stop
            && (part, start) < (stop_part, stop_seq)
        {
            kept.push((
                part,
                start,
                if part == stop_part {
                    end.min(stop_seq)
                } else {
                    end
                },
            ));
        }
        if let Some((next_part, next_seq)) = next
            && (part, end) > (next_part, next_seq)
        {
            kept.push((
                part,
                if part == next_part {
                    start.max(next_seq)
                } else {
                    start
                },
                end,
            ));
        }
    }
    runs.spans = kept;
    Ok(Some(Omission {
        count: tail_rank - head,
        next,
    }))
}
