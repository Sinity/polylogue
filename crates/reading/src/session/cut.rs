//! Cutting a composed reading into a page: the entries a cursor starts at,
//! as many whole entries as the budget and transfer size hold, fragments of
//! an entry too large for a page alone, and the continuations in the
//! footer.
use super::bind::{seen_marks, session_read};
use super::entries::PageEntry;
use super::{Reading, still_wanted};
use crate::{Error, ReadRequest, ReadServices};
use polylogue_model::read::*;
use polylogue_store::Reader;
use std::sync::Arc;

/// The page of `reading` its cursor names, or its first page.
pub(super) fn page(
    reader: &Reader<'_>,
    services: &impl ReadServices,
    original: &ReadRequest,
    operation: &Arc<crate::operation::Context>,
    reading: &Reading,
    opened: Option<PageCursor>,
    key: CursorSeal,
) -> Result<(ReadPage, Vec<polylogue_model::user::SeenMark>), Error> {
    let (format, limits, digest) = crate::page_terms(original)?;
    let target = &original.target;
    let request = &reading.request;
    let continuation = &reading.continuation;
    let composed = &reading.composed;
    let versions = PageCursor::version(&composed.versions(reader)?);
    if let Some(opened) = &opened
        && opened.rewritten(&versions)
    {
        return Err(Error::stale_cursor(
            "this session or one of its linked sessions changed since this page was issued",
        ));
    }
    still_wanted()?;
    let shown = reading.shown(reader)?;
    still_wanted()?;
    // Where a key sits in the reading: the first part that holds it.
    let locate = |key: &EntryKey| -> Result<Option<(usize, usize)>, Error> {
        for (at, (part, index)) in composed.parts.iter().zip(&composed.indexes).enumerate() {
            if let Some(seq) = reader.item_place(index.pk, key)?
                && (part.start..part.end).contains(&(seq as usize))
            {
                return Ok(Some((at, seq as usize)));
            }
        }
        Ok(None)
    };
    let lead = &reading.lead;
    let tokens = limits.tokens.get();
    let items = limits.items.get();
    let (page_entries, next_entry): (Vec<PageEntry>, Option<PageEntry>) = match &opened {
        // The first page's lead lines spend its budget first; when they
        // use all of it, the page is those lines alone.
        None => {
            let lead_tokens: u64 = lead
                .iter()
                .map(|item| {
                    serde_json::to_string(item).map_or(0, |text| estimate_tokens(&text) as u64)
                })
                .sum();
            if lead_tokens >= tokens || lead.len() >= items {
                (Vec::new(), shown.first_at(shown.start())?)
            } else {
                let page =
                    shown.page_after(shown.start(), tokens - lead_tokens, items - lead.len())?;
                (page.entries, page.next)
            }
        }
        Some(opened) => {
            let PagePosition::Entry { key, inside } = &opened.position else {
                return Err(Error::invalid_query(
                    "the cursor names a row position, not a session entry",
                ));
            };
            let place = locate(key)?.ok_or_else(|| {
                Error::stale_cursor(
                    "the message or event this page continues from is no longer in the reading",
                )
            })?;
            let after = (place.0, place.1 + 1);
            match opened.direction {
                // The rest of an entry too large for one page is a page of
                // its own.
                ReadDirection::After if inside.is_some() => {
                    let entry = shown
                        .first_at(place)?
                        .filter(|entry| entry.place() == place)
                        .ok_or_else(|| {
                            Error::stale_cursor("the message or event this page continues from is no longer shown by these options")
                        })?;
                    (vec![entry], shown.first_at(after)?)
                }
                ReadDirection::After => {
                    let page = shown.page_after(place, tokens, items)?;
                    (page.entries, page.next)
                }
                ReadDirection::Before => (
                    shown.page_before(place, tokens, items)?,
                    shown.first_at(place)?,
                ),
                ReadDirection::Past => {
                    let page = shown.page_after(after, tokens, items)?;
                    (page.entries, page.next)
                }
            }
        }
    };
    still_wanted()?;
    let start = match page_entries.first().or(next_entry.as_ref()) {
        Some(first) => shown.position(first.place())?,
        None => shown.len(),
    };
    let header = reading.header.clone();
    let mut items_out: Vec<ReadItem> = Vec::new();
    // Lead lines open the reading: on its first page, and on a backward
    // page that reaches the start; never again after a lead-only page.
    let continues_forward = opened.as_ref().is_some_and(|opened| {
        matches!(opened.direction, ReadDirection::After | ReadDirection::Past)
    });
    // A forward page at the start that left out the lead lines has them
    // on the page before it.
    let lead_before = start == 0 && continues_forward;
    if start == 0 && !continues_forward {
        items_out.extend(lead.iter().cloned());
    }
    let mut cache = None;
    let laid = reading
        .lay_out(
            reader,
            services,
            &shown,
            &page_entries,
            operation.ran_at,
            &mut cache,
            false,
        )?
        .items;
    // Whole entries up to the transfer size; an entry too large to travel
    // alone is replaced by the ready call that reads its block in pages.
    let room = limits.bytes.get();
    let used = serde_json::to_vec(&header)
        .map(|bytes| bytes.len())
        .unwrap_or(0)
        + items_out
            .iter()
            .map(|item| serde_json::to_vec(item).map_or(0, |bytes| bytes.len()))
            .sum::<usize>();
    let sizes: Vec<usize> = laid
        .iter()
        .map(|items| {
            items
                .iter()
                .map(|item| serde_json::to_vec(item).map_or(0, |bytes| bytes.len()))
                .sum()
        })
        .collect();
    // Whole entries that fit the transfer, at least one. A backward page
    // keeps the entries next to its cursor and gives up its first ones.
    let backward = opened
        .as_ref()
        .is_some_and(|opened| opened.direction == ReadDirection::Before);
    let fit = |order: &mut dyn Iterator<Item = usize>| {
        let mut total = used;
        let mut count = 0;
        for index in order {
            if count > 0 && total + sizes[index] > room {
                break;
            }
            total += sizes[index];
            count += 1;
        }
        count
    };
    let count = if backward {
        fit(&mut (0..laid.len()).rev())
    } else {
        fit(&mut (0..laid.len()))
    };
    let keep = if backward {
        laid.len() - count..laid.len()
    } else {
        0..count
    };
    let start = start + keep.start as u64;
    // An entry continued inside, or too large for the page alone, is sent
    // as fragments of its items' encodings; the page ends where they stop.
    let inside = opened.as_ref().and_then(|opened| match &opened.position {
        PagePosition::Entry { inside, .. } => *inside,
        _ => None,
    });
    let mut unfinished = None;
    for (offset, entry_items) in laid.into_iter().enumerate() {
        if !keep.contains(&offset) {
            continue;
        }
        if inside.is_some() || (keep.len() == 1 && sizes[offset] > room) {
            let entry = &page_entries[offset];
            let kind = match entry.entry.key {
                EntryKey::Message { .. } => ReadRecordKind::Message,
                EntryKey::Event { .. } => ReadRecordKind::Event,
            };
            // `used` already holds the header and the lead lines, the only
            // items before an entry that is alone on its page.
            unfinished = fragments(
                entry_items,
                inside,
                target,
                kind,
                room.saturating_sub(used),
                tokens,
                &mut items_out,
            )?
            .map(|place| (entry.entry.key.clone(), place));
        } else {
            items_out.extend(entry_items);
        }
    }
    let kept = &page_entries[keep.clone()];
    // The entry after the page: the first one left out, or the one after
    // every entry read.
    let after_page = page_entries.get(keep.end).or(next_entry.as_ref());
    let call = |entry: &PageEntry, direction: ReadDirection| -> Result<ReadyCall, Error> {
        let next = PageCursor {
            session_read: session_read(request).cloned(),
            target: continuation.target.clone(),
            digest: digest.clone(),
            alternates: request.alternates.clone(),
            versions,
            position: PagePosition::Entry {
                key: entry.entry.key.clone(),
                inside: None,
            },
            direction,
        };
        crate::next_call(
            services,
            continuation,
            next.seal(&key),
            crate::deadline(operation),
        )
    };
    let positions = reader.change_position()?;
    let marks = seen_marks(
        reader,
        composed,
        request,
        kept,
        unfinished.as_ref().map(|(key, _)| key),
        &positions,
    )?;
    let finished = unfinished.is_none();
    let next = match unfinished {
        Some((entry_key, place)) => {
            let cursor = PageCursor {
                session_read: session_read(request).cloned(),
                target: continuation.target.clone(),
                digest: digest.clone(),
                alternates: request.alternates.clone(),
                versions,
                position: PagePosition::Entry {
                    key: entry_key,
                    inside: Some(place),
                },
                direction: ReadDirection::After,
            };
            Some(crate::next_call(
                services,
                continuation,
                cursor.seal(&key),
                crate::deadline(operation),
            )?)
        }
        None => after_page
            .map(|entry| call(entry, ReadDirection::After))
            .transpose()?,
    };
    let first = kept.first().or(after_page);
    let footer = ReadFooter {
        estimator: Some(TOKEN_ESTIMATOR.into()),
        follow: match (finished && after_page.is_none(), kept.last()) {
            (true, Some(last)) => Some(call(last, ReadDirection::Past)?),
            _ => None,
        },
        next,
        previous: match first {
            Some(first) if start > 0 || lead_before => Some(call(first, ReadDirection::Before)?),
            _ => None,
        },
        shown: items_out.len() as u64,
        position: Some(start),
        count: ValueState::known(shown.len()),
        ..ReadFooter::default()
    };
    let snapshot = ReadPageSnapshot {
        query: target.clone(),
        ran_at: operation.ran_at,
        read_position: positions,
        freshness: services.freshness(reader)?,
    };
    Ok((
        ReadPage {
            format,
            snapshot,
            document: ReadDocument {
                header,
                items: items_out,
                footer,
            },
        },
        marks,
    ))
}

/// The items of one entry as whole items while they fit `room` bytes, and
/// from the first that does not as fragments of its archive JSON encoding,
/// starting at `from` when the page continues inside the entry. Returns the
/// place the next page starts at when the entry is not finished.
/// Concatenating the fragments gives back each item's encoding.
fn fragments(
    items: Vec<ReadItem>,
    from: Option<ItemPlace>,
    target: &str,
    kind: ReadRecordKind,
    room: usize,
    tokens: u64,
    out: &mut Vec<ReadItem>,
) -> Result<Option<ItemPlace>, Error> {
    let first = from.map_or(0, |place| place.item as usize);
    if first >= items.len() && !items.is_empty() {
        return Err(Error::stale_cursor(
            "the part this page continues from is no longer in its message or event",
        ));
    }
    let mut used = 0;
    let mut placed = false;
    for (index, item) in items.into_iter().enumerate().skip(first) {
        let encoded = polylogue_render::json_bytes(&item).map_err(Error::internal)?;
        let text = String::from_utf8(encoded).map_err(Error::internal)?;
        let item_hash = polylogue_model::BlobHash::of(text.as_bytes());
        let place = |character: usize| ItemPlace {
            selection: SelectionLocator {
                row: 0,
                record: ReadRecordLocator { kind, ordinal: 0 },
            },
            item: index as u64,
            character: character as u64,
            item_hash,
        };
        let start = match from {
            Some(from) if index == first => {
                if from.item_hash != item_hash {
                    return Err(Error::stale_cursor(
                        "the part this page continues from changed since this page was issued",
                    ));
                }
                from.character as usize
            }
            _ => 0,
        };
        if start == 0 && used + text.len() <= room {
            used += text.len();
            placed = true;
            out.push(item);
            continue;
        }
        if placed {
            return Ok(Some(place(0)));
        }
        let boundaries: Vec<usize> = text
            .char_indices()
            .map(|(at, _)| at)
            .chain(std::iter::once(text.len()))
            .collect();
        let total = boundaries.len() - 1;
        if start >= total {
            return Err(Error::stale_cursor(
                "the part this page continues from is shorter than when this page was issued",
            ));
        }
        // A fragment is sourced at the reading, as the row pager's are;
        // its position names the item.
        let make = |until: usize| ReadItem {
            source: Source::record(target),
            kind: ItemKind::Fragment {
                text: text[boundaries[start]..boundaries[until]].into(),
                position: place(start),
                until: until as u64,
                total: total as u64,
            },
        };
        let fits = |until: usize| -> Result<bool, Error> {
            let candidate = polylogue_render::json_bytes(&make(until)).map_err(Error::internal)?;
            let payload = &text[boundaries[start]..boundaries[until]];
            Ok(used + candidate.len() <= room && estimate_tokens(payload) as u64 <= tokens)
        };
        // At least one character, so every page moves forward.
        let (mut low, mut high) = (start + 1, total);
        while low < high {
            let middle = low + (high - low).div_ceil(2);
            if fits(middle)? {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        let fragment = make(low);
        used += polylogue_render::json_bytes(&fragment)
            .map_err(Error::internal)?
            .len();
        placed = true;
        out.push(fragment);
        if low < total {
            return Ok(Some(place(low)));
        }
    }
    Ok(None)
}
