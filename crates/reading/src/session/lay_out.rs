//! Laying out a composed reading: the lines that open it, each entry's
//! items through the record projection, and what the reading adds around
//! them (`continues` lines, cards, turn dividers, the context meter, stored
//! bytes for the raw view).
use super::compose::{Composed, Runs};
use super::entries::{PageEntry, ShownEntries};
use super::still_wanted;
use crate::records::SelectedRecord;
use crate::{Error, ReadRequest, ReadServices};
use crate::{layout_facts, output};
use polylogue_model::{MessageId, SessionId, bundle::Record, read::*};
use polylogue_store::Reader;
use polylogue_store::items::ItemLookup;
use std::collections::HashSet;

/// The first page begins with what the reading leaves out: an inherited
/// prefix shown as one line, link problems, and for the context view what
/// was not recorded and what the newest boundary removed.
pub(super) fn lead_items(
    composed: &Composed,
    runs: &Runs,
    options: &ReadOptions,
    target: &str,
    session: &polylogue_model::Session,
    inherited: u64,
    annotations: Vec<AnnotationMark>,
) -> Vec<ReadItem> {
    // The read session's card opens the reading, with whether its lineage
    // composed without a problem.
    let mut out = vec![ReadItem {
        source: Source::record(format!("session:{}", session.id)),
        kind: ItemKind::Card {
            card: Box::new(Card::Session {
                annotations,
                session: Box::new(session.clone()),
                // Whether the parents composed, when the reading shows them,
                // as the stream's session record says.
                lineage_complete: if options.inherited() {
                    ValueState::known(composed.issues.is_empty())
                } else {
                    ValueState::NotApplicable
                },
                open: None,
            }),
        },
    }];
    if let Some(text) = layout_facts::annotation_notice(options) {
        out.push(ReadItem {
            source: Source::record(target),
            kind: ItemKind::Text { text },
        });
    }
    let source = Source::record(target);
    for issue in composed.issues.iter().filter(|_| options.inherited()) {
        out.push(ReadItem {
            source: source.clone(),
            kind: ItemKind::LinkIssue {
                issue: issue.clone(),
            },
        });
    }
    if composed.unstored_copy {
        out.push(ReadItem {
            source: source.clone(),
            kind: ItemKind::Indicator {
                indicator: IndicatorKind::Continues,
                text: "(repeated from a session not in the archive, counted as own)".into(),
                expand: None,
            },
        });
    }
    if !options.inherited()
        && options.view != View::Context
        && let Some(parent) = composed.parts.iter().rev().find(|part| part.inherited)
    {
        out.push(ReadItem {
            source: source.clone(),
            kind: ItemKind::Indicator {
                indicator: IndicatorKind::Continues,
                text: format!(
                    "continues {} ({inherited} earlier messages; --show inherited to show)",
                    parent.session
                ),
                expand: None,
            },
        });
    }
    if let Some((complete, reason)) = &runs.context {
        out.push(ReadItem {
            source: source.clone(),
            kind: ItemKind::Omission {
                count: ValueState::NotRecorded,
                unit: "system prompt, tool definitions".into(),
                why: "not recorded in context".into(),
                continue_call: None,
            },
        });
        for key in &runs.evicted {
            out.push(ReadItem {
                source: Source::record(key.reference()),
                kind: ItemKind::Omission {
                    count: ValueState::known(1),
                    unit: "message".into(),
                    why: if runs.left {
                        "left the context between the two places".into()
                    } else {
                        "removed from context".into()
                    },
                    continue_call: None,
                },
            });
        }
        if !complete {
            out.push(ReadItem {
                source,
                kind: ItemKind::Omission {
                    count: ValueState::Unknown {
                        reason: reason.clone(),
                    },
                    unit: "context".into(),
                    why: reason
                        .clone()
                        .unwrap_or_else(|| "context is incomplete".into()),
                    continue_call: None,
                },
            });
        }
    }
    out
}

/// The card for a fork or a spawned session: a delegation card for a
/// spawn, a session card for a fork, each with the call that reads it in
/// the reading's view.
fn card(
    services: &impl ReadServices,
    reader: &Reader<'_>,
    link: &polylogue_store::lineage::SessionLink,
    child: &SessionId,
    call: Option<&IndexEntry>,
    view: View,
) -> Result<ReadItem, Error> {
    let reference = format!("session:{child}");
    let open = services.continuation(
        &ReadRequest::new(
            reference.clone(),
            ReadOptions {
                view,
                ..ReadOptions::default()
            },
        ),
        None,
        command_surface::Deadline::Default,
    )?;
    let kind = if link.kind == "spawn" {
        let objective = call
            .and_then(|entry| entry.target.clone())
            .map_or(ValueState::NotRecorded, ValueState::known);
        Card::Delegation {
            card: Box::new(DelegationCard {
                child: reference.clone(),
                objective,
                result: ValueState::Unknown {
                    reason: Some(format!("read {reference}")),
                },
                duration_ms: ValueState::Unknown { reason: None },
                cost: ValueState::Unknown { reason: None },
                retries: Vec::new(),
                result_use: ValueState::Unknown { reason: None },
                open: Some(open),
            }),
        }
    } else {
        match reader.session(child)? {
            Some(session) => Card::Session {
                annotations: Vec::new(),
                session: Box::new(session),
                lineage_complete: ValueState::NotApplicable,
                open: Some(Box::new(open)),
            },
            None => Card::Fields {
                title: format!("{} {reference} (not in the archive)", link.kind),
                figures: Vec::new(),
            },
        }
    };
    Ok(ReadItem {
        source: Source::record(reference),
        kind: ItemKind::Card {
            card: Box::new(kind),
        },
    })
}

/// The entry an item belongs to, from its source.
fn item_key(item: &ReadItem) -> Option<EntryKey> {
    match (&item.source.message, item.source.block) {
        (Some(message), block) => Some(EntryKey::Message {
            message: message.clone(),
            block,
        }),
        (None, _) => {
            let reference = &item.source.reference;
            if let Some(event) = reference.strip_prefix("event:") {
                Some(EntryKey::Event {
                    event: event.to_owned(),
                })
            } else {
                // A line beside a message (a collapsed harness message's
                // indicator) belongs to that message's header.
                reference
                    .strip_prefix("message:")
                    .and_then(|id| MessageId::parse(id).ok())
                    .map(|message| EntryKey::header(&message))
            }
        }
    }
}

/// Keep only the current source body while traversing its indexed entries.
/// Page and stream layout share this cache; neither retains the reading's bodies.
pub(super) struct PreparedRecord {
    key: EntryKey,
    inherited: bool,
    record: Option<ReadRecord>,
    items: std::collections::HashMap<EntryKey, Vec<ReadItem>>,
}

pub(super) struct LaidOut {
    pub items: Vec<Vec<ReadItem>>,
    pub records: Vec<(usize, ReadRecord)>,
}

/// Lay out indexed entries in order through the same record projection. Streams
/// retain original frames beside their first entry; pages transfer the items.
#[allow(clippy::too_many_arguments)]
pub(super) fn lay_out(
    reader: &Reader<'_>,
    services: &impl ReadServices,
    composed: &Composed,
    reading: &Runs,
    page: &[PageEntry],
    options: &ReadOptions,
    at: polylogue_model::Timestamp,
    cache: &mut Option<PreparedRecord>,
    bounded: bool,
    originals: bool,
) -> Result<LaidOut, Error> {
    let mut laid = LaidOut {
        items: vec![Vec::new(); page.len()],
        records: Vec::new(),
    };
    // A raw reading is the stored records, which `decorate` adds.
    if options.view == View::Raw {
        return Ok(laid);
    }
    let selected = |source: &Source| -> Result<bool, Error> {
        let Some(message) = &source.message else {
            return Ok(false);
        };
        let key = EntryKey::Message {
            message: message.clone(),
            block: source.block,
        };
        for (part, start, end) in &reading.spans {
            if reader
                .item_place(composed.indexes[*part].pk, &key)?
                .is_some_and(|entry| (*start..*end).contains(&(entry as usize)))
            {
                return Ok(true);
            }
        }
        Ok(false)
    };
    let ready_attachment = |message: &MessageId, block, index, input: Option<&str>| {
        crate::attachment_call(services, message, block, index, input)
    };
    let mut footer = ReadFooter::default();
    let mut captured = HashSet::new();
    for (slot, entry) in page.iter().enumerate() {
        still_wanted()?;
        let inherited = composed.parts[entry.part].inherited;
        let key = match &entry.entry.key {
            EntryKey::Message { message, .. } => EntryKey::header(message),
            key => key.clone(),
        };
        if cache
            .as_ref()
            .is_none_or(|cached| cached.key != key || cached.inherited != inherited)
        {
            let record = match &key {
                EntryKey::Message { message, .. } => {
                    crate::records::message(reader, message, None)?
                }
                EntryKey::Event { event } => {
                    Record::Event(reader.event_record(event)?.ok_or_else(|| {
                        Error::stale_cursor(format!("event {event} is no longer in this session"))
                    })?)
                }
            };
            let mut items: std::collections::HashMap<EntryKey, Vec<ReadItem>> = Default::default();
            let record = project_layout(
                reader,
                record,
                LayoutFacts {
                    inherited: inherited && matches!(&key, EntryKey::Message { .. }),
                    ..Default::default()
                },
                options,
                at,
                &selected,
                bounded,
                &ready_attachment,
                &mut footer,
                &mut |item| {
                    if let Some(key) = item_key(&item) {
                        items.entry(key).or_default().push(item);
                    }
                },
            )?;
            *cache = Some(PreparedRecord {
                key: key.clone(),
                inherited,
                record,
                items,
            });
        }
        let cached = cache.as_ref().expect("the entry's source was prepared");
        if originals
            && captured.insert(key)
            && let Some(record) = &cached.record
        {
            laid.records.push((slot, record.clone()));
        }
        if let Some(items) = cached.items.get(&entry.entry.key) {
            laid.items[slot].extend(items.iter().cloned());
        }
    }
    Ok(laid)
}

/// What a reading adds around laid-out entries: `continues` lines and
/// cards from composition, turn dividers, the context meter on model
/// headers, and for the raw view each record's stored bytes.
pub(super) fn decorate(
    services: &impl ReadServices,
    reader: &Reader<'_>,
    composed: &Composed,
    shown: &ShownEntries,
    page: &[PageEntry],
    laid: &mut [Vec<ReadItem>],
    options: &ReadOptions,
) -> Result<(), Error> {
    let human_prompt = |entry: &IndexEntry| {
        entry.kind == EntryKind::Header
            && entry.written_by == polylogue_model::WrittenBy::Human
            && entry.prompt_kind == Some(polylogue_model::PromptKind::Prose)
    };
    let mut placed = place_cards(services, reader, composed, shown, page, options)?;
    for (offset, items) in laid.iter_mut().enumerate() {
        let PageEntry {
            part,
            seq: at,
            entry,
        } = &page[offset];
        let (part, at) = (*part, *at);
        let mut before: Vec<ReadItem> =
            composed.before.get(&entry.key).cloned().unwrap_or_default();
        if options.turns() && human_prompt(entry) {
            let (number, calls) = turn(reader, composed, part, at)?;
            before.push(ReadItem {
                source: Source::record(entry.key.reference()),
                kind: ItemKind::Indicator {
                    indicator: IndicatorKind::Turn,
                    text: format!("turn {number} · {calls} calls"),
                    expand: None,
                },
            });
        }
        if options.context_meter() && entry.written_by == polylogue_model::WrittenBy::Model {
            let meter = context_meter(reader, composed, part, at, entry)?;
            for item in items.iter_mut() {
                if let ItemKind::MessageHeader { header } = &mut item.kind {
                    header.context = Some(match meter {
                        ContextMeter::Reported(tokens) => ValueState::known(tokens),
                        ContextMeter::Estimated(tokens) => ValueState::Inferred { value: tokens },
                        ContextMeter::Unknown => ValueState::Unknown {
                            reason: Some("no usage recorded".into()),
                        },
                    });
                }
            }
        }
        if options.view == View::Raw {
            let record = match &entry.key {
                EntryKey::Message {
                    message,
                    block: None,
                } => Some((
                    polylogue_store::items::RecordTable::Message,
                    message.to_string(),
                )),
                EntryKey::Event { event } => {
                    Some((polylogue_store::items::RecordTable::Event, event.clone()))
                }
                _ => None,
            };
            if let Some((table, id)) = record {
                let stored = reader.stored_record(table, &id)?.ok_or_else(|| {
                    Error::not_yet_available(format!(
                        "{} has no stored record",
                        entry.key.reference()
                    ))
                })?;
                let media_type = if stored.bytes.first() == Some(&b'{') {
                    "application/json"
                } else {
                    "application/octet-stream"
                };
                items.push(ReadItem {
                    source: Source::record(entry.key.reference()),
                    kind: ItemKind::Raw {
                        record: Box::new(RawRecord {
                            reference: entry.key.reference(),
                            input: stored.input,
                            media_type: media_type.into(),
                            bytes: stored.bytes,
                        }),
                    },
                });
            }
        }
        let (ahead, after) = placed.remove(&offset).unwrap_or_default();
        let mut joined = before;
        joined.extend(ahead);
        joined.append(items);
        joined.extend(after);
        *items = joined;
    }
    Ok(())
}

/// The number of the turn a prompt opens, and the calls up to the next
/// prompt. Turns are counted over the composed reading within each part's
/// bounds; an inline spawned session numbers its own turns. The calls run
/// across the pieces of a part split at a spawning call.
fn turn(
    reader: &Reader<'_>,
    composed: &Composed,
    part: usize,
    at: usize,
) -> Result<(u64, u64), Error> {
    let bounds = &composed.parts[part];
    let same = composed.same_context(part);
    let mut earlier = 0;
    for (index, other) in composed.parts[..part].iter().enumerate() {
        if same(other) {
            earlier += composed.count(
                reader,
                index,
                (other.start, other.end),
                EntryClass::is_prompt,
            )?;
        }
    }
    let number =
        earlier + composed.count(reader, part, (bounds.start, at), EntryClass::is_prompt)? + 1;
    let mut ranges = vec![(part, at + 1, bounds.end)];
    for (offset, other) in composed.parts[part + 1..].iter().enumerate() {
        if other.session == bounds.session {
            ranges.push((part + 1 + offset, other.start, other.end));
        }
    }
    let mut calls = 0;
    for (index, start, end) in ranges {
        if start >= end {
            continue;
        }
        let next = reader.item_find(
            composed.indexes[index].pk,
            ItemLookup::Prompt,
            start as u64..end as u64,
            true,
        )?;
        let end = next.map_or(end, |next| next as usize);
        calls += composed.count(reader, index, (start, end), EntryClass::is_call)?;
        if next.is_some() {
            break;
        }
    }
    Ok((number, calls))
}

/// The context meter of a model message: the provider-reported input of its
/// response, or else the estimate of every entry in context at it (since
/// the newest compaction, less the results a later microcompaction
/// removed), from the stored running sums.
fn context_meter(
    reader: &Reader<'_>,
    composed: &Composed,
    part: usize,
    at: usize,
    entry: &IndexEntry,
) -> Result<ContextMeter, Error> {
    if let Some(tokens) = entry.context_tokens {
        return Ok(ContextMeter::Reported(tokens));
    }
    let parts = composed.context_parts(part);
    let mut ranges: Vec<(usize, usize, usize)> = parts
        .iter()
        .enumerate()
        .take(part + 1)
        .map(|(index, other)| {
            let end = if index == part {
                (at + 1).min(other.end)
            } else {
                other.end
            };
            (index, other.start, end)
        })
        .filter(|(_, start, end)| start < end)
        .collect();
    for which in (0..ranges.len()).rev() {
        let (index, start, end) = ranges[which];
        if let Some(boundary) = reader.item_find(
            composed.indexes[index].pk,
            ItemLookup::Compaction,
            start as u64..end as u64,
            false,
        )? {
            ranges.drain(..which);
            ranges[0].1 = boundary as usize + 1;
            break;
        }
    }
    let mut estimate: u64 = 0;
    for &(index, start, end) in &ranges {
        if start < end {
            estimate += composed
                .counts_at(reader, index, end)?
                .since(&composed.counts_at(reader, index, start)?)
                .tokens_full;
        }
    }
    // A microcompaction removes the results it names that came before it.
    let mut removed: HashSet<(usize, u64)> = HashSet::new();
    for &(index, start, end) in &ranges {
        if start >= end {
            continue;
        }
        let pk = composed.indexes[index].pk;
        for micro in
            reader.item_find_all(pk, ItemLookup::Microcompaction, start as u64..end as u64)?
        {
            let Some(cleared) = composed.entry(reader, index, micro as usize)? else {
                continue;
            };
            for call in &cleared.cleared {
                for &(other, other_start, other_end) in &ranges {
                    for stored in reader.item_calls(composed.indexes[other].pk, call)? {
                        let place = (other, stored.seq as usize);
                        if stored.entry.kind == EntryKind::Result
                            && (other_start..other_end).contains(&place.1)
                            && place < (index, micro as usize)
                            && removed.insert((other, stored.seq))
                        {
                            estimate = estimate.saturating_sub(stored.entry.tokens_full);
                        }
                    }
                }
            }
        }
    }
    Ok(if estimate == 0 {
        ContextMeter::Unknown
    } else {
        ContextMeter::Estimated(estimate)
    })
}

/// Per page slot, the cards placed before its entry and after it.
type PlacedCards = std::collections::HashMap<usize, (Vec<ReadItem>, Vec<ReadItem>)>;

/// Cards for each page slot, placed after the entry they follow when it is
/// shown, or before the next shown entry when the reading hides that entry
/// (a spawning call in the dialogue view), so a card is never lost. A card
/// belongs to this page when that entry comes after the shown entry before
/// the page and no later than the page's last entry.
fn place_cards(
    services: &impl ReadServices,
    reader: &Reader<'_>,
    composed: &Composed,
    shown: &ShownEntries,
    page: &[PageEntry],
    options: &ReadOptions,
) -> Result<PlacedCards, Error> {
    let mut placed = PlacedCards::new();
    let (Some(first), Some(last)) = (page.first(), page.last()) else {
        return Ok(placed);
    };
    if composed.after.is_empty() {
        return Ok(placed);
    }
    let previous = shown.last_before(first.place())?.map(|entry| entry.place());
    for card_slot in &composed.after {
        let Some(place) = composed
            .parts
            .iter()
            .position(|bounds| {
                bounds.session == card_slot.session
                    && (bounds.start..bounds.end).contains(&card_slot.seq)
            })
            .map(|part| (part, card_slot.seq))
        else {
            continue;
        };
        if place > last.place() || previous.is_some_and(|previous| place <= previous) {
            continue;
        }
        // A spawned session's card is a delegation.
        if options.spawns() == SpawnDisplay::Hidden && card_slot.link.kind == "spawn" {
            continue;
        }
        let slot = page.partition_point(|entry| entry.place() < place);
        // A spawned session's card shows its spawning call's objective.
        let call = if card_slot.link.kind == "spawn" {
            composed.entry(reader, place.0, place.1)?
        } else {
            None
        };
        let card = card(
            services,
            reader,
            &card_slot.link,
            &card_slot.child,
            call.as_ref(),
            options.view,
        )?;
        let held = placed.entry(slot).or_default();
        if page[slot].place() == place {
            held.1.push(card);
        } else {
            held.0.push(card);
        }
    }
    Ok(placed)
}

#[allow(clippy::too_many_arguments)]
fn project_layout(
    reader: &Reader<'_>,
    record: Record,
    facts: LayoutFacts,
    options: &ReadOptions,
    at: polylogue_model::Timestamp,
    selected: &dyn Fn(&Source) -> Result<bool, Error>,
    bounded: bool,
    ready_attachment: &crate::attachment_facts::ReadyAttachment<'_>,
    footer: &mut ReadFooter,
    out: &mut dyn FnMut(ReadItem),
) -> Result<Option<ReadRecord>, Error> {
    // A context reading's entries were chosen from the index as members, so
    // each one laid out is selected; what it leaves out is said once, in
    // the reading's first lines.
    let mut record = record;
    // The index owns block membership. Preserve source ordinals and identity
    // while excluding evicted blocks from the original frame as well as layout.
    if (options.view == View::Context || bounded)
        && let Record::Message(row) = &mut record
    {
        let mut blocks = Vec::new();
        let mut ordinals = Vec::new();
        for (block, ordinal) in std::mem::take(&mut row.message.blocks)
            .into_iter()
            .zip(std::mem::take(&mut row.block_ordinals))
        {
            if selected(&Source::block(
                &row.message.id,
                &row.message.session,
                ordinal,
                facts.inherited,
            ))? {
                blocks.push(block);
                ordinals.push(ordinal);
            }
        }
        row.message.blocks = blocks;
        row.block_ordinals = ordinals;
        row.injections
            .retain(|item| row.block_ordinals.binary_search(&item.block).is_ok());
        row.bindings
            .retain(|item| row.block_ordinals.binary_search(&item.block).is_ok());
    }
    let mut facts = facts;
    if options.annotations() {
        layout_facts::annotate(reader, &record, &mut facts, at, options.annotation_at())?;
    }
    if options.view == View::Context {
        facts.context = Some(ContextState {
            selected: true,
            complete: true,
            reason: None,
            evicted: Vec::new(),
            not_recorded: Vec::new(),
        });
    }
    let Some(record) = output::project(
        SelectedRecord::Original(Box::new(record)),
        options,
        &facts,
        false,
        footer,
    )?
    else {
        return Ok(None);
    };
    let layout = layout_facts::Layout {
        options,
        facts: &facts,
        selected,
        ready_attachment,
        per_item_source: true,
        backwards: false,
        page_start: None,
    };
    layout_facts::visit_parts(reader, &record, &layout, |item, _| {
        out(item);
        Ok(true)
    })?;
    Ok(Some(record))
}
