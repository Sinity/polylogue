//! Indexed tool relationships enrich layout one pair at a time. Stored
//! records remain unchanged; offsets describe the existing emitted item order.
use super::Error;
use polylogue_model::{MessageId, PromptKind, Role, read::*};
use polylogue_store::Reader;
use std::ops::ControlFlow;

/// Current notes and marks are prepared from the same reader as their target.
pub(super) fn annotations(
    reader: &Reader<'_>,
    reference: &polylogue_model::Reference,
    at: polylogue_model::Timestamp,
) -> Result<Vec<Annotation>, Error> {
    let mut annotations = Vec::new();
    let _ = reader.visit_assertion_references_linked(
        reference,
        &[polylogue_model::assertion::LinkRole::About],
        |revision| {
            if reader.assertion_is_unexpired(&revision, at)?
                && reader
                    .assertion_revision_state(&revision)?
                    .is_some_and(|state| !state.withdrawn)
                && let Some(assertion) = reader.assertion(&revision)?
            {
                let fact = assertion.fact;
                annotations.push(Annotation {
                    reference: polylogue_model::Reference::Assertion(revision).to_string(),
                    key: fact.key,
                    author: fact.author,
                    kind: fact.kind.as_str().into(),
                    value: fact.value.unwrap_or(serde_json::Value::Null),
                    open: None,
                });
            }
            Ok(ControlFlow::Continue(()))
        },
    )?;
    Ok(annotations)
}

pub(super) fn annotate(
    reader: &Reader<'_>,
    record: &polylogue_model::bundle::Record,
    facts: &mut LayoutFacts,
    at: polylogue_model::Timestamp,
) -> Result<(), Error> {
    if matches!(
        record,
        polylogue_model::bundle::Record::Session(_) | polylogue_model::bundle::Record::Message(_)
    ) {
        let reference = polylogue_model::Reference::parse(
            record.reference().expect("selected record has a reference"),
        )
        .map_err(Error::internal)?;
        facts.assertions = annotations(reader, &reference, at)?;
    }
    Ok(())
}

pub(super) struct Layout<'a> {
    pub options: &'a ReadOptions,
    pub facts: &'a LayoutFacts,
    pub selected: &'a dyn Fn(&Source) -> Result<bool, Error>,
    pub ready_attachment: &'a super::attachment_facts::ReadyAttachment<'a>,
    pub backwards: bool,
    pub page_start: Option<u64>,
}
struct Segment {
    item: ReadItem,
    pair: Option<(MessageId, u64)>,
    start: u64,
    count: u64,
    attachment_index: Option<u64>,
}
fn check(reader: &Reader<'_>) -> Result<(), Error> {
    reader.scope_token().check()?;
    if let Some(limit) = polylogue_store::read::limit::ReadLimit::current() {
        limit.check()?;
    }
    Ok(())
}
fn visible(reader: &Reader<'_>, source: &Source, options: &ReadOptions) -> Result<bool, Error> {
    let (Some(message), Some(session)) = (&source.message, &source.source_session) else {
        return Err(Error::internal(
            "a stored tool call has no message identity",
        ));
    };
    check(reader)?;
    let selected = reader
        .select_messages(session, false)?
        .for_blocks(&[], false);
    let Some(from) = selected.locate(Some(message))? else {
        return Ok(false);
    };
    let mut shown = false;
    selected.visit(from, |message| {
        let harness =
            message.role == Role::System || message.prompt_kind == Some(PromptKind::Harness);
        shown = options
            .view
            .keeps_message_metadata(message.role, message.prompt_kind)
            && (!harness || options.harness() != Expansion::None);
        ControlFlow::Break(())
    })?;
    Ok(shown)
}
fn segments(
    reader: &Reader<'_>,
    record: &ReadRecord,
    layout: &Layout<'_>,
) -> Result<Vec<Segment>, Error> {
    // These parts contain only the current source record's own fields.
    // Remote counterpart bodies are never collected here.
    let base = if let Some(record) = record.record() {
        polylogue_render::layout_record(record, layout.options, layout.facts)
            .map_err(Error::from)?
    } else {
        record.parts().to_vec()
    };
    let mut segments = Vec::new();
    let mut offset = 0u64;
    let mut replaced = None;
    let mut attachment_indices = std::collections::BTreeMap::<String, u64>::new();
    for part in base {
        let ReadPart::Item(item) = part else { continue };
        if replaced.as_ref() == Some(&item.source.reference) {
            continue;
        }
        replaced = None;
        let mut pair_at = None;
        let mut count = 1u64;
        if matches!(item.kind, ItemKind::Call { .. })
            && let (Some(message), Some(ordinal)) = (&item.source.message, item.source.block)
        {
            check(reader)?;
            if let Some((call, result)) = reader.message_tool_pair_sources(message, ordinal)?
                && (call.reference == item.source.reference
                    || visible(reader, &call, layout.options)?)
            {
                // The source's result extras are replaced by the same indexed pair's
                // extras, or suppressed with its already-selected call.
                replaced = Some(item.source.reference.clone());
                if call.reference != item.source.reference && (layout.selected)(&call)? {
                    continue;
                }
                pair_at = Some((message.clone(), ordinal));
                if layout.options.tools() == Expansion::Full
                    && let Some(result) = &result
                    && (layout.options.view != View::Context || (layout.selected)(result)?)
                {
                    let (data, attachments) = reader
                        .message_tool_pair_result_shape(message, ordinal)?
                        .ok_or_else(|| {
                            Error::stale_cursor(
                                "a tool result in this session changed since this page was issued",
                            )
                        })?;
                    count = count
                        .checked_add(u64::from(data))
                        .and_then(|value| value.checked_add(attachments))
                        .ok_or_else(|| {
                            Error::internal("a tool call has more than 2^64 layout items")
                        })?;
                }
            }
        }
        let attachment_index = if matches!(item.kind, ItemKind::Attachment { .. }) {
            let index = attachment_indices
                .entry(item.source.reference.clone())
                .or_default();
            let current = *index;
            *index = index
                .checked_add(1)
                .ok_or_else(|| Error::internal("a record has more than 2^64 attachments"))?;
            Some(current)
        } else {
            None
        };
        segments.push(Segment {
            item,
            pair: pair_at,
            start: offset,
            count,
            attachment_index,
        });
        offset = offset
            .checked_add(count)
            .ok_or_else(|| Error::internal("a record has more than 2^64 layout items"))?;
    }
    Ok(segments)
}
fn pair_parts(
    reader: &Reader<'_>,
    segment: &Segment,
    layout: &Layout<'_>,
    mut visit: impl FnMut(ReadItem, Option<u64>) -> Result<bool, Error>,
) -> Result<bool, Error> {
    let Some((message, ordinal)) = &segment.pair else {
        return visit(segment.item.clone(), segment.attachment_index);
    };
    check(reader)?;
    let pair = if layout.options.tools() == Expansion::Full {
        reader.message_tool_pair(message, *ordinal)
    } else {
        reader.message_tool_pair_header(message, *ordinal)
    }?;
    let mut pair = pair.ok_or_else(|| {
        Error::stale_cursor("a tool call or its result changed since this page was issued")
    })?;
    if layout.options.view == View::Context
        && let Some(result) = &pair.result
        && !(layout.selected)(&result.source)?
    {
        pair.result = None;
    }
    pair.call.source.inherited = layout.facts.inherited;
    if let Some(result) = &mut pair.result {
        result.source.inherited = layout.facts.inherited;
    }
    let source = if pair.call.source.reference == segment.item.source.reference {
        pair.call.clone()
    } else {
        pair.result
            .as_ref()
            .filter(|result| result.source.reference == segment.item.source.reference)
            .cloned()
            .ok_or_else(|| {
                Error::stale_cursor(
                    "the result attached to a tool call changed since this page was issued",
                )
            })?
    };
    pair.emit_at = source.source.reference.clone();
    let original = source
        .source
        .block
        .ok_or_else(|| Error::internal("a stored tool call has no original block number"))?;
    let mut detail = layout.facts.clone();
    detail.tool_pairs.clear();
    detail.tool_pairs.insert(original, pair);
    let mut failure = None;
    let mut continuing = true;
    let mut attachment_index = 0u64;
    polylogue_render::visit_source_block(&source, layout.options, &detail, |mut item| {
        if item.source.reference == segment.item.source.reference
            && let ItemKind::Call { call: selected } = &segment.item.kind
            && let ItemKind::Call { call } = &mut item.kind
        {
            // The selected call already supplied this display target;
            // collapsed remote facts deliberately leaves its input unread.
            call.target = selected.target.clone();
        }
        let index = if matches!(item.kind, ItemKind::Attachment { .. }) {
            let index = attachment_index;
            attachment_index += 1;
            Some(index)
        } else {
            None
        };
        let result = check(reader).and_then(|()| visit(item, index));
        match result {
            Ok(keep) => {
                continuing = keep;
                keep
            }
            Err(error) => {
                failure = Some(error);
                false
            }
        }
    })
    .map_err(Error::from)?;
    if let Some(error) = failure {
        return Err(error);
    }
    Ok(continuing)
}
pub(super) fn visit_parts(
    reader: &Reader<'_>,
    record: &ReadRecord,
    layout: &Layout<'_>,
    mut visit: impl FnMut(ReadItem, u64) -> Result<bool, Error>,
) -> Result<bool, Error> {
    check(reader)?;
    let segments = segments(reader, record, layout)?;
    if layout.backwards {
        for segment in segments.iter().rev() {
            if layout.page_start.is_some_and(|start| segment.start > start) {
                continue;
            }
            // At most one pair's parts are held for descending delivery. Counts
            // of earlier pairs came from headers, never their remote payloads.
            let mut parts = Vec::new();
            pair_parts(reader, segment, layout, |item, attachment_index| {
                parts.push((item, attachment_index));
                Ok(true)
            })?;
            if parts.len() as u64 != segment.count {
                return Err(Error::stale_cursor(
                    "a tool call on this page is laid out differently since its result was re-read",
                ));
            }
            for (index, (item, attachment_index)) in parts.into_iter().enumerate().rev() {
                let ordinal = segment.start + index as u64;
                if layout.page_start.is_some_and(|start| ordinal > start) {
                    continue;
                }
                let item = attachment_item(reader, item, attachment_index, layout)?;
                if !visit(item, ordinal)? {
                    return Ok(false);
                }
            }
        }
    } else {
        for segment in &segments {
            if layout
                .page_start
                .is_some_and(|start| segment.start + segment.count <= start)
            {
                continue;
            }
            let mut offset = 0u64;
            let keep = pair_parts(reader, segment, layout, |item, attachment_index| {
                let ordinal = segment.start + offset;
                offset += 1;
                if layout.page_start.is_some_and(|start| ordinal < start) {
                    return Ok(true);
                }
                let item = attachment_item(reader, item, attachment_index, layout)?;
                visit(item, ordinal)
            })?;
            if !keep {
                return Ok(false);
            }
            if offset != segment.count {
                return Err(Error::stale_cursor(
                    "a tool call on this page is laid out differently since its result was re-read",
                ));
            }
        }
    }
    Ok(true)
}

fn attachment_item(
    reader: &Reader<'_>,
    item: ReadItem,
    index: Option<u64>,
    layout: &Layout<'_>,
) -> Result<ReadItem, Error> {
    if let Some(index) = index {
        super::attachment_facts::prepare(reader, item, index, layout.ready_attachment)
    } else {
        Ok(item)
    }
}
