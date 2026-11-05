//! Indexed tool relationships enrich layout one pair at a time. Canonical
//! records remain unchanged; offsets describe the existing emitted item order.
use super::{Failure, output};
use polylogue_model::{MessageId, PromptKind, Role, read::*};
use polylogue_store::Reader;
use std::ops::ControlFlow;

pub(super) struct Layout<'a> {
    pub options: &'a ReadOptions,
    pub evidence: &'a ProjectionEvidence,
    pub selected: &'a dyn Fn(&Source) -> Result<bool, Failure>,
    pub ready_attachment: &'a super::attachment_evidence::ReadyAttachment<'a>,
    pub backwards: bool,
    pub anchor: Option<u64>,
}
struct Segment {
    item: ReadItem,
    pair: Option<(MessageId, u64)>,
    start: u64,
    count: u64,
    attachment_index: Option<u64>,
}
fn check(reader: &Reader<'_>) -> Result<(), Failure> {
    reader.scope_token().check()?;
    if let Some(limit) = polylogue_store::read::limit::ReadLimit::current() {
        limit.check()?;
    }
    Ok(())
}
fn visible(reader: &Reader<'_>, source: &Source, options: &ReadOptions) -> Result<bool, Failure> {
    let (Some(message), Some(session)) = (&source.message, &source.source_session) else {
        return Err(Failure::new(
            "not_yet_available",
            "tool source has no canonical message identity",
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
    selected.visit_from(from, |message| {
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
) -> Result<Vec<Segment>, Failure> {
    // These parts contain only the current canonical record's own fields.
    // Remote counterpart bodies are never collected here.
    let base = if let Some(record) = record.record() {
        polylogue_render::layout_record(record, layout.options, layout.evidence)
            .map_err(output::read_error)?
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
                // Native result extras are replaced by the same indexed pair's
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
                            Failure::new("stale_cursor", "paired result geometry changed")
                        })?;
                    count = count
                        .checked_add(u64::from(data))
                        .and_then(|value| value.checked_add(attachments))
                        .ok_or_else(|| {
                            Failure::new(
                                "not_yet_available",
                                "tool layout position exceeds its ordinal range",
                            )
                        })?;
                }
            }
        }
        let attachment_index = if matches!(item.kind, ItemKind::Attachment { .. }) {
            let index = attachment_indices
                .entry(item.source.reference.clone())
                .or_default();
            let current = *index;
            *index = index.checked_add(1).ok_or_else(|| {
                Failure::new(
                    "not_yet_available",
                    "attachment index exceeds its ordinal range",
                )
            })?;
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
        offset = offset.checked_add(count).ok_or_else(|| {
            Failure::new(
                "not_yet_available",
                "read layout position exceeds its ordinal range",
            )
        })?;
    }
    Ok(segments)
}
fn pair_parts(
    reader: &Reader<'_>,
    segment: &Segment,
    layout: &Layout<'_>,
    mut visit: impl FnMut(ReadItem, Option<u64>) -> Result<bool, Failure>,
) -> Result<bool, Failure> {
    let Some((message, ordinal)) = &segment.pair else {
        return visit(segment.item.clone(), segment.attachment_index);
    };
    check(reader)?;
    let pair = if layout.options.tools() == Expansion::Full {
        reader.message_tool_pair(message, *ordinal)
    } else {
        reader.message_tool_pair_header(message, *ordinal)
    }?;
    let mut pair =
        pair.ok_or_else(|| Failure::new("stale_cursor", "indexed tool relationship changed"))?;
    if layout.options.view == View::Context
        && let Some(result) = &pair.result
        && !(layout.selected)(&result.source)?
    {
        pair.result = None;
    }
    pair.call.source.inherited = layout.evidence.inherited;
    if let Some(result) = &mut pair.result {
        result.source.inherited = layout.evidence.inherited;
    }
    let source = if pair.call.source.reference == segment.item.source.reference {
        pair.call.clone()
    } else {
        pair.result
            .as_ref()
            .filter(|result| result.source.reference == segment.item.source.reference)
            .cloned()
            .ok_or_else(|| Failure::new("stale_cursor", "paired result source changed"))?
    };
    pair.emit_at = source.source.reference.clone();
    let original = source.source.block.ok_or_else(|| {
        Failure::new(
            "not_yet_available",
            "tool source has no original block ordinal",
        )
    })?;
    let mut detail = layout.evidence.clone();
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
            // The selected canonical call already supplied this display target;
            // collapsed remote evidence deliberately leaves its input unread.
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
    .map_err(output::read_error)?;
    if let Some(error) = failure {
        return Err(error);
    }
    Ok(continuing)
}
pub(super) fn visit_parts(
    reader: &Reader<'_>,
    record: &ReadRecord,
    layout: &Layout<'_>,
    mut visit: impl FnMut(ReadItem, u64) -> Result<bool, Failure>,
) -> Result<bool, Failure> {
    check(reader)?;
    let segments = segments(reader, record, layout)?;
    if layout.backwards {
        for segment in segments.iter().rev() {
            if layout.anchor.is_some_and(|anchor| segment.start > anchor) {
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
                return Err(Failure::new("stale_cursor", "tool layout geometry changed"));
            }
            for (index, (item, attachment_index)) in parts.into_iter().enumerate().rev() {
                let ordinal = segment.start + index as u64;
                if layout.anchor.is_some_and(|anchor| ordinal > anchor) {
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
                .anchor
                .is_some_and(|anchor| segment.start + segment.count <= anchor)
            {
                continue;
            }
            let mut offset = 0u64;
            let keep = pair_parts(reader, segment, layout, |item, attachment_index| {
                let ordinal = segment.start + offset;
                offset += 1;
                if layout.anchor.is_some_and(|anchor| ordinal < anchor) {
                    return Ok(true);
                }
                let item = attachment_item(reader, item, attachment_index, layout)?;
                visit(item, ordinal)
            })?;
            if !keep {
                return Ok(false);
            }
            if offset != segment.count {
                return Err(Failure::new("stale_cursor", "tool layout geometry changed"));
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
) -> Result<ReadItem, Failure> {
    if let Some(index) = index {
        super::attachment_evidence::prepare(reader, item, index, layout.ready_attachment)
    } else {
        Ok(item)
    }
}
