//! Relationships are supplied by the service, never inferred from order.
use polylogue_model::{Block, ErrorReason, ToolOutcome, WrittenBy, bundle::Record, read::*};
use serde_json::{Value, json};

/// A bounded convenience adapter. Full exports use `visit_record` and release
/// each record after delivery. Input must be the result of `project_record`.
pub fn layout_record(
    record: &Record,
    options: &ReadOptions,
    evidence: &ProjectionEvidence,
) -> Result<Vec<ReadPart>, ReadError> {
    let mut parts = Vec::new();
    visit_record(record, options, evidence, |item| {
        parts.push(ReadPart::Item(item));
        true
    })?;
    Ok(parts)
}
/// Visit one projected canonical record. Returning false stops layout without
/// visiting another block. There is no archive-sized collection or recursion.
pub fn visit_record(
    record: &Record,
    options: &ReadOptions,
    evidence: &ProjectionEvidence,
    mut visit: impl FnMut(ReadItem) -> bool,
) -> Result<(), ReadError> {
    options.validate()?;
    let reference = record.reference().unwrap_or("manifest");
    let source = Source::record(reference);
    if options.view == View::Raw {
        if evidence.raw.is_empty() {
            return Err(ReadError::MissingEvidence {
                view: View::Raw,
                evidence: "original input bytes".into(),
                reference: reference.into(),
            });
        }
        for raw in &evidence.raw {
            if !visit(ReadItem {
                source: Source::record(&raw.reference),
                kind: ItemKind::Raw {
                    record: Box::new(raw.clone()),
                },
            }) {
                break;
            }
        }
        return Ok(());
    }
    for indicator in &evidence.before {
        if !visit(indicator_item(indicator)) {
            return Ok(());
        }
    }
    if let Some(context) = &evidence.context {
        for reference in &context.not_recorded {
            if !visit(ReadItem {
                source: source.clone(),
                kind: ItemKind::Omission {
                    count: ValueState::NotRecorded,
                    unit: reference.clone(),
                    why: "not recorded in context".into(),
                    continue_call: None,
                },
            }) {
                return Ok(());
            }
        }
        for reference in &context.evicted {
            if !visit(ReadItem {
                source: Source::record(reference),
                kind: ItemKind::Omission {
                    count: ValueState::known(1),
                    unit: "record".into(),
                    why: "removed from context".into(),
                    continue_call: None,
                },
            }) {
                return Ok(());
            }
        }
        if !context.complete
            && !visit(ReadItem {
                source: source.clone(),
                kind: ItemKind::Omission {
                    count: ValueState::Unknown {
                        reason: context.reason.clone(),
                    },
                    unit: "context".into(),
                    why: context
                        .reason
                        .clone()
                        .unwrap_or_else(|| "context is incomplete".into()),
                    continue_call: None,
                },
            })
        {
            return Ok(());
        }
    }
    for card in &evidence.cards {
        if !visit(ReadItem {
            source: source.clone(),
            kind: ItemKind::Card {
                card: Box::new(card.clone()),
            },
        }) {
            return Ok(());
        }
    }
    match options.view {
        View::Summary => {
            if let Record::Session(row) = record {
                visit(ReadItem {
                    source,
                    kind: ItemKind::Card {
                        card: Box::new(Card::Session {
                            session: Box::new(row.session.clone()),
                            open: None,
                        }),
                    },
                });
            }
            return Ok(());
        }
        View::Live => {
            if evidence.cards.is_empty() {
                return Err(ReadError::MissingEvidence {
                    view: options.view,
                    evidence: "live observation".into(),
                    reference: reference.into(),
                });
            }
            return Ok(());
        }
        View::Links => {
            if let Record::Link(row) = record {
                let (relation, how) = row.link.kind.columns();
                visit(ReadItem {
                    source,
                    kind: ItemKind::Card {
                        card: Box::new(Card::LinkEdge {
                            from: format!("session:{}", row.link.parent),
                            to: format!("session:{}", row.link.child),
                            relation: relation.into(),
                            how: Some(how.into()),
                            cross_link: false,
                        }),
                    },
                });
            }
            return Ok(());
        }
        _ => {}
    }
    match record {
        Record::Message(row) => {
            row.validate()
                .map_err(|e| ReadError::InvalidEvidence(e.to_string()))?;
            let message = &row.message;
            let by = WrittenBy::of(message.role, message.prompt_kind);
            let author = match by {
                WrittenBy::Human => {
                    ValueState::known(if options.for_others { "user" } else { "you" }.into())
                }
                WrittenBy::Model => ValueState::known("model".into()),
                WrittenBy::Harness => ValueState::known("[harness]".into()),
                WrittenBy::Unknown => ValueState::Unknown {
                    reason: Some("author not established by source".into()),
                },
            };
            let mark =
                row.block_ordinals
                    .iter()
                    .zip(&message.blocks)
                    .find_map(|(ordinal, block)| {
                        if !visible_text(block, options) {
                            return None;
                        }
                        let source = Source::block(
                            &message.id,
                            &message.session,
                            *ordinal,
                            evidence.inherited,
                        );
                        evidence.variants.mark_for(&source, options).cloned()
                    });
            if !visit(ReadItem {
                source: source.clone(),
                kind: ItemKind::MessageHeader {
                    header: Box::new(MessageHeader {
                        time: TimeValue {
                            at: message.at,
                            evidence: message.time_evidence(),
                        },
                        author,
                        model: message
                            .model
                            .clone()
                            .map(ValueState::known)
                            .unwrap_or(ValueState::NotRecorded),
                        assertions: if options.assertions {
                            evidence.assertions.clone()
                        } else {
                            vec![]
                        },
                        variant: mark,
                        inherited: evidence.inherited,
                        branch: evidence.branch.clone(),
                    }),
                },
            }) {
                return Ok(());
            }
            if by == WrittenBy::Harness && options.harness() == Expansion::Collapsed {
                let text = message
                    .blocks
                    .iter()
                    .filter_map(Block::text)
                    .next()
                    .unwrap_or_default();
                visit(ReadItem {
                    source,
                    kind: ItemKind::Indicator {
                        indicator: IndicatorKind::Harness,
                        text: text
                            .lines()
                            .next()
                            .unwrap_or_default()
                            .chars()
                            .take(80)
                            .collect(),
                        expand: evidence.expand_inherited.clone().map(Box::new),
                    },
                });
                return Ok(());
            }
            let mut absent: Option<(Source, u64)> = None;
            for (ordinal, block) in row.block_ordinals.iter().zip(&message.blocks) {
                let source =
                    Source::block(&message.id, &message.session, *ordinal, evidence.inherited);
                let empty = matches!(
                    block,
                    Block::Thinking {
                        text: None,
                        signature: None,
                        redacted: false
                    }
                ) || matches!(block,Block::Thinking {text:Some(text),signature:None,redacted:false} if text.is_empty());
                if empty && options.thinking() != Expansion::None {
                    if let Some((first, count)) = &mut absent
                        && first.block.is_some_and(|first| first + *count == *ordinal)
                    {
                        *count += 1;
                        continue;
                    }
                    if let Some((source, blocks)) = absent.take()
                        && !visit(absent_thinking(source, blocks))
                    {
                        return Ok(());
                    }
                    absent = Some((source, 1));
                    continue;
                }
                if let Some((source, blocks)) = absent.take()
                    && !visit(absent_thinking(source, blocks))
                {
                    return Ok(());
                }
                if !block_items(block, source, *ordinal, options, evidence, &mut visit)? {
                    return Ok(());
                }
            }
            if let Some((source, blocks)) = absent
                && !visit(absent_thinking(source, blocks))
            {
                return Ok(());
            }
        }
        Record::Document(row) => {
            if options.view == View::Dialogue {
                return Err(ReadError::UnsupportedTarget {
                    view: options.view,
                    target: row.reference.clone(),
                    accepted_views: vec![View::Transcript, View::Raw],
                });
            }
            for (ordinal, block) in row.document.blocks.iter().enumerate() {
                let source = Source {
                    reference: row.reference.clone(),
                    block: Some(ordinal as u64),
                    ..source.clone()
                };
                if !block_items(block, source, ordinal as u64, options, evidence, &mut visit)? {
                    return Ok(());
                }
            }
        }
        Record::Event(row) if options.view != View::Changes => {
            if !visit(ReadItem {
                source,
                kind: ItemKind::Other {
                    type_name: row.event.kind.clone(),
                    data: json!({"event":row.event,"identity":row.identity}),
                },
            }) {
                return Ok(());
            }
        }
        Record::Authored(row) if options.assertions => {
            if !visit(ReadItem {
                source,
                kind: ItemKind::Other {
                    type_name: "authored".into(),
                    data: json!(row),
                },
            }) {
                return Ok(());
            }
        }
        Record::Session(row) if options.view != View::Changes => {
            let accepted = visit(ReadItem {
                source,
                kind: ItemKind::Card {
                    card: Box::new(Card::Session {
                        session: Box::new(row.session.clone()),
                        open: None,
                    }),
                },
            });
            if !accepted {
                return Ok(());
            }
        }
        _ => {}
    }
    for indicator in &evidence.after {
        if !visit(indicator_item(indicator)) {
            break;
        }
    }
    Ok(())
}
fn indicator_item(indicator: &PreparedIndicator) -> ReadItem {
    ReadItem {
        source: indicator.source.clone(),
        kind: ItemKind::Indicator {
            indicator: indicator.kind,
            text: indicator.text.clone(),
            expand: indicator.expand.clone().map(Box::new),
        },
    }
}
fn block_items(
    block: &Block,
    source: Source,
    ordinal: u64,
    options: &ReadOptions,
    evidence: &ProjectionEvidence,
    visit: &mut impl FnMut(ReadItem) -> bool,
) -> Result<bool, ReadError> {
    if !options.view.block_kinds().contains(&block.kind()) {
        return Ok(true);
    }
    if !evidence.tool_pairs.contains_key(&ordinal)
        && visible_text(block, options)
        && !variant_indicator(&source, options, evidence, visit)
    {
        return Ok(false);
    }
    let presented = presentation_block(block, &source, options, evidence)?;
    let block = presented.as_ref();
    let kind = match block {
        Block::Text { text } => ItemKind::Text { text: text.clone() },
        Block::Thinking {
            text,
            redacted,
            signature,
        } => {
            if options.thinking() == Expansion::None {
                return Ok(true);
            }
            let content = match text {
                Some(text) if options.thinking() == Expansion::Full => {
                    ThinkingContent::Text { text: text.clone() }
                }
                Some(_) => ThinkingContent::Collapsed,
                None if *redacted => ThinkingContent::Redacted,
                None if signature.is_some() => ThinkingContent::Encrypted,
                None => ThinkingContent::Absent { blocks: 1 },
            };
            ItemKind::Thinking {
                words: text
                    .as_ref()
                    .map(|text| ValueState::known(text.split_whitespace().count() as u64))
                    .unwrap_or_default(),
                content,
            }
        }
        Block::ToolCall { .. } | Block::ToolResult { .. } => {
            if options.tools() == Expansion::None {
                return Ok(true);
            }
            if let Some(pair) = evidence.tool_pairs.get(&ordinal) {
                pair.validate()?;
                if pair.emit_at != source.reference {
                    return Ok(true);
                }
                let result = pair
                    .result
                    .as_ref()
                    .map(|result| {
                        presentation_block(&result.block, &result.source, options, evidence)
                    })
                    .transpose()?;
                if options.tools() == Expansion::Full
                    && let Some(result) = &pair.result
                    && !variant_indicator(&result.source, options, evidence, visit)
                {
                    return Ok(false);
                }
                let mut call = call_item(
                    &pair.call.block,
                    result.as_deref(),
                    options.tools() == Expansion::Full,
                    pair.running,
                );
                if !matches!(pair.duration_ms, ValueState::Unknown { .. }) {
                    call.duration_ms = pair.duration_ms.clone();
                }
                call.error_line = pair.error_line.clone().or(call.error_line);
                call.child = pair.child.clone().map(Box::new);
                call.edit = pair.edit.clone();
                call.result_source = pair.result.as_ref().map(|r| r.source.clone());
                excerpt_output(
                    &mut call,
                    options.output_lines,
                    evidence.output_calls.get(&ordinal),
                );
                if !visit(ReadItem {
                    source: pair.call.source.clone(),
                    kind: ItemKind::Call {
                        call: Box::new(call),
                    },
                }) {
                    return Ok(false);
                }
                if let Some(result) = &pair.result {
                    return result_extras(&result.block, &result.source, options, evidence, visit);
                }
                return Ok(true);
            }
            let mut call = match block {
                Block::ToolCall { .. } => {
                    call_item(block, None, options.tools() == Expansion::Full, false)
                }
                _ => call_item(
                    &Block::ToolCall {
                        call_id: None,
                        tool: None,
                        action: Default::default(),
                        input: Value::Null,
                        command: None,
                        path: None,
                        run_by_script: false,
                    },
                    Some(block),
                    options.tools() == Expansion::Full,
                    false,
                ),
            };
            call.orphan = matches!(block, Block::ToolResult { .. });
            excerpt_output(
                &mut call,
                options.output_lines,
                evidence.output_calls.get(&ordinal),
            );
            if !visit(ReadItem {
                source: source.clone(),
                kind: ItemKind::Call {
                    call: Box::new(call),
                },
            }) {
                return Ok(false);
            }
            return result_extras(block, &source, options, evidence, visit);
        }
        Block::Image(attachment) | Block::File(attachment) => {
            attachment_kind(attachment, &source, 0, evidence)?
        }
        Block::Other { type_name, data } => ItemKind::Other {
            type_name: type_name.clone(),
            data: data.clone(),
        },
    };
    Ok(visit(ReadItem { source, kind }))
}
fn attachment_kind(
    attachment: &polylogue_model::Attachment,
    source: &Source,
    index: usize,
    evidence: &ProjectionEvidence,
) -> Result<ItemKind, ReadError> {
    let prepared = evidence
        .attachments
        .get(&source.reference)
        .and_then(|blocks| source.block.and_then(|ordinal| blocks.get(&ordinal)))
        .and_then(|attachments| attachments.get(index));
    let inline_bytes = prepared.and_then(|prepared| prepared.inline_bytes.clone());
    if let Some(bytes) = &inline_bytes
        && attachment.blob.as_ref() != Some(&polylogue_model::BlobHash::of(bytes))
    {
        return Err(ReadError::InvalidEvidence(
            "attachment inline bytes do not match its canonical blob hash".into(),
        ));
    }
    Ok(ItemKind::Attachment {
        attachment: Box::new(attachment.clone()),
        supplied_by: prepared.map(|p| p.supplied_by.clone()).unwrap_or_default(),
        availability: prepared.map(|p| p.availability.clone()).unwrap_or_default(),
        link: prepared.and_then(|p| p.link.clone()),
        inline_bytes,
        open: prepared.and_then(|p| p.open.clone()),
    })
}

fn result_extras(
    block: &Block,
    source: &Source,
    options: &ReadOptions,
    evidence: &ProjectionEvidence,
    visit: &mut impl FnMut(ReadItem) -> bool,
) -> Result<bool, ReadError> {
    if let Block::ToolResult {
        data, attachments, ..
    } = block
    {
        if options.tools() == Expansion::Full
            && let Some(data) = data
            && !visit(ReadItem {
                source: source.clone(),
                kind: ItemKind::Other {
                    type_name: "tool result data".into(),
                    data: data.clone(),
                },
            })
        {
            return Ok(false);
        }
        for (index, attachment) in attachments.iter().enumerate() {
            if !visit(ReadItem {
                source: source.clone(),
                kind: attachment_kind(attachment, source, index, evidence)?,
            }) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}
fn call_item(call: &Block, result: Option<&Block>, full: bool, running: bool) -> CallItem {
    let Block::ToolCall {
        call_id,
        tool,
        action,
        input,
        command,
        path,
        ..
    } = call
    else {
        unreachable!("validated call")
    };
    let target = command.clone().or_else(|| path.clone()).or_else(|| {
        input
            .as_object()
            .and_then(|o| o.values().find_map(Value::as_str).map(str::to_owned))
    });
    let (outcome, exit, duration_ms, output) = match result {
        Some(Block::ToolResult {
            outcome,
            reported_duration_ms,
            output,
            ..
        }) => {
            let (state, exit) = match outcome {
                ToolOutcome::Ok => (ValueState::known("ok".into()), None),
                ToolOutcome::Error { reason } => (
                    ValueState::known("error".into()),
                    match reason {
                        Some(ErrorReason::ExitCode(code)) => Some(*code),
                        _ => None,
                    },
                ),
                ToolOutcome::NoResult => (
                    ValueState::Unknown {
                        reason: Some("no result recorded".into()),
                    },
                    None,
                ),
                ToolOutcome::Unknown { reason } => (
                    ValueState::Unknown {
                        reason: Some(format!("{reason:?}")),
                    },
                    None,
                ),
            };
            (
                state,
                exit,
                reported_duration_ms
                    .map(ValueState::known)
                    .unwrap_or_default(),
                if full { output.clone() } else { None },
            )
        }
        _ => (
            if running {
                ValueState::Pending
            } else {
                ValueState::Unknown {
                    reason: Some("no result recorded".into()),
                }
            },
            None,
            ValueState::default(),
            None,
        ),
    };
    CallItem {
        orphan: false,
        call_id: call_id.clone().or_else(|| match result {
            Some(Block::ToolResult { call_id, .. }) => call_id.clone(),
            _ => None,
        }),
        tool: tool.clone().map(ValueState::known).unwrap_or_default(),
        target: target.map(ValueState::known).unwrap_or_default(),
        action: *action,
        outcome,
        exit,
        duration_ms,
        error_line: match result {
            Some(Block::ToolResult {
                outcome:
                    ToolOutcome::Error {
                        reason: Some(reason),
                    },
                ..
            }) => Some(match reason {
                ErrorReason::ExitCode(0) => "exit 0 but marked error".into(),
                ErrorReason::ExitCode(code) => format!("exit {code}"),
                ErrorReason::Denied(reason) => format!("denied ({reason})"),
                ErrorReason::Interrupted => "interrupted".into(),
                ErrorReason::TimedOut => "timed out".into(),
            }),
            _ => None,
        },
        input: full.then(|| input.clone()),
        output,
        output_excerpt: None,
        result_source: None,
        child: None,
        edit: None,
    }
}

/// Retained lines and each long line's retained ends remain exact substrings.
/// Every omitted interval names original Unicode offsets and exact counts.
fn excerpt_output(call: &mut CallItem, lines: u64, open: Option<&ReadyCall>) {
    if lines == 0 {
        return;
    }
    let Some(output) = &call.output else { return };
    let source_lines: Vec<&str> = output.split_inclusive('\n').collect();
    let n = usize::try_from(lines).unwrap_or(usize::MAX);
    let trim_lines = n < source_lines.len().div_ceil(2);
    let mut parts = Vec::new();
    let mut offset = 0u64;
    let mut omitted = false;
    for (index, line) in source_lines.iter().enumerate() {
        let count = line.chars().count() as u64;
        if trim_lines && index >= n && index < source_lines.len() - n {
            match parts.last_mut() {
                Some(OutputPart::Omission {
                    lines,
                    characters,
                    until,
                    ..
                }) => {
                    *lines += 1;
                    *characters += count;
                    *until += count;
                }
                _ => parts.push(OutputPart::Omission {
                    lines: 1,
                    characters: count,
                    from: offset,
                    until: offset + count,
                }),
            }
            omitted = true;
            offset += count;
            continue;
        }
        let content_characters = line.strip_suffix('\n').unwrap_or(line).chars().count();
        if content_characters > 4000 {
            let from = line
                .char_indices()
                .nth(2000)
                .map(|(i, _)| i)
                .expect("long line");
            let tail = line
                .char_indices()
                .rev()
                .nth(1999)
                .map(|(i, _)| i)
                .expect("long line");
            let before = &line[..from];
            let middle = &line[from..tail];
            let after = &line[tail..];
            let first = before.chars().count() as u64;
            let missing = middle.chars().count() as u64;
            append_output_text(&mut parts, before, offset, offset + first);
            parts.push(OutputPart::Omission {
                lines: 0,
                characters: missing,
                from: offset + first,
                until: offset + first + missing,
            });
            append_output_text(&mut parts, after, offset + first + missing, offset + count);
            omitted = true;
        } else {
            append_output_text(&mut parts, line, offset, offset + count);
        }
        offset += count;
    }
    if omitted {
        call.output_excerpt = Some(OutputExcerpt {
            parts,
            open: open.cloned(),
        });
        call.output = None;
    }
}
fn append_output_text(parts: &mut Vec<OutputPart>, text: &str, from: u64, until: u64) {
    if let Some(OutputPart::Text {
        text: previous,
        until: end,
        ..
    }) = parts.last_mut()
    {
        previous.push_str(text);
        *end = until;
    } else {
        parts.push(OutputPart::Text {
            text: text.into(),
            from,
            until,
        });
    }
}

fn absent_thinking(source: Source, blocks: u64) -> ReadItem {
    ReadItem {
        source,
        kind: ItemKind::Thinking {
            words: ValueState::NotRecorded,
            content: ThinkingContent::Absent { blocks },
        },
    }
}

fn visible_text(block: &Block, options: &ReadOptions) -> bool {
    options.view.block_kinds().contains(&block.kind())
        && match block {
            Block::Text { .. } => true,
            Block::Thinking { text: Some(_), .. } => options.thinking() == Expansion::Full,
            Block::ToolResult {
                output: Some(_), ..
            } => options.tools() == Expansion::Full,
            _ => false,
        }
}
fn presentation_block<'a>(
    block: &'a Block,
    source: &Source,
    options: &ReadOptions,
    evidence: &ProjectionEvidence,
) -> Result<std::borrow::Cow<'a, Block>, ReadError> {
    use std::borrow::Cow;
    if !visible_text(block, options) {
        return Ok(Cow::Borrowed(block));
    }
    let text = match block {
        Block::Text { text }
        | Block::Thinking {
            text: Some(text), ..
        }
        | Block::ToolResult {
            output: Some(text), ..
        } => text,
        _ => return Ok(Cow::Borrowed(block)),
    };
    match evidence.variants.apply(source, text, options)? {
        Cow::Borrowed(_) => Ok(Cow::Borrowed(block)),
        Cow::Owned(text) => {
            let mut presented = block.clone();
            match &mut presented {
                Block::Text { text: field }
                | Block::Thinking {
                    text: Some(field), ..
                }
                | Block::ToolResult {
                    output: Some(field),
                    ..
                } => *field = text,
                _ => unreachable!("text body kind is unchanged"),
            }
            Ok(Cow::Owned(presented))
        }
    }
}
fn variant_indicator(
    source: &Source,
    options: &ReadOptions,
    evidence: &ProjectionEvidence,
    visit: &mut impl FnMut(ReadItem) -> bool,
) -> bool {
    let mut shown = std::collections::BTreeSet::new();
    for (original, variant) in evidence.variants.for_source(source) {
        if !shown.insert(original)
            || variant.mark.stale && !options.variants.contains(&variant.mark.key)
        {
            continue;
        }
        if !visit(ReadItem {
            source: source.clone(),
            kind: ItemKind::Indicator {
                indicator: IndicatorKind::Variant,
                text: format!(
                    "[variant {} · {} · {}{}{}]",
                    variant.mark.key,
                    variant
                        .relations
                        .iter()
                        .map(|relation| relation.as_str())
                        .collect::<Vec<_>>()
                        .join(", "),
                    variant.mark.reference,
                    if variant.mark.stale { " · stale" } else { "" }
                ),
                expand: variant.mark.show_original.clone().map(Box::new),
            },
        }) {
            return false;
        }
    }
    true
}
