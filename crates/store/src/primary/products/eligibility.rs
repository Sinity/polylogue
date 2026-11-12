//! Native reading eligibility never supplies the local model's values.
//! The original parser reproduces each value before its native identity is used.
use super::*;
use crate::{Result, StoreError};
use rusqlite::{Connection, params};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SourceRecord {
    pub family: String,
    pub reference: String,
    pub value_hash: BlobHash,
    pub start: u64,
    pub end: u64,
    pub session: Option<String>,
    pub blobs: Vec<BlobHash>,
}

impl SourceRecord {
    pub fn of(product: &Product) -> Result<Option<Self>> {
        if let Product::Interpreted { record, .. } = product {
            return Ok(Some(record.clone()));
        }
        let Some(reading) = product.reading() else {
            return Ok(None);
        };
        let (family, reference, range, session, blocks) = match product {
            Product::Session { facts, .. } => (
                "session",
                format!("session:{}", facts.id),
                None,
                Some(facts.id.to_string()),
                &[][..],
            ),
            Product::Message { facts, .. } => (
                "message",
                format!("message:{}:{}", facts.message.id, facts.offset),
                Some((facts.offset, facts.end.max(facts.offset))),
                Some(facts.message.session.to_string()),
                facts.message.blocks.as_slice(),
            ),
            Product::ToolCall { facts, .. } => (
                "tool_call",
                format!("call:{}:{}", facts.message, facts.block),
                Some((facts.source_start, facts.source_end)),
                Some(facts.session.to_string()),
                &[][..],
            ),
            Product::Document { facts, .. } => (
                "document",
                format!("document:{}:{}", facts.document.id, facts.offset),
                Some((facts.offset, facts.end.max(facts.offset))),
                None,
                facts.document.blocks.as_slice(),
            ),
            Product::Fragment { facts, .. } => (
                "fragment",
                format!("fragment:{}:{}", facts.record.reference, facts.offset),
                Some((facts.offset, facts.end.max(facts.offset))),
                None,
                &[][..],
            ),
            Product::Event { id, facts, .. } => (
                "event",
                format!("event:{id}"),
                None,
                facts.session.as_ref().map(ToString::to_string),
                &[][..],
            ),
            Product::Link { facts, .. } => (
                "link",
                facts.selection_tag(),
                None,
                Some(facts.child.to_string()),
                &[][..],
            ),
            Product::PrefixCopy { facts, .. } => (
                "prefix_copy",
                format!("prefix:{}:{}", facts.parent, facts.child),
                None,
                Some(facts.child.to_string()),
                &[][..],
            ),
            Product::PrefixMember { facts, .. } => (
                "prefix_member",
                format!("member:{}:{}:{}", facts.parent, facts.child, facts.message),
                None,
                Some(facts.child.to_string()),
                &[][..],
            ),
            Product::Background { facts, .. } => (
                "background",
                format!("background:{}:{}", facts.session, facts.call_id),
                None,
                Some(facts.session.to_string()),
                &[][..],
            ),
            Product::Repository { facts, .. } => (
                "repository",
                format!("repository:{}", facts.path),
                None,
                None,
                &[][..],
            ),
            Product::ParseProgress { .. } => (
                "parse_progress",
                format!("progress:{}:{}", reading.start, reading.parsed_upto),
                None,
                None,
                &[][..],
            ),
            _ => unreachable!("all source interpretations listed above"),
        };
        let mut value = serde_json::to_value(product)?;
        value
            .as_object_mut()
            .expect("tagged product object")
            .remove("reading");
        let value_hash = BlobHash::of(&serde_json::to_vec(&value)?);
        let mut blobs = std::collections::BTreeSet::new();
        for block in blocks {
            let attachments = match block {
                polylogue_model::Block::File(value) | polylogue_model::Block::Image(value) => {
                    std::slice::from_ref(value)
                }
                polylogue_model::Block::ToolResult { attachments, .. } => attachments,
                _ => &[],
            };
            blobs.extend(attachments.iter().filter_map(|value| value.blob));
        }
        let (start, end) = range.unwrap_or((reading.start, reading.parsed_upto));
        Ok(Some(Self {
            family: family.into(),
            reference,
            value_hash,
            start,
            end,
            session,
            blobs: blobs.into_iter().collect(),
        }))
    }
}

/// A covered interpretation's absent outputs stay absent. Otherwise a cold
/// reconstruction would republish a reading explicitly archived or erased.
pub(crate) fn allowed(
    index: &Connection,
    reading: &Reading,
    record: &SourceRecord,
) -> Result<bool> {
    let (known, live): (bool,bool) = index.query_row(
        "SELECT count(*)>0,coalesce(max(state='live'),0) FROM primary_fact WHERE input=?1 AND chain=?2 AND reading_version=?3 AND reference=?4 AND value_hash=?5",
        params![reading.input.id,reading.chain,reading.reading_version,record.reference,record.value_hash.0.as_slice()], |row| Ok((row.get(0)?,row.get(1)?)),
    )?;
    if known {
        return Ok(live);
    }
    Ok(!index.query_row(
        "SELECT EXISTS(SELECT 1 FROM primary_fact WHERE kind='reading_progress' AND input=?1 AND chain=?2 AND reading_version=?3 AND step_start<=?4 AND step_end>=?5)",
        params![reading.input.id,reading.chain,reading.reading_version,record.start,record.end], |row| row.get::<_,bool>(0),
    )?)
}

fn retain<T>(values: &mut Vec<T>, mask: &[bool]) {
    let mut index = 0;
    values.retain(|_| {
        let keep = mask[index];
        index += 1;
        keep
    });
}
fn reindex<T>(values: &mut std::collections::BTreeMap<usize, T>, mask: &[bool]) {
    let mut positions = vec![None; mask.len()];
    let mut next = 0;
    for (index, keep) in mask.iter().copied().enumerate() {
        if keep {
            positions[index] = Some(next);
            next += 1;
        }
    }
    *values = std::mem::take(values)
        .into_iter()
        .filter_map(|(index, value)| {
            positions
                .get(index)
                .copied()
                .flatten()
                .map(|index| (index, value))
        })
        .collect();
}

/// Keep the parser's prepared values and stable event identities; only native
/// eligibility decides which source contributions reach the ordinary writer.
pub(crate) fn retain_source_records(
    index: &Connection,
    input: &InputKey,
    chain: i64,
    length: u64,
    step: &mut crate::ParseStep,
) -> Result<()> {
    let products = super::parse(input, chain, Some(length), step)?;
    let mut masks = std::collections::BTreeMap::<String, Vec<bool>>::new();
    for product in products {
        let reading = product.reading().expect("parser fact has source");
        let record = SourceRecord::of(&product)?.expect("parser fact has identity");
        let keep = allowed(index, reading, &record)?;
        if record.family == "parse_progress" {
            step.source_prefix_missing |= !keep;
        }
        masks.entry(record.family).or_default().push(keep);
    }
    let all = masks.values().all(|mask| mask.iter().all(|keep| *keep));
    if all {
        return Ok(());
    }
    let event_ids = step.event_identities(&input.id, chain)?;
    let output = &mut step.output;
    let mask = |family: &str| masks.get(family).map(Vec::as_slice).unwrap_or(&[]);
    retain(&mut output.sessions, mask("session"));
    reindex(&mut output.session_receipts, mask("session"));
    let messages = mask("message");
    retain(&mut output.messages, messages);
    retain(&mut step.prepared.messages, messages);
    reindex(&mut output.message_order, messages);
    reindex(&mut output.message_projections, messages);
    reindex(&mut output.message_receipts, messages);
    reindex(&mut output.native_hook_fields, messages);
    reindex(&mut output.declared_injections, messages);
    reindex(&mut output.declared_attachment_bindings, messages);
    let documents = mask("document");
    retain(&mut output.documents, documents);
    retain(&mut step.prepared.documents, documents);
    reindex(&mut output.document_receipts, documents);
    let fragments = mask("fragment");
    retain(&mut output.fragments, fragments);
    retain(&mut step.prepared.fragments, fragments);
    reindex(&mut output.fragment_receipts, fragments);
    let events = mask("event");
    retain(&mut output.events, events);
    retain(&mut step.prepared.events, events);
    output.event_ids = event_ids.into_iter().enumerate().collect();
    reindex(&mut output.event_ids, events);
    reindex(&mut output.event_receipts, events);
    reindex(&mut output.imported_events, events);
    let links = mask("link");
    retain(&mut output.links, links);
    reindex(&mut output.link_receipts, links);
    let copies = mask("prefix_copy");
    retain(&mut output.prefix_copies, copies);
    reindex(&mut output.prefix_copy_receipts, copies);
    let members = mask("prefix_member");
    retain(&mut output.prefix_members, members);
    reindex(&mut output.prefix_member_receipts, members);
    retain(&mut output.background, mask("background"));
    retain(&mut output.repo_lookups, mask("repository"));
    step.prepared.validate(output).map_err(StoreError::from)
}
