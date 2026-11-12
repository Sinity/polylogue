//! Native reading eligibility never supplies the local model's values.
//! The original parser reproduces each value before its native identity is used.
use super::*;
use crate::Result;
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
                format!(
                    "background:{}",
                    serde_json::to_string(&(&facts.session, &facts.call_id, &facts.task_id))?
                ),
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
