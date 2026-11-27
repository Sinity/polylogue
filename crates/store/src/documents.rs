//! Text documents: one shown version each, with source time before route
//! precedence, and reads in the reader's read transaction.
use crate::ConnExt as _;
use crate::encoding::{restore_text, unpack_block, unpack_json};
use crate::row::{Rows as _, Statement, row};
use crate::{Reader, Result};
use polylogue_model::hex;
use polylogue_model::{Block, Document, DocumentId, Json};
use rusqlite::params;
use serde::Serialize;

use std::ops::ControlFlow;

/// A document selection belongs to its Reader; budgeted consumers decode blocks
/// in order and can stop before later blocks. No input is reopened.
pub struct DocumentRead<'r, 'a> {
    reader: &'r Reader<'a>,
    pk: i64,
    fields: Document,
    hash: Vec<u8>,
}

row! {
    /// A shown document's key, version and packed fields.
    struct DocumentHead {
        pk: i64,
        version_hash: Vec<u8>,
        fields: Vec<u8>,
    }
    statements {
        const OF_ID = "SELECT pk, version_hash, fields FROM document WHERE id=?1";
    }
}

row! {
    /// A reading of a document: its input, byte range and version, and
    /// whether it is the shown one.
    struct DocumentReading {
        input: String,
        chain: i64,
        source_start: i64,
        source_end: i64,
        hash: Vec<u8>,
        shown: bool,
    }
    statements {
        const OF_DOCUMENT = "SELECT r.id AS input, s.chain, s.source_start, s.source_start + s.source_length AS source_end, s.hash,
            (s.input_pk=d.input_pk AND s.chain=d.source_chain) AS shown
            FROM document d JOIN document_parse s ON s.document_id=d.id JOIN input_ref r ON r.pk=s.input_pk
            WHERE d.pk=?1 ORDER BY r.id, s.chain, s.source_start, s.hash";
    }
}

row! {
    /// Where and when a machine saw an input.
    struct InputLocation {
        machine: String,
        path_bytes: Vec<u8>,
        seen_at: i64,
    }
    statements {
        const OF_INPUT = "SELECT machine, path_bytes, seen_at FROM source_observation WHERE input=?1
            ORDER BY seen_at, machine, path_bytes";
    }
}

row! {
    /// A document block's packed body, its text's group body and place, its
    /// packed data and its position.
    struct DocumentBlockRow {
        body: Vec<u8>,
        group_body: Option<Vec<u8>>,
        start: Option<i64>,
        length: Option<i64>,
        data: Option<Vec<u8>>,
        position: u64,
    }
}

impl<'a> Reader<'a> {
    pub fn select_document<'r>(&'r self, id: &DocumentId) -> Result<Option<DocumentRead<'r, 'a>>> {
        self.conn()?
            .maybe(&DocumentHead::OF_ID, [id.to_string()])?
            .map(|head| {
                Ok(DocumentRead {
                    reader: self,
                    pk: head.pk,
                    fields: unpack_json(&head.fields)?,
                    hash: head.version_hash,
                })
            })
            .transpose()
    }

    pub fn document(&self, id: &DocumentId) -> Result<Option<Document>> {
        self.select_document(id)?
            .map(|selected| selected.document())
            .transpose()
    }
}

impl DocumentRead<'_, '_> {
    pub fn id(&self) -> &DocumentId {
        &self.fields.id
    }
    pub fn title(&self) -> Option<&str> {
        self.fields.title.as_deref()
    }
    pub fn purpose(&self) -> polylogue_model::DocumentPurpose {
        self.fields.purpose
    }
    pub fn version(&self) -> String {
        hex(&self.hash)
    }
    pub fn document(&self) -> Result<Document> {
        let mut document = self.fields.clone();
        self.visit_blocks(crate::read::VisitFrom::First, |_, block| {
            document.blocks.push(block);
            ControlFlow::Continue(())
        })?;
        Ok(document)
    }
    /// Original source ordinals stay attached to blocks in either direction.
    /// Returning Break prevents the next body from being decoded.
    pub fn visit_blocks(
        &self,
        from: crate::read::VisitFrom<u64>,
        visit: impl FnMut(u64, Block) -> ControlFlow<()>,
    ) -> Result<()> {
        use crate::read::VisitFrom;
        let (start, end, backwards) = match from {
            VisitFrom::First => (0, None, false),
            VisitFrom::At(at) => (at, None, false),
            VisitFrom::After(after) => {
                let Some(start) = after.checked_add(1) else {
                    return Ok(());
                };
                (start, None, false)
            }
            VisitFrom::Before(before) => (0, usize::try_from(before).ok(), true),
            VisitFrom::Last => (0, None, true),
        };
        visit_blocks_selected_from(
            (self.reader.conn()?, Some(self.reader)),
            self.pk,
            start,
            end,
            false,
            backwards,
            visit,
        )
    }

    pub fn has_block(&self, ordinal: u64) -> Result<bool> {
        let Ok(ordinal) = i64::try_from(ordinal) else {
            return Ok(false);
        };
        self.reader.conn()?.exists(
            "SELECT EXISTS(SELECT 1 FROM document_block WHERE document_pk=?1 AND position=?2)",
            params![self.pk, ordinal],
        )
    }

    pub fn sources(&self) -> Result<Vec<DocumentSource>> {
        let conn = self.reader.conn()?;
        let mut out = Vec::new();
        for reading in conn.all(&DocumentReading::OF_DOCUMENT, [self.pk])? {
            let locations = conn
                .all(&InputLocation::OF_INPUT, [&reading.input])?
                .into_iter()
                .map(|seen| {
                    serde_json::json!({"machine": seen.machine, "path": String::from_utf8_lossy(&seen.path_bytes), "seen_at_ms": seen.seen_at})
                })
                .collect();
            out.push(DocumentSource {
                input: reading.input,
                chain: reading.chain,
                start: reading.source_start,
                end: reading.source_end,
                version: hex(&reading.hash),
                shown: reading.shown,
                locations,
            });
        }
        Ok(out)
    }
}

fn decoded_block(row: DocumentBlockRow) -> Result<Block> {
    let mut block = unpack_block(&row.body)?;
    if let Some(body) = row.group_body {
        let text = crate::text_rows::slice(
            &body,
            row.start.unwrap_or_default(),
            row.length.unwrap_or_default(),
        )?;
        restore_text(&mut block, text);
    }
    if let Some(data) = row.data {
        crate::encoding::set_data(&mut block, unpack_json(&data)?);
    }
    Ok(block)
}

#[derive(Debug, Clone, Serialize)]
pub struct DocumentSource {
    pub input: String,
    pub chain: i64,
    pub start: i64,
    pub end: i64,
    pub version: String,
    pub shown: bool,
    pub locations: Vec<Json>,
}

/// A borrowed view used by the source text walker during preparation.
/// DocumentRead and this view share the same indexed block codec.
pub(crate) struct TextRead<'a> {
    conn: &'a rusqlite::Connection,
    pk: i64,
}
pub(crate) fn text_read<'a>(
    conn: &'a rusqlite::Connection,
    id: &DocumentId,
) -> Result<Option<TextRead<'a>>> {
    let pk =
        conn.scalar_optional::<i64>("SELECT pk FROM document WHERE id=?1", [id.to_string()])?;
    Ok(pk.map(|pk| TextRead { conn, pk }))
}
impl TextRead<'_> {
    pub(crate) fn visit_text_blocks_from(
        &self,
        from: u64,
        visit: impl FnMut(u64, Block) -> ControlFlow<()>,
    ) -> Result<()> {
        visit_blocks_selected_from((self.conn, None), self.pk, from, None, true, false, visit)
    }
}

fn visit_blocks_selected_from(
    (conn, reader): (&rusqlite::Connection, Option<&Reader<'_>>),
    pk: i64,
    from: u64,
    end: Option<usize>,
    text_only: bool,
    backwards: bool,
    mut visit: impl FnMut(u64, Block) -> ControlFlow<()>,
) -> Result<()> {
    let Ok(from) = i64::try_from(from) else {
        return Ok(());
    };
    // Every stored ordinal is a nonnegative SQLite integer. A larger
    // upper bound includes every row rather than wrapping or refusing it.
    let end = end.and_then(|end| i64::try_from(end).ok());
    let bound = if end.is_some() {
        "AND b.position<?3"
    } else {
        ""
    };
    let kinds = if text_only {
        "AND b.kind IN ('text','thinking','tool_result')"
    } else {
        ""
    };
    let data = if text_only { "NULL" } else { "b.data" };
    let order = if backwards { "DESC" } else { "ASC" };
    let statement = Statement::<DocumentBlockRow>::built(format!(
        "SELECT b.body, g.body AS group_body, t.start, t.length, {data} AS data, b.position FROM document_block b LEFT JOIN text t ON t.id=b.text_id LEFT JOIN content_group g ON g.id=t.group_id WHERE b.document_pk=?1 AND b.position>=?2 {bound} {kinds} ORDER BY b.position {order}"
    ));
    let mut each = |row: DocumentBlockRow| {
        if let Some(reader) = reader {
            reader.check()?;
        }
        if let Some(limit) = crate::read::limit::ReadLimit::current() {
            limit.check()?;
        }
        let ordinal = row.position;
        Ok(visit(ordinal, decoded_block(row)?))
    };
    let _ = match end {
        Some(end) => conn.each(&statement, params![pk, from, end], &mut each)?,
        None => conn.each(&statement, params![pk, from], &mut each)?,
    };
    Ok(())
}

/// The rows this module declares statements for, for the every-statement test.
#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<DocumentHead>();
    check.row::<DocumentReading>();
    check.row::<InputLocation>();
}
