//! Text documents: one shown version each, with source time before route
//! precedence, and reads in the reader's read transaction.
use crate::ConnExt as _;
use crate::ResultExt as _;
use crate::encoding::{restore_text, unpack_block, unpack_json};
use crate::{Reader, Result};
use polylogue_model::hex;
use polylogue_model::{Block, Document, DocumentId, Json};
use rusqlite::{OptionalExtension, params};
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

impl<'a> Reader<'a> {
    pub fn select_document<'r>(&'r self, id: &DocumentId) -> Result<Option<DocumentRead<'r, 'a>>> {
        let row: Option<(i64, Vec<u8>, Vec<u8>)> = self
            .conn()?
            .query_row(
                "SELECT pk,version_hash,fields FROM document WHERE id=?1",
                [id.to_string()],
                |r| <(_, _, _)>::try_from(r),
            )
            .optional()?;
        row.map(|(pk, hash, fields)| {
            Ok(DocumentRead {
                reader: self,
                pk,
                fields: unpack_json(&fields)?,
                hash,
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
            "SELECT EXISTS(SELECT 1 FROM document_block WHERE document_pk=?1 AND idx=?2)",
            params![self.pk, ordinal],
        )
    }

    pub fn sources(&self) -> Result<Vec<DocumentSource>> {
        let mut statement=self.reader.conn()?.prepare("SELECT r.id,s.chain,s.source_start,s.source_end,s.hash,(s.input_pk=d.input_pk AND s.chain=d.source_chain) FROM document d JOIN document_parse s ON s.document_id=d.id JOIN input_ref r ON r.pk=s.input_pk WHERE d.pk=?1 ORDER BY r.id,s.chain,s.source_start,s.hash")?;
        let mut rows = statement.query([self.pk])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let input: String = row.get(0)?;
            let locations=self.reader.conn()?.prepare("SELECT machine,path,seen_at FROM source_observation WHERE input=?1 ORDER BY seen_at,machine,path")?.query_map([&input],|r|Ok(serde_json::json!({"machine":r.get::<_,String>(0)?,"path":r.get::<_,String>(1)?,"seen_at_ms":r.get::<_,i64>(2)?})))?.collect::<rusqlite::Result<Vec<_>>>()?;
            out.push(DocumentSource {
                input,
                chain: row.get(1)?,
                start: row.get(2)?,
                end: row.get(3)?,
                version: hex(&row.get::<_, Vec<u8>>(4)?),
                shown: row.get(5)?,
                locations,
            });
        }
        Ok(out)
    }
}

fn decoded_block(row: &rusqlite::Row<'_>) -> Result<Block> {
    let mut block = unpack_block(&row.get::<_, Vec<u8>>(0)?)?;
    if let Some(text) = crate::text_rows::column(row, 1)? {
        restore_text(&mut block, text);
    }
    if let Some(data) = row.get::<_, Option<Vec<u8>>>(4)? {
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
    let bound = if end.is_some() { "AND b.idx<?3" } else { "" };
    let kinds = if text_only {
        "AND b.kind IN ('text','thinking','tool_result')"
    } else {
        ""
    };
    let data = if text_only { "NULL" } else { "b.data" };
    let order = if backwards { "DESC" } else { "ASC" };
    let sql = format!(
        "SELECT b.body,g.body,t.start,t.length,{data},b.idx FROM document_block b LEFT JOIN text t ON t.id=b.text_id LEFT JOIN content_group g ON g.id=t.group_id WHERE b.document_pk=?1 AND b.idx>=?2 {bound} {kinds} ORDER BY b.idx {order}"
    );
    let mut statement = conn.prepare_cached(&sql)?;
    let mut rows = if let Some(end) = end {
        statement.query(params![pk, from, end])?
    } else {
        statement.query(params![pk, from])?
    };
    while let Some(row) = rows.next()? {
        if let Some(reader) = reader {
            reader.check()?;
        }
        if let Some(limit) = crate::read::limit::ReadLimit::current() {
            limit.check()?;
        }
        let ordinal =
            u64::try_from(row.get::<_, i64>(5)?).or_invalid("negative document block ordinal")?;
        if visit(ordinal, decoded_block(row)?).is_break() {
            break;
        }
    }
    Ok(())
}
