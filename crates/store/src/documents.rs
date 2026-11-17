//! Text documents: one shown version each, with source time before route
//! precedence, and reads in the reader's read transaction.
use crate::ConnExt as _;
use crate::encoding::{PreparedDocument, restore_text, unpack_block, unpack_json};
use crate::{OptionExt as _, ResultExt as _};
use crate::{Reader, Result, StoreError};
use polylogue_model::hex;
use polylogue_model::{Block, Document, DocumentId, DocumentRow, Json};
use rusqlite::{OptionalExtension, Transaction, params};
use serde::Serialize;

use std::ops::ControlFlow;

pub(crate) struct Source<'a> {
    pub input_pk: i64,
    pub chain: i64,
    pub route: polylogue_model::Route,
    pub counter: i64,
    pub packed: &'a crate::text_rows::PreparedTexts,
    pub texts: &'a crate::text_rows::TextIds,
    /// The writer's index upkeep for this transaction.
    pub upkeep: &'a crate::writer::resolve::IndexUpkeep,
    pub keys: &'a crate::writer::resolve::ResolvedKeys,
}

pub(crate) fn write(
    tx: &Transaction<'_>,
    source: &Source<'_>,
    row: &DocumentRow,
    prepared: &PreparedDocument,
    committed: Option<i64>,
) -> Result<()> {
    if prepared.id != row.document.id || prepared.blocks.len() != row.document.blocks.len() {
        return Err(StoreError::Invalid(
            "prepared Document differs from its source row".into(),
        ));
    }
    let doc = &row.document;
    let time = doc.source_time.map(|t| t.unix_ms());
    let keys = source.keys;
    let reserved = keys.get(
        crate::writer::resolve::Kind::Document,
        doc.id.to_string().as_bytes(),
    )?;
    let existing = tx
        .prepare_cached("SELECT pk FROM document WHERE pk=?1")?
        .query_row([reserved.pk], |row| row.get::<_, i64>(0))
        .optional()?;
    crate::versions::store_document(
        tx,
        &prepared.hash,
        &prepared.body,
        &doc.blocks,
        &prepared.texts,
        crate::versions::TextRows {
            packed: source.packed,
            texts: source.texts,
            membership: None,
            upkeep: source.upkeep,
        },
    )?;
    let pk = match existing {
        Some(pk) => pk,
        None => {
            tx.execute("INSERT INTO document(pk,id,origin,native_id,purpose,title,source_time,authorship,version_hash,input_pk,source_chain,source_start,source_end,changed,fields)
                VALUES (?15,?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)",params![doc.id.to_string(),doc.id.origin().as_str(),doc.id.native().as_str(),doc.purpose.as_str(),doc.title,time,doc.authorship.as_str(),&prepared.hash[..],source.input_pk,source.chain,row.offset as i64,row.end as i64,source.counter,prepared.fields,Some(reserved.pk)])?;
            tx.last_insert_rowid()
        }
    };
    set_source(tx, pk, source, row, time, prepared, committed)?;
    Ok(())
}

pub(crate) fn apply_choice(
    tx: &Transaction<'_>,
    pk: i64,
    choice: Option<&crate::writer::resolve::membership::DocumentChoice>,
    counter: i64,
    upkeep: &crate::writer::resolve::IndexUpkeep,
    texts: &crate::text_rows::TextIds,
    keys: &crate::writer::resolve::ResolvedKeys,
) -> Result<()> {
    let Some(choice) = choice else {
        let old=tx.prepare("SELECT DISTINCT text_id FROM document_block WHERE document_pk=?1 AND text_id IS NOT NULL")?.query_map([pk],|r|r.get::<_,i64>(0))?.collect::<rusqlite::Result<Vec<_>>>()?;
        tx.execute("DELETE FROM document WHERE pk=?1", [pk])?;
        for text in old {
            crate::text_rows::unindex_unused(tx, texts, text, upkeep.at_build_end())?;
        }
        return Ok(());
    };
    let packed = crate::text_rows::PreparedTexts::default();
    let source = Source {
        input_pk: choice.input,
        chain: choice.chain,
        route: choice.route,
        counter,
        packed: &packed,
        upkeep,
        texts,
        keys,
    };
    if choice.changed {
        let document = choice.document.as_ref().ok_or_else(|| {
            StoreError::Invalid("a changed document has no prepared value".into())
        })?;
        let prepared = choice
            .prepared
            .as_ref()
            .or_invalid("a changed document has no prepared body")?;
        let row = DocumentRow {
            offset: choice.start as u64,
            end: choice.end as u64,
            document: document.clone(),
        };
        show(tx, pk, &source, &row, prepared)?;
    } else {
        tx.prepare_cached("UPDATE document SET input_pk=?2,source_chain=?3,source_start=?4,source_end=?5,changed=?6 WHERE pk=?1")?.execute(params![pk,choice.input,choice.chain,choice.start,choice.end,counter])?;
    }
    tx.execute(
        "UPDATE document_source SET shown=(input_pk=?2 AND chain=?3) WHERE document_pk=?1",
        params![pk, choice.input, choice.chain],
    )?;
    Ok(())
}

fn set_source(
    tx: &Transaction<'_>,
    pk: i64,
    source: &Source<'_>,
    row: &DocumentRow,
    time: Option<i64>,
    prepared: &PreparedDocument,
    committed: Option<i64>,
) -> Result<()> {
    tx.execute("INSERT INTO document_source(document_pk,input_pk,chain,source_start,source_end,route,source_time,version_hash,shown,time,committed)
        VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)
        ON CONFLICT(document_pk,input_pk,chain) DO UPDATE SET source_start=excluded.source_start,source_end=excluded.source_end,
            route=excluded.route,source_time=excluded.source_time,version_hash=excluded.version_hash,shown=excluded.shown,time=excluded.time,committed=excluded.committed",
        params![pk,source.input_pk,source.chain,row.offset as i64,row.end as i64,source.route.as_str(),time,&prepared.hash[..],false,prepared.time,committed])?;
    Ok(())
}

fn show(
    tx: &Transaction<'_>,
    pk: i64,
    source: &Source<'_>,
    row: &DocumentRow,
    prepared: &PreparedDocument,
) -> Result<()> {
    let document = &row.document;
    tx.execute("UPDATE document SET purpose=?2,title=?3,source_time=?4,authorship=?5,version_hash=?6,input_pk=?7,source_chain=?8,source_start=?9,source_end=?10,changed=?11,fields=?12 WHERE pk=?1",params![pk,document.purpose.as_str(),document.title,document.source_time.map(|t|t.unix_ms()),document.authorship.as_str(),&prepared.hash[..],source.input_pk,source.chain,row.offset as i64,row.end as i64,source.counter,prepared.fields])?;
    let old: Vec<i64> = tx
        .prepare("SELECT DISTINCT text_id FROM document_block WHERE document_pk=?1 AND text_id IS NOT NULL")?
        .query_map([pk], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    tx.execute("DELETE FROM document_block WHERE document_pk=?1", [pk])?;
    insert_blocks(tx, pk, source, document, prepared)?;
    for text in old {
        crate::text_rows::unindex_unused(tx, source.texts, text, source.upkeep.at_build_end())?;
    }
    Ok(())
}

fn insert_blocks(
    tx: &Transaction<'_>,
    pk: i64,
    source: &Source<'_>,
    document: &Document,
    prepared: &PreparedDocument,
) -> Result<()> {
    let stored_texts = crate::versions::document_text_ids(tx, &prepared.hash)?;
    for (idx, ((block, body), data)) in document
        .blocks
        .iter()
        .zip(&prepared.blocks)
        .zip(&prepared.data)
        .enumerate()
    {
        let text = prepared.texts[idx]
            .as_ref()
            .map(|_| {
                stored_texts.get(&idx).copied().ok_or_else(|| {
                    StoreError::Invalid("stored document text reference is missing".into())
                })
            })
            .transpose()?;
        if let Some(id) = text {
            crate::text_rows::activate(tx, source.texts, id, 1, source.upkeep.at_build_end())?;
        }
        let blob = match block {
            Block::Image(a) | Block::File(a) => a.blob.map(|h| h.0.to_vec()),
            _ => None,
        };
        tx.execute("INSERT INTO document_block(document_pk,idx,kind,body,text_id,blob_hash,data) VALUES (?1,?2,?3,?4,?5,?6,?7)",params![pk,idx as i64,block.kind().as_str(),body,text,blob,data])?;
    }
    Ok(())
}

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
    pub fn metadata(&self) -> Result<Json> {
        let mut value = serde_json::to_value(&self.fields)?;
        value
            .as_object_mut()
            .expect("Document is an object")
            .remove("blocks");
        Ok(value)
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
        let mut statement=self.reader.conn()?.prepare("SELECT r.id,s.chain,s.source_start,s.source_end,s.version_hash,s.shown FROM document_source s JOIN input_ref r ON r.pk=s.input_pk WHERE s.document_pk=?1 ORDER BY r.id,s.chain,s.source_start,s.version_hash")?;
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
        crate::encoding::restore_data(&mut block, &data)?;
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
        "SELECT b.body,g.body,t.start,t.length,{data},b.idx FROM document_block b LEFT JOIN text t ON t.id=b.text_id LEFT JOIN text_group g ON g.id=t.group_id WHERE b.document_pk=?1 AND b.idx>=?2 {bound} {kinds} ORDER BY b.idx {order}"
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
