//! Text documents: one shown version each, with source time before arrival
//! rank, and reads in the reader's read transaction.
use crate::encoding::hex;
use crate::encoding::{PreparedDocument, restore_text, unpack_block, unpack_json, unpack_text};
use crate::versions::VersionKind;
use crate::{Reader, Result, StoreError};
use polylogue_model::{Block, Document, DocumentId, DocumentRow, Json};
use rusqlite::{OptionalExtension, Transaction, params};
use serde::Serialize;
use std::collections::HashMap;
use std::ops::ControlFlow;

pub(crate) struct Source<'a> {
    pub input_pk: i64,
    pub chain: i64,
    pub rank: i64,
    pub counter: i64,
    pub packed: &'a HashMap<[u8; 32], Vec<u8>>,
    /// Set during a first build ([`crate::first_build`]).
    pub first_build: Option<&'a crate::first_build::FirstBuild>,
    pub keys: Option<&'a crate::writer::resolve::ResolvedKeys>,
}

pub(crate) fn write(
    tx: &Transaction<'_>,
    source: &Source<'_>,
    row: &DocumentRow,
    prepared: &PreparedDocument,
) -> Result<()> {
    if prepared.id != row.document.id || prepared.blocks.len() != row.document.blocks.len() {
        return Err(StoreError::Invalid(
            "prepared Document differs from its source row".into(),
        ));
    }
    let doc = &row.document;
    let time = doc.source_time.map(|t| t.unix_ms());
    let keys = source
        .keys
        .ok_or_else(|| StoreError::Invalid("document reading has no resolved key".into()))?;
    let reserved = keys.get(
        crate::writer::resolve::Kind::Document,
        doc.id.to_string().as_bytes(),
    )?;
    let existing = tx
        .prepare_cached("SELECT pk FROM document WHERE pk=?1")?
        .query_row([reserved.pk], |row| row.get::<_, i64>(0))
        .optional()?;
    crate::versions::store(
        tx,
        VersionKind::Document,
        &prepared.hash,
        &prepared.body,
        &doc.blocks,
        &prepared.texts,
        crate::versions::TextRows {
            packed: source.packed,
            first_build: source.first_build,
            keys,
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
    set_source(tx, pk, source, row, time, &prepared.hash, false)?;
    Ok(())
}

pub(crate) fn apply_choice(
    tx: &Transaction<'_>,
    pk: i64,
    choice: Option<&crate::writer::resolve::versions::DocumentChoice>,
    counter: i64,
    first_build: Option<&crate::first_build::FirstBuild>,
    keys: &crate::writer::resolve::ResolvedKeys,
) -> Result<()> {
    let Some(choice) = choice else {
        let texts=tx.prepare("SELECT DISTINCT text_id FROM document_block WHERE document_pk=?1 AND text_id IS NOT NULL")?.query_map([pk],|r|r.get::<_,i64>(0))?.collect::<rusqlite::Result<Vec<_>>>()?;
        tx.execute("DELETE FROM document WHERE pk=?1", [pk])?;
        for text in texts {
            crate::text_rows::unindex_unused(tx, text, first_build.is_some())?;
        }
        return Ok(());
    };
    let packed = HashMap::new();
    let source = Source {
        input_pk: choice.input,
        chain: choice.chain,
        rank: choice.rank,
        counter,
        packed: &packed,
        first_build,
        keys: Some(keys),
    };
    if choice.changed {
        let document = choice.document.as_ref().ok_or_else(|| {
            StoreError::Invalid("a changed document has no prepared value".into())
        })?;
        let prepared = choice
            .prepared
            .as_ref()
            .ok_or_else(|| StoreError::Invalid("a changed document has no prepared body".into()))?;
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
    hash: &[u8; 32],
    shown: bool,
) -> Result<()> {
    tx.execute("INSERT INTO document_source(document_pk,input_pk,chain,source_start,source_end,arrival_rank,source_time,version_hash,shown)
        VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)
        ON CONFLICT(document_pk,input_pk,chain) DO UPDATE SET source_start=excluded.source_start,source_end=excluded.source_end,
            arrival_rank=excluded.arrival_rank,source_time=excluded.source_time,version_hash=excluded.version_hash,shown=excluded.shown",
        params![pk,source.input_pk,source.chain,row.offset as i64,row.end as i64,source.rank,time,&hash[..],shown])?;
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
        crate::text_rows::unindex_unused(tx, text, source.first_build.is_some())?;
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
    for (idx, ((block, body), data)) in document
        .blocks
        .iter()
        .zip(&prepared.blocks)
        .zip(&prepared.data)
        .enumerate()
    {
        let keys = source.keys.ok_or_else(|| {
            StoreError::Invalid("document blocks have no resolved text keys".into())
        })?;
        let text = prepared.texts[idx]
            .as_ref()
            .map(|hash| {
                keys.get(crate::writer::resolve::Kind::Text, hash)
                    .map(|key| key.pk)
            })
            .transpose()?;
        if let Some(id) = text {
            crate::text_rows::activate(tx, id, 1, source.first_build.is_some())?;
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
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
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
        self.visit_blocks(|block| {
            document.blocks.push(block);
            ControlFlow::Continue(())
        })?;
        Ok(document)
    }
    pub fn visit_blocks(&self, visit: impl FnMut(Block) -> ControlFlow<()>) -> Result<()> {
        self.visit_blocks_until(None, visit)
    }

    fn visit_blocks_until(
        &self,
        end: Option<usize>,
        mut visit: impl FnMut(Block) -> ControlFlow<()>,
    ) -> Result<()> {
        let bound = if end.is_some() { "AND b.idx<?2" } else { "" };
        let sql = format!(
            "SELECT b.body,t.body,b.data FROM document_block b LEFT JOIN text t ON t.id=b.text_id WHERE b.document_pk=?1 {bound} ORDER BY b.idx"
        );
        let mut statement = self.reader.conn()?.prepare_cached(&sql)?;
        let mut rows = if let Some(end) = end {
            statement.query(params![self.pk, end as i64])?
        } else {
            statement.query([self.pk])?
        };
        while let Some(row) = rows.next()? {
            if visit(decoded_block(row)?).is_break() {
                break;
            }
        }
        Ok(())
    }

    pub fn visit_blocks_before(
        &self,
        end: usize,
        visit: impl FnMut(Block) -> ControlFlow<()>,
    ) -> Result<()> {
        self.visit_blocks_until(Some(end), visit)
    }

    /// Seek to the selected source block immediately before `before`.
    pub fn previous_block(
        &self,
        before: usize,
        kinds: &[polylogue_model::BlockKind],
    ) -> Result<Option<usize>> {
        let kinds = kinds
            .iter()
            .map(|kind| format!("'{}'", kind.as_str()))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT idx FROM document_block WHERE document_pk=?1 AND idx<?2 AND kind IN ({kinds}) ORDER BY idx DESC LIMIT 1"
        );
        let index = self
            .reader
            .conn()?
            .prepare_cached(&sql)?
            .query_row(params![self.pk, before as i64], |row| row.get::<_, i64>(0))
            .optional()?;
        index
            .map(|index| {
                usize::try_from(index).map_err(|_| {
                    StoreError::Invalid("the document block index is outside address space".into())
                })
            })
            .transpose()
    }
    pub fn sources(&self) -> Result<Vec<DocumentSource>> {
        let mut statement=self.reader.conn()?.prepare("SELECT r.id,s.chain,s.source_start,s.source_end,s.version_hash,s.shown FROM document_source s JOIN input_ref r ON r.pk=s.input_pk WHERE s.document_pk=?1 ORDER BY r.id,s.chain,s.source_start,s.version_hash")?;
        let mut rows = statement.query([self.pk])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let input: String = row.get(0)?;
            let locations=self.reader.conn()?.prepare("SELECT machine,path,seen_at FROM a.observation WHERE input=?1 ORDER BY seen_at,machine,path")?.query_map([&input],|r|Ok(serde_json::json!({"machine":r.get::<_,String>(0)?,"path":r.get::<_,String>(1)?,"seen_at_ms":r.get::<_,i64>(2)?})))?.collect::<rusqlite::Result<Vec<_>>>()?;
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
    if let Some(text) = row.get::<_, Option<Vec<u8>>>(1)? {
        restore_text(&mut block, unpack_text(&text)?);
    }
    if let Some(data) = row.get::<_, Option<Vec<u8>>>(2)? {
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
