//! Text documents: one shown version each, with source time before arrival
//! rank, and reads in the reader's read transaction.
use crate::encoding::hex;
use crate::encoding::{
    PreparedDocument, block_text, restore_text, unpack_block, unpack_json, unpack_text,
};
use crate::{Reader, Result, StoreError};
use polylogue_model::{Block, Document, DocumentId, DocumentRow, Json};
use rusqlite::{OptionalExtension, Transaction, params};
use serde::Serialize;
use std::collections::HashMap;
use std::ops::ControlFlow;

pub(crate) struct Source<'a> {
    pub input_pk: i64,
    pub chain: i64,
    /// Later records of the chain restate earlier ones.
    pub repeats_whole_conversations: bool,
    pub rank: i64,
    pub counter: i64,
    pub packed: &'a HashMap<[u8; 32], Vec<u8>>,
    /// Set during a first build ([`crate::first_build`]).
    pub first_build: Option<&'a crate::first_build::FirstBuild>,
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
    let existing: Option<i64> = tx
        .query_row(
            "SELECT pk FROM document WHERE id=?1",
            [doc.id.to_string()],
            |r| r.get(0),
        )
        .optional()?;
    if let Some(pk) = existing {
        let prior:Option<i64>=tx.query_row("SELECT source_start FROM document_source WHERE document_pk=?1 AND input_pk=?2 AND chain=?3",params![pk,source.input_pk,source.chain],|r|r.get(0)).optional()?;
        if prior.is_some_and(|start| {
            if source.repeats_whole_conversations {
                (row.offset as i64) < start
            } else {
                (row.offset as i64) > start
            }
        }) {
            return Ok(());
        }
        let earlier:Option<(Option<i64>,i64,Vec<u8>)>=tx.query_row(
            "SELECT source_time,arrival_rank,version_hash FROM document_source WHERE document_pk=?1 AND input_pk=?2 AND chain=?3",
            params![pk,source.input_pk,source.chain],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?;
        if earlier.is_some_and(|(old_time, old_rank, old_hash)| {
            (
                time,
                std::cmp::Reverse(source.rank),
                prepared.hash.as_slice(),
            ) < (old_time, std::cmp::Reverse(old_rank), old_hash.as_slice())
        }) {
            return Ok(());
        }
    }
    crate::versions::store(
        tx,
        "document",
        &prepared.hash,
        &prepared.body,
        &doc.blocks,
        source.packed,
        source.first_build,
    )?;
    let pk = match existing {
        Some(pk) => pk,
        None => {
            tx.execute("INSERT INTO document(id,origin,native_id,purpose,title,source_time,authorship,version_hash,input_pk,source_chain,source_start,source_end,changed,fields)
                VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)",params![doc.id.to_string(),doc.id.origin().as_str(),doc.id.native().as_str(),doc.purpose.as_str(),doc.title,time,doc.authorship.as_str(),&prepared.hash[..],source.input_pk,source.chain,row.offset as i64,row.end as i64,source.counter,prepared.fields])?;
            tx.last_insert_rowid()
        }
    };
    set_source(tx, pk, source, row, time, &prepared.hash, false)?;
    refresh(tx, pk, source.counter, source.first_build)
}

pub(crate) fn refresh(
    tx: &Transaction<'_>,
    pk: i64,
    counter: i64,
    first_build: Option<&crate::first_build::FirstBuild>,
) -> Result<()> {
    // Source time is authoritative for documents, including within one input.
    let winner=tx.query_row(
        "SELECT s.input_pk,i.id,s.chain,s.source_start,s.source_end,s.arrival_rank,s.version_hash
         FROM document_source s JOIN input_ref i ON i.pk=s.input_pk WHERE s.document_pk=?1
         ORDER BY s.source_time IS NULL,s.source_time DESC,s.arrival_rank,s.version_hash DESC,i.id,s.chain DESC LIMIT 1",
         [pk],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,String>(1)?,r.get::<_,i64>(2)?,r.get::<_,i64>(3)?,r.get::<_,i64>(4)?,r.get::<_,i64>(5)?,r.get::<_,Vec<u8>>(6)?))).optional()?;
    let Some((input_pk, _input_id, chain, start, end, rank, hash)) = winner else {
        let texts = tx
            .prepare(
                "SELECT text_id FROM document_block WHERE document_pk=?1 AND text_id IS NOT NULL",
            )?
            .query_map([pk], |r| r.get::<_, i64>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        tx.execute("DELETE FROM document WHERE pk=?1", [pk])?;
        for text in texts {
            crate::text_rows::unindex_unused(tx, text, first_build.is_some())?;
        }
        return Ok(());
    };
    let document: Document = crate::versions::read(tx, "document", &hash)?;
    let prepared = PreparedDocument::new(&document)?;
    let packed = HashMap::new();
    let source = Source {
        input_pk,
        chain,
        repeats_whole_conversations: false,
        rank,
        counter,
        packed: &packed,
        first_build,
    };
    let row = DocumentRow {
        offset: start as u64,
        end: end as u64,
        document,
    };
    show(tx, pk, &source, &row, &prepared)?;
    tx.execute(
        "UPDATE document_source SET shown=(input_pk=?2 AND chain=?3) WHERE document_pk=?1",
        params![pk, input_pk, chain],
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
        let text = block_text(block)
            .map(|text| {
                crate::text_rows::intern(tx, text, 1, Some(source.packed), source.first_build)
            })
            .transpose()?;
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
    pub fn visit_blocks(&self, mut visit: impl FnMut(Block) -> ControlFlow<()>) -> Result<()> {
        let mut statement=self.reader.conn()?.prepare("SELECT b.body,t.body,b.data FROM document_block b LEFT JOIN text t ON t.id=b.text_id WHERE b.document_pk=?1 ORDER BY b.idx")?;
        let mut rows = statement.query([self.pk])?;
        while let Some(row) = rows.next()? {
            let mut block = unpack_block(&row.get::<_, Vec<u8>>(0)?)?;
            if let Some(text) = row.get::<_, Option<Vec<u8>>>(1)? {
                restore_text(&mut block, unpack_text(&text)?);
            }
            if let Some(data) = row.get::<_, Option<Vec<u8>>>(2)? {
                crate::encoding::restore_data(&mut block, &data)?;
            }
            if visit(block).is_break() {
                break;
            }
        }
        Ok(())
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
