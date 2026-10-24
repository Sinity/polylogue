//! Text documents, their retained versions and reads on the admitted Store view.
use crate::encoding::{PreparedDocument, restore_text, unpack_block, unpack_json, unpack_text};
use crate::{Reader, Result, StoreError};
use polylogue_model::{Block, Document, DocumentId, DocumentRow, Json};
use rusqlite::{OptionalExtension, Transaction, params};
use serde::Serialize;
use std::collections::HashMap;
use std::ops::ControlFlow;

pub(crate) struct Source<'a> {
    pub input_pk: i64,
    pub chain: i64,
    pub parser_version: i64,
    pub rank: i64,
    pub counter: i64,
    pub packed: &'a HashMap<[u8; 32], Vec<u8>>,
    pub bulk: bool,
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
    crate::versions::retain_content(
        tx,
        "document",
        prepared,
        &doc.blocks,
        source.packed,
        source.bulk,
    )?;
    tx.execute("INSERT INTO document(id,origin,native_id,purpose,title,source_time,authorship,version_hash,input_pk,source_chain,source_start,source_end,changed)
        VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13) ON CONFLICT(id) DO NOTHING",params![doc.id.to_string(),doc.id.origin().as_str(),doc.id.native().as_str(),doc.purpose.as_str(),doc.title,doc.source_time.map(|t|t.unix_ms()),doc.authorship.as_str(),&prepared.hash[..],source.input_pk,source.chain,row.offset as i64,row.end as i64,source.counter])?;
    let pk: i64 = tx.query_row(
        "SELECT pk FROM document WHERE id=?1",
        [doc.id.to_string()],
        |r| r.get(0),
    )?;
    tx.execute(
        "UPDATE document_source SET current=0 WHERE document_pk=?1 AND input_pk=?2 AND chain<?3",
        params![pk, source.input_pk, source.chain],
    )?;
    tx.execute("INSERT INTO document_source(document_pk,input_pk,chain,source_start,source_end,parser_version,route_rank,source_time,version_hash,current)
        VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,NOT EXISTS(SELECT 1 FROM document_source WHERE document_pk=?1 AND input_pk=?2 AND chain>?3))
        ON CONFLICT(document_pk,input_pk,chain,source_start,source_end,version_hash) DO UPDATE SET parser_version=excluded.parser_version,route_rank=excluded.route_rank,current=excluded.current",params![pk,source.input_pk,source.chain,row.offset as i64,row.end as i64,source.parser_version,source.rank,doc.source_time.map(|t|t.unix_ms()),&prepared.hash[..]])?;
    project(tx, pk, source.counter, source.bulk)
}

pub(crate) fn project(tx: &Transaction<'_>, pk: i64, counter: i64, bulk: bool) -> Result<()> {
    let preferred:Option<(Vec<u8>,i64,i64,i64,i64)>=tx.query_row("SELECT s.version_hash,s.input_pk,s.chain,s.source_start,s.source_end FROM document_source s JOIN input_ref r ON r.pk=s.input_pk
        WHERE s.document_pk=?1 AND s.current=1 AND r.excised=0
        ORDER BY (s.source_time IS NOT NULL) DESC,s.source_time DESC,s.route_rank,s.version_hash DESC,r.id,s.chain DESC,s.source_start LIMIT 1",[pk],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?))).optional()?;
    let old:Vec<i64>=tx.prepare("SELECT DISTINCT text_id FROM document_block WHERE document_pk=?1 AND text_id IS NOT NULL")?.query_map([pk],|r|r.get(0))?.collect::<rusqlite::Result<_>>()?;
    let Some((hash, input, chain, start, end)) = preferred else {
        tx.execute(
            "UPDATE document SET present=0,changed=?2 WHERE pk=?1",
            params![pk, counter],
        )?;
        tx.execute("DELETE FROM document_block WHERE document_pk=?1", [pk])?;
        for text in old {
            crate::text_rows::unindex_unused(tx, text, bulk)?;
        }
        return Ok(());
    };
    let (version, document) = crate::versions::load_document(tx, &hash)?;
    tx.execute("UPDATE document SET purpose=?2,title=?3,source_time=?4,authorship=?5,version_hash=?6,input_pk=?7,source_chain=?8,source_start=?9,source_end=?10,present=1,changed=?11 WHERE pk=?1",params![pk,document.purpose.as_str(),document.title,document.source_time.map(|t|t.unix_ms()),document.authorship.as_str(),hash,input,chain,start,end,counter])?;
    tx.execute("DELETE FROM document_block WHERE document_pk=?1", [pk])?;
    let mut statement = tx.prepare(
        "SELECT idx,body,text_id,blob_hash FROM version_block WHERE version_pk=?1 ORDER BY idx",
    )?;
    let mut rows = statement.query([version])?;
    for (idx, block) in document.blocks.iter().enumerate() {
        let row = rows
            .next()?
            .ok_or_else(|| StoreError::Invalid("Document version has too few blocks".into()))?;
        if row.get::<_, i64>(0)? != idx as i64 {
            return Err(StoreError::Invalid(
                "Document version block order is not consecutive".into(),
            ));
        }
        let text: Option<i64> = row.get(2)?;
        if let Some(text) = text {
            crate::text_rows::activate(tx, text, 1, bulk)?;
        }
        tx.execute("INSERT INTO document_block(document_pk,idx,kind,body,text_id,blob_hash) VALUES (?1,?2,?3,?4,?5,?6)",params![pk,idx as i64,block.kind().as_str(),row.get::<_,Vec<u8>>(1)?,text,row.get::<_,Option<Vec<u8>>>(3)?])?;
    }
    if rows.next()?.is_some() {
        return Err(StoreError::Invalid(
            "Document version has too many blocks".into(),
        ));
    }
    for text in old {
        crate::text_rows::unindex_unused(tx, text, bulk)?;
    }
    Ok(())
}

/// Publish a parser replacement only after it reaches the captured source EOF.
pub(crate) fn retire_previous_parser(tx: &Transaction<'_>, source: &Source<'_>) -> Result<()> {
    let affected:Vec<i64>=tx.prepare("SELECT DISTINCT document_pk FROM document_source WHERE input_pk=?1 AND (chain<?2 OR (chain=?2 AND parser_version<?3)) AND current=1")?.query_map(params![source.input_pk,source.chain,source.parser_version],|r|r.get(0))?.collect::<rusqlite::Result<_>>()?;
    tx.execute(
        "UPDATE document_source SET current=0 WHERE input_pk=?1 AND (chain<?2 OR (chain=?2 AND parser_version<?3))",
        params![source.input_pk, source.chain, source.parser_version],
    )?;
    for pk in affected {
        project(tx, pk, source.counter, source.bulk)?;
    }
    Ok(())
}

/// A document selection belongs to its Reader; budgeted consumers decode blocks
/// in order and can stop before later blocks. No input is reopened.
pub struct DocumentRead<'r, 'a> {
    reader: &'r Reader<'a>,
    pk: i64,
    attrs: Document,
    hash: Vec<u8>,
}

impl<'a> Reader<'a> {
    pub fn select_document<'r>(&'r self, id: &DocumentId) -> Result<Option<DocumentRead<'r, 'a>>> {
        let row:Option<(i64,Vec<u8>,Vec<u8>)>=self.conn()?.query_row("SELECT d.pk,d.version_hash,v.attrs FROM current_document d JOIN content_version v ON v.hash=d.version_hash AND v.kind='document' WHERE d.id=?1",[id.to_string()],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?;
        row.map(|(pk, hash, attrs)| {
            Ok(DocumentRead {
                reader: self,
                pk,
                attrs: unpack_json(&attrs)?,
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

    pub fn document_versions(&self, id: &DocumentId) -> Result<Vec<DocumentVersion>> {
        let mut statement=self.conn()?.prepare("SELECT DISTINCT s.version_hash,d.version_hash=s.version_hash FROM document_source s JOIN document d ON d.pk=s.document_pk WHERE d.id=?1 ORDER BY s.version_hash")?;
        let mut rows = statement.query([id.to_string()])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let hash: Vec<u8> = row.get(0)?;
            out.push(DocumentVersion {
                version: hex(&hash),
                preferred: row.get(1)?,
                document: crate::versions::load_document(self.conn()?, &hash)?.1,
            });
        }
        Ok(out)
    }
}

impl DocumentRead<'_, '_> {
    pub fn id(&self) -> &DocumentId {
        &self.attrs.id
    }
    pub fn title(&self) -> Option<&str> {
        self.attrs.title.as_deref()
    }
    pub fn purpose(&self) -> polylogue_model::DocumentPurpose {
        self.attrs.purpose
    }
    pub fn version(&self) -> String {
        hex(&self.hash)
    }
    pub fn metadata(&self) -> Result<Json> {
        let mut value = serde_json::to_value(&self.attrs)?;
        value
            .as_object_mut()
            .expect("Document is an object")
            .remove("blocks");
        Ok(value)
    }
    pub fn document(&self) -> Result<Document> {
        let mut document = self.attrs.clone();
        self.visit_blocks(|block| {
            document.blocks.push(block);
            ControlFlow::Continue(())
        })?;
        Ok(document)
    }
    pub fn visit_blocks(&self, mut visit: impl FnMut(Block) -> ControlFlow<()>) -> Result<()> {
        let mut statement=self.reader.conn()?.prepare("SELECT b.body,t.body FROM document_block b LEFT JOIN text t ON t.id=b.text_id WHERE b.document_pk=?1 ORDER BY b.idx")?;
        let mut rows = statement.query([self.pk])?;
        while let Some(row) = rows.next()? {
            let mut block = unpack_block(&row.get::<_, Vec<u8>>(0)?)?;
            if let Some(text) = row.get::<_, Option<Vec<u8>>>(1)? {
                restore_text(&mut block, unpack_text(&text)?);
            }
            if visit(block).is_break() {
                break;
            }
        }
        Ok(())
    }
    pub fn sources(&self) -> Result<Vec<DocumentSource>> {
        let mut statement=self.reader.conn()?.prepare("SELECT r.id,s.chain,s.source_start,s.source_end,s.version_hash,s.parser_version,s.current FROM document_source s JOIN input_ref r ON r.pk=s.input_pk WHERE s.document_pk=?1 ORDER BY r.id,s.chain,s.source_start,s.version_hash")?;
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
                parser_version: row.get(5)?,
                current: row.get(6)?,
                locations,
            });
        }
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DocumentVersion {
    pub version: String,
    pub preferred: bool,
    pub document: Document,
}
#[derive(Debug, Clone, Serialize)]
pub struct DocumentSource {
    pub input: String,
    pub chain: i64,
    pub start: i64,
    pub end: i64,
    pub version: String,
    pub parser_version: i64,
    pub current: bool,
    pub locations: Vec<Json>,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
