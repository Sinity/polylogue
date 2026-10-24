//! Document removal uses every retained source citation and the shared text owner.
use super::*;

impl Plan<'_> {
    pub(super) fn document(&mut self, t: &Transactions<'_, '_>, pk: i64, id: &str) -> Result<()> {
        self.documents.insert(pk, id.to_owned());
        t.index.execute(
            "INSERT OR IGNORE INTO temp.excision_documents VALUES (?1)",
            [pk],
        )?;
        let mut stmt=t.index.prepare("SELECT r.id,s.chain,s.source_start,s.source_end FROM document_source s JOIN input_ref r ON r.pk=s.input_pk WHERE s.document_pk=?1 ORDER BY r.id,s.chain,s.source_start")?;
        for row in stmt.query_map([pk], |r| {
            Ok((
                r.get::<_, String>(0)?,
                Span {
                    chain: r.get(1)?,
                    start: r.get(2)?,
                    end: r.get(3)?,
                },
            ))
        })? {
            let (input, span) = row?;
            if span.end <= span.start {
                return Err(StoreError::Invalid(format!(
                    "document {id} has no removable source boundary"
                )));
            }
            self.spans.insert(&input, span)?;
        }
        Ok(())
    }
}

pub(super) fn remove(
    t: &Transactions<'_, '_>,
    pk: i64,
    id: &str,
    req: &ExciseRequest,
    hashes: &RemovalHashes<'_>,
    done: &mut Excised,
) -> Result<()> {
    let mut stmt=t.index.prepare("SELECT DISTINCT v.text_id FROM version_block v JOIN content_version c ON c.pk=v.version_pk JOIN document_source s ON s.version_hash=c.hash WHERE s.document_pk=?1 AND v.text_id IS NOT NULL")?;
    let texts = stmt
        .query_map([pk], |r| r.get::<_, i64>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    if req.dry_run {
        for tid in texts {
            let hash:Option<Vec<u8>>=t.index.query_row("SELECT hash FROM text WHERE id=?1 AND NOT EXISTS(SELECT 1 FROM block WHERE text_id=?1) AND NOT EXISTS(SELECT 1 FROM document_block WHERE text_id=?1 AND document_pk<>?2) AND NOT EXISTS(SELECT 1 FROM version_block b JOIN content_version c ON c.pk=b.version_pk WHERE b.text_id=?1 AND (c.kind='message' OR EXISTS(SELECT 1 FROM document_source s WHERE s.version_hash=c.hash AND s.document_pk<>?2)))",params![tid,pk],|r|r.get(0)).optional()?;
            if let Some(hash) = hash {
                hashes.text(&hash)?;
            }
        }
        return Ok(());
    }
    t.archive.execute(
        "INSERT OR IGNORE INTO excision_document(excision_id,document_id) VALUES (?1,?2)",
        params![req.id, id],
    )?;
    let reference = format!("document:{id}");
    done.notes_removed += super::notes::remove_document(t.archive, &reference)?;
    t.archive
        .execute("DELETE FROM variant WHERE target=?1", [&reference])?;
    t.index.execute("DELETE FROM document WHERE pk=?1", [pk])?;
    crate::versions::prune(t.index)?;
    for tid in texts {
        let row:Option<(Vec<u8>,i64)>=t.index.query_row("SELECT hash,indexed FROM text WHERE id=?1 AND NOT EXISTS(SELECT 1 FROM block WHERE text_id=?1) AND NOT EXISTS(SELECT 1 FROM document_block WHERE text_id=?1) AND NOT EXISTS(SELECT 1 FROM version_block WHERE text_id=?1)",[tid],|r|Ok((r.get(0)?,r.get(1)?))).optional()?;
        if let Some((hash, indexed)) = row {
            crate::text_rows::remove(t.index, tid, indexed)?;
            hashes.text(&hash)?;
        }
    }
    Ok(())
}
