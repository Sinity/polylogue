use super::{Plan, position};
use crate::excise::stored_text::{Extent, extent};
use crate::{Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::{Connection, params};

impl Plan {
    pub(in super::super) fn stage_affected(&self, archive: &Connection) -> Result<()> {
        self.conn
            .execute_batch("DELETE FROM affected; DELETE FROM observations;")?;
        let mut affected = archive.prepare(
            "SELECT blob,offset,length,blob_offset FROM extent WHERE input=?1 AND chain=?2
             AND offset<?4 AND offset+length>?3",
        )?;
        let mut blob = self
            .conn
            .prepare_cached("INSERT OR IGNORE INTO affected(hash) VALUES (?1)")?;
        let mut observation = self
            .conn
            .prepare_cached("INSERT INTO observations(input,span) VALUES (?1,?2)")?;
        self.changed(|input, chain, start, end| {
            let mut rows =
                affected.query(params![input, chain, position(start)?, position(end)?])?;
            while let Some(row) = rows.next()? {
                let hash: Vec<u8> = row.get(0)?;
                let offset = row.get::<_, i64>(1)? as u64;
                let length = row.get::<_, i64>(2)? as u64;
                let blob_offset = row.get::<_, i64>(3)? as u64;
                let from = start.max(offset) - offset + blob_offset;
                let to = end.min(offset + length) - offset + blob_offset;
                let span = serde_json::json!([crate::excise::hex(&hash), from, to - from]);
                observation.execute(params![input, serde_json::to_string(&span)?])?;
                blob.execute([hash])?;
            }
            Ok(())
        })
    }

    pub(in super::super) fn visit_blobs(
        &self,
        mut visit: impl FnMut(BlobHash) -> Result<()>,
    ) -> Result<()> {
        let mut statement = self
            .conn
            .prepare("SELECT hash FROM affected ORDER BY hash")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let hash = row.get_ref(0)?;
            let hash = hash
                .as_blob()
                .map_err(|e| StoreError::Invalid(e.to_string()))?;
            visit(BlobHash(hash.try_into().map_err(|_| {
                StoreError::Invalid("blob hash is not 32 bytes".into())
            })?))?;
        }
        Ok(())
    }

    /// Freeze one blob's referring extents before changing any of those rows.
    pub(in super::super) fn freeze_extents(
        &self,
        archive: &Connection,
        hash: BlobHash,
    ) -> Result<()> {
        self.conn.execute("DELETE FROM referring", [])?;
        let mut statement = archive.prepare(
            "SELECT input,chain,offset,length,blob,blob_offset FROM extent WHERE blob=?1",
        )?;
        let mut insert = self.conn.prepare_cached(
            "INSERT INTO referring(input,chain,offset,length,blob,blob_offset)
             VALUES (?1,?2,?3,?4,?5,?6)",
        )?;
        for row in statement.query_map([&hash.0[..]], extent)? {
            let e = row?;
            insert.execute(params![
                e.input,
                e.chain,
                position(e.offset)?,
                position(e.length)?,
                &e.blob.0[..],
                position(e.blob_offset)?
            ])?;
        }
        Ok(())
    }

    pub(in super::super) fn visit_extents(
        &self,
        mut visit: impl FnMut(Extent) -> Result<()>,
    ) -> Result<()> {
        let mut statement = self.conn.prepare(
            "SELECT input,chain,offset,length,blob,blob_offset FROM referring ORDER BY ordinal",
        )?;
        for row in statement.query_map([], extent)? {
            visit(row?)?;
        }
        Ok(())
    }

    pub(in super::super) fn visit_observations(
        &self,
        mut visit: impl FnMut(&str, serde_json::Value) -> Result<()>,
    ) -> Result<()> {
        let mut inputs = self
            .conn
            .prepare("SELECT DISTINCT input FROM observations ORDER BY input")?;
        let mut spans = self
            .conn
            .prepare_cached("SELECT span FROM observations WHERE input=?1 ORDER BY ordinal")?;
        let mut rows = inputs.query([])?;
        while let Some(row) = rows.next()? {
            let input: String = row.get(0)?;
            let mut out = Vec::new();
            for span in spans.query_map([&input], |r| r.get::<_, String>(0))? {
                out.push(serde_json::from_str(&span?)?);
            }
            visit(&input, serde_json::Value::Array(out))?;
        }
        Ok(())
    }
}
