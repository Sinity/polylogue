//! Completed content-addressed imports keep their confirmed source receipt.
use super::*;

impl Inner {
    pub(super) fn confirmed_immutable_import(
        &self,
        item: &PreparedBatch,
    ) -> Result<Option<StoredInput>> {
        let batch = &item.batch;
        let Some(read) = &batch.capture_read else {
            return Ok(None);
        };
        if read.route != crate::primary::ReadRoute::ExplicitImport
            || batch.input.id != read.hash.to_string()
            || read.range.start != 0
            || read.range.end != batch.length
            || batch.continues
            || !batch.blobs.is_empty()
            || !batch.parse.is_empty()
            || batch.required_outcome.is_some()
            || batch.base.is_some()
        {
            return Ok(None);
        }
        let Some(blob) = &batch.new_blob else {
            return Ok(None);
        };
        if blob.hash != read.hash || blob.size != batch.length {
            return Ok(None);
        }
        if crate::source::SourceBoundary::read(&self.index, &self.index, &self.location)?.is_none()
        {
            return Ok(None);
        }
        let Some(revision) = latest(&self.index, &batch.input.id)? else {
            return Ok(None);
        };
        if !self.index.scalar::<bool>(
            "SELECT EXISTS(SELECT 1 FROM source_observation WHERE input=?1 AND machine=?2 AND class='imported')",
            params![batch.input.id, read.source.machine],
        )? {
            return Ok(None);
        }
        let format: String = self.index.query_row(
            "SELECT format FROM source_input WHERE id=?1",
            [&batch.input.id],
            |row| row.get(0),
        )?;
        if format != batch.input.format || revision.length != batch.length as i64 {
            return Err(StoreError::Conflict(batch.input.id.clone()));
        }
        let extents = self
            .index
            .prepare_cached(
                "SELECT chain,offset,length,blob,blob_offset FROM source_extent
             WHERE input=?1 AND chain=?2 ORDER BY offset",
            )?
            .query_map(params![batch.input.id, revision.chain], |row| {
                let hash: Vec<u8> = row.get(3)?;
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    hash,
                    row.get::<_, i64>(4)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let expected: Vec<_> = bytes::extents(blob, revision.chain, 0)
            .into_iter()
            .map(|extent| {
                (
                    extent.chain,
                    extent.offset,
                    extent.length,
                    extent.blob.0.to_vec(),
                    extent.blob_offset,
                )
            })
            .collect();
        if extents != expected {
            return Err(StoreError::Conflict(batch.input.id.clone()));
        }
        // Extents alone can be historical metadata. A nonempty body must also
        // have its original material in the live confirmed source population.
        if batch.length != 0 && !self.index.scalar::<bool>(
            "SELECT EXISTS(SELECT 1 FROM source_material m JOIN confirmed_event e ON e.event=m.event
             WHERE m.input=?1 AND m.chain=?2 AND m.offset=0 AND m.source_length=?3
             AND m.length=?3 AND m.hash=?4 AND m.blob_offset=0 AND e.state='live')",
            params![batch.input.id, revision.chain, batch.length as i64, &read.hash.0[..]],
        )? {
            return Ok(None);
        }
        Ok(Some(StoredInput {
            input: batch.input.id.clone(),
            chain: revision.chain,
            length: batch.length,
            already_stored: true,
            bytes_written: 0,
        }))
    }
}
