//! Groups append prepared frames and sync them once before extents and
//! the rows that cite them commit. The index then saves its parsed position.
use super::*;
use crate::staging::{PreparedBatch, PreparedBytes, StoredInput};
use std::collections::BTreeSet;

/// A parse-pool group waiting for the writer, with the store its files go
/// to and where its results are answered.
pub(super) type WaitingGroup = (
    Vec<PreparedBatch>,
    Arc<crate::blobs::Blobs>,
    SyncSender<Vec<Result<StoredInput>>>,
);

impl Inner {
    /// Stores one import's bytes: a batch with an observation and no
    /// parsed rows.
    pub(super) fn store_bytes(
        &mut self,
        file: PreparedBytes,
        mut batch: Batch,
    ) -> Result<StoredInput> {
        if file.owner != self.location.root
            || batch.new_blob.is_some()
            || !batch.blobs.is_empty()
            || !batch.parse.is_empty()
            || batch.observation.is_none()
        {
            return Err(StoreError::Invalid(
                "storing bytes takes one private file of this archive and a batch without rows"
                    .into(),
            ));
        }
        batch.new_blob = Some(NewBlob {
            hash: file.hash(),
            size: file.len(),
        });
        let store = std::sync::Arc::new(self.location.blobs()?);
        self.store_group(
            vec![PreparedBatch {
                batch,
                files: vec![file],
            }],
            store,
        )
        .pop()
        .expect("one result")
    }

    /// Stores the groups that were waiting together, so their files and
    /// rows share syncs and commits ([`Self::store_group`]), and answers
    /// each. A group with a container member is stored alone: a failure keeps the
    /// container's group whole, which would fail the others with it.
    pub(super) fn store_groups(&mut self, groups: Vec<WaitingGroup>) {
        let (alone, together): (Vec<WaitingGroup>, Vec<WaitingGroup>) =
            groups.into_iter().partition(|(items, _, _)| {
                items
                    .iter()
                    .any(|item| crate::container_members::container_of(&item.batch).is_some())
            });
        for (items, store, reply) in alone {
            let _ = reply.send(self.store_group(items, store));
        }
        let Some(store) = together.first().map(|(_, store, _)| Arc::clone(store)) else {
            return;
        };
        let mut items = Vec::new();
        let mut replies = Vec::new();
        for (group, _, reply) in together {
            replies.push((group.len(), reply));
            items.extend(group);
        }
        let mut results = self.store_group(items, store).into_iter();
        for (count, reply) in replies {
            let _ = reply.send(results.by_ref().take(count).collect());
        }
    }

    /// Stores a group of batches whose new blobs include private files, then
    /// writes their parsed rows. A refused group is tried again batch by
    /// batch, so one bad batch costs only itself.
    pub(super) fn store_group(
        &mut self,
        mut group: Vec<PreparedBatch>,
        _store: std::sync::Arc<crate::blobs::Blobs>,
    ) -> Vec<Result<StoredInput>> {
        if group.is_empty() {
            return Vec::new();
        }
        if self.sinex.is_some() {
            return self.primary_store_group(group);
        }
        for item in &group {
            if let Err(error) = self.check_staged(item) {
                return errors(&group, error);
            }
        }
        let prepared = if self.sinex.is_none() {
            let prepared = group
                .iter_mut()
                .flat_map(|item| item.files.drain(..))
                .map(|file| match file.file {
                    crate::staging::Prepared::Local { prepared } => Ok(*prepared),
                    crate::staging::Prepared::Part { .. } => Err(StoreError::Invalid(
                        "Sinex bytes reached the segment writer".into(),
                    )),
                })
                .collect::<Result<Vec<_>>>();
            match prepared {
                Ok(prepared) => prepared,
                Err(error) => return errors(&group, error),
            }
        } else {
            Vec::new()
        };
        let synced = if self.sinex.is_none() && !prepared.is_empty() {
            if self.segment.is_none() {
                match crate::blobs::open_writer(&self.archive, &self.location.root) {
                    Ok(writer) => self.segment = Some(writer),
                    Err(error) => return errors(&group, error),
                }
            }
            let writer = self.segment.as_mut().expect("opened segment writer");
            let appended = writer.append_group(prepared, |hash| {
                crate::blobs::load_extent(&self.archive, "", hash)
                    .map_err(|error| hashed_blobs::Error::Io(std::io::Error::other(error)))
            });
            let pending = match appended {
                Ok(pending) => pending,
                Err(error) => {
                    self.segment.take();
                    return errors(&group, error.into());
                }
            };
            if let Err(error) = crate::fault::check("segment.after_append", "") {
                self.segment.take();
                return errors(&group, error);
            }
            let synced = match pending.wait() {
                Ok(synced) => synced,
                Err(error) => {
                    self.segment.take();
                    return errors(&group, error.into());
                }
            };
            if let Err(error) = crate::fault::check("segment.after_sync", "") {
                self.segment.take();
                return errors(&group, error);
            }
            Some(synced)
        } else {
            None
        };
        let stored = match self.write_rows_first(&mut group, synced.as_ref()) {
            Ok(stored) => stored,
            Err(error) => {
                if synced.is_some() {
                    self.segment.take();
                }
                return errors(&group, error);
            }
        };
        if let Some(synced) = synced {
            if let Err(error) = crate::fault::check("segment.after_commit", "") {
                self.segment.take();
                return errors(&group, error);
            }
            self.report.segment_syncs += synced.metrics.syncs;
            self.report.segment_bytes_written += synced.metrics.appended_bytes;
            self.report.blobs_appended += synced.metrics.appended_materials;
            self.report.blobs_deduplicated += synced.metrics.deduplicated_materials;
            if let Err(error) = self
                .segment
                .as_mut()
                .expect("appended writer")
                .acknowledge(&synced)
            {
                self.segment.take();
                return errors(&group, error.into());
            }
        }
        let mut parsed = Vec::new();
        for (mut item, stored) in group.into_iter().zip(&stored) {
            if !item.batch.parse.is_empty() {
                item.batch.change = Change::Reparse {
                    chain: stored.chain,
                };
                item.batch.continues = false;
                item.batch.new_blob = None;
                item.batch.observation = None;
                item.batch.blobs.clear();
                parsed.push(item.batch);
            }
        }
        if !parsed.is_empty()
            && let Err(error) = self.write(parsed.iter())
        {
            let reason = error.to_string();
            return stored
                .into_iter()
                .map(|_| Err(StoreError::Invalid(reason.clone())))
                .collect();
        }
        stored.into_iter().map(Ok).collect()
    }

    /// Commits the durable extents and the revisions and observations that cite them.
    fn write_rows_first(
        &mut self,
        group: &mut [PreparedBatch],
        synced: Option<&hashed_blobs::SyncedGroup>,
    ) -> Result<Vec<StoredInput>> {
        // A member read from a container needs its container stored, earlier in this group
        // or before.
        let staged: BTreeSet<&str> = group
            .iter()
            .map(|item| item.batch.input.id.as_str())
            .collect();
        for item in group.iter() {
            if let Some(zip) = crate::container_members::container_of(&item.batch)
                && !staged.contains(zip.input.as_str())
            {
                crate::container_members::require_original(&self.archive, &zip.input)?;
            }
        }
        let tx = self.archive.transaction()?;
        tx.execute_batch("PRAGMA defer_foreign_keys=ON")?;
        let seq = bump_counter(&tx)?;
        if let Some(synced) = synced {
            for extent in &synced.extents {
                crate::blobs::insert_extent(&tx, extent)?;
            }
            tx.execute("INSERT INTO segment(name,committed_end) VALUES (?1,?2) ON CONFLICT(name) DO UPDATE SET committed_end=excluded.committed_end",
                params!["segments/current.hb",synced.end as i64])?;
        }
        let mut stored = Vec::new();
        let mut seen = BTreeSet::new();
        for item in group.iter_mut() {
            let PreparedBatch { batch: b, files } = item;
            // An immutable import named by its bytes may be stored already, or
            // appear twice in a group (equal container members).
            let known: Option<(i64, i64)> = tx
                .query_row(
                    "SELECT chain, length FROM revision WHERE input = ?1 ORDER BY number DESC LIMIT 1",
                    [&b.input.id],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()?;
            let again = !seen.insert(b.input.id.clone());
            if b.input.arrival_rank == 2 && b.new_blob.is_some() && (known.is_some() || again) {
                let new_blob = b.new_blob.as_ref().expect("checked new_blob");
                let same: bool = tx.query_row(
                    "SELECT count(*) = 1 AND min(offset) = 0 AND min(length) = ?3 AND min(blob) = ?4
                     FROM extent WHERE input = ?1 AND chain = (SELECT chain FROM revision WHERE input = ?1 ORDER BY number DESC LIMIT 1)
                     AND ?2 = ?3",
                    params![b.input.id, known.map_or(b.length as i64, |k| k.1), b.length as i64, &new_blob.hash.0[..]],
                    |r| r.get(0),
                )?;
                if !same {
                    return Err(StoreError::Conflict(
                        "an import named by its bytes holds other bytes".into(),
                    ));
                }
                b.change = Change::Unchanged;
                b.new_blob = None;
            }
            let (chain, _) = write_bytes(&tx, b, seq)?;
            let chain = chain
                .ok_or_else(|| StoreError::Invalid("stored bytes produced no chain".into()))?;
            crate::fault::check("import.rows", &b.input.id)?;
            let already_stored =
                again || matches!(b.change, Change::Unchanged | Change::Observed { .. });
            stored.push(StoredInput {
                input: b.input.id.clone(),
                chain,
                length: b.length,
                already_stored,
            });
        }
        crate::fault::check("write.archive_commit", "")?;
        crate::write_lock::commit(tx)?;
        self.commits.committed();
        self.report.transactions += 1;
        self.report.batches += group.iter().filter(|g| g.batch.parse.is_empty()).count() as u64;
        Ok(stored)
    }

    /// What a batch must satisfy before any of its rows are written.
    fn check_staged(&self, item: &PreparedBatch) -> Result<()> {
        let b = &item.batch;
        if b.input.id.is_empty() {
            return Err(StoreError::Invalid("stored bytes need an input".into()));
        }
        i64::try_from(b.length)
            .map_err(|_| StoreError::Invalid("input exceeds archive address space".into()))?;
        check_import_format(
            &self.archive,
            &b.input.id,
            b.observation.as_ref().and_then(|o| o.import.as_ref()),
        )?;
        for new_blob in b.new_blob.iter().chain(&b.blobs) {
            if !item
                .files
                .iter()
                .any(|file| file.hash() == new_blob.hash && file.len() == new_blob.size)
            {
                return Err(StoreError::Invalid(
                    "a file new_blob has no private file".into(),
                ));
            }
        }
        for file in &item.files {
            if file.owner != self.location.root {
                return Err(StoreError::Invalid(
                    "a private file belongs to another archive".into(),
                ));
            }
            if !b
                .new_blob
                .iter()
                .chain(&b.blobs)
                .any(|new_blob| new_blob.hash == file.hash() && new_blob.size == file.len())
            {
                return Err(StoreError::Invalid(
                    "a private file belongs to no new_blob of its batch".into(),
                ));
            }
        }
        if b.input.arrival_rank == 2
            && let Some(new_blob) = &b.new_blob
            && (b.input.id != new_blob.hash.to_string() || new_blob.size != b.length || b.continues)
        {
            return Err(StoreError::Invalid(
                "an import must be named by its complete bytes".into(),
            ));
        }
        Ok(())
    }
}

fn errors(group: &[PreparedBatch], error: StoreError) -> Vec<Result<StoredInput>> {
    if group.len() == 1 {
        return vec![Err(error)];
    }
    let reason = error.to_string();
    group
        .iter()
        .map(|_| Err(StoreError::Invalid(reason.clone())))
        .collect()
}

#[cfg(test)]
mod tests;
