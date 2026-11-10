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

/// Where a stored group's results are answered: each caller's share, in order.
pub(super) type Replies = Vec<(usize, SyncSender<Vec<Result<StoredInput>>>)>;

/// A group whose files are appended and whose sync may still be running.
/// It keeps its files until it commits.
pub(super) struct AppendedGroup {
    items: Vec<PreparedBatch>,
    files: Vec<PreparedBytes>,
    pending: Option<hashed_blobs::PendingGroup>,
}

impl AppendedGroup {
    fn subject(&self) -> &str {
        subject(&self.items)
    }
}

enum Appending {
    Appended(AppendedGroup),
    Answered(Vec<Result<StoredInput>>),
}

/// The fault-point subject of a group: its first input.
fn subject(items: &[PreparedBatch]) -> &str {
    items
        .first()
        .map_or("", |item| item.batch.input.id.as_str())
}

fn answer(replies: Replies, results: Vec<Result<StoredInput>>) {
    let mut results = results.into_iter();
    for (count, reply) in replies {
        let _ = reply.send(results.by_ref().take(count).collect());
    }
}

// A package or one consistent database read is one logical acquisition.
// A physical pool flush may contain several independent file samples.
pub(super) fn atomic_acquisition(items: &[PreparedBatch]) -> bool {
    items.iter().any(|item| {
        crate::container_members::container_of(&item.batch).is_some()
            || item
                .batch
                .observation
                .as_ref()
                .is_some_and(|observation| observation.import.is_some())
            || item.batch.acquisition.as_ref().is_some_and(|receipt| {
                receipt.route == crate::primary::AcquisitionRoute::ExplicitImport
                    || matches!(
                        receipt.generation,
                        Some(crate::primary::SourceGeneration::Database { .. })
                    )
            })
    })
}

pub(super) fn independent_files(items: &[PreparedBatch]) -> bool {
    items.len() > 1
        && !atomic_acquisition(items)
        && items.iter().any(|item| {
            item.batch.acquisition.as_ref().is_some_and(|receipt| {
                receipt.route == crate::primary::AcquisitionRoute::NativeCapture
            })
        })
}

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
    /// rows share syncs and commits, and answers each. Logical package and
    /// database groups retain the caller's boundary. In primary mode each
    /// caller's group also stays whole before its file samples receive their
    /// separate acquisition dispositions.
    ///
    /// Each group's bytes are appended before the previous group's sync is
    /// waited for, so the next append overlaps that sync. The last group stays
    /// appended ([`Inner::appended`]) until the writer's next turn appends
    /// after it, or the writer has nothing else to do; it is answered only
    /// after it commits.
    pub(super) fn store_groups(&mut self, groups: Vec<WaitingGroup>) {
        let (alone, together): (Vec<WaitingGroup>, Vec<WaitingGroup>) = groups
            .into_iter()
            .partition(|(items, _, _)| self.sinex.is_some() || atomic_acquisition(items));
        let mut units: Vec<(Vec<PreparedBatch>, Replies)> = alone
            .into_iter()
            .map(|(items, _, reply)| {
                let count = items.len();
                (items, vec![(count, reply)])
            })
            .collect();
        if !together.is_empty() {
            let mut items = Vec::new();
            let mut replies = Vec::new();
            for (group, _, reply) in together {
                replies.push((group.len(), reply));
                items.extend(group);
            }
            units.push((items, replies));
        }
        for (items, replies) in units {
            let next = self.append_group(items);
            self.finish_appended();
            match next {
                Appending::Answered(results) => {
                    answer(replies, results);
                    self.refresh_after_group();
                }
                Appending::Appended(mut group) => {
                    if group.pending.is_some() && self.segment.is_none() {
                        // The previous group's failure dropped the segment
                        // writer, and reopening it truncates this group's
                        // append; append its retained files again.
                        group.pending = None;
                        if let Err(error) = self.append_files(&mut group) {
                            answer(replies, errors(&group.items, error));
                            self.refresh_after_group();
                            continue;
                        }
                    }
                    self.appended = Some((group, replies));
                }
            }
        }
    }

    /// Commits, writes the rows of and answers the group left appended by
    /// [`Self::store_groups`].
    pub(super) fn finish_appended(&mut self) {
        if let Some((group, replies)) = self.appended.take() {
            let results = self.commit_group(group);
            answer(replies, results);
            self.refresh_after_group();
        }
    }

    /// Stores one group of batches whose new blobs include private files,
    /// then writes their parsed rows, before returning ([`Self::append_group`],
    /// [`Self::commit_group`]). A group left appended commits first.
    pub(super) fn store_group(
        &mut self,
        group: Vec<PreparedBatch>,
        _store: std::sync::Arc<crate::blobs::Blobs>,
    ) -> Vec<Result<StoredInput>> {
        self.finish_appended();
        let results = match self.append_group(group) {
            Appending::Answered(results) => results,
            Appending::Appended(group) => self.commit_group(group),
        };
        self.refresh_after_group();
        results
    }

    /// The index counts the stored chains before the answer, also when no
    /// parsed rows followed them; a failure is retried after the turn.
    fn refresh_after_group(&mut self) {
        if let Err(error) = self.refresh_chain_progress() {
            self.report.errors.push(format!("chain progress: {error}"));
        }
    }

    /// Checks a group and appends its files to the segment without waiting
    /// for their sync. A group that needs no append is answered here.
    fn append_group(&mut self, mut group: Vec<PreparedBatch>) -> Appending {
        if group.is_empty() {
            return Appending::Answered(Vec::new());
        }
        if self.sinex.is_some() {
            return Appending::Answered(self.primary_store_group(group));
        }
        for item in &group {
            if let Err(error) = self.check_staged(item) {
                return Appending::Answered(errors(&group, error));
            }
        }
        match self.repeated_imports(&group) {
            Ok(Some(stored)) => return Appending::Answered(stored.into_iter().map(Ok).collect()),
            Ok(None) => {}
            Err(error) => return Appending::Answered(errors(&group, error)),
        }
        let files = group
            .iter_mut()
            .flat_map(|item| item.files.drain(..))
            .collect::<Vec<_>>();
        let mut appended = AppendedGroup {
            items: group,
            files,
            pending: None,
        };
        if appended.files.is_empty() {
            return Appending::Appended(appended);
        }
        match self.append_files(&mut appended) {
            Ok(()) => Appending::Appended(appended),
            Err(error) => Appending::Answered(errors(&appended.items, error)),
        }
    }

    /// Appends the group's files; it keeps them, so a group whose append
    /// a failed earlier group truncated can be appended again.
    fn append_files(&mut self, group: &mut AppendedGroup) -> Result<()> {
        if self.segment.is_none() {
            self.segment = Some(crate::blobs::open_writer(
                &self.archive,
                &self.location.root,
            )?);
        }
        let writer = self.segment.as_mut().expect("opened segment writer");
        let prepared = group
            .files
            .iter()
            .cloned()
            .map(PreparedBytes::into_prepared);
        let appended = writer.append_group(prepared, |hash| {
            crate::blobs::load_extent(&self.archive, "", hash)
                .map_err(|error| hashed_blobs::Error::Io(std::io::Error::other(error)))
        });
        match appended {
            Ok(pending) => group.pending = Some(pending),
            Err(error) => {
                self.segment.take();
                return Err(error.into());
            }
        }
        if let Err(error) = crate::fault::check("segment.after_append", group.subject()) {
            self.segment.take();
            return Err(error);
        }
        Ok(())
    }

    /// Waits for the group's sync, commits its extents with the revisions
    /// and observations that cite them, then writes its parsed rows. Parsed
    /// rows of independent inputs that are refused together are written
    /// again batch by batch, so one bad batch costs only itself; one logical
    /// acquisition ([`atomic_acquisition`]) succeeds or fails as a whole.
    fn commit_group(&mut self, group: AppendedGroup) -> Vec<Result<StoredInput>> {
        let AppendedGroup {
            items: mut group,
            files: _,
            pending,
        } = group;
        let independent = !atomic_acquisition(&group);
        let synced = match pending {
            Some(pending) => {
                let synced = match pending.wait() {
                    Ok(synced) => synced,
                    Err(error) => {
                        self.segment.take();
                        return errors(&group, error.into());
                    }
                };
                if let Err(error) = crate::fault::check("segment.after_sync", subject(&group)) {
                    self.segment.take();
                    return errors(&group, error);
                }
                Some(synced)
            }
            None => None,
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
            if let Err(error) = crate::fault::check("segment.after_commit", subject(&group)) {
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
        for (position, (mut item, stored)) in group.into_iter().zip(&stored).enumerate() {
            if !item.batch.parse.is_empty() {
                item.batch.change = Change::Reparse {
                    chain: stored.chain,
                };
                item.batch.continues = false;
                item.batch.new_blob = None;
                item.batch.observation = None;
                item.batch.blobs.clear();
                parsed.push((position, item.batch));
            }
        }
        let mut results: Vec<Result<StoredInput>> = stored.into_iter().map(Ok).collect();
        if parsed.is_empty() {
            return results;
        }
        let Err(error) = self.write(parsed.iter().map(|(_, batch)| batch)) else {
            return results;
        };
        if independent && parsed.len() > 1 {
            for (position, batch) in &parsed {
                if let Err(error) = self.write(std::iter::once(batch)) {
                    results[*position] = Err(error);
                }
            }
            return results;
        }
        if results.len() == 1 {
            return vec![Err(error)];
        }
        let reason = error.to_string();
        results
            .into_iter()
            .map(|_| Err(StoreError::Invalid(reason.clone())))
            .collect()
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
            let PreparedBatch { batch: b, files: _ } = item;
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
            if b.input.route.is_imported() && b.new_blob.is_some() && (known.is_some() || again) {
                let new_blob = b.new_blob.as_ref().expect("checked new_blob");
                // Empty bytes are stored with no extent.
                let same: bool = tx.query_row(
                    "SELECT CASE WHEN ?2 <> ?3 THEN 0 WHEN ?3 = 0 THEN count(*) = 0
                            ELSE count(*) = 1 AND min(offset) = 0 AND min(length) = ?3 AND min(blob) = ?4 END
                     FROM extent WHERE input = ?1 AND chain = (SELECT chain FROM revision WHERE input = ?1 ORDER BY number DESC LIMIT 1)",
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

    /// An exact retry of an immutable import group has no new fact to write.
    /// Format validation has already run through `check_staged`; a new path,
    /// receiver or import membership is a new observation even for known bytes.
    fn repeated_imports(&self, group: &[PreparedBatch]) -> Result<Option<Vec<StoredInput>>> {
        let mut stored = Vec::with_capacity(group.len());
        for item in group {
            let b = &item.batch;
            let (Some(observation), Some(blob)) = (&b.observation, &b.new_blob) else {
                return Ok(None);
            };
            if !b.input.route.is_imported() || !b.parse.is_empty() || !b.blobs.is_empty() {
                return Ok(None);
            }
            let details = observation
                .import
                .as_ref()
                .map(|member| {
                    serde_json::to_string(
                        &serde_json::json!({"import":member,"import_format":member.format}),
                    )
                })
                .transpose()?;
            let known: Option<(i64, i64)> = self.archive.query_row(
                "SELECT r.chain,r.length FROM revision r WHERE r.input=?1
                   AND r.number=(SELECT max(number) FROM revision WHERE input=?1)
                   AND r.length=?2
                   AND (SELECT count(*) FROM extent WHERE input=?1 AND chain=r.chain)=1
                   AND EXISTS(SELECT 1 FROM extent x WHERE x.input=?1 AND x.chain=r.chain AND x.offset=0 AND x.length=?2 AND x.blob=?3)
                   AND EXISTS(SELECT 1 FROM observation o WHERE o.input=?1 AND o.machine=?4 AND o.path_bytes=?5 AND o.class='imported' AND json_extract(o.details_json,'$.import') IS json_extract(?6,'$.import') AND json_extract(o.details_json,'$.import_format') IS json_extract(?6,'$.import_format'))",
                params![b.input.id,b.length as i64,&blob.hash.0[..],observation.machine,observation.path_bytes,details],
                |row| Ok((row.get(0)?,row.get(1)?)),
            ).optional()?;
            let Some((chain, length)) = known else {
                return Ok(None);
            };
            stored.push(StoredInput {
                input: b.input.id.clone(),
                chain,
                length: length as u64,
                already_stored: true,
            });
        }
        Ok(Some(stored))
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
        if b.input.route.is_imported()
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
