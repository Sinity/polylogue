//! Bytes into the archive in one order: the blob's row first, in state
//! `writing`, then the file under its content name, then the row `present`.
//! A stop between them leaves `writing` rows, which [`Inner::finish_writing`]
//! finishes. In primary mode the bytes go to Sinex before the transaction
//! that writes their rows commits, and those rows are `present` at once.
use super::*;
use crate::staging::{Finished, PreparedBatch, PreparedBytes, StoredInput};
use polylogue_model::BlobHash;
use std::collections::{BTreeMap, BTreeSet};

/// A parse-pool group waiting for the writer, with the store its files go
/// to and where its results are answered.
pub(super) type WaitingGroup = (
    Vec<PreparedBatch>,
    Arc<hashed_blobs::Store>,
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
            stored: Stored::File { stored_size: 0 },
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
    /// each. A group with a ZIP member is stored alone: a failure keeps the
    /// ZIP's group whole, which would fail the others with it.
    pub(super) fn store_groups(&mut self, groups: Vec<WaitingGroup>) {
        let (alone, together): (Vec<WaitingGroup>, Vec<WaitingGroup>) =
            groups.into_iter().partition(|(items, _, _)| {
                items
                    .iter()
                    .any(|item| crate::zip_members::zip_of(&item.batch).is_some())
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
        store: std::sync::Arc<hashed_blobs::Store>,
    ) -> Vec<Result<StoredInput>> {
        if group.is_empty() {
            return Vec::new();
        }
        let previous: i64 = match self
            .archive
            .pragma_query_value(None, "synchronous", |r| r.get(0))
        {
            Ok(value) => value,
            Err(error) => return errors(&group, error.into()),
        };
        if let Err(error) = self.archive.pragma_update(None, "synchronous", "full") {
            return errors(&group, error.into());
        }
        let stored = match self.write_rows_first(&mut group) {
            Ok(stored) => stored,
            Err(error) => {
                let _ = self.archive.pragma_update(None, "synchronous", previous);
                if group.len() > 1
                    && !group
                        .iter()
                        .any(|item| crate::zip_members::zip_of(&item.batch).is_some())
                {
                    return group
                        .into_iter()
                        .flat_map(|item| {
                            self.store_group(vec![item], std::sync::Arc::clone(&store))
                        })
                        .collect();
                }
                return errors(&group, error);
            }
        };
        let mut installed = BTreeSet::new();
        let mut failed = BTreeMap::<BlobHash, String>::new();
        // In primary mode Sinex holds the bytes already: the rows committed
        // after it acknowledged them.
        if self.sinex.is_some() {
            for item in &mut group {
                item.files.clear();
            }
        }
        // The files were synced where they were prepared; one batch links
        // them and syncs each directory it touched once.
        let mut files = store.batch(hashed_blobs::Durability::PerFile);
        let mut adding = BTreeSet::new();
        for item in &mut group {
            let subject = item.batch.input.id.clone();
            for file in item.files.drain(..) {
                let hash = file.hash();
                if adding.contains(&hash) || failed.contains_key(&hash) {
                    continue;
                }
                // Bytes stored before need no second copy.
                let present: bool = match self.archive.query_row(
                    "SELECT state = 'present' FROM blob WHERE hash = ?1",
                    [&hash.0[..]],
                    |r| r.get(0),
                ) {
                    Ok(present) => present,
                    Err(error) => {
                        failed.insert(hash, error.to_string());
                        continue;
                    }
                };
                if present {
                    continue;
                }
                let crate::staging::Prepared::Local { prepared, .. } = file.file else {
                    failed.insert(hash, "bytes to send to Sinex reached the blob store".into());
                    continue;
                };
                let added = crate::fault::check("import.install", &subject)
                    .and_then(|()| files.add(*prepared).map_err(StoreError::from));
                match added {
                    Ok(_) => {
                        adding.insert(hash);
                    }
                    Err(error) => {
                        failed.insert(hash, error.to_string());
                    }
                }
            }
        }
        match crate::timing::time(crate::timing::Stage::Sync, || files.commit()) {
            Ok(_) => installed = adding,
            Err(error) => {
                for hash in adding {
                    failed.insert(hash, error.to_string());
                }
            }
        }
        let mut outcomes: Vec<Option<String>> = group
            .iter()
            .map(|item| {
                item.batch
                    .new_blob
                    .iter()
                    .chain(&item.batch.blobs)
                    .find_map(|new_blob| failed.get(&new_blob.hash).cloned())
            })
            .collect();
        let finished: Result<()> = (|| {
            let tx = self.archive.transaction()?;
            for hash in &installed {
                let size = i64::try_from(store.stored_size(&hashed_blobs::Hash(hash.0))?).map_err(
                    |_| StoreError::Invalid("blob exceeds archive address space".into()),
                )?;
                tx.execute(
                    "UPDATE blob SET state = 'present', stored_size = ?2 WHERE hash = ?1",
                    params![&hash.0[..], size],
                )?;
            }
            for item in &group {
                crate::fault::check("import.installed", &item.batch.input.id)?;
            }
            crate::write_lock::commit(tx)?;
            self.report.transactions += 1;
            Ok(())
        })();
        if let Err(error) = finished {
            for outcome in outcomes.iter_mut().filter(|o| o.is_none()) {
                *outcome = Some(error.to_string());
            }
        }
        if let Err(error) = self.archive.pragma_update(None, "synchronous", previous) {
            for outcome in outcomes.iter_mut().filter(|o| o.is_none()) {
                *outcome = Some(error.to_string());
            }
        }
        let mut parsed = Vec::new();
        for ((mut item, stored), outcome) in group.into_iter().zip(&stored).zip(&outcomes) {
            if outcome.is_none() && !item.batch.parse.is_empty() {
                item.batch.change = Change::Reparse {
                    chain: stored.chain,
                };
                item.batch.continues = false;
                item.batch.new_blob = None;
                item.batch.observation = None;
                item.batch.blobs.clear();
                parsed.push(Box::new(item.batch));
            }
        }
        if !parsed.is_empty() {
            crate::timing::time(crate::timing::Stage::Rows, || self.write_group(&parsed));
        }
        drop(parsed);
        stored
            .into_iter()
            .zip(outcomes)
            .map(|(stored, error)| {
                if let Some(reason) = error {
                    Err(StoreError::InputPending {
                        input: stored.input,
                        reason,
                    })
                } else if self.report.failed_inputs.contains(&stored.input) {
                    Err(StoreError::Invalid(format!(
                        "input {}: parsed rows were refused",
                        stored.input
                    )))
                } else {
                    Ok(stored)
                }
            })
            .collect()
    }

    /// Checks every batch, then in one transaction writes the rows of its
    /// bytes: a `writing` row for each private file, and the revision,
    /// extents and observation that use them.
    fn write_rows_first(&mut self, group: &mut [PreparedBatch]) -> Result<Vec<StoredInput>> {
        for item in group.iter() {
            self.check_staged(item)?;
        }
        // A member read from a ZIP needs its ZIP stored, earlier in this group
        // or before.
        let staged: BTreeSet<&str> = group
            .iter()
            .map(|item| item.batch.input.id.as_str())
            .collect();
        for item in group.iter() {
            if let Some(zip) = crate::zip_members::zip_of(&item.batch)
                && !staged.contains(zip.input.as_str())
            {
                crate::zip_members::require_original(&self.archive, &zip.input)?;
            }
        }
        let tx = self.archive.transaction()?;
        let seq = bump_counter(&tx)?;
        if self.sinex.is_none() {
            for item in group.iter() {
                for file in &item.files {
                    tx.execute(
                        "INSERT OR IGNORE INTO blob(hash, size, stored_size, place, encoding, state)
                         VALUES (?1, ?2, 0, 'file', 'plain', 'writing')",
                        params![&file.hash().0[..], file.len() as i64],
                    )?;
                }
            }
        }
        let mut stored = Vec::new();
        let mut seen = BTreeSet::new();
        for item in group.iter_mut() {
            let PreparedBatch { batch: b, files } = item;
            // An immutable import named by its bytes may be stored already, or
            // appear twice in a group (equal ZIP members).
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
            let (chain, rows) = write_bytes(&tx, b, seq)?;
            let chain = chain
                .ok_or_else(|| StoreError::Invalid("stored bytes produced no chain".into()))?;
            if let (Some(sinex), Some(rows)) = (&self.sinex, &rows) {
                let new_blob = b.new_blob.as_ref().map(|new_blob| new_blob.hash);
                let file = new_blob.and_then(|hash| files.iter().find(|f| f.hash() == hash));
                send_to_sinex(sinex.as_ref(), &tx, rows, file, b.new_blob.as_ref())?;
            }
            crate::fault::check("import.rows", &b.input.id)?;
            let already_stored = again
                || (matches!(b.change, Change::Unchanged)
                    && match crate::staging::require_present(&tx, "", &b.input.id, Some(chain)) {
                        Ok(()) => true,
                        Err(StoreError::InputPending { .. }) => false,
                        Err(error) => return Err(error),
                    });
            stored.push(StoredInput {
                input: b.input.id.clone(),
                chain,
                length: b.length,
                already_stored,
            });
        }
        crate::write_lock::commit(tx)?;
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
            if matches!(new_blob.stored, Stored::File { .. })
                && !item
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
        if self.excisions
            && let Some(observation) = &b.observation
        {
            for new_blob in b.new_blob.iter().chain(&b.blobs) {
                if crate::excise::raw_import(
                    &self.archive,
                    &b.input.id,
                    &new_blob.hash,
                    observation,
                )? {
                    return Err(StoreError::InputRefused {
                        input: b.input.id.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Finishes the blob rows a stop or a failed install left behind
    /// ([`crate::blobs::finish_pending`]) and reports the inputs whose bytes are
    /// still `writing` without a file; the loop reads them again.
    pub(super) fn finish_writing(&mut self, store: &hashed_blobs::Store) -> Result<Finished> {
        let write = crate::write_lock::hold(&self.archive)?;
        let completed = crate::blobs::finish_pending(&self.archive, store)?;
        drop(write);
        if completed > 0 {
            self.report.transactions += 1;
        }
        let mut pending = crate::staging::pending_blobs(&self.archive, "")?;
        for item in &mut pending {
            item.state = crate::staging::PendingState::Missing;
        }
        Ok(Finished { completed, pending })
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

/// Sends what one step stored to Sinex, before its rows commit: its new
/// bytes as a part of the input, or, when it found none, the rows alone.
pub(super) fn send_to_sinex(
    sinex: &dyn crate::primary::Sinex,
    tx: &Connection,
    rows: &crate::input_rows::InputRows,
    file: Option<&PreparedBytes>,
    new_blob: Option<&NewBlob>,
) -> Result<()> {
    let input = input_name(tx, rows)?;
    let Some(extent) = &rows.extent else {
        if rows.observation.is_some() {
            let change = crate::primary::RegisterChange::Observation {
                input,
                stored: Box::new(rows.clone()),
            };
            let material = sinex.send_change(&change)?;
            crate::primary::change_material(tx, &change, &material)?;
        }
        return Ok(());
    };
    let (index, previous) = crate::primary::next_part(tx, &rows.input, extent.chain)?;
    let mut bytes: Box<dyn std::io::Read> =
        match (file.map(|f| &f.file), new_blob.map(|p| &p.stored)) {
            (Some(crate::staging::Prepared::Part { file, .. }), _) => Box::new(file.reopen()?),
            (_, Some(Stored::Inline(data))) => {
                Box::new(std::io::Cursor::new(zstd::bulk::decompress(
                    data,
                    usize::try_from(extent.length)
                        .map_err(|_| StoreError::Invalid("a new blob exceeds memory".into()))?,
                )?))
            }
            _ => {
                return Err(StoreError::Invalid(
                    "Sinex input has no bytes to send".into(),
                ));
            }
        };
    let material = sinex.send_part(
        &crate::primary::Part {
            input,
            index,
            previous,
            hash: extent.blob,
            length: extent.length as u64,
            rows: rows.clone(),
        },
        &mut bytes,
    )?;
    crate::primary::part_material(tx, rows, index, &material)
}

/// The input's origin, its latest path and the host it was seen on.
fn input_name(
    tx: &Connection,
    rows: &crate::input_rows::InputRows,
) -> Result<crate::primary::InputName> {
    let (path, host) = match &rows.observation {
        Some(o) => (o.path.clone(), o.host.clone()),
        None => tx
            .prepare_cached(
                "SELECT o.path, coalesce((SELECT host FROM machine m WHERE m.id = o.machine
                     ORDER BY first_seen DESC LIMIT 1), '')
                 FROM observation o WHERE o.input = ?1 ORDER BY o.rowid DESC LIMIT 1",
            )?
            .query_row([&rows.input], |r| Ok((r.get(0)?, r.get(1)?)))
            .optional()?
            .ok_or_else(|| {
                StoreError::Invalid(format!("input {} has no observation", rows.input))
            })?,
    };
    Ok(crate::primary::InputName {
        origin: rows.format.clone(),
        path,
        host,
    })
}

#[cfg(test)]
mod tests;
