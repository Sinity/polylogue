//! Bytes into the archive in one order: the blob's row first, in state
//! `writing`, then the file under its content name, then the row `present`.
//! A stop between them leaves `writing` rows, which [`Inner::finish_writing`]
//! completes when the file is there and otherwise reports.
use super::*;
use crate::staging::{Finished, PreparedBatch, PreparedBytes, StoredInput};
use polylogue_model::BlobHash;
use std::collections::{BTreeMap, BTreeSet};

impl Inner {
    /// Stores one acquisition's bytes: a batch with an observation and no
    /// parsed rows.
    pub(super) fn store_bytes(
        &mut self,
        file: PreparedBytes,
        mut batch: Batch,
    ) -> Result<StoredInput> {
        if file.owner != self.location.root
            || batch.piece.is_some()
            || !batch.blobs.is_empty()
            || !batch.parse.is_empty()
            || batch.observation.is_none()
        {
            return Err(StoreError::Invalid(
                "storing bytes takes one private file of this archive and a batch without rows"
                    .into(),
            ));
        }
        batch.piece = Some(Piece {
            hash: file.hash(),
            size: file.len(),
            stored: Stored::File { stored_size: 0 },
        });
        let store = std::sync::Arc::clone(&file.store);
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

    /// Stores a group of batches whose pieces include private files, then
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
        let write = match crate::publication::write(&self.archive) {
            Ok(write) => write,
            Err(error) => return errors(&group, error),
        };
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
                drop(write);
                if group.len() > 1
                    && !group
                        .iter()
                        .any(|item| crate::containers::coordinate(&item.batch).is_some())
                {
                    return group
                        .into_iter()
                        .flat_map(|item| self.store_group(vec![item], std::sync::Arc::clone(&store)))
                        .collect();
                }
                return errors(&group, error);
            }
        };
        let mut installed = BTreeSet::new();
        let mut failed = BTreeMap::<BlobHash, String>::new();
        for item in &mut group {
            let subject = item.batch.input.id.clone();
            for file in item.files.drain(..) {
                let hash = file.hash();
                if installed.contains(&hash) || failed.contains_key(&hash) {
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
                match file
                    .prepared
                    .install_checked::<StoreError>(|| crate::fault::check("import.install", &subject))
                {
                    Ok(_) => {
                        installed.insert(hash);
                    }
                    Err(error) => {
                        failed.insert(hash, error.to_string());
                    }
                }
            }
        }
        let mut outcomes: Vec<Option<String>> = group
            .iter()
            .map(|item| {
                item.batch
                    .piece
                    .iter()
                    .chain(&item.batch.blobs)
                    .find_map(|piece| failed.get(&piece.hash).cloned())
            })
            .collect();
        let finished: Result<()> = (|| {
            let tx = self.archive.transaction()?;
            for hash in &installed {
                let size = i64::try_from(store.stored_size(&hashed_blobs::Hash(hash.0))?)
                    .map_err(|_| StoreError::Invalid("blob exceeds archive address space".into()))?;
                tx.execute(
                    "UPDATE blob SET state = 'present', stored_size = ?2 WHERE hash = ?1",
                    params![&hash.0[..], size],
                )?;
            }
            for item in &group {
                crate::fault::check("import.installed", &item.batch.input.id)?;
            }
            tx.commit()?;
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
        drop(write);
        let mut parsed = Vec::new();
        for ((mut item, stored), outcome) in group.into_iter().zip(&stored).zip(&outcomes) {
            if outcome.is_none() && !item.batch.parse.is_empty() {
                item.batch.change = Change::Reparse {
                    chain: stored.chain,
                };
                item.batch.continues = false;
                item.batch.piece = None;
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
                    Err(StoreError::InputUnfinished {
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
            if let Some(container) = crate::containers::coordinate(&item.batch)
                && !staged.contains(container.input.as_str())
            {
                crate::containers::require_original(&self.archive, &container.input)?;
            }
        }
        let tx = self.archive.transaction()?;
        let seq = bump_counter(&tx)?;
        for item in group.iter() {
            for file in &item.files {
                tx.execute(
                    "INSERT OR IGNORE INTO blob(hash, size, stored_size, place, encoding, state)
                     VALUES (?1, ?2, 0, 'file', 'plain', 'writing')",
                    params![&file.hash().0[..], file.len() as i64],
                )?;
            }
        }
        let mut stored = Vec::new();
        let mut seen = BTreeSet::new();
        for item in group.iter_mut() {
            let b = &mut item.batch;
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
            if b.input.route_rank == 2 && b.piece.is_some() && (known.is_some() || again) {
                let piece = b.piece.as_ref().expect("checked piece");
                let same: bool = tx.query_row(
                    "SELECT count(*) = 1 AND min(offset) = 0 AND min(length) = ?3 AND min(blob) = ?4
                     FROM extent WHERE input = ?1 AND chain = (SELECT chain FROM revision WHERE input = ?1 ORDER BY number DESC LIMIT 1)
                     AND ?2 = ?3",
                    params![b.input.id, known.map_or(b.length as i64, |k| k.1), b.length as i64, &piece.hash.0[..]],
                    |r| r.get(0),
                )?;
                if !same {
                    return Err(StoreError::Conflict(
                        "an import named by its bytes holds other bytes".into(),
                    ));
                }
                b.change = Change::Unchanged;
                b.piece = None;
            }
            let chain = write_bytes(&tx, b, seq)?
                .ok_or_else(|| StoreError::Invalid("stored bytes produced no chain".into()))?;
            crate::fault::check("import.rows", &b.input.id)?;
            let already_stored = again
                || (matches!(b.change, Change::Unchanged)
                    && match crate::staging::require_present(&tx, "", &b.input.id, Some(chain)) {
                        Ok(()) => true,
                        Err(StoreError::InputUnfinished { .. }) => false,
                        Err(error) => return Err(error),
                    });
            stored.push(StoredInput {
                input: b.input.id.clone(),
                chain,
                length: b.length,
                already_stored,
            });
        }
        tx.commit()?;
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
        for piece in b.piece.iter().chain(&b.blobs) {
            if matches!(piece.stored, Stored::File { .. })
                && !item
                    .files
                    .iter()
                    .any(|file| file.hash() == piece.hash && file.len() == piece.size)
            {
                return Err(StoreError::Invalid(
                    "a file piece has no private file".into(),
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
                .piece
                .iter()
                .chain(&b.blobs)
                .any(|piece| piece.hash == file.hash() && piece.size == file.len())
            {
                return Err(StoreError::Invalid(
                    "a private file belongs to no piece of its batch".into(),
                ));
            }
        }
        if b.input.route_rank == 2
            && let Some(piece) = &b.piece
            && (b.input.id != piece.hash.to_string() || piece.size != b.length || b.continues)
        {
            return Err(StoreError::Invalid(
                "an import must be named by its complete bytes".into(),
            ));
        }
        if let Some(observation) = &b.observation {
            for piece in b.piece.iter().chain(&b.blobs) {
                if self
                    .refused
                    .raw_import(&b.input.id, &piece.hash, observation)?
                {
                    return Err(StoreError::InputRefused {
                        input: b.input.id.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Completes the blobs a stop left `writing`: one whose file is under its
    /// content name becomes `present` (installation is a rename, so a file
    /// there is whole); the others are reported, and the loop reads their
    /// inputs again.
    pub(super) fn finish_writing(&mut self, store: &hashed_blobs::Store) -> Result<Finished> {
        let write = crate::publication::write(&self.archive)?;
        let mut report = Finished::default();
        let waiting = crate::staging::unfinished(&self.archive, "")?;
        let mut missing = BTreeSet::new();
        let tx = self.archive.transaction()?;
        for item in &waiting {
            let hash = hashed_blobs::Hash(item.blob.0);
            if !store.contains(&hash) {
                missing.insert(item.blob);
                continue;
            }
            let size = i64::try_from(store.stored_size(&hash)?)
                .map_err(|_| StoreError::Invalid("blob exceeds archive address space".into()))?;
            report.completed += tx.execute(
                "UPDATE blob SET state = 'present', stored_size = ?2 WHERE hash = ?1 AND state = 'writing'",
                params![&item.blob.0[..], size],
            )? as u64;
        }
        tx.commit()?;
        drop(write);
        if report.completed > 0 {
            self.report.transactions += 1;
        }
        report.unfinished = crate::staging::unfinished(&self.archive, "")?;
        for item in &mut report.unfinished {
            if missing.contains(&item.blob) {
                item.state = crate::staging::UnfinishedState::Missing;
            }
        }
        Ok(report)
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
