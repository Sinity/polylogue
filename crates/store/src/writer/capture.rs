//! Captured bytes reach the index in the one order the two writers need:
//! their blobs are put into the byte home ([`crate::bytes`]), their input,
//! revision, extent and observation rows commit in the archive
//! ([`crate::archive::writer`]), and only then are their parsed rows handed
//! to the index writer, as parses of stored bytes. The first two steps run
//! on the calling thread; puts waiting together share one sync and groups
//! waiting together share one archive commit.
use super::*;
use crate::archive::writer::{Group, Outcome};
use crate::staging::{PreparedBatch, PreparedBytes, StoredInput};
use polylogue_model::LockExt as _;

/// Parsed rows of bytes the archive has committed, handed to the index
/// writer together. An atomic group's rows commit together or not at all.
/// A group with no rows waits for the index to take in its archive commit.
pub(super) struct Parsed {
    pub batches: Vec<Batch>,
    pub atomic: bool,
    pub reply: Reply,
}

/// Who waits for a [`Parsed`] group.
pub(super) enum Reply {
    /// Nobody: a failure is reported at the next flush.
    Nobody,
    /// A capture group, answered with where each of its batches is stored:
    /// `stored` for every batch of the group, `positions` naming the batches
    /// whose rows these are.
    Stored {
        stored: Vec<StoredInput>,
        positions: Vec<usize>,
        reply: SyncSender<Vec<Result<StoredInput>>>,
    },
}

// A package or one consistent database read is one capture group. A pool
// flush may contain several independent file samples.
fn is_atomic_capture_group(items: &[PreparedBatch]) -> bool {
    items.iter().any(|item| {
        crate::container_members::container_of(&item.batch).is_some()
            || item
                .batch
                .observation
                .as_ref()
                .is_some_and(|observation| observation.import.is_some())
            || item
                .batch
                .capture_read
                .as_ref()
                .is_some_and(|capture_read| {
                    capture_read.route == crate::capture_read::ReadRoute::ExplicitImport
                        || matches!(
                            capture_read.generation,
                            Some(crate::capture_read::ObservedState::Database { .. })
                        )
                })
    })
}

impl Batch {
    /// The batch's archive rows without its parsed rows.
    fn archive_rows(&self) -> Self {
        Self {
            input: self.input.clone(),
            change: self.change.clone(),
            observation: self.observation.clone(),
            continues: self.continues,
            new_blob: self.new_blob.clone(),
            blobs: self.blobs.clone(),
            length: self.length,
            parse: Vec::new(),
            release: InFlight::default(),
            capture_read: self.capture_read.clone(),
        }
    }

    /// Whether the batch writes archive rows, beyond naming stored bytes.
    fn stores(&self) -> bool {
        !matches!(self.change, Change::Reparse { .. })
            || self.new_blob.is_some()
            || self.observation.is_some()
    }

    /// The batch as parsed rows of the stored chain `chain`.
    fn into_parsed(mut self, chain: i64) -> Self {
        self.change = Change::Reparse { chain };
        self.continues = false;
        self.new_blob = None;
        self.observation = None;
        self.blobs.clear();
        self
    }
}

impl WriterSender {
    /// Hands one batch over without waiting for its commit. A batch with
    /// archive rows and parsed rows commits its archive rows first, here; a
    /// failure after the hand-over is reported at the next flush.
    pub fn send(&self, mut batch: Batch) -> Result<()> {
        prepare_batches(&self.0.location, &mut [&mut batch])?;
        if batch.stores() {
            if batch.parse.is_empty() {
                return self.0.archive.hand_over(Group::Captured {
                    batches: vec![batch],
                    added: Default::default(),
                    atomic: false,
                });
            }
            let committed = self.0.archive.commit(Group::Captured {
                batches: vec![batch.archive_rows()],
                added: Default::default(),
                atomic: false,
            })?;
            let Outcome::Stored(stored) = committed.outcome else {
                return Err(StoreError::internal("a captured group restored a record"));
            };
            let chain = stored
                .first()
                .or_internal("a captured batch was not stored")?
                .chain;
            batch = batch.into_parsed(chain);
        }
        self.0
            .send(Msg::Batch(Parsed {
                batches: vec![batch],
                atomic: false,
                reply: Reply::Nobody,
            }))
            .map_err(|_| StoreError::WriterGone)
    }

    /// Stores a group of batches and their prepared files and waits for
    /// their parsed rows to commit.
    pub fn store_group(&self, group: Vec<PreparedBatch>) -> Result<Vec<Result<StoredInput>>> {
        self.enqueue_group(group)?
            .recv()
            .map_err(|_| StoreError::WriterGone)
    }

    /// Puts the group's files and commits its archive rows, then hands its
    /// parsed rows to the index writer; the receiver answers once they are
    /// committed, with where each batch is stored. Parsed rows of independent
    /// inputs that are refused together are written again batch by batch, so
    /// one bad batch costs only itself; one capture group (a package, one
    /// consistent database read) succeeds or fails as a whole.
    pub fn enqueue_group(
        &self,
        mut group: Vec<PreparedBatch>,
    ) -> Result<Receiver<Vec<Result<StoredInput>>>> {
        let mut batches: Vec<_> = group.iter_mut().map(|item| &mut item.batch).collect();
        prepare_batches(&self.0.location, &mut batches)?;
        let (reply, wait) = sync_channel(1);
        if group.is_empty() {
            let _ = reply.send(Vec::new());
            return Ok(wait);
        }
        let atomic = is_atomic_capture_group(&group);
        let stored = match self.store_rows(&mut group, atomic) {
            Ok(stored) => stored,
            Err(StoreError::WriterGone) => return Err(StoreError::WriterGone),
            Err(error) => {
                let _ = reply.send(errors(group.len(), error));
                return Ok(wait);
            }
        };
        let mut parsed = Vec::new();
        let mut positions = Vec::new();
        for (position, (item, stored)) in group.into_iter().zip(&stored).enumerate() {
            // A batch without parsed rows is done with here, which releases it.
            if !item.batch.parse.is_empty() {
                parsed.push(item.batch.into_parsed(stored.chain));
                positions.push(position);
            }
        }
        // A group without parsed rows is answered once the index has taken
        // in what its commit changed (the progress of the chains it stored),
        // as one with parsed rows is once they commit.
        self.0
            .send(Msg::Batch(Parsed {
                batches: parsed,
                atomic,
                reply: Reply::Stored {
                    stored,
                    positions,
                    reply,
                },
            }))
            .map_err(|_| StoreError::WriterGone)?;
        Ok(wait)
    }

    /// Puts the group's files and commits its archive rows.
    fn store_rows(&self, group: &mut [PreparedBatch], atomic: bool) -> Result<Vec<StoredInput>> {
        for item in group.iter() {
            check_files(item, &self.0.location.root)?;
        }
        let files: Vec<PreparedBytes> = group
            .iter_mut()
            .flat_map(|item| item.files.drain(..))
            .collect();
        let put = if files.is_empty() {
            crate::bytes::Put::default()
        } else {
            self.0.bytes.put_all(files)?
        };
        put.report(&mut self.0.archive.report().locked());
        crate::fault::check("capture.between_put_and_commit", "")?;
        let committed = self.0.archive.commit(Group::Captured {
            batches: group.iter().map(|item| item.batch.archive_rows()).collect(),
            added: put.added,
            atomic,
        })?;
        crate::fault::check("capture.after_commit", "")?;
        let Outcome::Stored(stored) = committed.outcome else {
            return Err(StoreError::internal("a captured group restored a record"));
        };
        self.0.archive.report().locked().batches += group
            .iter()
            .filter(|item| item.batch.parse.is_empty())
            .count() as u64;
        Ok(stored)
    }

    /// Stores one import's bytes: a batch with an observation and no parsed
    /// rows.
    pub fn store_bytes(&self, file: PreparedBytes, mut batch: Batch) -> Result<StoredInput> {
        if file.owner != self.0.location.root
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
        batch.new_blob = Some(NewBlob::whole(file.hash(), file.len()));
        self.store_group(vec![PreparedBatch {
            batch,
            files: vec![file],
        }])?
        .pop()
        .or_internal("a stored group answered nothing")?
    }

    /// Stores bytes no input holds as one blob; `file` is the private file
    /// [`crate::Archive::prepare_blob`] wrote for them.
    pub fn store_blob(
        &self,
        new_blob: crate::NewBlob,
        file: Option<PreparedBytes>,
    ) -> Result<BlobHash> {
        let Some(file) = file else {
            let conn = crate::archive::open_read(&self.0.location.archive_db())?;
            return match crate::blobs::load_extent(&conn, "", &hashed_blobs::Hash(new_blob.hash.0))?
            {
                Some(_) => Ok(new_blob.hash),
                None => Err(StoreError::Invalid("new blob has no prepared bytes".into())),
            };
        };
        if file.owner != self.0.location.root
            || file.hash() != new_blob.hash
            || file.len() != new_blob.size
        {
            return Err(StoreError::Invalid(
                "prepared bytes do not match the new blob".into(),
            ));
        }
        let put = self.0.bytes.put_all(vec![file])?;
        put.report(&mut self.0.archive.report().locked());
        Ok(new_blob.hash)
    }
}

/// What a batch's private files must satisfy before any of them is put.
fn check_files(item: &PreparedBatch, root: &std::path::Path) -> Result<()> {
    let b = &item.batch;
    let parts = || b.new_blob.iter().chain(&b.blobs).flat_map(NewBlob::parts);
    for new_blob in parts() {
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
        if file.owner != root {
            return Err(StoreError::Invalid(
                "a private file belongs to another archive".into(),
            ));
        }
        if !parts().any(|new_blob| new_blob.hash == file.hash() && new_blob.size == file.len()) {
            return Err(StoreError::Invalid(
                "a private file belongs to no new_blob of its batch".into(),
            ));
        }
    }
    Ok(())
}

fn errors(count: usize, error: StoreError) -> Vec<Result<StoredInput>> {
    if count == 1 {
        return vec![Err(error)];
    }
    (0..count).map(|_| Err(error.for_group_member())).collect()
}

/// The answer for a batch whose bytes are `stored` and whose rows `error`
/// lost: a refusal of the rows' content keeps where the bytes are.
fn rows_refused(stored: &StoredInput, error: StoreError) -> Result<StoredInput> {
    if polylogue_model::Caused::cause(&error).recurs() {
        return Err(StoreError::RowsRefused {
            stored: stored.clone(),
            reason: error.to_string(),
            cause: polylogue_model::Caused::cause(&error),
        });
    }
    Err(error)
}

/// Where each batch of a capture group is stored, given what became of the
/// parsed rows of the batches at `positions`. When the rows of an atomic
/// group fail, every batch of the group answers with the failure.
pub(super) fn stored_answer(
    stored: Vec<StoredInput>,
    positions: &[usize],
    rows: Vec<Result<()>>,
    atomic: bool,
) -> Vec<Result<StoredInput>> {
    let failed = rows.iter().position(Result::is_err);
    if atomic && let Some(at) = failed {
        let Some(Err(error)) = rows.into_iter().nth(at) else {
            unreachable!("a failed row")
        };
        if stored.len() == 1 {
            return vec![rows_refused(&stored[0], error)];
        }
        return stored
            .iter()
            .map(|stored| rows_refused(stored, error.for_group_member()))
            .collect();
    }
    let mut answers: Vec<Option<Result<()>>> = stored.iter().map(|_| None).collect();
    for (position, row) in positions.iter().zip(rows) {
        answers[*position] = Some(row);
    }
    stored
        .into_iter()
        .zip(answers)
        .map(|(stored, row)| match row {
            Some(Err(error)) => rows_refused(&stored, error),
            _ => Ok(stored),
        })
        .collect()
}

#[cfg(test)]
mod tests;
