//! Where blob bytes are kept, behind two operations: `put` makes encoded
//! bytes durable and known to the archive before it returns, and `open`
//! reads a range of a blob's decoded bytes back. Nothing above this module
//! knows which home answers: every blob row's storage descriptor names the
//! file and range that hold it, so the catalog reader ([`crate::blobs::Blobs`])
//! and the stored-chain reader read either home the same way.
//!
//! Two homes: [`Segments`], the append-only segment log of the shared
//! `hashed-blobs` container, and [`Directory`], one file per blob hash. Each
//! home has one thread that takes every put waiting, stores their bytes with
//! one sync, and records their blob rows (and the segment's committed end) in
//! one small `archive.db` transaction of its own before any of them returns.
//! A blob put before a crash is therefore found by the next put of the same
//! bytes, which stores nothing. This thread writes only the `blob` and
//! `segment` tables; every other archive row is the archive writer's
//! ([`crate::archive::writer`]).
//!
//! A small cache keeps the decoded bytes of blobs put in the last moments
//! once they are first opened, so the reads that follow a capture (the
//! parse, then the next capture's proof of the stored tail) are answered
//! from memory. It fills at the first open, not at the put: a put copies
//! the prepared encoding as it is and never decodes it.
use crate::{Archive, Result, StoreError};
use polylogue_model::{BlobHash, LockExt as _};
use rusqlite::Connection;
use std::collections::{BTreeMap, VecDeque};
use std::io::Read;
use std::ops::Range;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

mod directory;
mod segments;
pub use directory::Directory;
pub(crate) use segments::CURRENT as CURRENT_SEGMENT;
pub use segments::Segments;

/// Bytes hashed and encoded off the writers, ready for a home to store.
pub type PreparedBlob = crate::staging::PreparedBytes;

/// Where blob bytes are kept. Nothing above this trait knows which home
/// answers.
pub trait Bytes: Send + Sync {
    /// Stores the encoded blob; durable on this host, and known to the
    /// archive, before it returns. Putting bytes already stored returns their
    /// hash and writes nothing.
    fn put(&self, blob: PreparedBlob) -> Result<BlobHash> {
        let hash = blob.hash();
        self.put_all(vec![blob])?;
        Ok(hash)
    }
    /// Stores several blobs with one sync, as [`Bytes::put`] stores one.
    fn put_all(&self, blobs: Vec<PreparedBlob>) -> Result<Put>;
    /// Opens a range of a blob's decoded bytes. A blob put a moment ago may
    /// be answered from memory.
    fn open(&self, hash: &BlobHash, range: Range<u64>) -> Result<Box<dyn Read + Send>>;
    /// Ends the home's thread once the puts handed to it are answered;
    /// later puts fail with [`StoreError::WriterGone`]. Its files are
    /// released, so another writer can open them.
    fn close(&self);
}

/// What one [`Bytes::put_all`] stored.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Put {
    /// The blobs this put added to the archive, with their decoded sizes;
    /// blobs it found stored are not named.
    pub added: BTreeMap<BlobHash, u64>,
    /// Syncs of the home this put caused; a sync shared with other puts
    /// waiting at the same time is counted by the first of them only.
    pub syncs: u64,
    pub bytes_written: u64,
    pub appended: u64,
    pub deduplicated: u64,
}

impl Put {
    /// Adds this put's figures to a writer's report.
    pub fn report(&self, report: &mut crate::WriteReport) {
        report.segment_syncs += self.syncs;
        report.segment_bytes_written += self.bytes_written;
        report.blobs_appended += self.appended;
        report.blobs_deduplicated += self.deduplicated;
    }
}

/// What a home stored for one group of blobs, before their rows commit.
struct Stored {
    /// One descriptor for each blob, in the order given.
    extents: Vec<hashed_blobs::Extent>,
    /// The segment's end once these rows commit; `None` for a home without one.
    end: Option<u64>,
    syncs: u64,
    bytes_written: u64,
    /// A token the home needs back once the rows committed.
    acknowledge: Option<hashed_blobs::SyncedGroup>,
}

/// One kind of place blob bytes are kept, driven by [`PutThread`].
trait Home: Send + 'static {
    /// Stores `blobs` durably, opening each prepared encoding only as it
    /// is copied, so a group of any size holds no descriptor per member.
    fn store(&mut self, conn: &Connection, blobs: Vec<PreparedBlob>) -> Result<Stored>;
    /// The rows of `stored` committed.
    fn recorded(&mut self, stored: Stored) -> Result<()>;
    /// A store or its rows failed: the home starts again from what the
    /// archive records.
    fn reset(&mut self);
}

/// Blobs put in the last moments, and the decoded bytes of those opened
/// since, at most [`RECENT_BYTES`] of them.
#[derive(Default)]
struct Recent {
    put: VecDeque<BlobHash>,
    blobs: VecDeque<(BlobHash, Arc<[u8]>)>,
    bytes: usize,
}

/// How many recent puts are remembered.
const RECENT_PUTS: usize = 4096;
/// The decoded size of the largest blob kept in memory.
const RECENT_BLOB: u64 = 256 << 10;
/// The memory recent blobs may take together.
const RECENT_BYTES: usize = 8 << 20;

impl Recent {
    fn put(&mut self, hash: BlobHash) {
        self.put.push_back(hash);
        while self.put.len() > RECENT_PUTS {
            self.put.pop_front();
        }
    }

    fn was_put(&self, hash: &BlobHash) -> bool {
        self.put.contains(hash)
    }

    fn keep(&mut self, hash: BlobHash, bytes: Arc<[u8]>) {
        if self.blobs.iter().any(|(known, _)| *known == hash) {
            return;
        }
        self.bytes += bytes.len();
        self.blobs.push_back((hash, bytes));
        while self.bytes > RECENT_BYTES
            && let Some((_, dropped)) = self.blobs.pop_front()
        {
            self.bytes -= dropped.len();
        }
    }

    fn get(&self, hash: &BlobHash) -> Option<Arc<[u8]>> {
        self.blobs
            .iter()
            .find(|(known, _)| known == hash)
            .map(|(_, bytes)| Arc::clone(bytes))
    }
}

struct Request {
    blobs: Vec<PreparedBlob>,
    reply: Sender<Result<Put>>,
}

/// The thread of one home: it takes every put waiting, stores them with one
/// sync and commits their rows before answering any of them.
struct PutThread {
    requests: Mutex<Option<Sender<Request>>>,
    handle: Mutex<Option<std::thread::JoinHandle<()>>>,
    recent: Arc<Mutex<Recent>>,
    root: std::path::PathBuf,
    catalog: crate::blobs::Blobs,
}

impl PutThread {
    fn start(archive: &Archive, name: &str, home: impl Home) -> Result<Self> {
        let conn = crate::archive::open_archive_writer(archive)?;
        let (tx, rx) = channel();
        let recent = Arc::new(Mutex::new(Recent::default()));
        let thread_recent = Arc::clone(&recent);
        let handle = polylogue_model::spawn_named(name, move || {
            run(conn, home, &rx, &thread_recent);
        })?;
        Ok(Self {
            requests: Mutex::new(Some(tx)),
            handle: Mutex::new(Some(handle)),
            recent,
            root: archive.root.clone(),
            catalog: archive.blobs()?,
        })
    }

    fn put_all(&self, blobs: Vec<PreparedBlob>) -> Result<Put> {
        if blobs.is_empty() {
            return Ok(Put::default());
        }
        for blob in &blobs {
            if blob.owner != self.root {
                return Err(StoreError::Invalid(
                    "a prepared blob belongs to another archive".into(),
                ));
            }
        }
        let (reply, wait) = channel();
        self.requests
            .locked()
            .as_ref()
            .ok_or(StoreError::WriterGone)?
            .send(Request { blobs, reply })
            .map_err(|_| StoreError::WriterGone)?;
        wait.recv().map_err(|_| StoreError::WriterGone)?
    }

    fn open(&self, hash: &BlobHash, range: Range<u64>) -> Result<Box<dyn Read + Send>> {
        if range.start > range.end {
            return Err(StoreError::Invalid(
                "an empty or reversed byte range".into(),
            ));
        }
        let cached = self.recent.locked().get(hash);
        let cached = match cached {
            Some(bytes) => Some(bytes),
            None if self.recent.locked().was_put(hash) => {
                let extent = self.catalog.extent(&hashed_blobs::Hash(hash.0))?;
                if extent.decoded_len <= RECENT_BLOB {
                    let mut bytes = Vec::new();
                    self.catalog
                        .read(&hashed_blobs::Hash(hash.0))?
                        .read_to_end(&mut bytes)?;
                    let bytes: Arc<[u8]> = bytes.into();
                    self.recent.locked().keep(*hash, Arc::clone(&bytes));
                    Some(bytes)
                } else {
                    None
                }
            }
            None => None,
        };
        if let Some(bytes) = cached {
            let end = usize::try_from(range.end.min(bytes.len() as u64))
                .map_err(|_| StoreError::Invalid("range exceeds address space".into()))?;
            let start = usize::try_from(range.start)
                .map_err(|_| StoreError::Invalid("range exceeds address space".into()))?
                .min(end);
            return Ok(Box::new(std::io::Cursor::new(bytes[start..end].to_vec())));
        }
        let mut reader = self.catalog.read(&hashed_blobs::Hash(hash.0))?;
        std::io::Seek::seek(&mut reader, std::io::SeekFrom::Start(range.start))?;
        Ok(Box::new(reader.take(range.end - range.start)))
    }

    fn stop(&self) {
        self.requests.locked().take();
        if let Some(handle) = self.handle.locked().take() {
            let _ = handle.join();
        }
    }
}

impl Drop for PutThread {
    fn drop(&mut self) {
        self.stop();
    }
}

fn run(
    mut conn: Connection,
    mut home: impl Home,
    requests: &Receiver<Request>,
    recent: &Mutex<Recent>,
) {
    let _worker = crate::timing::worker();
    while let Ok(first) = requests.recv() {
        let mut waiting = vec![first];
        waiting.extend(requests.try_iter());
        let _busy = crate::timing::start_stage(crate::timing::Stage::Writer);
        let answers: Vec<Result<Put>> = match put_group(&mut conn, &mut home, &waiting) {
            Ok(answers) => {
                let mut recent = recent.locked();
                for answer in &answers {
                    for hash in answer.added.keys() {
                        recent.put(*hash);
                    }
                }
                answers.into_iter().map(Ok).collect()
            }
            Err(error) => {
                home.reset();
                let mut answers: Vec<Result<Put>> = (1..waiting.len())
                    .map(|_| Err(error.for_group_member()))
                    .collect();
                answers.insert(0, Err(error));
                answers
            }
        };
        for (request, answer) in waiting.into_iter().zip(answers) {
            let _ = request.reply.send(answer);
        }
    }
}

/// Stores every blob of `waiting` with one sync and commits their rows. A
/// blob the archive holds already is not stored again, and a put of only
/// such blobs writes and syncs nothing.
fn put_group(conn: &mut Connection, home: &mut impl Home, waiting: &[Request]) -> Result<Vec<Put>> {
    let mut answers: Vec<Put> = vec![Put::default(); waiting.len()];
    let mut prepared = Vec::new();
    let mut owners = Vec::new();
    for (n, request) in waiting.iter().enumerate() {
        for blob in &request.blobs {
            if crate::blobs::load_extent(conn, "", &hashed_blobs::Hash(blob.hash().0))?.is_some() {
                answers[n].deduplicated += 1;
                continue;
            }
            prepared.push(blob.clone());
            owners.push(n);
        }
    }
    if prepared.is_empty() {
        return Ok(answers);
    }
    let stored = home.store(conn, prepared)?;
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    answers[owners[0]].syncs = stored.syncs;
    answers[owners[0]].bytes_written = stored.bytes_written;
    for (extent, owner) in stored.extents.iter().zip(owners) {
        let answer = &mut answers[owner];
        if crate::blobs::insert_extent(&tx, extent)? {
            answer
                .added
                .insert(BlobHash(extent.hash.0), extent.decoded_len);
            answer.appended += 1;
        } else {
            answer.deduplicated += 1;
        }
    }
    if let Some(end) = stored.end {
        crate::blobs::record_committed_end(&tx, end)?;
    }
    crate::fault::check("bytes.rows", "")?;
    crate::write_lock::commit(tx)?;
    home.recorded(stored)?;
    Ok(answers)
}

#[cfg(test)]
mod tests;
