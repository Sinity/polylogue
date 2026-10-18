//! Blob capture on worker threads. Workers write blob files; only the Writer
//! records them in the archive. A capture holds the archive lock until the
//! blob is recorded or dropped.
use crate::{BlobHash, Error, Result, Writer, counter, db, private_dir};
use rusqlite::{OptionalExtension, params};
use std::{
    fs::{File, OpenOptions},
    io::{self, Read},
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

pub struct BlobCapture {
    root: PathBuf,
    lock: Arc<File>,
}
/// A blob written and verified in the store, ready for
/// `Writer::store_captured_blob`.
pub struct PreparedBlob {
    root: PathBuf,
    lock: Arc<File>,
    hash: BlobHash,
    length: u64,
    stored_size: u64,
}
impl PreparedBlob {
    pub fn hash(&self) -> BlobHash {
        self.hash
    }
    pub fn length(&self) -> u64 {
        self.length
    }
    pub fn stored_size(&self) -> u64 {
        self.stored_size
    }
}
struct Counted<R> {
    reader: R,
    length: u64,
}
impl<R: Read> Read for Counted<R> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let count = self.reader.read(bytes)?;
        self.length = self
            .length
            .checked_add(
                u64::try_from(count).map_err(|_| io::Error::other("blob length overflow"))?,
            )
            .ok_or_else(|| io::Error::other("blob length overflow"))?;
        Ok(count)
    }
}
impl BlobCapture {
    pub(crate) fn read_handle(&self) -> Result<crate::ReadHandle> {
        crate::ReadHandle::owned(self.root.clone(), Arc::clone(&self.lock))
    }
    /// Capture one regular file. The nonblocking, no-follow open keeps a FIFO
    /// or final symlink from blocking the worker before the type check.
    pub fn capture_regular_file(
        &self,
        path: &Path,
        cancelled: &AtomicBool,
    ) -> Result<PreparedBlob> {
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW)
            .open(path)?;
        if !file.metadata()?.is_file() {
            return Err(Error::Invariant("capture requires a regular file".into()));
        }
        self.capture(file, cancelled)
    }
    /// Run on a worker. Streams the input; memory grows only with the blob's
    /// frame count.
    pub fn capture(&self, reader: impl Read, cancelled: &AtomicBool) -> Result<PreparedBlob> {
        self.capture_with_check(reader, cancelled, |_| Ok(()))
    }
    /// `work` is charged per step: an estimated 64 KiB per store poll and the
    /// exact size of each verified chunk.
    pub(crate) fn capture_with_check(
        &self,
        reader: impl Read,
        cancelled: &AtomicBool,
        mut work: impl FnMut(u64) -> Result<()>,
    ) -> Result<PreparedBlob> {
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        work(0)?;
        let blobs = hashed_blobs::Store::open(&self.root, vec![], 0o600)?;
        let mut counted = Counted { reader, length: 0 };
        let result = blobs.put_plain_checked::<Error>(&mut counted, 3, || {
            if cancelled.load(Ordering::Acquire) {
                Err(Error::Cancelled)
            } else {
                work(65536)
            }
        });
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        let hash = result?;
        if !matches!(blobs.kind(&hash)?, hashed_blobs::Kind::Plain) {
            return Err(Error::Invariant(
                "captured blob is not stored as a plain file".into(),
            ));
        }
        // put may find an existing file with this hash. Verify that file, not
        // only the incoming stream.
        blobs.verify_plain::<Error>(&hash, counted.length, |bytes| {
            if cancelled.load(Ordering::Acquire) {
                Err(Error::Cancelled)
            } else {
                work(bytes.len() as u64)
            }
        })?;
        private_dir(&self.root.join("blobs").join(&hash.hex()[..2]))?;
        let stored_size = blobs.stored_size(&hash)?;
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        Ok(PreparedBlob {
            root: self.root.clone(),
            lock: Arc::clone(&self.lock),
            hash: BlobHash::from_bytes(hash.0),
            length: counted.length,
            stored_size,
        })
    }
}
impl Writer {
    /// Does no IO; the worker opens the blob store itself.
    pub fn blob_capture(&self) -> Result<BlobCapture> {
        Ok(BlobCapture {
            root: self.root.clone(),
            lock: Arc::clone(&self._lock),
        })
    }
    /// Record a captured blob in the archive. Only metadata work happens
    /// here; the capture already wrote and verified the bytes.
    pub fn store_captured_blob(&mut self, proof: PreparedBlob) -> Result<BlobHash> {
        if proof.root != self.root || !Arc::ptr_eq(&proof.lock, &self._lock) {
            return Err(Error::Invariant(
                "blob was captured for a different archive".into(),
            ));
        }
        let length = i64::try_from(proof.length).map_err(|_| Error::Range)?;
        let stored = i64::try_from(proof.stored_size).map_err(|_| Error::Range)?;
        let file = self.root.join("archive.db");
        let tx = db(&file, || self.archive.transaction())?;
        type Existing = (i64, i64, String, String, bool, String, bool);
        let old: Option<Existing> = db(&file, || {
            tx.query_row(
                "SELECT size,stored_size,place,encoding,base IS NOT NULL,state,data IS NOT NULL FROM blob WHERE hash=?1",
                [proof.hash.bytes().as_slice()],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                        r.get(6)?,
                    ))
                },
            )
            .optional()
        })?;
        let changed = if let Some((size, physical, place, encoding, base, state, data)) = old {
            if size != length
                || place != "file"
                || encoding != "plain"
                || base
                || data
                || !matches!(state.as_str(), "writing" | "present")
            {
                return Err(Error::Invariant(
                    "captured blob metadata conflicts with archive".into(),
                ));
            }
            if physical != stored || state != "present" {
                db(&file, || {
                    tx.execute(
                        "UPDATE blob SET stored_size=?2,state='present' WHERE hash=?1",
                        params![proof.hash.bytes().as_slice(), stored],
                    )
                })?;
                true
            } else {
                false
            }
        } else {
            db(&file, || {
                tx.execute("INSERT INTO blob(hash,size,stored_size,place,encoding,state) VALUES(?1,?2,?3,'file','plain','present')",params![proof.hash.bytes().as_slice(),length,stored])
            })?;
            true
        };
        if changed {
            let next = counter(&tx, &file)?
                .checked_add(1)
                .filter(|n| *n <= i64::MAX as u64)
                .ok_or_else(|| Error::Schema(file.clone()))?;
            db(&file, || {
                tx.execute(
                    "UPDATE meta SET value=?1 WHERE key='write_counter'",
                    [next.to_string()],
                )
            })?;
        }
        db(&file, || tx.commit())?;
        Ok(proof.hash)
    }
}

#[cfg(test)]
#[path = "capture/tests.rs"]
mod tests;
