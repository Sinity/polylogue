//! Worker-side CAS capture and opaque proof admission. SQLite remains owned by
//! Writer; capture has no SQL capability and retains the archive lock authority.
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
    authority: Arc<File>,
}
/// Minted only after the reviewed CAS put has finished durably. Not a claim of
/// provider parsing, immutable-input retention, or garbage-collection coverage.
pub struct PreparedBlob {
    root: PathBuf,
    authority: Arc<File>,
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
struct Counted<'a, R> {
    reader: R,
    cancelled: &'a AtomicBool,
    length: u64,
}
impl<R: Read> Read for Counted<'_, R> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if self.cancelled.load(Ordering::Acquire) {
            // Interrupted is retried by many Read helpers, potentially forever.
            return Err(io::Error::other("blob capture cancelled"));
        }
        let count = self.reader.read(bytes)?;
        if self.cancelled.load(Ordering::Acquire) {
            return Err(io::Error::other("blob capture cancelled"));
        }
        self.length = self
            .length
            .checked_add(
                u64::try_from(count).map_err(|_| io::Error::other("blob length overflow"))?,
            )
            .ok_or_else(|| io::Error::other("blob length overflow"))?;
        Ok(count)
    }
}
// The reviewed decoder owns the file format. This helper only streams its
// uncompressed bytes, checks cancellation, and verifies the content receipt.
fn verify_plain(
    mut reader: impl Read,
    expected: hashed_blobs::Hash,
    length: u64,
    cancelled: &AtomicBool,
) -> Result<()> {
    let mut digest = blake3::Hasher::new();
    let mut total = 0_u64;
    let mut chunk = [0_u8; 64 * 1024];
    loop {
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        let count = match reader.read(&mut chunk) {
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        if count == 0 {
            break;
        }
        total = total
            .checked_add(u64::try_from(count).map_err(|_| Error::Range)?)
            .ok_or(Error::Range)?;
        digest.update(&chunk[..count]);
    }
    if total != length || digest.finalize().as_bytes() != &expected.0 {
        return Err(hashed_blobs::Error::Corrupt {
            blob: expected,
            what: "captured CAS digest or length mismatch".into(),
        }
        .into());
    }
    Ok(())
}
impl BlobCapture {
    /// Explicit regular-file import only. Nonblocking/no-follow open prevents
    /// FIFO opens and final-component symlinks from trapping the worker before
    /// descriptor type validation. This is not a general collector policy.
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
    /// Run on a worker. The underlying store streams through bounded frames;
    /// its seek table still grows with frame count (no whole-input Vec here).
    pub fn capture(&self, reader: impl Read, cancelled: &AtomicBool) -> Result<PreparedBlob> {
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        let blobs = hashed_blobs::Store::open(&self.root, vec![], 0o600)?;
        let mut counted = Counted {
            reader,
            cancelled,
            length: 0,
        };
        let result = blobs.put(&mut counted, 3);
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        let hash = result?;
        if !matches!(blobs.kind(&hash)?, hashed_blobs::Kind::Plain) {
            return Err(Error::Invariant(
                "capture requires a plain CAS representation".into(),
            ));
        }
        // put may deduplicate against an existing filename. Certify those actual
        // retained bytes, not just the hash computed from the incoming stream.
        verify_plain(blobs.read(&hash)?, hash, counted.length, cancelled)?;
        private_dir(&self.root.join("blobs").join(&hash.hex()[..2]))?;
        let stored_size = blobs.stored_size(&hash)?;
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        Ok(PreparedBlob {
            root: self.root.clone(),
            authority: Arc::clone(&self.authority),
            hash: BlobHash::from_bytes(hash.0),
            length: counted.length,
            stored_size,
        })
    }
}
impl Writer {
    /// Minting performs no source/blob IO; the worker opens its write-only CAS.
    pub fn blob_capture(&self) -> Result<BlobCapture> {
        Ok(BlobCapture {
            root: self.root.clone(),
            authority: Arc::clone(&self._lock),
        })
    }
    /// Only bounded metadata work occurs on the actor. Completed file bytes,
    /// compression, digest and physical size were established by capture.
    pub fn retain_captured_blob(&mut self, proof: PreparedBlob) -> Result<BlobHash> {
        if proof.root != self.root || !Arc::ptr_eq(&proof.authority, &self._lock) {
            return Err(Error::Invariant(
                "blob capture belongs to another archive authority".into(),
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
mod tests {
    use super::*;
    use std::sync::mpsc;
    struct Small {
        remaining: usize,
        calls: usize,
    }
    impl Read for Small {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            self.calls += 1;
            let n = bytes.len().min(self.remaining).min(7);
            bytes[..n].fill(b'x');
            self.remaining -= n;
            Ok(n)
        }
    }
    fn proof(w: &Writer, bytes: &[u8]) -> PreparedBlob {
        w.blob_capture()
            .unwrap()
            .capture(bytes, &AtomicBool::new(false))
            .unwrap()
    }
    fn rows(w: &Writer) -> i64 {
        w.archive
            .query_row("SELECT count(*) FROM blob", [], |r| r.get(0))
            .unwrap()
    }
    #[test]
    fn many_small_reads_are_captured_and_metadata_is_idempotent() {
        let root = tempfile::tempdir().unwrap();
        let mut w = Writer::open(root.path()).unwrap();
        let mut reader = Small {
            remaining: 65537,
            calls: 0,
        };
        let p = w
            .blob_capture()
            .unwrap()
            .capture(&mut reader, &AtomicBool::new(false))
            .unwrap();
        assert!(reader.calls > 9000);
        assert_eq!(p.length(), 65537);
        assert!(p.stored_size() > 0);
        assert_eq!(p.hash(), BlobHash::of(&vec![b'x'; 65537]));
        assert_eq!(rows(&w), 0);
        let h = w.retain_captured_blob(p).unwrap();
        let counter = w.status().unwrap().write_counter;
        assert_eq!(
            w.retain_captured_blob(proof(&w, &vec![b'x'; 65537]))
                .unwrap(),
            h
        );
        assert_eq!(w.status().unwrap().write_counter, counter);
        assert_eq!(rows(&w), 1);
    }
    #[test]
    fn cancellation_and_original_source_failure_do_not_publish_metadata() {
        struct Failing;
        impl Read for Failing {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "synthetic source denied",
                ))
            }
        }
        struct Cancels<'a>(&'a AtomicBool);
        impl Read for Cancels<'_> {
            fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
                self.0.store(true, Ordering::Release);
                b[0] = 1;
                Ok(1)
            }
        }
        let root = tempfile::tempdir().unwrap();
        let w = Writer::open(root.path()).unwrap();
        let c = w.blob_capture().unwrap();
        let cancelled = AtomicBool::new(false);
        assert!(matches!(
            c.capture(Cancels(&cancelled), &cancelled),
            Err(Error::Cancelled)
        ));
        assert!(matches!(
            c.capture(&b"ignored"[..], &cancelled),
            Err(Error::Cancelled)
        ));
        match c.capture(Failing, &AtomicBool::new(false)) {
            Err(Error::Blob(hashed_blobs::Error::Io(e))) => {
                assert_eq!(e.kind(), io::ErrorKind::PermissionDenied);
                assert_eq!(e.to_string(), "synthetic source denied");
            }
            _ => panic!("original source failure was not retained"),
        }
        assert_eq!(rows(&w), 0);
    }
    #[test]
    fn proof_is_archive_bound_and_keeps_lock_authority_alive() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let wa = Writer::open(a.path()).unwrap();
        let mut wb = Writer::open(b.path()).unwrap();
        let p = proof(&wa, b"bound");
        assert!(matches!(
            wb.retain_captured_blob(p),
            Err(Error::Invariant(_))
        ));
        assert_eq!(rows(&wb), 0);
        let capture = wa.blob_capture().unwrap();
        drop(wa);
        assert!(matches!(
            Writer::open(a.path()),
            Err(Error::WriterHeld { .. })
        ));
        let p = capture
            .capture(&b"lock"[..], &AtomicBool::new(false))
            .unwrap();
        drop(capture);
        assert!(matches!(
            Writer::open(a.path()),
            Err(Error::WriterHeld { .. })
        ));
        drop(p);
        assert!(Writer::open(a.path()).is_ok());
    }
    #[test]
    fn worker_capture_does_not_block_actor_metadata_progress() {
        struct Blocked {
            entered: mpsc::Sender<()>,
            resume: mpsc::Receiver<()>,
            done: bool,
        }
        impl Read for Blocked {
            fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
                if self.done {
                    return Ok(0);
                }
                self.entered.send(()).map_err(io::Error::other)?;
                self.resume.recv().map_err(io::Error::other)?;
                b[..4].copy_from_slice(b"slow");
                self.done = true;
                Ok(4)
            }
        }
        let root = tempfile::tempdir().unwrap();
        let mut w = Writer::open(root.path()).unwrap();
        let quick = proof(&w, b"quick");
        let capture = w.blob_capture().unwrap();
        let (entered_rx_tx, entered) = mpsc::channel();
        let (resume, receiver) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            capture.capture(
                Blocked {
                    entered: entered_rx_tx,
                    resume: receiver,
                    done: false,
                },
                &AtomicBool::new(false),
            )
        });
        entered
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap();
        w.retain_captured_blob(quick).unwrap();
        assert_eq!(rows(&w), 1);
        resume.send(()).unwrap();
        let p = worker.join().unwrap().unwrap();
        w.retain_captured_blob(p).unwrap();
        assert_eq!(rows(&w), 2);
    }
    #[test]
    fn metadata_recovery_refuses_logical_conflict_without_resurrecting_deletions() {
        let root = tempfile::tempdir().unwrap();
        let mut w = Writer::open(root.path()).unwrap();
        let h = w.retain_captured_blob(proof(&w, b"body")).unwrap();
        w.archive
            .execute(
                "UPDATE blob SET state='writing',stored_size=0 WHERE hash=?1",
                [h.bytes().as_slice()],
            )
            .unwrap();
        let counter = w.status().unwrap().write_counter;
        w.retain_captured_blob(proof(&w, b"body")).unwrap();
        assert_eq!(w.status().unwrap().write_counter, counter + 1);
        let state: String = w
            .archive
            .query_row("SELECT state FROM blob", [], |r| r.get(0))
            .unwrap();
        assert_eq!(state, "present");
        w.archive.execute("UPDATE blob SET size=5", []).unwrap();
        assert!(matches!(
            w.retain_captured_blob(proof(&w, b"body")),
            Err(Error::Invariant(_))
        ));
        w.archive
            .execute("UPDATE blob SET size=4,state='deleting'", [])
            .unwrap();
        assert!(matches!(
            w.retain_captured_blob(proof(&w, b"body")),
            Err(Error::Invariant(_))
        ));
    }
    #[test]
    fn corrupt_existing_cas_is_not_certified_by_fresh_incoming_stream() {
        let root = tempfile::tempdir().unwrap();
        let w = Writer::open(root.path()).unwrap();
        let p = proof(&w, b"original");
        let hex = p.hash().to_string();
        let path = root
            .path()
            .join("blobs")
            .join(&hex[..2])
            .join(format!("{hex}.zst"));
        std::fs::write(&path, b"not a zstd object").unwrap();
        drop(p);
        assert!(
            w.blob_capture()
                .unwrap()
                .capture(&b"original"[..], &AtomicBool::new(false))
                .is_err()
        );
        assert_eq!(std::fs::read(path).unwrap(), b"not a zstd object");
        assert_eq!(rows(&w), 0);
    }
}

#[cfg(test)]
mod cooperative_tests {
    use super::*;
    #[test]
    fn verification_cancels_between_reads_and_preserves_read_failure() {
        struct Cancels<'a> {
            cancelled: &'a AtomicBool,
            calls: usize,
        }
        impl Read for Cancels<'_> {
            fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
                self.calls += 1;
                b[0] = b'x';
                self.cancelled.store(true, Ordering::Release);
                Ok(1)
            }
        }
        let flag = AtomicBool::new(false);
        let mut reader = Cancels {
            cancelled: &flag,
            calls: 0,
        };
        assert!(matches!(
            verify_plain(&mut reader, hashed_blobs::Hash::of(b"x"), 1, &flag),
            Err(Error::Cancelled)
        ));
        assert_eq!(reader.calls, 1);
        struct Fails;
        impl Read for Fails {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "verification denied",
                ))
            }
        }
        match verify_plain(
            Fails,
            hashed_blobs::Hash::of(b""),
            0,
            &AtomicBool::new(false),
        ) {
            Err(Error::Io(e)) => assert_eq!(e.kind(), io::ErrorKind::PermissionDenied),
            _ => panic!("verification IO failure was not preserved"),
        }
        assert!(
            verify_plain(
                &b"body"[..],
                hashed_blobs::Hash::of(b"body"),
                4,
                &AtomicBool::new(false)
            )
            .is_ok()
        );
        assert!(
            verify_plain(
                &b"body"[..],
                hashed_blobs::Hash::of(b"body"),
                5,
                &AtomicBool::new(false)
            )
            .is_err()
        );
    }
    #[test]
    fn explicit_file_capture_refuses_fifo_symlink_and_nonregular_descriptor() {
        use std::os::unix::fs::symlink;
        let root = tempfile::tempdir().unwrap();
        let input = tempfile::tempdir().unwrap();
        let mut w = Writer::open(root.path()).unwrap();
        let capture = w.blob_capture().unwrap();
        let regular = input.path().join("source");
        std::fs::write(&regular, b"regular").unwrap();
        let link = input.path().join("link");
        symlink(&regular, &link).unwrap();
        assert!(
            capture
                .capture_regular_file(&link, &AtomicBool::new(false))
                .is_err()
        );
        assert!(
            capture
                .capture_regular_file(input.path(), &AtomicBool::new(false))
                .is_err()
        );
        let fifo = input.path().join("pipe");
        let status = std::process::Command::new("mkfifo")
            .arg("--mode=600")
            .arg(&fifo)
            .status()
            .unwrap();
        assert!(status.success(), "mkfifo fixture failed: {status}");
        assert!(
            capture
                .capture_regular_file(&fifo, &AtomicBool::new(false))
                .is_err()
        );
        assert!(matches!(
            capture.capture_regular_file(&fifo, &AtomicBool::new(true)),
            Err(Error::Cancelled)
        ));
        let proof = capture
            .capture_regular_file(&regular, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(proof.length(), 7);
        w.retain_captured_blob(proof).unwrap();
    }
}
