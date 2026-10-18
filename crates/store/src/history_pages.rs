//! Transaction-consistent page capture, with flat immutable CAS ranges. No live
//! database descriptor is opened outside SQLite. Partial preparation is not input authority.
use crate::{
    BlobCapture, BlobHash, Error, ImportObservation, InputFormat, PreparedBlob, ReadHandle, Result,
    StoredInput, Writer, db,
};
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use std::{
    io::{self, Read},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
pub mod turns;
use turns::{Control, Turns};
const MAX_PARTS: usize = 8192;
type Part = (u64, u64, BlobHash, u64);
fn uint(r: &rusqlite::Row<'_>, n: usize) -> rusqlite::Result<u64> {
    u64::try_from(r.get::<_, i64>(n)?).map_err(|_| rusqlite::Error::InvalidQuery)
}
fn bad(s: &str) -> Error {
    Error::ObservedSourceRefused(s.into())
}
fn check(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}
struct Image {
    connection: Connection,
    path: PathBuf,
    pages: u64,
    size: u64,
}
impl Image {
    fn open(path: &Path) -> Result<Self> {
        if !std::fs::symlink_metadata(path)?.is_file() {
            return Err(bad(
                "history capture requires a regular non-symlink database",
            ));
        }
        let connection = db(path, || {
            Connection::open_with_flags(
                path,
                OpenFlags::SQLITE_OPEN_READ_ONLY
                    | OpenFlags::SQLITE_OPEN_NO_MUTEX
                    | OpenFlags::SQLITE_OPEN_NOFOLLOW,
            )
        })?;
        db(path, || connection.busy_timeout(Duration::from_millis(100)))?;
        db(path, || {
            connection.execute_batch("PRAGMA query_only=ON; BEGIN")
        })?;
        // Establish the read snapshot before reading size or pages.
        let _: i64 = db(path, || {
            connection.query_row("PRAGMA schema_version", [], |r| r.get(0))
        })?;
        let pages: u64 = db(path, || {
            connection.query_row("PRAGMA page_count", [], |r| uint(r, 0))
        })?;
        let size: u64 = db(path, || {
            connection.query_row("PRAGMA page_size", [], |r| uint(r, 0))
        })?;
        if size.checked_mul(pages).is_none_or(|n| n > i64::MAX as u64) {
            return Err(bad("history page image length is invalid"));
        }
        // Fail explicitly when a build lacks the required extension.
        let _: i64 = db(path, || {
            connection.query_row(
                "SELECT count(*) FROM sqlite_dbpage('main') WHERE pgno=1",
                [],
                |r| r.get(0),
            )
        })?;
        Ok(Self {
            connection,
            path: path.into(),
            pages,
            size,
        })
    }
    fn page(&self, n: u64, cancel: &AtomicBool) -> Result<Vec<u8>> {
        check(cancel)?;
        let bytes: Vec<u8> = db(&self.path, || {
            self.connection.query_row(
                "SELECT data FROM sqlite_dbpage('main') WHERE pgno=?1",
                [n as i64],
                |r| r.get(0),
            )
        })?;
        if bytes.len() as u64 != self.size {
            return Err(bad("history page has wrong size"));
        }
        Ok(bytes)
    }
}
fn append(parts: &mut Vec<Part>, part: Part) {
    if let Some(last) = parts.last_mut() {
        if last.0 + last.1 == part.0 && last.2 == part.2 && last.3 + last.1 == part.3 {
            last.1 += part.1;
            return;
        }
    }
    parts.push(part);
}
pub struct Basis {
    reader: ReadHandle,
    hash: BlobHash,
    length: u64,
}

impl Writer {
    /// Select only already admitted history from this exact acquisition binding.
    pub fn known_history_snapshot(
        &mut self,
        hash: BlobHash,
        o: &ImportObservation<'_>,
    ) -> Result<Option<StoredInput>> {
        self.check_import_acquisition()?;
        let file = self.root.join("archive.db");
        let id: Option<String> = db(&file, || {
            self.archive.query_row("SELECT i.id FROM observation o JOIN input i ON i.id=o.input JOIN machine m ON m.id=o.machine AND m.host=?3 WHERE o.machine=?1 AND o.path=?2 AND i.format='codex-history-sqlite' AND o.class='imported' ORDER BY o.rowid DESC LIMIT 1",params![o.machine,o.path,o.host],|r|r.get(0)).optional()
        })?;
        let expected =
            polylogue_model::InputId::imported(&hash, InputFormat::CodexHistorySqlite.name(), &[])
                .to_string();
        if id.as_deref() != Some(&expected) {
            return Ok(None);
        }
        Ok(self.immutable_import(&expected)?.map(|input| input.input))
    }

    pub fn history_basis(&mut self, o: &ImportObservation<'_>) -> Result<Option<Basis>> {
        self.check_import_acquisition()?;
        let file = self.root.join("archive.db");
        let id: Option<String> = db(&file, || {
            self.archive.query_row("SELECT i.id FROM observation o JOIN input i ON i.id=o.input JOIN machine m ON m.id=o.machine AND m.host=?3 WHERE o.machine=?1 AND o.path=?2 AND i.format='codex-history-sqlite' AND o.class='imported' ORDER BY o.rowid DESC LIMIT 1",params![o.machine,o.path,o.host],|r|r.get(0)).optional()
        })?;
        let Some(id) = id else { return Ok(None) };
        let base: Option<(Vec<u8>, i64)> = db(&file, || {
            self.archive.query_row("SELECT b.hash,b.size FROM extent e JOIN blob b ON b.hash=e.blob WHERE e.input=?1 AND e.chain=1 AND e.offset=e.blob_offset AND b.state='present' AND b.encoding='plain' AND EXISTS(SELECT 1 FROM revision r JOIN input i ON i.id=r.input WHERE r.blake3=b.hash AND r.length=b.size AND i.format='codex-history-sqlite') ORDER BY b.size DESC,b.hash LIMIT 1",[id],|r|Ok((r.get(0)?,r.get(1)?))).optional()
        })?;
        let Some((hash, length)) = base else {
            return Ok(None);
        };
        Ok(Some(Basis {
            reader: self.read_handle()?,
            hash: BlobHash::from_bytes(
                hash.try_into()
                    .map_err(|_| bad("invalid complete base hash"))?,
            ),
            length: u64::try_from(length).map_err(|_| Error::Range)?,
        }))
    }

    pub fn store_history_pages(
        &mut self,
        p: PreparedHistory,
        o: &ImportObservation<'_>,
        allowance: usize,
    ) -> Result<StoredInput> {
        // PreparedBlob remains archive-bound; certify patch custody before any
        // input is selectable. A crash here leaves only unselected physical data.
        let patch = self.store_captured_blob(p.patch)?;
        let parts = p
            .parts
            .into_iter()
            .map(|(at, n, h, start)| {
                (
                    at,
                    n,
                    if h == BlobHash::from_bytes([0; 32]) {
                        patch
                    } else {
                        h
                    },
                    start,
                )
            })
            .collect::<Vec<_>>();
        self.store_immutable_parts(
            p.hash,
            p.length,
            &parts,
            InputFormat::CodexHistorySqlite,
            &[],
            o,
            allowance,
        )
    }
}
pub struct PreparedHistory {
    patch: PreparedBlob,
    hash: BlobHash,
    length: u64,
    parts: Vec<Part>,
    pub read_bytes: u64,
    pub changed_bytes: u64,
    pub predecessor_read_bytes: u64,
    pub rebased: bool,
    pub base_verification_bytes: u64,
    pub canonical_read_bytes: u64,
}
impl PreparedHistory {
    pub(crate) fn into_plain(self) -> PreparedBlob {
        self.patch
    }
    pub fn hash(&self) -> BlobHash {
        self.hash
    }
    pub fn length(&self) -> u64 {
        self.length
    }
}
struct Changes<'a, 'b> {
    control: Control<'b>,
    image: Image,
    basis: Option<Basis>,
    cancel: &'a AtomicBool,
    page: u64,
    buffer: Vec<u8>,
    at: usize,
    parts: Vec<Part>,
    digest: blake3::Hasher,
    changed: u64,
    old_read_bytes: u64,
    old_window: Vec<u8>,
    old_start: u64,
    part_limit: usize,
    failure: Option<Error>,
}
impl Changes<'_, '_> {
    fn next(&mut self) -> Result<bool> {
        while self.page < self.image.pages {
            let offset = self.page * self.image.size;
            self.control
                .borrow_mut()
                .work("provider_pages", self.image.size)?;
            let bytes = self.image.page(self.page + 1, self.cancel)?;
            self.control.borrow_mut().page_read(self.image.size)?;
            self.digest.update(&bytes);
            self.page += 1;
            let equal = if let Some(old) = &self.basis {
                if offset + self.image.size > old.length {
                    false
                } else {
                    if self.old_window.is_empty()
                        || offset < self.old_start
                        || offset + self.image.size > self.old_start + self.old_window.len() as u64
                    {
                        self.old_start = offset;
                        let length = (old.length - offset).min(self.control.borrow().window());
                        self.old_window =
                            old.reader
                                .read_blob_range_checked(old.hash, offset, length, || {
                                    check(self.cancel)?;
                                    self.control.borrow_mut().work("base_ranges", 65536)
                                })?;
                        self.old_read_bytes += length;
                    }
                    let start = (offset - self.old_start) as usize;
                    bytes == self.old_window[start..start + self.image.size as usize]
                }
            } else {
                false
            };
            if equal {
                let old = self.basis.as_ref().expect("equal basis");
                append(&mut self.parts, (offset, self.image.size, old.hash, offset));
                if self.parts.len() > self.part_limit {
                    return Err(Error::SnapshotFragmented);
                }
                continue;
            }
            append(
                &mut self.parts,
                (
                    offset,
                    self.image.size,
                    BlobHash::from_bytes([0; 32]),
                    self.changed,
                ),
            );
            if self.parts.len() > self.part_limit {
                return Err(Error::SnapshotFragmented);
            }

            self.changed += self.image.size;
            self.buffer = bytes;
            self.at = 0;
            return Ok(true);
        }
        Ok(false)
    }
}
impl Read for Changes<'_, '_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        if self.at == self.buffer.len() {
            match self.next() {
                Ok(false) => return Ok(0),
                Ok(true) => {}
                Err(e) => {
                    self.failure = Some(e);
                    return Err(io::Error::other("history page capture failed"));
                }
            }
        }
        let n = out.len().min(self.buffer.len() - self.at);
        out[..n].copy_from_slice(&self.buffer[self.at..self.at + n]);
        self.at += n;
        Ok(n)
    }
}
impl BlobCapture {
    pub fn capture_history_pages(
        &self,
        path: &Path,
        basis: Option<Basis>,
        cancel: &AtomicBool,
    ) -> Result<PreparedHistory> {
        self.capture_history_with_limit(path, basis, cancel, MAX_PARTS)
    }
    pub fn capture_history_pages_in_turns(
        &self,
        path: &Path,
        basis: Option<Basis>,
        cancel: &AtomicBool,
        budget: turns::Budget,
        mut yield_turn: impl FnMut(turns::Progress) -> Result<()>,
    ) -> Result<PreparedHistory> {
        let control = Turns::new(budget, &mut yield_turn)?;
        self.capture_history_controlled(path, basis, cancel, MAX_PARTS, control)
    }
    fn canonical_layout(
        &self,
        mut prepared: PreparedHistory,
        cancel: &AtomicBool,
        control: &Control<'_>,
    ) -> Result<PreparedHistory> {
        let reader = self.read_handle()?;
        let id = polylogue_model::InputId::imported(
            &prepared.hash,
            InputFormat::CodexHistorySqlite.name(),
            &[],
        )
        .to_string();
        let Some(source) = reader.source(&id)? else {
            return Ok(prepared);
        };
        if source.binding != prepared.hash || source.input.length != prepared.length {
            return Err(bad("existing immutable history identity differs"));
        }
        let mut digest = blake3::Hasher::new();
        let mut at = 0;
        while at < prepared.length {
            let amount = (prepared.length - at).min(control.borrow().window());
            let bytes =
                reader.read_source_range_checked(&id, 1, source.binding, at, amount, || {
                    check(cancel)?;
                    control.borrow_mut().work("canonical_layout", 65536)
                })?;
            if bytes.is_empty() {
                return Err(bad("existing immutable history layout is incomplete"));
            }
            control
                .borrow_mut()
                .work("canonical_hash", bytes.len() as u64)?;
            digest.update(&bytes);
            at += bytes.len() as u64;
        }
        if digest.finalize().as_bytes() != prepared.hash.bytes() {
            return Err(bad("existing immutable history bytes differ"));
        }
        let file = reader.root.join("archive.db");
        let parts: Vec<Part> = db(&file, || {
            let mut q=reader.archive.prepare("SELECT offset,length,blob,blob_offset FROM extent WHERE input=?1 AND chain=1 ORDER BY offset LIMIT 8194")?;
            q.query_map([id], |r| {
                let hash: Vec<u8> = r.get(2)?;
                Ok((
                    uint(r, 0)?,
                    uint(r, 1)?,
                    BlobHash::from_bytes(
                        hash.try_into().map_err(|_| rusqlite::Error::InvalidQuery)?,
                    ),
                    uint(r, 3)?,
                ))
            })?
            .collect()
        })?;
        if parts.len() > MAX_PARTS {
            return Err(bad("existing history exceeds extent publication admission"));
        }
        prepared.parts = parts;
        prepared.canonical_read_bytes = at;
        Ok(prepared)
    }
    fn capture_history_with_limit(
        &self,
        path: &Path,
        basis: Option<Basis>,
        cancel: &AtomicBool,
        part_limit: usize,
    ) -> Result<PreparedHistory> {
        let mut yield_turn = |_| Ok(());
        let control = Turns::new(turns::Budget::default(), &mut yield_turn)?;
        self.capture_history_controlled(path, basis, cancel, part_limit, control)
    }
    fn capture_history_controlled(
        &self,
        path: &Path,
        basis: Option<Basis>,
        cancel: &AtomicBool,
        part_limit: usize,
        control: Control<'_>,
    ) -> Result<PreparedHistory> {
        check(cancel)?;
        control.borrow_mut().work("opening", 0)?;
        let image = Image::open(path)?;
        let length = image.size * image.pages;
        let mut base_verification_bytes = 0;
        if let Some(base) = &basis {
            let stats = base.reader.blob_reader.verify_plain::<Error>(
                &hashed_blobs::Hash(*base.hash.bytes()),
                base.length,
                |bytes| {
                    check(cancel)?;
                    control
                        .borrow_mut()
                        .work("base_verification", bytes.len() as u64)?;
                    base_verification_bytes += bytes.len() as u64;
                    Ok(())
                },
            )?;
            let _ = stats;
        }
        let mut stream = Changes {
            control: control.clone(),
            image,
            basis,
            cancel,
            page: 0,
            buffer: Vec::new(),
            at: 0,
            parts: Vec::new(),
            digest: blake3::Hasher::new(),
            changed: 0,
            old_read_bytes: 0,
            old_window: Vec::new(),
            old_start: 0,
            part_limit,
            failure: None,
        };
        let mut result = self.capture_with_check(&mut stream, cancel, |n| {
            control.borrow_mut().work("patch_custody", n)
        });
        let mut rebased = false;
        let mut prefix_read = 0;
        let mut previous_read = 0;
        if matches!(stream.failure, Some(Error::SnapshotFragmented)) {
            prefix_read = stream.page * stream.image.size;
            previous_read = stream.old_read_bytes;
            stream = Changes {
                control: control.clone(),
                image: stream.image,
                basis: None,
                cancel,
                page: 0,
                buffer: Vec::new(),
                at: 0,
                parts: Vec::new(),
                digest: blake3::Hasher::new(),
                changed: 0,
                old_read_bytes: 0,
                old_window: Vec::new(),
                old_start: 0,
                part_limit,
                failure: None,
            };
            result = self.capture_with_check(&mut stream, cancel, |n| {
                control.borrow_mut().work("patch_custody", n)
            });
            rebased = true;
        }
        if let Some(e) = stream.failure.take() {
            return Err(e);
        }
        let patch = result?;
        check(cancel)?;
        if stream.page != stream.image.pages || stream.at != stream.buffer.len() {
            return Err(bad("incomplete history page capture"));
        }
        let prepared = PreparedHistory {
            patch,
            hash: BlobHash::from_bytes(*stream.digest.finalize().as_bytes()),
            length,
            parts: stream.parts,
            read_bytes: length + prefix_read,
            changed_bytes: stream.changed,
            predecessor_read_bytes: stream.old_read_bytes + previous_read,
            rebased,
            base_verification_bytes,
            canonical_read_bytes: 0,
        };
        drop(stream.image);
        self.canonical_layout(prepared, cancel, &control)
    }
}
#[cfg(test)]
mod tests;

#[cfg(test)]
mod turn_tests;

#[cfg(test)]
mod process_turn_tests;
