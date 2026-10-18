//! Stored inputs: acquiring observed Claude files and reading any input back
//! by byte range. Provider paths are opened only while acquiring; reads use
//! the archive alone.
use crate::{
    BlobHash, Error, ImmutableImport, InputFormat, ReadHandle, StoredInput, Writer, counter, db,
};
use rusqlite::{OptionalExtension, params};
pub(crate) mod context;
use std::{
    io::{self, Read, Seek, SeekFrom},
    os::unix::fs::MetadataExt,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceIdentity {
    Immutable {
        original_hash: BlobHash,
    },
    Observed {
        machine: String,
        path: String,
        device: u64,
        inode: u64,
    },
}
#[derive(Clone, Debug)]
pub struct StoredSource {
    pub input: StoredInput,
    /// An immutable input's original hash, or a hash of an observed input's ID
    /// and chain. Range reads take it back to check they read the same input.
    pub binding: BlobHash,
    pub identity: SourceIdentity,
    pub format: InputFormat,
    pub context: Vec<String>,
    pub revision: u64,
    pub companions: Vec<super::claude_bodies::Companion>,
}
impl From<ImmutableImport> for StoredSource {
    fn from(v: ImmutableImport) -> Self {
        Self {
            identity: SourceIdentity::Immutable {
                original_hash: v.hash,
            },
            binding: v.hash,
            input: v.input,
            format: v.format,
            context: v.context,
            revision: 1,
            companions: vec![],
        }
    }
}
impl StoredSource {
    pub fn original_hash(&self) -> Option<BlobHash> {
        match self.identity {
            SourceIdentity::Immutable { original_hash } => Some(original_hash),
            _ => None,
        }
    }
    pub fn observed(&self) -> bool {
        matches!(self.identity, SourceIdentity::Observed { .. })
    }
}
pub fn observed_binding(id: &str, chain: i64) -> BlobHash {
    let mut h = blake3::Hasher::new();
    h.update(b"polylogue.observed-source.v1\0");
    h.update(&(id.len() as u64).to_be_bytes());
    h.update(id.as_bytes());
    h.update(&chain.to_be_bytes());
    BlobHash::from_bytes(*h.finalize().as_bytes())
}
fn refusal(message: &str) -> Error {
    Error::ObservedSourceRefused(message.into())
}

/// Decode one stored input from archive rows only, never from the provider's
/// current layout.
pub(crate) fn decode(
    connection: &rusqlite::Connection,
    file: &Path,
    id: &str,
) -> crate::Result<Option<StoredSource>> {
    type Row = (
        String,
        Option<Vec<u8>>,
        Option<String>,
        i64,
        i64,
        i64,
        Option<Vec<u8>>,
    );
    let row: Option<Row> = db(file, || {
        connection.query_row("SELECT i.format,i.original_hash,CASE WHEN length(CAST(i.context_json AS BLOB))<=65536 THEN i.context_json END,r.number,r.chain,r.length,r.blake3 FROM input i JOIN revision r ON r.input=i.id WHERE i.id=?1 ORDER BY r.number DESC LIMIT 1",[id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?))).optional()
    })?;
    let Some((format, original, context, revision, chain, length, digest)) = row else {
        return Ok(None);
    };
    if original.is_some() {
        let (id, hash, format, context, length) = crate::imports::decode_import_identity(
            id.into(),
            format,
            length,
            digest.ok_or_else(|| refusal("immutable input has no digest"))?,
            original.unwrap(),
            context.ok_or_else(|| refusal("immutable input has no context"))?,
        )?;
        if revision != 1 || chain != 1 {
            return Err(refusal("immutable input has more than one revision"));
        }
        return Ok(Some(StoredSource {
            input: StoredInput {
                input_pk: 0,
                id,
                chain,
                length,
            },
            binding: hash,
            identity: SourceIdentity::Immutable {
                original_hash: hash,
            },
            format,
            context,
            revision: 1,
            companions: vec![],
        }));
    }
    if context.is_some() || format != "claude-code-jsonl" || chain != 1 {
        return Err(refusal("unsupported observed input"));
    }
    let (machine, path, device, inode): (String, String, i64, i64) = db(file, || {
        connection.query_row("SELECT CASE WHEN length(CAST(machine AS BLOB))<=65536 THEN machine END,CASE WHEN length(CAST(path AS BLOB))<=65536 THEN path END,device,inode FROM observation WHERE input=?1 AND class='first' ORDER BY rowid LIMIT 1",[id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?)))
    })?;
    let context = context::read(connection, file, id)?;
    Ok(Some(StoredSource {
        input: StoredInput {
            input_pk: 0,
            id: id.into(),
            chain,
            length: u64::try_from(length).map_err(|_| Error::Range)?,
        },
        binding: observed_binding(id, chain),
        identity: SourceIdentity::Observed {
            machine,
            path,
            device: device as u64,
            inode: inode as u64,
        },
        format: InputFormat::ClaudeCodeJsonl,
        context,
        revision: u64::try_from(revision).map_err(|_| Error::Range)?,
        companions: super::claude_bodies::manifest(connection, file, id)?,
    }))
}
impl Writer {
    pub fn stored_source(&mut self, id: &str) -> crate::Result<Option<StoredSource>> {
        self.check_import_acquisition()?;
        let Some(mut source) = decode(&self.archive, &self.root.join("archive.db"), id)? else {
            return Ok(None);
        };
        source.input = self.register_input_ref(source.input.id, source.input.length)?;
        Ok(Some(source))
    }
    pub fn next_stored_source(&mut self, after: &str) -> crate::Result<Option<StoredSource>> {
        self.check_import_acquisition()?;
        let id: Option<String> = db(&self.root.join("archive.db"), || {
            self.archive
                .query_row(
                    "SELECT id FROM input WHERE id>?1 AND format<>'claude-code-tool-result' ORDER BY id LIMIT 1",
                    [after],
                    |r| r.get(0),
                )
                .optional()
        })?;
        id.map(|id| self.stored_source(&id))
            .transpose()
            .map(Option::flatten)
    }
    /// Store new bytes of one Claude main file; used before the owner starts.
    /// The whole stored prefix is compared with the file. If the file changes
    /// while being read the attempt fails, and blobs already written stay
    /// unreferenced.
    pub fn acquire_claude_main(
        &mut self,
        path: &Path,
        machine: &str,
        host: &str,
        id: &str,
        seen_at: i64,
        cancelled: &AtomicBool,
    ) -> crate::Result<StoredSource> {
        let path_text = path
            .to_str()
            .ok_or_else(|| refusal("non-UTF8 path is unsupported"))?;
        if !path.is_absolute()
            || machine.is_empty()
            || host.is_empty()
            || id.is_empty()
            || path_text.len() + machine.len() + host.len() + id.len() > 65536
        {
            return Err(refusal(
                "need an absolute path and nonempty machine, host and ID under 64 KiB",
            ));
        }
        let old = self.claude_input_at(path, machine)?;
        let capture = self.blob_capture()?;
        let prepared = capture.prepare_claude_main(path, old, cancelled, |_| Ok(()), |_| Ok(()))?;
        self.store_claude_main(prepared, machine, host, id, seen_at)
    }
    /// The input last observed at `path` on `machine`, if any.
    pub fn claude_input_at(
        &mut self,
        path: &Path,
        machine: &str,
    ) -> crate::Result<Option<StoredSource>> {
        self.check_import_acquisition()?;
        let path_text = path.to_str().ok_or(Error::Range)?;
        let archive_file = self.root.join("archive.db");
        let old_id: Option<String> = db(&archive_file, || {
            self.archive.query_row("SELECT input FROM observation WHERE machine=?1 AND path=?2 AND class IN ('first','appended') ORDER BY rowid DESC LIMIT 1",params![machine,path_text],|r|r.get(0)).optional()
        })?;
        old_id
            .as_ref()
            .map(|id| self.stored_source(id))
            .transpose()
            .map(Option::flatten)
    }
    pub fn store_claude_main(
        &mut self,
        prepared: PreparedClaudeMain,
        machine: &str,
        host: &str,
        id: &str,
        seen_at: i64,
    ) -> crate::Result<StoredSource> {
        self.check_import_acquisition()?;
        let PreparedClaudeMain {
            location,
            old,
            meta,
            proof,
        } = prepared;
        location.check()?;
        let path = location.path();
        let path_text = path.to_str().ok_or(Error::Range)?;
        if machine.is_empty()
            || host.is_empty()
            || id.is_empty()
            || path_text.len() + machine.len() + host.len() + id.len() > 65536
        {
            return Err(refusal("need nonempty machine, host and ID under 64 KiB"));
        }
        let current = self.claude_input_at(path, machine)?;
        if current
            .as_ref()
            .is_some_and(|s| s.context != location.context())
        {
            return Err(refusal("file context differs from the stored input's"));
        }
        if current
            .as_ref()
            .map(|s| (&s.input.id, s.revision, s.input.length))
            != old
                .as_ref()
                .map(|s| (&s.input.id, s.revision, s.input.length))
        {
            return Err(refusal("stored input changed during acquisition"));
        }
        let previous = old.as_ref().map(|s| s.input.length).unwrap_or(0);
        let added = meta.len().checked_sub(previous).ok_or(Error::Range)?;
        if old.is_some() && added == 0 {
            return Ok(current.unwrap());
        }
        let archive_file = self.root.join("archive.db");
        let hash =
            self.store_captured_blob(proof.ok_or_else(|| refusal("new bytes were not captured"))?)?;
        let input_id = old.as_ref().map(|s| s.input.id.as_str()).unwrap_or(id);
        let revision = i64::try_from(
            old.as_ref()
                .map(|s| s.revision.checked_add(1).ok_or(Error::Range))
                .transpose()?
                .unwrap_or(1),
        )
        .map_err(|_| Error::Range)?;
        let length = i64::try_from(meta.len()).map_err(|_| Error::Range)?;
        let tx = db(&archive_file, || self.archive.transaction())?;
        let next = counter(&tx, &archive_file)?
            .checked_add(1)
            .ok_or(Error::Range)?;
        let next_i = i64::try_from(next).map_err(|_| Error::Range)?;
        db(&archive_file, || {
            tx.execute(
                "INSERT OR IGNORE INTO input(id,format) VALUES(?1,'claude-code-jsonl')",
                [input_id],
            )
        })?;
        if old.is_none() {
            context::insert(&tx, &archive_file, input_id, location.context())?;
        }
        db(&archive_file, || {
            tx.execute("INSERT INTO revision(input,number,produced,chain,length,local_seq,blake3) VALUES(?1,?2,?3,1,?4,?5,?6)",params![input_id,revision,if old.is_some(){"appended"}else{"first"},length,next_i,old.is_none().then_some(hash.bytes().as_slice())])
        })?;
        if added > 0 {
            db(&archive_file, || {
                tx.execute("INSERT INTO extent(input,chain,offset,length,blob,blob_offset) VALUES(?1,1,?2,?3,?4,0)",params![input_id,previous as i64,added as i64,hash.bytes().as_slice()])
            })?;
        }
        db(&archive_file, || {
            tx.execute(
                "INSERT OR IGNORE INTO machine(id,host,first_seen) VALUES(?1,?2,?3)",
                params![machine, host, seen_at],
            )
        })?;
        db(&archive_file, || {
            tx.execute("INSERT INTO observation(id,input,machine,path,class,revision,length,device,inode,size,mtime_ns,ctime_ns,seen_at,polylogue_version) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?7,?10,?11,?12,?13)",params![id,input_id,machine,path_text,if old.is_some(){"appended"}else{"first"},revision,length,meta.dev() as i64,meta.ino() as i64,meta.mtime().saturating_mul(1_000_000_000).saturating_add(meta.mtime_nsec()),meta.ctime().saturating_mul(1_000_000_000).saturating_add(meta.ctime_nsec()),seen_at,env!("CARGO_PKG_VERSION")])
        })?;
        db(&archive_file, || {
            tx.execute(
                "UPDATE meta SET value=?1 WHERE key='write_counter'",
                [next.to_string()],
            )
        })?;
        db(&archive_file, || tx.commit())?;
        self.stored_source(input_id)?
            .ok_or_else(|| refusal("acquired input is missing after commit"))
    }
}

pub struct PreparedClaude {
    pub main: PreparedClaudeMain,
    pub bodies: Vec<(std::path::PathBuf, crate::claude_bodies::PreparedClaudeBody)>,
}
pub struct PreparedClaudeMain {
    location: crate::configured_claude::Location,
    old: Option<StoredSource>,
    meta: std::fs::Metadata,
    proof: Option<crate::PreparedBlob>,
}
impl crate::BlobCapture {
    /// Prepare a main file and its companions in cooperative turns. One work
    /// counter covers the prefix check, `inspect` and every companion; the
    /// engine supplies `inspect` and the companion list.
    pub fn prepare_claude_location_in_turns(
        &self,
        location: crate::configured_claude::Location,
        old: Option<StoredSource>,
        cancelled: &AtomicBool,
        budget: crate::history_pages::turns::Budget,
        mut yield_turn: impl FnMut(crate::history_pages::turns::Progress) -> crate::Result<()>,
        inspect: impl FnMut(&[u8]) -> crate::Result<()>,
        finish_references: impl FnOnce() -> crate::Result<Vec<(String, Option<(BlobHash, u64)>)>>,
    ) -> crate::Result<PreparedClaude> {
        let control = crate::history_pages::turns::Turns::new(budget, &mut yield_turn)?;
        let main = self.prepare_claude_location(
            location,
            old,
            cancelled,
            |n| control.borrow_mut().work("claude_main", n),
            inspect,
        )?;
        let references = finish_references()?;
        if references.len() > crate::claude_bodies::MAX_COMPANIONS {
            return Err(refusal("Claude companion reference count exceeds 32"));
        }
        let mut bodies = Vec::new();
        for (name, inline) in references {
            let body = std::path::PathBuf::from(name);
            if let Some(prepared) =
                self.prepare_claude_body_location(&main.location, &body, cancelled, inline, |n| {
                    control.borrow_mut().work("claude_body", n)
                })?
            {
                bodies.push((body, prepared));
            }
        }
        Ok(PreparedClaude { main, bodies })
    }
    pub(crate) fn prepare_claude_main(
        &self,
        path: &Path,
        old: Option<StoredSource>,
        cancelled: &AtomicBool,
        work: impl FnMut(u64) -> crate::Result<()>,
        inspect: impl FnMut(&[u8]) -> crate::Result<()>,
    ) -> crate::Result<PreparedClaudeMain> {
        let location = crate::configured_claude::Location::explicit_main(path.to_path_buf())?;
        self.prepare_claude_location(location, old, cancelled, work, inspect)
    }
    fn prepare_claude_location(
        &self,
        location: crate::configured_claude::Location,
        old: Option<StoredSource>,
        cancelled: &AtomicBool,
        mut work: impl FnMut(u64) -> crate::Result<()>,
        mut inspect: impl FnMut(&[u8]) -> crate::Result<()>,
    ) -> crate::Result<PreparedClaudeMain> {
        let path = location.path();
        if !path.is_absolute() {
            return Err(refusal("absolute path required"));
        }
        if old
            .as_ref()
            .is_some_and(|s| s.context != location.context())
        {
            return Err(refusal("file context differs from the stored input's"));
        }
        let mut file = location.open()?;
        let meta = file.metadata()?;
        if !meta.is_file() {
            return Err(refusal("regular file required"));
        }
        let previous = old.as_ref().map(|s| s.input.length).unwrap_or(0);
        if let Some(source) = &old {
            let SourceIdentity::Observed { device, inode, .. } = source.identity else {
                return Err(refusal("location belongs to an immutable import"));
            };
            if device != meta.dev() || inode != meta.ino() {
                return Err(refusal(
                    "file was replaced; replaced files are not supported",
                ));
            }
            if meta.len() < previous {
                return Err(refusal("file shrank; rewritten files are not supported"));
            }
            let reader = self.read_handle()?;
            let mut offset = 0;
            while offset < previous {
                if cancelled.load(Ordering::Acquire) {
                    return Err(Error::Cancelled);
                }
                let length = (previous - offset).min(65536);
                let expected = reader.read_source_range_checked(
                    &source.input.id,
                    source.input.chain,
                    source.binding,
                    offset,
                    length,
                    || work(65536),
                )?;
                let mut actual = vec![0; length as usize];
                file.read_exact(&mut actual)?;
                if actual != expected {
                    return Err(refusal("stored prefix of the file changed"));
                }
                work(length)?;
                offset += length;
            }
        }
        file.seek(SeekFrom::Start(previous))?;
        let added = meta.len().checked_sub(previous).ok_or(Error::Range)?;
        let proof = if added > 0 || old.is_none() {
            Some(self.capture_with_check(
                Exact {
                    reader: &mut file,
                    remaining: added,
                },
                cancelled,
                &mut work,
            )?)
        } else {
            None
        };
        file.seek(SeekFrom::Start(0))?;
        let mut left = meta.len();
        let mut buffer = vec![0; 65536];
        while left > 0 {
            if cancelled.load(Ordering::Acquire) {
                return Err(Error::Cancelled);
            }
            let n = left.min(65536) as usize;
            file.read_exact(&mut buffer[..n])?;
            inspect(&buffer[..n])?;
            work(n as u64)?;
            left -= n as u64;
        }
        let after = file.metadata()?;
        if (
            after.dev(),
            after.ino(),
            after.len(),
            after.mtime(),
            after.mtime_nsec(),
            after.ctime(),
            after.ctime_nsec(),
        ) != (
            meta.dev(),
            meta.ino(),
            meta.len(),
            meta.mtime(),
            meta.mtime_nsec(),
            meta.ctime(),
            meta.ctime_nsec(),
        ) {
            return Err(refusal("file changed while being read; retry"));
        }
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        location.check()?;
        Ok(PreparedClaudeMain {
            location,
            old,
            meta,
            proof,
        })
    }
}

struct Exact<R> {
    reader: R,
    remaining: u64,
}
impl<R: Read> Read for Exact<R> {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        if self.remaining == 0 || b.is_empty() {
            return Ok(0);
        }
        let limit = b
            .len()
            .min(usize::try_from(self.remaining).unwrap_or(usize::MAX));
        let n = self.reader.read(&mut b[..limit])?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "file ended before its stat size",
            ));
        }
        self.remaining -= n as u64;
        Ok(n)
    }
}
impl ReadHandle {
    /// Read a byte range of a stored input. A gap or overlap in its extents is
    /// an error.
    pub fn read_source_range(
        &self,
        id: &str,
        chain: i64,
        binding: BlobHash,
        offset: u64,
        length: u64,
    ) -> crate::Result<Vec<u8>> {
        self.read_source_range_checked(id, chain, binding, offset, length, || Ok(()))
    }
    pub fn read_source_range_checked(
        &self,
        id: &str,
        chain: i64,
        binding: BlobHash,
        offset: u64,
        length: u64,
        mut check: impl FnMut() -> crate::Result<()>,
    ) -> crate::Result<Vec<u8>> {
        check()?;
        let file = self.root.join("archive.db");
        let source = self.source(id)?.ok_or_else(|| refusal("unknown source"))?;
        if source.input.chain != chain
            || source.binding != binding
            || offset
                .checked_add(length)
                .is_none_or(|end| end > source.input.length)
        {
            return Err(Error::Range);
        }
        let end = offset + length;
        let mut result = Vec::new();
        result
            .try_reserve_exact(usize::try_from(length).map_err(|_| Error::Range)?)
            .map_err(|e| Error::Io(io::Error::other(e)))?;
        let mut position = offset;
        while position < end {
            let row: Option<(i64, i64, Vec<u8>, i64)> = db(&file, || {
                self.archive.query_row("SELECT offset,length,blob,blob_offset FROM extent WHERE input=?1 AND chain=?2 AND offset<=?3 AND offset+length>?3 ORDER BY offset LIMIT 1",params![id,chain,position as i64],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).optional()
            })?;
            let Some((start, size, hash, blob_start)) = row else {
                return Err(refusal("stored input has a gap in its extents"));
            };
            let start = u64::try_from(start).map_err(|_| Error::Range)?;
            let size = u64::try_from(size).map_err(|_| Error::Range)?;
            let extent_end = start.checked_add(size).ok_or(Error::Range)?;
            if extent_end > source.input.length {
                return Err(Error::Range);
            }
            let overlaps: bool = db(&file, || {
                self.archive.query_row("SELECT EXISTS(SELECT 1 FROM extent WHERE input=?1 AND chain=?2 AND offset<>?3 AND offset<?4 AND offset+length>?3)",params![id,chain,start as i64,extent_end as i64],|r|r.get(0))
            })?;
            if overlaps {
                return Err(refusal("stored input has overlapping extents"));
            }
            let take = (end - position).min(extent_end - position);
            let hash = BlobHash::from_bytes(
                hash.try_into()
                    .map_err(|_| refusal("invalid extent digest"))?,
            );
            let (reader, verify_first) = self.source_readers.select(id, chain, binding, hash)?;
            result.extend_from_slice(
                &self.read_blob_range_using(
                    reader,
                    verify_first,
                    hash,
                    u64::try_from(blob_start)
                        .map_err(|_| Error::Range)?
                        .checked_add(position - start)
                        .ok_or(Error::Range)?,
                    take,
                    &mut check,
                )?,
            );
            position += take;
        }
        Ok(result)
    }
    /// Verify every blob behind a stored input. Range reads check extent
    /// coverage separately.
    pub fn verify_source(
        &self,
        source: &StoredSource,
        mut check: impl FnMut() -> crate::Result<()>,
    ) -> crate::Result<()> {
        let file = self.root.join("archive.db");
        let mut after = -1i64;
        loop {
            check()?;
            let row: Option<(i64, Vec<u8>)> = db(&file, || {
                self.archive.query_row("SELECT e.offset,e.blob FROM extent e JOIN blob b ON b.hash=e.blob AND b.state='present' WHERE e.input=?1 AND e.chain=?2 AND e.offset>?3 AND e.offset<?4 ORDER BY e.offset LIMIT 1",params![source.input.id,source.input.chain,after,source.input.length as i64],|r|Ok((r.get(0)?,r.get(1)?))).optional()
            })?;
            let Some((offset, hash)) = row else {
                break;
            };
            after = offset;
            let hash = BlobHash::from_bytes(
                hash.try_into()
                    .map_err(|_| refusal("invalid backing digest"))?,
            );
            let (reader, verify_first) = self.source_readers.select(
                &source.input.id,
                source.input.chain,
                source.binding,
                hash,
            )?;
            // A zero-length read still re-checks the blob's state in the archive.
            self.read_blob_range_using(reader, verify_first, hash, 0, 0, &mut check)?;
        }
        Ok(())
    }
    pub fn source(&self, id: &str) -> crate::Result<Option<StoredSource>> {
        let file = self.root.join("archive.db");
        let excisions: bool = db(&file, || {
            self.archive
                .query_row("SELECT EXISTS(SELECT 1 FROM excision)", [], |r| r.get(0))
        })?;
        if excisions {
            return Err(Error::ExcisionCheckRequired);
        }
        decode(&self.archive, &file, id)
    }
}

#[cfg(test)]
#[path = "observed/tests.rs"]
mod tests;
