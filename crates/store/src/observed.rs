//! Retained input identity and contiguous source reads. Provider paths are used
//! only during explicit startup acquisition, never by interpretation/readback.
use crate::{
    BlobHash, Error, ImmutableImport, InputFormat, ReadHandle, RetainedInput, Writer, counter, db,
};
use rusqlite::{OptionalExtension, params};
use std::{
    fs::OpenOptions,
    io::{self, Read, Seek, SeekFrom},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
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
pub struct RetainedSource {
    pub input: RetainedInput,
    /// Immutable original hash, or domain-separated observed input/chain key.
    /// This is an interpretation binding, not a full observed-content digest.
    pub binding: BlobHash,
    pub identity: SourceIdentity,
    pub format: InputFormat,
    pub context: Vec<String>,
    pub revision: u64,
}
impl From<ImmutableImport> for RetainedSource {
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
        }
    }
}
impl RetainedSource {
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

/// Decode one retained identity from archive facts only. Observed context is
/// the first location's stem, independently of its current provider contents.
pub(crate) fn decode(
    connection: &rusqlite::Connection,
    file: &Path,
    id: &str,
) -> crate::Result<Option<RetainedSource>> {
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
        let (id, hash, format, context, length) = crate::retained::decode_import_identity(
            id.into(),
            format,
            length,
            digest.ok_or_else(|| refusal("immutable digest missing"))?,
            original.unwrap(),
            context.ok_or_else(|| refusal("immutable context missing"))?,
        )?;
        if revision != 1 || chain != 1 {
            return Err(refusal("unsupported immutable chain"));
        }
        return Ok(Some(RetainedSource {
            input: RetainedInput {
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
        }));
    }
    if context.is_some() || format != "claude-code-jsonl" || chain != 1 {
        return Err(refusal("unsupported location identity"));
    }
    let (machine, path, device, inode): (String, String, i64, i64) = db(file, || {
        connection.query_row("SELECT CASE WHEN length(CAST(machine AS BLOB))<=65536 THEN machine END,CASE WHEN length(CAST(path AS BLOB))<=65536 THEN path END,device,inode FROM observation WHERE input=?1 AND class='first' ORDER BY rowid LIMIT 1",[id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?)))
    })?;
    let stem = Path::new(&path)
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| refusal("first source stem is invalid"))?;
    let context = vec!["claude-main".to_string(), stem.to_string()];
    InputFormat::ClaudeCodeJsonl
        .validate_context(&context.iter().map(String::as_str).collect::<Vec<_>>())?;
    Ok(Some(RetainedSource {
        input: RetainedInput {
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
    }))
}
impl Writer {
    pub fn retained_source(&mut self, id: &str) -> crate::Result<Option<RetainedSource>> {
        self.check_import_acquisition()?;
        let Some(mut source) = decode(&self.archive, &self.root.join("archive.db"), id)? else {
            return Ok(None);
        };
        source.input = self.register_retained_reference(source.input.id, source.input.length)?;
        Ok(Some(source))
    }
    pub fn next_retained_source(&mut self, after: &str) -> crate::Result<Option<RetainedSource>> {
        self.check_import_acquisition()?;
        let id: Option<String> = db(&self.root.join("archive.db"), || {
            self.archive
                .query_row(
                    "SELECT id FROM input WHERE id>?1 ORDER BY id LIMIT 1",
                    [after],
                    |r| r.get(0),
                )
                .optional()
        })?;
        id.map(|id| self.retained_source(&id))
            .transpose()
            .map(Option::flatten)
    }
    /// Explicit single regular location, performed before starting the actor.
    /// Full-prefix comparison deliberately avoids sampled-prefix certification.
    /// Any in-flight descriptor change refuses this attempt; bytes already in
    /// CAS remain recoverable but acquire no input/frontier authority.
    pub fn acquire_claude_main(
        &mut self,
        path: &Path,
        machine: &str,
        host: &str,
        id: &str,
        seen_at: i64,
        cancelled: &AtomicBool,
    ) -> crate::Result<RetainedSource> {
        self.check_import_acquisition()?;
        if !path.is_absolute() || machine.is_empty() || host.is_empty() || id.is_empty() {
            return Err(refusal("absolute path and acquisition identities required"));
        }
        let path_text = path
            .to_str()
            .ok_or_else(|| refusal("non-UTF8 path is unsupported"))?;
        if path_text.len() + machine.len() + host.len() + id.len() > 65536 {
            return Err(refusal("source metadata exceeds admission"));
        }
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW)
            .open(path)?;
        let meta = file.metadata()?;
        if !meta.is_file() {
            return Err(refusal("regular file required"));
        }
        let archive_file = self.root.join("archive.db");
        let old_id: Option<String> = db(&archive_file, || {
            self.archive.query_row("SELECT input FROM observation WHERE machine=?1 AND path=?2 AND class IN ('first','appended') ORDER BY rowid DESC LIMIT 1",params![machine,path_text],|r|r.get(0)).optional()
        })?;
        let old = old_id
            .as_ref()
            .map(|id| self.retained_source(id))
            .transpose()?
            .flatten();
        let previous = old.as_ref().map(|s| s.input.length).unwrap_or(0);
        if let Some(source) = &old {
            let SourceIdentity::Observed { device, inode, .. } = source.identity else {
                return Err(refusal("location belongs to an immutable import"));
            };
            if device != meta.dev() || inode != meta.ino() {
                return Err(refusal(
                    "replacement requires unsupported revision handling",
                ));
            }
            if meta.len() < previous {
                return Err(refusal("source shrank; rewrite handling is unsupported"));
            }
            let reader = self.read_handle()?;
            let mut offset = 0;
            while offset < previous {
                if cancelled.load(Ordering::Acquire) {
                    return Err(Error::Cancelled);
                }
                let length = (previous - offset).min(65536);
                let expected = reader.read_source_range(
                    &source.input.id,
                    source.input.chain,
                    source.binding,
                    offset,
                    length,
                )?;
                let mut actual = vec![0; length as usize];
                file.read_exact(&mut actual)?;
                if actual != expected {
                    return Err(refusal("retained predecessor prefix changed"));
                }
                offset += length;
            }
        }
        file.seek(SeekFrom::Start(previous))?;
        let added = meta.len().checked_sub(previous).ok_or(Error::Range)?;
        let proof = if added > 0 || old.is_none() {
            let capture = self.blob_capture()?;
            Some(capture.capture(
                Exact {
                    reader: &mut file,
                    remaining: added,
                },
                cancelled,
            )?)
        } else {
            None
        };
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
            return Err(refusal(
                "descriptor changed during acquisition; retry unchanged action",
            ));
        }
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        if old.is_some() && added == 0 {
            return Ok(old.unwrap());
        }
        let hash =
            self.retain_captured_blob(proof.ok_or_else(|| refusal("missing capture proof"))?)?;
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
        self.retained_source(input_id)?
            .ok_or_else(|| refusal("committed acquisition missing"))
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
                "observed descriptor ended before stat size",
            ));
        }
        self.remaining -= n as u64;
        Ok(n)
    }
}
impl ReadHandle {
    /// Resolve exact retained coordinates, checking every covering extent. No
    /// provider-path fallback and no implicit gap/overlap selection.
    pub fn read_source_range(
        &self,
        id: &str,
        chain: i64,
        binding: BlobHash,
        offset: u64,
        length: u64,
    ) -> crate::Result<Vec<u8>> {
        let file = self.root.join("archive.db");
        let source = self
            .source(id)?
            .ok_or_else(|| refusal("unknown retained source"))?;
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
                return Err(refusal("missing retained predecessor range"));
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
                return Err(refusal("overlapping retained extents"));
            }
            let take = (end - position).min(extent_end - position);
            let hash = BlobHash::from_bytes(
                hash.try_into()
                    .map_err(|_| refusal("invalid extent digest"))?,
            );
            result.extend_from_slice(
                &self.read_blob_range(
                    hash,
                    u64::try_from(blob_start)
                        .map_err(|_| Error::Range)?
                        .checked_add(position - start)
                        .ok_or(Error::Range)?,
                    take,
                )?,
            );
            position += take;
        }
        Ok(result)
    }
    /// Cryptographically verify every backing piece before a reconstruction is
    /// exposed. Range reads separately check exact coverage and frame integrity.
    pub fn verify_source(
        &self,
        source: &RetainedSource,
        mut check: impl FnMut() -> crate::Result<()>,
    ) -> crate::Result<()> {
        let file = self.root.join("archive.db");
        let mut after = -1i64;
        loop {
            check()?;
            let row: Option<(i64, Vec<u8>, i64)> = db(&file, || {
                self.archive.query_row("SELECT e.offset,e.blob,b.size FROM extent e JOIN blob b ON b.hash=e.blob AND b.state='present' WHERE e.input=?1 AND e.chain=?2 AND e.offset>?3 AND e.offset<?4 ORDER BY e.offset LIMIT 1",params![source.input.id,source.input.chain,after,source.input.length as i64],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()
            })?;
            let Some((offset, hash, length)) = row else {
                break;
            };
            after = offset;
            let hash = BlobHash::from_bytes(
                hash.try_into()
                    .map_err(|_| refusal("invalid backing digest"))?,
            );
            let length = u64::try_from(length).map_err(|_| Error::Range)?;
            let mut digest = blake3::Hasher::new();
            let mut cursor = 0;
            while cursor < length {
                check()?;
                let bytes = self.read_blob_range(hash, cursor, (length - cursor).min(65536))?;
                digest.update(&bytes);
                cursor += bytes.len() as u64;
            }
            if digest.finalize().as_bytes() != hash.bytes() {
                return Err(refusal("backing piece content hash mismatch"));
            }
        }
        Ok(())
    }
    pub fn source(&self, id: &str) -> crate::Result<Option<RetainedSource>> {
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
mod tests {
    use super::*;
    use std::io::Write;
    fn acquire(w: &mut Writer, p: &Path, id: &str) -> RetainedSource {
        w.acquire_claude_main(
            p,
            "synthetic-machine",
            "synthetic-host",
            id,
            1,
            &AtomicBool::new(false),
        )
        .unwrap()
    }
    #[test]
    fn first_nonempty_observed_revision_records_full_digest_without_immutable_identity() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("session.jsonl");
        let bytes = b"synthetic first observed bytes\n";
        std::fs::write(&path, bytes).unwrap();
        let mut w = Writer::open(&root.path().join("archive")).unwrap();
        let source = acquire(&mut w, &path, "first");
        let (digest, original): (Vec<u8>, Option<Vec<u8>>) = w.archive.query_row("SELECT r.blake3,i.original_hash FROM revision r JOIN input i ON i.id=r.input WHERE r.input=?1 AND r.number=1", [&source.input.id], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
        assert_eq!(digest, blake3::hash(bytes).as_bytes());
        assert!(original.is_none());
        assert!(matches!(source.identity, SourceIdentity::Observed { .. }));
    }
    #[test]
    fn same_location_retains_empty_genesis_separator_and_split_record_extents() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("session.jsonl");
        std::fs::write(&path, b"").unwrap();
        let archive = root.path().join("archive");
        let mut w = Writer::open(&archive).unwrap();
        let first = acquire(&mut w, &path, "genesis");
        assert_eq!(first.input.length, 0);
        let digest: Vec<u8> = w
            .archive
            .query_row(
                "SELECT blake3 FROM revision WHERE input=?1 AND number=1",
                [&first.input.id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(digest, blake3::hash(b"").as_bytes());
        let original: Option<Vec<u8>> = w
            .archive
            .query_row(
                "SELECT original_hash FROM input WHERE id=?1",
                [&first.input.id],
                |r| r.get(0),
            )
            .unwrap();
        assert!(original.is_none());
        assert_eq!(first.context, ["claude-main", "session"]);
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        let parts = [
            b"{\"a\":1}\n".as_slice(),
            b"\n \r\n",
            b"{\"partial\":",
            b"true}\n",
        ];
        let mut whole = Vec::new();
        for (i, part) in parts.iter().enumerate() {
            file.write_all(part).unwrap();
            file.sync_all().unwrap();
            whole.extend_from_slice(part);
            let next = acquire(&mut w, &path, &format!("step-{i}"));
            assert_eq!(next.input.id, first.input.id);
            assert_eq!(next.input.chain, 1);
            assert_eq!(next.binding, first.binding);
            assert_eq!(next.revision, i as u64 + 2);
            let digest: Option<Vec<u8>> = w
                .archive
                .query_row(
                    "SELECT blake3 FROM revision WHERE input=?1 AND number=?2",
                    params![next.input.id, i64::try_from(next.revision).unwrap()],
                    |r| r.get(0),
                )
                .unwrap();
            assert!(digest.is_none());
            assert_eq!(
                w.read_handle()
                    .unwrap()
                    .read_source_range(&next.input.id, 1, next.binding, 0, next.input.length)
                    .unwrap(),
                whole
            );
            drop(w);
            w = Writer::open(&archive).unwrap();
        }
        let before = w.status().unwrap().write_counter;
        let retry = acquire(&mut w, &path, "unchanged");
        assert_eq!(retry.revision, 5);
        assert_eq!(w.status().unwrap().write_counter, before);
        assert_eq!(
            w.archive
                .query_row("SELECT count(*) FROM extent", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            4
        );
    }
    #[test]
    fn mutation_replacement_gap_overlap_and_missing_blob_refuse_without_frontier_advance() {
        let root = tempfile::tempdir().unwrap();
        let p = root.path().join("s.jsonl");
        std::fs::write(&p, b"first\n").unwrap();
        let mut w = Writer::open(&root.path().join("archive")).unwrap();
        let s = acquire(&mut w, &p, "first");
        let counter = w.status().unwrap().write_counter;
        std::fs::write(&p, b"wrong\nmore\n").unwrap();
        assert!(
            w.acquire_claude_main(
                &p,
                "synthetic-machine",
                "synthetic-host",
                "bad",
                2,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert_eq!(w.status().unwrap().write_counter, counter);
        let replacement = root.path().join("replacement");
        std::fs::write(&replacement, b"first\nmore\n").unwrap();
        std::fs::rename(&replacement, &p).unwrap();
        assert!(
            w.acquire_claude_main(
                &p,
                "synthetic-machine",
                "synthetic-host",
                "replaced",
                2,
                &AtomicBool::new(false)
            )
            .is_err()
        );
        assert_eq!(w.status().unwrap().write_counter, counter);
        w.archive.execute("UPDATE extent SET offset=1", []).unwrap();
        assert!(
            w.read_handle()
                .unwrap()
                .read_source_range(&s.input.id, 1, s.binding, 0, 6)
                .is_err()
        );
        w.archive.execute("UPDATE extent SET offset=0", []).unwrap();
        w.archive.execute("INSERT INTO extent(input,chain,offset,length,blob,blob_offset) SELECT input,chain,2,2,blob,0 FROM extent",[]).unwrap();
        assert!(
            w.read_handle()
                .unwrap()
                .read_source_range(&s.input.id, 1, s.binding, 0, 6)
                .is_err()
        );
        w.archive
            .execute("DELETE FROM extent WHERE offset=2", [])
            .unwrap();
        w.archive
            .execute("UPDATE blob SET state='deleting'", [])
            .unwrap();
        assert!(
            w.read_handle()
                .unwrap()
                .read_source_range(&s.input.id, 1, s.binding, 0, 6)
                .is_err()
        );
    }
    #[test]
    fn archive_commit_before_index_registration_recovers_without_provider() {
        let root = tempfile::tempdir().unwrap();
        let p = root.path().join("s.jsonl");
        std::fs::write(&p, b"retained\n").unwrap();
        let archive = root.path().join("archive");
        let mut w = Writer::open(&archive).unwrap();
        w._index.execute_batch("CREATE TRIGGER refuse_ref BEFORE INSERT ON input_ref BEGIN SELECT RAISE(ABORT,'registration interrupted'); END;").unwrap();
        assert!(
            w.acquire_claude_main(&p, "m", "h", "id", 1, &AtomicBool::new(false))
                .is_err()
        );
        let before = w.status().unwrap().write_counter;
        assert!(before > 0);
        w._index.execute_batch("DROP TRIGGER refuse_ref;").unwrap();
        drop(w);
        std::fs::remove_file(p).unwrap();
        let mut w = Writer::open(&archive).unwrap();
        let recovered = w.next_retained_source("").unwrap().unwrap();
        assert_eq!(recovered.input.id, "id");
        assert_eq!(w.status().unwrap().write_counter, before);
        assert_eq!(
            w.read_handle()
                .unwrap()
                .read_source_range("id", 1, recovered.binding, 0, 9)
                .unwrap(),
            b"retained\n"
        );
    }
    #[test]
    fn stat_bounded_reader_refuses_short_capture_instead_of_successful_eof() {
        let mut bytes = Vec::new();
        let result = Exact {
            reader: b"short".as_slice(),
            remaining: 9,
        }
        .read_to_end(&mut bytes);
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
        let mut bounded = Vec::new();
        Exact {
            reader: b"longer".as_slice(),
            remaining: 3,
        }
        .read_to_end(&mut bounded)
        .unwrap();
        assert_eq!(bounded, b"lon");
    }
    #[test]
    fn retained_source_reads_refuse_unhandled_archive_excisions() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("s.jsonl");
        std::fs::write(&path, b"retained\n").unwrap();
        let mut writer = Writer::open(&root.path().join("archive")).unwrap();
        let source = acquire(&mut writer, &path, "input");
        writer.archive.execute("INSERT INTO excision(id,scope,created_at,local_seq) VALUES('e','input:unknown',1,1)",[]).unwrap();
        let reader = writer.read_handle().unwrap();
        assert!(matches!(
            reader.source(&source.input.id),
            Err(Error::ExcisionCheckRequired)
        ));
        assert!(matches!(
            reader.read_source_range(&source.input.id, 1, source.binding, 0, source.input.length),
            Err(Error::ExcisionCheckRequired)
        ));
    }
}
