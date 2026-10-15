//! One canonical normalized payload store. Appends and range reads touch one
//! admitted chunk (at most 64 KiB); integrity hashing runs outside write
//! transactions. A lost process replays hashing from zero in bounded steps.
//! Stored chunks and their declared total length are immutable through this API.
//! Ready proves completeness/BLAKE3 integrity, not JSON syntax or raw-source
//! version evidence. JSON canonical encoding remains its producer's contract.
use crate::{Error, ReadHandle, Result, Writer};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use std::path::Path;

pub const MAX_CHUNK: usize = 65_536;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    Utf8,
    Json,
    Bytes,
}
impl Codec {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Utf8 => "utf8",
            Self::Json => "json",
            Self::Bytes => "bytes",
        }
    }
    fn parse(name: &str) -> Result<Self> {
        match name {
            "utf8" => Ok(Self::Utf8),
            "json" => Ok(Self::Json),
            "bytes" => Ok(Self::Bytes),
            _ => Err(invalid("unknown payload codec")),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Building,
    Ready,
    Retiring,
}
/// Local canonical payload identity, fenced by disposable index incarnation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueId {
    pub(crate) pk: i64,
    pub(crate) index_generation: String,
    pub(crate) index_path: std::path::PathBuf,
}
impl ValueId {
    pub fn pk(&self) -> i64 {
        self.pk
    }
    pub fn index_generation(&self) -> &str {
        &self.index_generation
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Head {
    pub id: ValueId,
    pub codec: Codec,
    pub length: u64,
    /// Durable append progress, not a trusted saved BLAKE3 position.
    pub appended: u64,
    pub state: State,
    pub hash: Option<[u8; 32]>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadyValue {
    pub id: ValueId,
    pub codec: Codec,
    pub length: u64,
    pub hash: [u8; 32],
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Seal {
    pub ready: ReadyValue,
    pub retired_duplicate: Option<ValueId>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Range {
    pub bytes: Vec<u8>,
    pub next_offset: u64,
    pub eof: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidationProgress {
    pub hashed_upto: u64,
    pub complete: bool,
}
/// Private in-process hash proof. Its offset is never restored without its
/// corresponding library Hasher. After restart create a fresh proof at zero.
/// Each step is retryable by starting again; failures fuse the attempted proof.
pub struct Validation {
    id: ValueId,
    codec: Codec,
    length: u64,
    offset: u64,
    chunk: Option<(u64, u64)>,
    hasher: blake3::Hasher,
    digest: Option<[u8; 32]>,
    utf8: Utf8,
    failed: bool,
}
impl Validation {
    pub fn progress(&self) -> ValidationProgress {
        ValidationProgress {
            hashed_upto: self.offset,
            complete: !self.failed
                && self.offset == self.length
                && self.utf8.len == 0
                && self.digest.is_some(),
        }
    }
    pub fn value(&self) -> &ValueId {
        &self.id
    }
    pub fn advance(&mut self, reader: &ReadHandle, allowance: usize) -> Result<ValidationProgress> {
        if self.failed {
            return Err(invalid("payload validation attempt already failed"));
        }
        let result = self.advance_inner(reader, allowance);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn advance_inner(
        &mut self,
        reader: &ReadHandle,
        allowance: usize,
    ) -> Result<ValidationProgress> {
        let path = reader.root.join("index.db");
        let h = head(&reader.index, &path, &self.id)?;
        if h.state != State::Building
            || h.codec != self.codec
            || h.length != self.length
            || h.appended != h.length
            || h.hash.is_some()
        {
            return Err(Error::ValuePredecessorChanged);
        }
        if self.offset == self.length {
            return Ok(self.progress());
        }
        let piece = piece(&reader.index, &path, &h, self.offset, allowance)?;
        if let Some((start, end)) = self.chunk {
            if (self.offset < end && (piece.start, piece.end) != (start, end))
                || (self.offset == end && piece.start != end)
            {
                return Err(invalid(
                    "payload chunks overlap or changed during validation",
                ));
            }
        } else if piece.start != 0 {
            return Err(invalid("payload does not start at zero"));
        }
        if self.codec != Codec::Bytes {
            self.utf8.update(&piece.bytes)?;
        }
        self.hasher.update(&piece.bytes);
        self.offset = self
            .offset
            .checked_add(piece.bytes.len() as u64)
            .ok_or(Error::Range)?;
        self.chunk = Some((piece.start, piece.end));
        if self.offset == self.length {
            if self.utf8.len != 0 {
                return Err(invalid("payload ends in an incomplete UTF-8 codepoint"));
            }
            self.digest = Some(*self.hasher.finalize().as_bytes());
        }
        Ok(self.progress())
    }
}
#[derive(Default)]
struct Utf8 {
    tail: [u8; 4],
    len: usize,
}
impl Utf8 {
    fn update(&mut self, mut bytes: &[u8]) -> Result<()> {
        if self.len > 0 {
            let width = match self.tail[0] {
                0xc2..=0xdf => 2,
                0xe0..=0xef => 3,
                0xf0..=0xf4 => 4,
                _ => return Err(invalid("invalid payload UTF-8 prefix")),
            };
            let take = (width - self.len).min(bytes.len());
            self.tail[self.len..self.len + take].copy_from_slice(&bytes[..take]);
            self.len += take;
            bytes = &bytes[take..];
            if self.len < width {
                return Ok(());
            }
            std::str::from_utf8(&self.tail[..width])
                .map_err(|_| invalid("invalid payload UTF-8"))?;
            self.len = 0;
        }
        match std::str::from_utf8(bytes) {
            Ok(_) => Ok(()),
            Err(e) if e.error_len().is_none() => {
                let tail = &bytes[e.valid_up_to()..];
                if tail.len() > 3 {
                    return Err(invalid("invalid payload UTF-8 tail"));
                }
                self.tail[..tail.len()].copy_from_slice(tail);
                self.len = tail.len();
                Ok(())
            }
            Err(_) => Err(invalid("invalid payload UTF-8")),
        }
    }
}
fn invalid(message: &str) -> Error {
    Error::Invariant(message.into())
}
fn integer(n: u64) -> Result<i64> {
    i64::try_from(n).map_err(|_| Error::Range)
}
fn admission(resource: &'static str, n: usize) -> Error {
    Error::ParserAdmission {
        resource,
        required: u64::try_from(n).unwrap_or(u64::MAX),
    }
}
fn generation(db: &Connection, path: &Path) -> Result<String> {
    crate::db(path, || {
        db.query_row("SELECT value FROM meta WHERE key='generation'", [], |r| {
            r.get(0)
        })
    })
}
pub(crate) fn head(db: &Connection, path: &Path, id: &ValueId) -> Result<Head> {
    if path != id.index_path || generation(db, path)? != id.index_generation {
        return Err(Error::ValuePredecessorChanged);
    }
    let row = crate::db(path, || {
        db.query_row(
            "SELECT codec,length,hash,state,incoming FROM body_value WHERE pk=?1",
            [id.pk],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, Option<Vec<u8>>>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, i64>(4)?,
                ))
            },
        )
        .optional()
    })?
    .ok_or(Error::ValuePredecessorChanged)?;
    let codec = Codec::parse(&row.0)?;
    let length = u64::try_from(row.1).map_err(|_| Error::Range)?;
    let state = match row.3.as_str() {
        "building" => State::Building,
        "ready" => State::Ready,
        "retiring" => State::Retiring,
        _ => return Err(invalid("unknown payload state")),
    };
    let hash = row
        .2
        .map(|v| {
            v.try_into()
                .map_err(|_| invalid("invalid payload digest length"))
        })
        .transpose()?;
    if row.4 < 0
        || (state == State::Ready && hash.is_none())
        || (state == State::Retiring && row.4 != 0)
    {
        return Err(invalid("invalid canonical payload metadata"));
    }
    let last = crate::db(path, || {
        db.query_row("SELECT offset,length(bytes) FROM body_value_chunk WHERE value_pk=?1 ORDER BY offset DESC LIMIT 1",[id.pk],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?))).optional()
    })?;
    let appended = match last {
        None => 0,
        Some((offset, size)) => {
            let offset = u64::try_from(offset).map_err(|_| Error::Range)?;
            let size = u64::try_from(size).map_err(|_| Error::Range)?;
            if size == 0 || size > MAX_CHUNK as u64 {
                return Err(invalid("invalid canonical payload chunk length"));
            }
            offset.checked_add(size).ok_or(Error::Range)?
        }
    };
    if appended > length {
        return Err(invalid("payload chunks exceed declared length"));
    }
    if state == State::Ready && appended != length {
        return Err(invalid("ready payload has incomplete durable chunks"));
    }
    Ok(Head {
        id: id.clone(),
        codec,
        length,
        appended,
        state,
        hash,
    })
}
struct Piece {
    start: u64,
    end: u64,
    bytes: Vec<u8>,
}
fn piece(db: &Connection, path: &Path, h: &Head, offset: u64, allowance: usize) -> Result<Piece> {
    if offset >= h.length {
        return Err(Error::Range);
    }
    if allowance == 0 {
        return Err(admission("normalized value read bytes", 1));
    }
    let offset_i = integer(offset)?;
    let allowance = allowance.min(MAX_CHUNK);
    let row=crate::db(path,||db.query_row("SELECT offset,length(bytes),substr(bytes,?2-offset+1,?3) FROM body_value_chunk WHERE value_pk=?1 AND offset<=?2 ORDER BY offset DESC LIMIT 1",params![h.id.pk,offset_i,allowance as i64],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?,r.get::<_,Vec<u8>>(2)?))).optional())?.ok_or_else(||invalid("missing canonical payload chunk"))?;
    let start = u64::try_from(row.0).map_err(|_| Error::Range)?;
    let size = u64::try_from(row.1).map_err(|_| Error::Range)?;
    let end = start.checked_add(size).ok_or(Error::Range)?;
    if size == 0
        || size > MAX_CHUNK as u64
        || start > offset
        || end <= offset
        || end > h.length
        || row.2.is_empty()
        || row.2.len() > allowance
        || row.2.len() as u64 > end - offset
    {
        return Err(invalid("invalid or gapped canonical payload chunk"));
    }
    Ok(Piece {
        start,
        end,
        bytes: row.2,
    })
}
impl Writer {
    pub fn begin_value(&mut self, codec: Codec, length: u64) -> Result<Head> {
        let path = self.root.join("index.db");
        let tx = crate::db(&path, || self._index.transaction())?;
        let h = begin_in(&tx, &path, codec, length, 0)?;
        crate::db(&path, || tx.commit())?;
        Ok(h)
    }
    /// One immutable contiguous append. Retry by inspecting Head.appended; an
    /// old offset is rejected, never reinterpreted as a second append.
    pub fn append_value_chunk(
        &mut self,
        id: &ValueId,
        offset: u64,
        bytes: &[u8],
        allowance: usize,
    ) -> Result<Head> {
        if bytes.is_empty() {
            return Err(invalid("zero-byte canonical payload chunks are forbidden"));
        }
        if bytes.len() > MAX_CHUNK {
            return Err(Error::ValueChunkSplitRequired {
                required: bytes.len(),
                maximum: MAX_CHUNK,
            });
        }
        if bytes.len() > allowance {
            return Err(admission("normalized value append bytes", bytes.len()));
        }
        let offset_i = integer(offset)?;
        let end = offset.checked_add(bytes.len() as u64).ok_or(Error::Range)?;
        let path = self.root.join("index.db");
        let tx = crate::db(&path, || self._index.transaction())?;
        let h = head(&tx, &path, id)?;
        if h.state != State::Building || h.hash.is_some() || h.appended != offset {
            return Err(Error::ValuePredecessorChanged);
        }
        if end > h.length {
            return Err(Error::Range);
        }
        crate::db(&path, || {
            tx.execute(
                "INSERT INTO body_value_chunk(value_pk,offset,bytes) VALUES(?1,?2,?3)",
                params![id.pk, offset_i, bytes],
            )
        })?;
        let h = Head { appended: end, ..h };
        crate::db(&path, || tx.commit())?;
        Ok(h)
    }
    pub fn value_head(&self, id: &ValueId) -> Result<Head> {
        head(&self._index, &self.root.join("index.db"), id)
    }
    /// A fresh proof starts at zero. Complete durable bytes survive restart;
    /// ephemeral digest progress deliberately does not pretend to survive.
    pub fn validate_value(&self, id: &ValueId) -> Result<Validation> {
        validation(&self._index, &self.root.join("index.db"), id)
    }
    /// Consume the precomputed digest and CAS metadata. This method never
    /// reads payload chunks or performs hashing inside its transaction.
    pub fn seal_value(&mut self, proof: &Validation) -> Result<Seal> {
        if !proof.progress().complete {
            return Err(Error::ValueNotReady);
        }
        let path = self.root.join("index.db");
        let tx = crate::db(&path, || self._index.transaction())?;
        let result = seal_in(&tx, &path, proof, |_, _| Ok(()))?;
        crate::db(&path, || tx.commit())?;
        Ok(result)
    }
}
/// Allocate inside the owning transaction, including its producer lease count.
/// The caller must install matching ownership and commit before exposing Head.
/// Any error must roll back that transaction; no external receipt exists yet.
pub(crate) fn begin_in(
    tx: &Transaction<'_>,
    path: &Path,
    codec: Codec,
    length: u64,
    incoming: u64,
) -> Result<Head> {
    let length = integer(length)?;
    let incoming = integer(incoming)?;
    let index_generation = generation(tx, path)?;
    crate::db(path, || {
        tx.execute("INSERT INTO body_value(codec,length,hash,state,incoming) VALUES(?1,?2,NULL,'building',?3)",params![codec.as_str(),length,incoming])
    })?;
    let id = ValueId {
        pk: tx.last_insert_rowid(),
        index_generation,
        index_path: path.to_path_buf(),
    };
    head(tx, path, &id)
}
/// Consume an already-complete private digest proof without hashing or reading
/// payload bytes. The owner callback must verify its relation and move leases
/// atomically, without modifying payload bytes. A duplicate is retired only
/// after that callback leaves incoming0.
/// The caller MUST roll back on any error and commit before exposing Seal/IDs.
pub(crate) fn seal_in<F>(
    tx: &Transaction<'_>,
    path: &Path,
    proof: &Validation,
    transfer: F,
) -> Result<Seal>
where
    F: FnOnce(&Transaction<'_>, &ReadyValue) -> Result<()>,
{
    if !proof.progress().complete {
        return Err(Error::ValueNotReady);
    }
    let digest = proof.digest.ok_or(Error::ValueNotReady)?;
    if path != proof.id.index_path || generation(tx, path)? != proof.id.index_generation {
        return Err(Error::ValuePredecessorChanged);
    }
    let row = crate::db(path, || {
        tx.query_row(
            "SELECT codec,length,state,hash FROM body_value WHERE pk=?1",
            [proof.id.pk],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, Option<Vec<u8>>>(3)?,
                ))
            },
        )
        .optional()
    })?
    .ok_or(Error::ValuePredecessorChanged)?;
    if row.0 != proof.codec.as_str()
        || row.1 != integer(proof.length)?
        || row.2 != "building"
        || row.3.is_some()
    {
        return Err(Error::ValuePredecessorChanged);
    }
    let existing = crate::db(path, || {
        tx.query_row(
            "SELECT pk,length FROM body_value WHERE codec=?1 AND hash=?2 AND state='ready'",
            params![proof.codec.as_str(), digest.as_slice()],
            |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)),
        )
        .optional()
    })?;
    let (id, retired_duplicate) = if let Some((pk, length)) = existing {
        if length != integer(proof.length)? {
            return Err(invalid("equal payload digest has inconsistent length"));
        }
        (
            ValueId {
                pk,
                index_generation: proof.id.index_generation.clone(),
                index_path: path.to_path_buf(),
            },
            Some(proof.id.clone()),
        )
    } else {
        let changed = crate::db(path, || {
            tx.execute("UPDATE body_value SET state='ready',hash=?2 WHERE pk=?1 AND state='building' AND hash IS NULL",params![proof.id.pk,digest.as_slice()])
        })?;
        if changed != 1 {
            return Err(Error::ValuePredecessorChanged);
        }
        (proof.id.clone(), None)
    };

    let ready = ReadyValue {
        id,
        codec: proof.codec,
        length: proof.length,
        hash: digest,
    };
    transfer(tx, &ready)?;
    if retired_duplicate.is_some() {
        let changed = crate::db(path, || {
            tx.execute("UPDATE body_value SET state='retiring' WHERE pk=?1 AND state='building' AND incoming=0",[proof.id.pk])
        })?;
        if changed != 1 {
            return Err(Error::ValuePredecessorChanged);
        }
    }
    Ok(Seal {
        ready,
        retired_duplicate,
    })
}

fn validation(db: &Connection, path: &Path, id: &ValueId) -> Result<Validation> {
    let h = head(db, path, id)?;
    if h.state != State::Building || h.hash.is_some() {
        return Err(Error::ValuePredecessorChanged);
    }
    if h.appended != h.length {
        return Err(Error::ValueNotReady);
    }
    let hasher = blake3::Hasher::new();
    let digest = (h.length == 0).then(|| *hasher.finalize().as_bytes());
    Ok(Validation {
        id: id.clone(),
        codec: h.codec,
        length: h.length,
        offset: 0,
        chunk: None,
        hasher,
        digest,
        utf8: Utf8::default(),
        failed: false,
    })
}
impl ReadHandle {
    pub fn value_head(&self, id: &ValueId) -> Result<Head> {
        head(&self.index, &self.root.join("index.db"), id)
    }
    pub fn validate_value(&self, id: &ValueId) -> Result<Validation> {
        validation(&self.index, &self.root.join("index.db"), id)
    }
    /// Returns at most allowance bytes from one stored chunk. Byte boundaries
    /// may split UTF-8; callers must not decode a range with unchecked UTF-8.
    pub fn value_range(&self, id: &ValueId, offset: u64, allowance: usize) -> Result<Range> {
        range(
            &self.index,
            &self.root.join("index.db"),
            id,
            offset,
            allowance,
        )
    }
}

pub(crate) fn range(
    connection: &Connection,
    path: &Path,
    id: &ValueId,
    offset: u64,
    allowance: usize,
) -> Result<Range> {
    let h = head(connection, path, id)?;
    if h.state != State::Ready {
        return Err(Error::ValueNotReady);
    }
    if offset > h.length {
        return Err(Error::Range);
    }
    if offset == h.length {
        return Ok(Range {
            bytes: Vec::new(),
            next_offset: offset,
            eof: true,
        });
    }
    let p = piece(connection, path, &h, offset, allowance)?;
    let next_offset = offset + p.bytes.len() as u64;
    Ok(Range {
        bytes: p.bytes,
        next_offset,
        eof: next_offset == h.length,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn world() -> (tempfile::TempDir, std::path::PathBuf, Writer) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("archive");
        let writer = Writer::open(&root).unwrap();
        (dir, root, writer)
    }
    fn append(writer: &mut Writer, codec: Codec, bytes: &[u8], chunk: usize) -> Head {
        let mut h = writer.begin_value(codec, bytes.len() as u64).unwrap();
        for bytes in bytes.chunks(chunk) {
            h = writer
                .append_value_chunk(&h.id, h.appended, bytes, bytes.len())
                .unwrap();
        }
        h
    }
    fn validate(writer: &Writer, id: &ValueId, allowance: usize) -> Validation {
        let reader = writer.read_handle().unwrap();
        let mut proof = reader.validate_value(id).unwrap();
        while !proof.progress().complete {
            proof.advance(&reader, allowance).unwrap();
        }
        proof
    }
    #[test]
    fn append_progress_and_hash_restart_are_honest_and_ready_ranges_are_partial() {
        let (_dir, root, mut writer) = world();
        let h = writer.begin_value(Codec::Utf8, 11).unwrap();
        let h = writer.append_value_chunk(&h.id, 0, b"hello", 5).unwrap();
        assert_eq!(h.appended, 5);
        assert!(matches!(
            writer.validate_value(&h.id),
            Err(Error::ValueNotReady)
        ));
        assert!(matches!(
            writer.read_handle().unwrap().value_range(&h.id, 0, 3),
            Err(Error::ValueNotReady)
        ));
        drop(writer);
        let mut writer = Writer::open(&root).unwrap();
        let h = writer.value_head(&h.id).unwrap();
        assert_eq!(h.appended, 5);
        let h = writer.append_value_chunk(&h.id, 5, b" world", 6).unwrap();
        let reader = writer.read_handle().unwrap();
        let mut proof = writer.validate_value(&h.id).unwrap();
        assert_eq!(proof.advance(&reader, 2).unwrap().hashed_upto, 2);
        assert!(matches!(
            writer.seal_value(&proof),
            Err(Error::ValueNotReady)
        ));
        drop(proof);
        drop(reader);
        drop(writer);
        let mut writer = Writer::open(&root).unwrap();
        let proof = writer.validate_value(&h.id).unwrap();
        assert_eq!(
            proof.progress().hashed_upto,
            0,
            "restart never trusts an offset without hash state"
        );
        let proof = validate(&writer, &h.id, 1);
        let sealed = writer.seal_value(&proof).unwrap();
        assert_eq!(sealed.ready.hash, *blake3::hash(b"hello world").as_bytes());
        let reader = writer.read_handle().unwrap();
        assert_eq!(
            reader.value_range(&sealed.ready.id, 3, 10).unwrap().bytes,
            b"lo"
        );
        assert_eq!(
            reader.value_range(&sealed.ready.id, 4, 1).unwrap().bytes,
            b"o"
        );
        assert_eq!(
            reader.value_range(&sealed.ready.id, 5, 3).unwrap().bytes,
            b" wo"
        );
        let eof = reader.value_range(&sealed.ready.id, 11, 0).unwrap();
        assert!(eof.eof && eof.bytes.is_empty());
        assert!(matches!(
            reader.value_range(&sealed.ready.id, 12, 1),
            Err(Error::Range)
        ));
        assert!(matches!(
            reader.value_range(&sealed.ready.id, 0, 0),
            Err(Error::ParserAdmission {
                resource: "normalized value read bytes",
                required: 1
            })
        ));
    }
    #[test]
    fn zero_length_and_deduplication_have_no_empty_chunks_or_hidden_hash_pass() {
        let (_dir, _root, mut writer) = world();
        let h = writer.begin_value(Codec::Bytes, 0).unwrap();
        assert!(writer.append_value_chunk(&h.id, 0, b"", 0).is_err());
        let proof = writer.validate_value(&h.id).unwrap();
        assert!(proof.progress().complete);
        let sealed = writer.seal_value(&proof).unwrap();
        assert_eq!(sealed.ready.hash, *blake3::hash(b"").as_bytes());
        let duplicate = writer.begin_value(Codec::Bytes, 0).unwrap();
        let proof = writer.validate_value(&duplicate.id).unwrap();
        let result = writer.seal_value(&proof).unwrap();
        assert_eq!(result.ready.id, sealed.ready.id);
        assert_eq!(result.retired_duplicate, Some(duplicate.id.clone()));
        assert_eq!(
            writer.value_head(&duplicate.id).unwrap().state,
            State::Retiring
        );
        let count: i64 = crate::db(&writer.root.join("index.db"), || {
            writer
                ._index
                .query_row("SELECT count(*) FROM body_value_chunk", [], |r| r.get(0))
        })
        .unwrap();
        assert_eq!(count, 0);
        let different_codec = writer.begin_value(Codec::Utf8, 0).unwrap();
        let proof = writer.validate_value(&different_codec.id).unwrap();
        assert_ne!(writer.seal_value(&proof).unwrap().ready.id, sealed.ready.id);
    }
    #[test]
    fn chunk_admission_offset_cas_and_ranges_never_materialize_whole_values() {
        let (_dir, _root, mut writer) = world();
        let bytes = vec![b'x'; MAX_CHUNK * 2 + 17];
        let h = writer
            .begin_value(Codec::Bytes, bytes.len() as u64)
            .unwrap();
        assert!(matches!(
            writer.append_value_chunk(&h.id, 0, &bytes[..MAX_CHUNK + 1], usize::MAX),
            Err(Error::ValueChunkSplitRequired {required,maximum}) if required==MAX_CHUNK+1 && maximum==MAX_CHUNK
        ));
        assert!(matches!(
            writer.append_value_chunk(&h.id, 0, b"xx", 1),
            Err(Error::ParserAdmission {
                resource: "normalized value append bytes",
                required: 2
            })
        ));
        assert_eq!(writer.value_head(&h.id).unwrap().appended, 0);
        writer
            .append_value_chunk(&h.id, 0, &bytes[..MAX_CHUNK], MAX_CHUNK)
            .unwrap();
        assert!(matches!(
            writer.append_value_chunk(&h.id, 0, b"x", 1),
            Err(Error::ValuePredecessorChanged)
        ));
        assert!(matches!(
            writer.append_value_chunk(&h.id, (MAX_CHUNK + 1) as u64, b"x", 1),
            Err(Error::ValuePredecessorChanged)
        ));
        writer
            .append_value_chunk(
                &h.id,
                MAX_CHUNK as u64,
                &bytes[MAX_CHUNK..2 * MAX_CHUNK],
                MAX_CHUNK,
            )
            .unwrap();
        writer
            .append_value_chunk(&h.id, (2 * MAX_CHUNK) as u64, &bytes[2 * MAX_CHUNK..], 17)
            .unwrap();
        assert!(matches!(
            writer.append_value_chunk(&h.id, bytes.len() as u64, b"x", 1),
            Err(Error::Range)
        ));
        let proof = validate(&writer, &h.id, 257);
        let ready = writer.seal_value(&proof).unwrap().ready;
        assert_eq!(ready.hash, *blake3::hash(&bytes).as_bytes());
        let reader = writer.read_handle().unwrap();
        let range = reader.value_range(&ready.id, 0, usize::MAX).unwrap();
        assert_eq!(range.bytes.len(), MAX_CHUNK);
        assert!(!range.eof);
        let range = reader
            .value_range(&ready.id, (MAX_CHUNK - 1) as u64, usize::MAX)
            .unwrap();
        assert_eq!(range.bytes.len(), 1);
    }
    #[test]
    fn utf8_checks_survive_every_split_and_never_decode_unchecked_json() {
        for codec in [Codec::Utf8, Codec::Json] {
            let (_dir, _root, mut writer) = world();
            let bytes = "aé日🦀z".as_bytes();
            let h = append(&mut writer, codec, bytes, 1);
            let proof = validate(&writer, &h.id, 1);
            writer.seal_value(&proof).unwrap();
            // Json is an encoding declaration; syntax/canonicality is the
            // typed producer's separate contract, not invented by this proof.
        }
        for bytes in [&b"\xf0\x9f"[..], &b"\xed\xa0\x80"[..], &b"\xc0\x80"[..]] {
            let (_dir, _root, mut writer) = world();
            let h = append(&mut writer, Codec::Utf8, bytes, 1);
            let reader = writer.read_handle().unwrap();
            let mut proof = writer.validate_value(&h.id).unwrap();
            while !proof.failed {
                let _ = proof.advance(&reader, 1);
            }
            let offset = proof.progress().hashed_upto;
            assert!(proof.advance(&reader, 1).is_err());
            assert_eq!(offset, proof.progress().hashed_upto);
            assert!(matches!(
                writer.seal_value(&proof),
                Err(Error::ValueNotReady)
            ));
            assert_eq!(writer.value_head(&h.id).unwrap().state, State::Building);
        }
    }
    #[test]
    fn missing_overlapping_chunks_fail_validation_without_readiness() {
        for overlap in [false, true] {
            let (_dir, _root, mut writer) = world();
            let h = append(&mut writer, Codec::Bytes, b"abcdefgh", 4);
            let path = writer.root.join("index.db");
            if overlap {
                crate::db(&path, || {
                    writer._index.execute(
                        "INSERT INTO body_value_chunk(value_pk,offset,bytes) VALUES(?1,2,?2)",
                        params![h.id.pk, b"XYZW".as_slice()],
                    )
                })
                .unwrap();
            } else {
                crate::db(&path, || {
                    writer._index.execute(
                        "DELETE FROM body_value_chunk WHERE value_pk=?1 AND offset=0",
                        [h.id.pk],
                    )
                })
                .unwrap();
            }
            let reader = writer.read_handle().unwrap();
            let mut proof = writer.validate_value(&h.id).unwrap();
            let mut failed = false;
            for _ in 0..12 {
                if proof.advance(&reader, 1).is_err() {
                    failed = true;
                    break;
                }
            }
            assert!(failed);
            assert!(matches!(
                writer.seal_value(&proof),
                Err(Error::ValueNotReady)
            ));
        }
    }
    #[test]
    fn native_sql_error_survives_and_deleted_ids_or_new_index_cannot_reuse_proof() {
        let (_dir, _root, mut writer) = world();
        let empty = writer.begin_value(Codec::Bytes, 0).unwrap();
        let proof = writer.validate_value(&empty.id).unwrap();
        let path = writer.root.join("index.db");
        crate::db(&path, || {
            writer
                ._index
                .execute("DELETE FROM body_value WHERE pk=?1", [empty.id.pk])
        })
        .unwrap();
        let replacement = writer.begin_value(Codec::Bytes, 0).unwrap();
        assert_ne!(empty.id.pk, replacement.id.pk);
        assert!(matches!(
            writer.seal_value(&proof),
            Err(Error::ValuePredecessorChanged)
        ));
        let proof = writer.validate_value(&replacement.id).unwrap();
        crate::db(&path, || {
            writer._index.execute(
                "UPDATE meta SET value='synthetic-new-index' WHERE key='generation'",
                [],
            )
        })
        .unwrap();
        assert!(matches!(
            writer.seal_value(&proof),
            Err(Error::ValuePredecessorChanged)
        ));
        let (_dir2, _root2, mut writer2) = world();
        let h = append(&mut writer2, Codec::Bytes, b"x", 1);
        let mut proof = writer2.validate_value(&h.id).unwrap();
        let reader = writer2.read_handle().unwrap();
        let path = writer2.root.join("index.db");
        crate::db(&path, || {
            writer2._index.execute_batch("DROP TABLE body_value_chunk")
        })
        .unwrap();
        assert!(matches!(proof.advance(&reader,1),Err(Error::Sqlite {file,..}) if file==path));
        assert!(proof.failed);
        assert_eq!(proof.progress().hashed_upto, 0);
    }
}

#[cfg(test)]
mod archive_identity_tests {
    use super::*;
    #[test]
    fn cloned_index_generation_does_not_let_another_archive_supply_hash_proof() {
        let dir = tempfile::tempdir().unwrap();
        let mut first = Writer::open(&dir.path().join("first")).unwrap();
        let mut second = Writer::open(&dir.path().join("second")).unwrap();
        let first_value = first.begin_value(Codec::Bytes, 1).unwrap();
        first
            .append_value_chunk(&first_value.id, 0, b"a", 1)
            .unwrap();
        let second_value = second.begin_value(Codec::Bytes, 1).unwrap();
        second
            .append_value_chunk(&second_value.id, 0, b"b", 1)
            .unwrap();
        let path = second.root.join("index.db");
        crate::db(&path, || {
            second._index.execute(
                "UPDATE meta SET value=?1 WHERE key='generation'",
                [&first_value.id.index_generation],
            )
        })
        .unwrap();
        // Simulates a copied index incarnation without filesystem mutation.
        let reader = second.read_handle().unwrap();
        let mut proof = first.validate_value(&first_value.id).unwrap();
        assert!(matches!(
            proof.advance(&reader, 1),
            Err(Error::ValuePredecessorChanged)
        ));
        assert!(matches!(
            reader.value_head(&first_value.id),
            Err(Error::ValuePredecessorChanged)
        ));
        assert_eq!(proof.progress().hashed_upto, 0);
    }
}

#[cfg(test)]
mod transaction_helper_tests {
    use super::*;
    fn world() -> (tempfile::TempDir, Writer) {
        let dir = tempfile::tempdir().unwrap();
        let writer = Writer::open(&dir.path().join("archive")).unwrap();
        (dir, writer)
    }
    fn incoming(writer: &Writer, id: &ValueId) -> i64 {
        crate::db(&writer.root.join("index.db"), || {
            writer._index.query_row(
                "SELECT incoming FROM body_value WHERE pk=?1",
                [id.pk],
                |r| r.get(0),
            )
        })
        .unwrap()
    }
    #[test]
    fn begin_helper_rollback_exposes_no_committed_allocation() {
        let (_dir, mut writer) = world();
        let path = writer.root.join("index.db");
        {
            let tx = crate::db(&path, || writer._index.transaction()).unwrap();
            let pending = begin_in(&tx, &path, Codec::Bytes, 0, 1).unwrap();
            let count: i64 = crate::db(&path, || {
                tx.query_row(
                    "SELECT incoming FROM body_value WHERE pk=?1",
                    [pending.id.pk],
                    |r| r.get(0),
                )
            })
            .unwrap();
            assert_eq!(count, 1);
            // Head never escapes this uncommitted scope; SQLite may legitimately
            // reuse a rolled-back pk. Only the committed wrapper exposes IDs.
            drop(pending);
            crate::db(&path, || tx.rollback()).unwrap();
        }
        let count: i64 = crate::db(&path, || {
            writer
                ._index
                .query_row("SELECT count(*) FROM body_value", [], |r| r.get(0))
        })
        .unwrap();
        assert_eq!(count, 0);
        let committed = writer.begin_value(Codec::Bytes, 0).unwrap();
        assert_eq!(
            writer.value_head(&committed.id).unwrap().state,
            State::Building
        );
    }
    #[test]
    fn leased_dedup_requires_atomic_owner_transfer_and_rolls_back_callback_errors() {
        let (_dir, mut writer) = world();
        let path = writer.root.join("index.db");
        let first = writer.begin_value(Codec::Bytes, 0).unwrap();
        let proof = writer.validate_value(&first.id).unwrap();
        let first = writer.seal_value(&proof).unwrap().ready;
        crate::db(&path,||writer._index.execute_batch("CREATE TEMP TABLE producer_lease(id INTEGER PRIMARY KEY,value_pk INTEGER NOT NULL)")).unwrap();
        let pending = {
            let tx = crate::db(&path, || writer._index.transaction()).unwrap();
            let h = begin_in(&tx, &path, Codec::Bytes, 0, 1).unwrap();
            crate::db(&path, || {
                tx.execute(
                    "INSERT INTO producer_lease(id,value_pk) VALUES(1,?1)",
                    [h.id.pk],
                )
            })
            .unwrap();
            crate::db(&path, || tx.commit()).unwrap();
            h
        };
        let proof = writer.validate_value(&pending.id).unwrap();
        {
            let tx = crate::db(&path, || writer._index.transaction()).unwrap();
            assert!(matches!(
                seal_in(&tx, &path, &proof, |_, _| Ok(())),
                Err(Error::ValuePredecessorChanged)
            ));
            crate::db(&path, || tx.rollback()).unwrap();
        }
        let transfer = |tx: &Transaction<'_>, ready: &ReadyValue| -> Result<()> {
            let changed = crate::db(&path, || {
                tx.execute(
                    "UPDATE producer_lease SET value_pk=?2 WHERE id=1 AND value_pk=?1",
                    params![pending.id.pk, ready.id.pk],
                )
            })?;
            if changed != 1 {
                return Err(Error::ValuePredecessorChanged);
            }
            let changed = crate::db(&path, || {
                tx.execute(
                    "UPDATE body_value SET incoming=incoming-1 WHERE pk=?1 AND incoming=1",
                    [pending.id.pk],
                )
            })?;
            if changed != 1 {
                return Err(Error::ValuePredecessorChanged);
            }
            crate::db(&path, || {
                tx.execute(
                    "UPDATE body_value SET incoming=incoming+1 WHERE pk=?1 AND state='ready'",
                    [ready.id.pk],
                )
            })?;
            Ok(())
        };
        {
            let tx = crate::db(&path, || writer._index.transaction()).unwrap();
            let result = seal_in(&tx, &path, &proof, |tx, ready| {
                transfer(tx, ready)?;
                Err(Error::Cancelled)
            });
            assert!(matches!(result, Err(Error::Cancelled)));
            crate::db(&path, || tx.rollback()).unwrap();
        }
        assert_eq!(
            writer.value_head(&pending.id).unwrap().state,
            State::Building
        );
        assert_eq!(incoming(&writer, &pending.id), 1);
        assert_eq!(incoming(&writer, &first.id), 0);
        let result = {
            let tx = crate::db(&path, || writer._index.transaction()).unwrap();
            let result = seal_in(&tx, &path, &proof, transfer).unwrap();
            crate::db(&path, || tx.commit()).unwrap();
            result
        };
        assert_eq!(result.ready.id, first.id);
        assert_eq!(result.retired_duplicate, Some(pending.id.clone()));
        assert_eq!(
            writer.value_head(&pending.id).unwrap().state,
            State::Retiring
        );
        assert_eq!(incoming(&writer, &pending.id), 0);
        assert_eq!(incoming(&writer, &first.id), 1);
        let owner: i64 = crate::db(&path, || {
            writer
                ._index
                .query_row("SELECT value_pk FROM producer_lease WHERE id=1", [], |r| {
                    r.get(0)
                })
        })
        .unwrap();
        assert_eq!(owner, first.id.pk);
    }
    #[test]
    fn nonduplicate_owned_seal_preserves_lease_and_rollback_does_not_publish_ready() {
        let (_dir, mut writer) = world();
        let path = writer.root.join("index.db");
        let pending = {
            let tx = crate::db(&path, || writer._index.transaction()).unwrap();
            let h = begin_in(&tx, &path, Codec::Utf8, 0, 1).unwrap();
            crate::db(&path, || tx.commit()).unwrap();
            h
        };
        let proof = writer.validate_value(&pending.id).unwrap();
        {
            let tx = crate::db(&path, || writer._index.transaction()).unwrap();
            assert!(matches!(
                seal_in(&tx, &path, &proof, |_, _| Err(Error::Cancelled)),
                Err(Error::Cancelled)
            ));
            crate::db(&path, || tx.rollback()).unwrap();
        }
        assert_eq!(
            writer.value_head(&pending.id).unwrap().state,
            State::Building
        );
        assert_eq!(incoming(&writer, &pending.id), 1);
        let result = {
            let tx = crate::db(&path, || writer._index.transaction()).unwrap();
            let result = seal_in(&tx, &path, &proof, |_, _| Ok(())).unwrap();
            crate::db(&path, || tx.commit()).unwrap();
            result
        };
        assert_eq!(result.ready.id, pending.id);
        assert!(result.retired_duplicate.is_none());
        assert_eq!(incoming(&writer, &pending.id), 1);
    }
}
