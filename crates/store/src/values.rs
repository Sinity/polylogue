//! Normalized values (message bodies and fields), stored as immutable chunks of
//! at most 64 KiB. A value is built by appending chunks, hashed by a
//! `Validation` outside any write transaction, then sealed as ready and
//! deduplicated by codec and BLAKE3 hash. Ready means complete and hashed; JSON
//! syntax is the producer's responsibility.
use crate::{Error, ReadHandle, Result, Writer};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use std::path::Path;

pub(crate) mod small;

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
            _ => Err(invalid("unknown value codec")),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Building,
    Ready,
    Retiring,
}
/// A value row, valid only in the index.db (path and generation) it came from.
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
    /// Bytes appended so far. Hash progress is never stored.
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
/// Incremental BLAKE3 hash and UTF-8 check of a fully appended value, kept in
/// memory only: after a restart, start a new one at zero. Any error makes it
/// unusable.
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
            return Err(invalid("this validation already failed; start a new one"));
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
            return Err(Error::StaleValue);
        }
        if self.offset == self.length {
            return Ok(self.progress());
        }
        let piece = piece(&reader.index, &path, &h, self.offset, allowance)?;
        if let Some((start, end)) = self.chunk {
            if (self.offset < end && (piece.start, piece.end) != (start, end))
                || (self.offset == end && piece.start != end)
            {
                return Err(invalid("value chunks overlap or changed during validation"));
            }
        } else if piece.start != 0 {
            return Err(invalid("value chunks do not start at offset 0"));
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
                return Err(invalid("value ends in an incomplete UTF-8 codepoint"));
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
                _ => return Err(invalid("invalid UTF-8 in value")),
            };
            let take = (width - self.len).min(bytes.len());
            self.tail[self.len..self.len + take].copy_from_slice(&bytes[..take]);
            self.len += take;
            bytes = &bytes[take..];
            if self.len < width {
                return Ok(());
            }
            std::str::from_utf8(&self.tail[..width])
                .map_err(|_| invalid("invalid UTF-8 in value"))?;
            self.len = 0;
        }
        match std::str::from_utf8(bytes) {
            Ok(_) => Ok(()),
            Err(e) if e.error_len().is_none() => {
                let tail = &bytes[e.valid_up_to()..];
                if tail.len() > 3 {
                    return Err(invalid("invalid UTF-8 in value"));
                }
                self.tail[..tail.len()].copy_from_slice(tail);
                self.len = tail.len();
                Ok(())
            }
            Err(_) => Err(invalid("invalid UTF-8 in value")),
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
        return Err(Error::StaleValue);
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
    .ok_or(Error::StaleValue)?;
    let codec = Codec::parse(&row.0)?;
    let length = u64::try_from(row.1).map_err(|_| Error::Range)?;
    let state = match row.3.as_str() {
        "building" => State::Building,
        "ready" => State::Ready,
        "retiring" => State::Retiring,
        _ => return Err(invalid("unknown value state")),
    };
    let hash = row
        .2
        .map(|v| {
            v.try_into()
                .map_err(|_| invalid("value hash is not 32 bytes"))
        })
        .transpose()?;
    if row.4 < 0
        || (state == State::Ready && hash.is_none())
        || (state == State::Retiring && row.4 != 0)
    {
        return Err(invalid("invalid value metadata"));
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
                return Err(invalid("invalid value chunk length"));
            }
            offset.checked_add(size).ok_or(Error::Range)?
        }
    };
    if appended > length {
        return Err(invalid("value chunks exceed the declared length"));
    }
    if state == State::Ready && appended != length {
        return Err(invalid("ready value is missing chunks"));
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
    let row=crate::db(path,||db.query_row("SELECT offset,length(bytes),substr(bytes,?2-offset+1,?3) FROM body_value_chunk WHERE value_pk=?1 AND offset<=?2 ORDER BY offset DESC LIMIT 1",params![h.id.pk,offset_i,allowance as i64],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?,r.get::<_,Vec<u8>>(2)?))).optional())?.ok_or_else(||invalid("value chunk missing"))?;
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
        return Err(invalid("value chunk is invalid or leaves a gap"));
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
    /// Appends one chunk at `offset`, which must equal `Head.appended`. A retry
    /// with an old offset fails with `StaleValue`.
    pub fn append_value_chunk(
        &mut self,
        id: &ValueId,
        offset: u64,
        bytes: &[u8],
        allowance: usize,
    ) -> Result<Head> {
        if bytes.is_empty() {
            return Err(invalid("empty value chunks are not allowed"));
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
            return Err(Error::StaleValue);
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
    /// Starts hashing a fully appended value at offset 0.
    pub fn validate_value(&self, id: &ValueId) -> Result<Validation> {
        validation(&self._index, &self.root.join("index.db"), id)
    }
    /// Marks a validated value ready. Reads no chunks.
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
/// Creates a building value with `incoming` producer leases inside the caller's
/// transaction. The caller must record the matching leases and commit before
/// exposing the returned `Head`, and roll back on any error.
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
/// Seals a value from a completed `Validation` without reading chunks. If a
/// ready value with the same codec and hash exists, that one is returned and the
/// new one becomes `retiring`; `transfer` must first move its leases (leaving
/// `incoming` at 0) without changing any bytes. The caller must roll back on
/// any error and commit before exposing the result.
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
        return Err(Error::StaleValue);
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
    .ok_or(Error::StaleValue)?;
    if row.0 != proof.codec.as_str()
        || row.1 != integer(proof.length)?
        || row.2 != "building"
        || row.3.is_some()
    {
        return Err(Error::StaleValue);
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
            return Err(invalid("values with the same hash have different lengths"));
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
            return Err(Error::StaleValue);
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
            return Err(Error::StaleValue);
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
        return Err(Error::StaleValue);
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
mod tests;
