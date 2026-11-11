//! `embeddings.db`: purchased vectors keyed by embedding key and NFC text
//! hash, never destroyed by a reindex.
//! The texts come from `index.db`'s `text` table; a vector outlives every
//! index generation because the hash, not the row, names its text.
//!
//! Search holds every vector of one key in memory as int8 copies
//! ([`Loaded`]) and scores only the texts a query selects, each by its best
//! chunk's dot product with a unit query vector, then re-scores the best
//! [`RESCORED`] of them from their stored f32 values ([`Similarity`]). The
//! daemon keeps loaded vectors by model revision read from the same SQLite
//! snapshot as their rows, and brings them to a newer revision by reading
//! only the rows stored since ([`Vectors::append`]), on a cue from
//! [`write_events`]. [`writes`] is a coarse process-local invalidation
//! counter for query continuations, not the identity of a loaded snapshot.
//! Query vectors the provider answered are stored too
//! ([`Vectors::query_vector`]), so a query text is paid for once and ranks
//! with model calls off.

use crate::archive::{Archive, open_read, open_write};
use crate::read::Reader;
use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Commits that stored vectors in this process. Only the daemon
/// writes `embeddings.db` (its embedding build), so a copy
/// loaded at the same count holds what the file holds.
static WRITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Characters per chunk: a text longer than this is embedded in chunks of
/// this many characters of its NFC form, and its best chunk ranks it.
pub const CHUNK_CHARS: usize = 6000;

pub fn writes() -> u64 {
    WRITES.load(std::sync::atomic::Ordering::Acquire)
}

pub(crate) fn wrote() {
    WRITES.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    WRITE_EVENTS.committed(crate::completion::Committed::CONTENT);
}

/// Cues after each commit that stored vectors in this process.
static WRITE_EVENTS: std::sync::LazyLock<crate::completion::CommitEvents> =
    std::sync::LazyLock::new(Default::default);

/// A coalesced cue after each commit that stores vectors; the daemon's
/// vectors worker extends its loaded set on it.
pub fn write_events() -> crate::completion::CommitSubscription {
    WRITE_EVENTS.subscribe()
}

/// One vector to store.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    pub text_hash: Vec<u8>,
    pub chunk: u32,
    /// Unit length; the scan's dot product is the cosine similarity.
    pub values: Vec<f32>,
}

/// Vectors per embedding key, for `status`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct VectorStatus {
    pub keys: Vec<KeyStatus>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KeyStatus {
    pub key: String,
    pub dims: u32,
    pub texts: u64,
    pub vectors: u64,
    pub failures: u64,
}

/// The embeddings database. One process writes it (the daemon's embedding
/// build); each reader owns its optional connection.
pub struct Vectors {
    conn: Connection,
    writable: bool,
}

impl Vectors {
    pub(crate) fn for_reader(
        archive: &Archive,
        cache_kib: i64,
        writable: bool,
        limit: Option<&crate::read::limit::ReadLimit>,
    ) -> Result<Self> {
        let conn = if writable {
            let conn = open_write(&archive.embeddings_db(), crate::archive::cache::SETUP)?;
            // Vectors are paid for; each committed batch must survive a power
            // loss, not only a process crash. One sync per batch costs little
            // next to the provider call that produced it.
            conn.pragma_update(None, "synchronous", "full")?;
            conn
        } else {
            open_read(&archive.embeddings_db())?
        };
        if !writable && let Some(limit) = limit {
            limit.install(&conn)?;
        }
        conn.pragma_update(None, "cache_size", -cache_kib)?;
        conn.pragma_update(None, "mmap_size", 0)?;
        if !writable {
            conn.execute_batch("BEGIN")?;
        }
        Ok(Self { conn, writable })
    }

    /// A read-only handle on its own snapshot of the embeddings database,
    /// for the daemon's vectors worker, which reads outside any request.
    pub fn snapshot(archive: &Archive, cache_kib: i64) -> Result<Self> {
        Self::for_reader(archive, cache_kib, false, None)
    }

    /// A writable handle outside any reader, for the daemon storing the
    /// query vectors the provider answered.
    pub fn writer(archive: &Archive, cache_kib: i64) -> Result<Self> {
        Self::for_reader(archive, cache_kib, true, None)
    }

    /// The stored vector of a query text (NFC) under an embedding key
    /// starting with `prefix`, if one was stored.
    pub fn query_vector(&self, prefix: &str, text: &str) -> Result<Option<Vec<f32>>> {
        let _read = self.read_transaction()?;
        let mut statement = self.conn.prepare_cached(
            "SELECT vector FROM query_vector
             WHERE substr(key, 1, length(?1)) = ?1 AND text_hash = ?2
             ORDER BY key LIMIT 1",
        )?;
        let bytes: Option<Vec<u8>> = statement
            .query_row(params![prefix, query_text_hash(text)], |row| row.get(0))
            .optional()?;
        Ok(bytes.map(|bytes| floats(&bytes)))
    }

    /// Stores the provider's vector for a query text (NFC) under its
    /// embedding key. A text already stored under the key keeps the vector
    /// stored first, the one every earlier ranking used.
    pub fn put_query_vector(&self, key: &str, text: &str, values: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
        let tx = rusqlite::Transaction::new_unchecked(
            &self.conn,
            rusqlite::TransactionBehavior::Immediate,
        )?;
        tx.execute(
            "INSERT INTO query_vector(key, text_hash, vector, embedded_at) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(key, text_hash) DO NOTHING",
            params![key, query_text_hash(text), bytes, now_ms()],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// The stored f32 vectors of `hashes` under `key`, every chunk of each.
    fn stored(&self, key: &str, hashes: &[&[u8]]) -> Result<HashMap<Vec<u8>, Vec<Vec<f32>>>> {
        let mut out: HashMap<Vec<u8>, Vec<Vec<f32>>> = HashMap::new();
        if hashes.is_empty() {
            return Ok(out);
        }
        let _read = self.read_transaction()?;
        let placeholders = (0..hashes.len())
            .map(|index| format!("?{}", index + 2))
            .collect::<Vec<_>>()
            .join(",");
        let mut arguments = vec![rusqlite::types::Value::Text(key.into())];
        arguments.extend(
            hashes
                .iter()
                .map(|hash| rusqlite::types::Value::Blob(hash.to_vec())),
        );
        let mut statement = self.conn.prepare_cached(&format!(
            "SELECT text_hash, vector FROM embedded WHERE model = ?1 AND text_hash IN ({placeholders})"
        ))?;
        let mut rows = statement.query(rusqlite::params_from_iter(arguments))?;
        while let Some(row) = rows.next()? {
            let bytes: Vec<u8> = row.get(1)?;
            out.entry(row.get(0)?).or_default().push(floats(&bytes));
        }
        Ok(out)
    }

    pub(crate) fn writable(&self) -> bool {
        self.writable
    }

    // Writable handles end each read transaction before their next put or fail.
    // Read-only handles keep the read transaction their reader started.
    fn read_transaction(&self) -> Result<Option<rusqlite::Transaction<'_>>> {
        let read = if self.writable {
            Some(rusqlite::Transaction::new_unchecked(
                &self.conn,
                rusqlite::TransactionBehavior::Deferred,
            )?)
        } else {
            None
        };
        Ok(read)
    }

    /// Missing chunk identities for one exact key and a page of whole texts.
    pub fn missing_chunks(
        &self,
        key: &str,
        texts: &[(Vec<u8>, usize)],
    ) -> Result<HashMap<Vec<u8>, HashSet<u32>>> {
        let _read = self.read_transaction()?;
        let mut missing = texts
            .iter()
            .map(|(hash, chunks)| {
                let chunks = (0..*chunks)
                    .map(|chunk| {
                        u32::try_from(chunk).map_err(|_| {
                            StoreError::Invalid("embedding chunk identity is out of range".into())
                        })
                    })
                    .collect::<Result<HashSet<_>>>()?;
                Ok((hash.clone(), chunks))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        if texts.is_empty() {
            return Ok(missing);
        }
        let placeholders = (0..texts.len())
            .map(|index| format!("?{}", index + 2))
            .collect::<Vec<_>>()
            .join(",");
        let mut arguments = vec![rusqlite::types::Value::Text(key.into())];
        arguments.extend(
            texts
                .iter()
                .map(|(hash, _)| rusqlite::types::Value::Blob(hash.clone())),
        );
        let mut statement = self.conn.prepare_cached(&format!(
            "SELECT text_hash,chunk FROM embedded WHERE model=?1 AND text_hash IN ({placeholders})"
        ))?;
        for row in statement.query_map(rusqlite::params_from_iter(arguments), |row| {
            Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, u32>(1)?))
        })? {
            let (hash, chunk) = row?;
            if let Some(chunks) = missing.get_mut(&hash) {
                chunks.remove(&chunk);
            }
        }
        Ok(missing)
    }

    /// Whole texts permanently refused for this exact embedding key.
    pub fn rejected(&self, key: &str) -> Result<HashSet<Vec<u8>>> {
        let _read = self.read_transaction()?;
        let mut statement = self
            .conn
            .prepare_cached("SELECT text_hash FROM embed_failure WHERE model=?1")?;
        let rejected = statement
            .query_map([key], |row| row.get::<_, Vec<u8>>(0))?
            .collect::<rusqlite::Result<_>>()?;
        Ok(rejected)
    }

    /// Stores every purchased vector, keyed by its content. A vector is
    /// written once: the same values again are already stored, and different
    /// values for the same text, key and chunk are refused, since the stored
    /// one was paid for. Nothing of a refused call is stored.
    pub fn put(&mut self, key: &str, vectors: &[Vector]) -> Result<()> {
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let mut stored = false;
        {
            let mut find = tx.prepare_cached(
                "SELECT vector FROM embedded WHERE text_hash = ?1 AND model = ?2 AND chunk = ?3",
            )?;
            let mut insert = tx.prepare_cached(
                "INSERT INTO embedded(text_hash, model, chunk, dims, vector, embedded_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            let now = now_ms();
            for v in vectors {
                let bytes: Vec<u8> = v.values.iter().flat_map(|f| f.to_le_bytes()).collect();
                let known: Option<Vec<u8>> = find
                    .query_row(params![v.text_hash, key, v.chunk], |r| r.get(0))
                    .optional()?;
                match known {
                    Some(known) if known == bytes => {}
                    Some(_) => {
                        return Err(StoreError::Conflict(format!(
                            "the embeddings store holds a different vector for chunk {} of this text under {key}",
                            v.chunk
                        )));
                    }
                    None => {
                        insert.execute(params![
                            v.text_hash,
                            key,
                            v.chunk,
                            v.values.len() as i64,
                            bytes,
                            now
                        ])?;
                        stored = true;
                    }
                }
            }
        }
        tx.commit()?;
        if stored {
            wrote();
        }
        Ok(())
    }

    /// Stores vectors bought for another archive exactly as they were: the
    /// same text hash, key, chunk, values and time. A vector this archive
    /// already holds identically is present; a different vector for the
    /// same text, key and chunk is refused rather than replaced, since both
    /// were paid for.
    pub fn restore(
        &mut self,
        vectors: &[crate::older_archive::StoredVector],
    ) -> Result<Vec<crate::writer::Restored>> {
        use crate::writer::Restored;
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let mut out = Vec::with_capacity(vectors.len());
        let mut stored = false;
        {
            let mut find = tx.prepare_cached(
                "SELECT dims, vector FROM embedded WHERE text_hash = ?1 AND model = ?2 AND chunk = ?3",
            )?;
            let mut insert = tx.prepare_cached(
                "INSERT INTO embedded(text_hash, model, chunk, dims, vector, embedded_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            for v in vectors {
                if v.dims < 1 || v.vector.len() as i64 != v.dims * 4 {
                    out.push(Restored::Refused(format!(
                        "it holds {} bytes, not {} four-byte values",
                        v.vector.len(),
                        v.dims
                    )));
                    continue;
                }
                let known: Option<(i64, Vec<u8>)> = find
                    .query_row(params![v.text_hash, v.model, v.chunk], |r| {
                        Ok((r.get(0)?, r.get(1)?))
                    })
                    .optional()?;
                out.push(match known {
                    Some((dims, vector)) if dims == v.dims && vector == v.vector => {
                        Restored::Present
                    }
                    Some(_) => Restored::Refused(
                        "this archive holds a different vector for the same text, key and chunk"
                            .into(),
                    ),
                    None => {
                        insert.execute(params![
                            v.text_hash,
                            v.model,
                            v.chunk,
                            v.dims,
                            v.vector,
                            v.embedded_at
                        ])?;
                        stored = true;
                        Restored::Stored
                    }
                });
            }
        }
        tx.commit()?;
        if stored {
            wrote();
        }
        Ok(out)
    }

    /// Stores permanent rejections recorded by another archive, with their
    /// original error and time. One this archive holds with another error
    /// is refused rather than replaced.
    pub fn restore_failures(
        &mut self,
        failures: &[crate::older_archive::VectorFailure],
    ) -> Result<Vec<crate::writer::Restored>> {
        use crate::writer::Restored;
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let mut out = Vec::with_capacity(failures.len());
        {
            let mut find = tx.prepare_cached(
                "SELECT error FROM embed_failure WHERE text_hash = ?1 AND model = ?2",
            )?;
            let mut insert = tx.prepare_cached(
                "INSERT INTO embed_failure(text_hash, model, error, failed_at) VALUES (?1, ?2, ?3, ?4)",
            )?;
            for f in failures {
                let known: Option<String> = find
                    .query_row(params![f.text_hash, f.model], |r| r.get(0))
                    .optional()?;
                out.push(match known {
                    Some(error) if error == f.error => Restored::Present,
                    Some(_) => Restored::Refused(
                        "this archive records another error for the same text and key".into(),
                    ),
                    None => {
                        insert.execute(params![f.text_hash, f.model, f.error, f.failed_at])?;
                        Restored::Stored
                    }
                });
            }
        }
        tx.commit()?;
        Ok(out)
    }

    /// A text the provider permanently rejected, with its error as given.
    pub fn fail(&self, key: &str, text_hash: &[u8], error: &str) -> Result<()> {
        self.record_failure(key, text_hash, error)
    }

    fn record_failure(&self, key: &str, text_hash: &[u8], error: &str) -> Result<()> {
        let tx = rusqlite::Transaction::new_unchecked(
            &self.conn,
            rusqlite::TransactionBehavior::Immediate,
        )?;
        tx.execute(
            "INSERT INTO embed_failure(text_hash, model, error, failed_at) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(text_hash, model) DO UPDATE SET error = excluded.error,
                 failed_at = excluded.failed_at",
            params![text_hash, key, error, now_ms()],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// An embedding key starting with `prefix` that has vectors, without
    /// counting them.
    pub fn key_with_prefix(&self, prefix: &str) -> Result<Option<String>> {
        use rusqlite::OptionalExtension;
        let _read = self.read_transaction()?;
        Ok(self
            .conn
            .query_row(
                "SELECT model FROM embedded WHERE substr(model, 1, length(?1)) = ?1 LIMIT 1",
                [prefix],
                |r| r.get(0),
            )
            .optional()?)
    }

    /// Revision of a model in this handle's SQLite read snapshot.
    pub fn revision(&self, key: &str) -> Result<i64> {
        Ok(self
            .conn
            .query_row(
                "SELECT revision FROM embedding_revision WHERE model = ?1",
                [key],
                |row| row.get(0),
            )
            .optional()?
            .unwrap_or(0))
    }

    /// Every vector of `key` in memory, for `similar:`. Each value is kept as
    /// one signed byte scaled by its vector's largest value: a quarter of
    /// the size, and similarities within about 1% of the stored ones,
    /// which keeps their order for ranking.
    pub fn load(&self, key: &str) -> Result<Loaded> {
        self.append(&Loaded::empty(key))
    }

    /// `base` with the vectors of its key stored since it was read. Stored
    /// vectors are never changed or removed, so the rows after the last one
    /// `base` holds are everything it lacks; `base`'s runs are shared, not
    /// copied.
    pub fn append(&self, base: &Loaded) -> Result<Loaded> {
        let _read = self.read_transaction()?;
        let key = base.key.as_str();
        let mut loaded = Loaded {
            key: key.to_owned(),
            revision: self.revision(key)?,
            dims: base.dims,
            through: base.through,
            rows_read: 0,
            runs: base.runs.clone(),
        };
        let mut run = Run::default();
        let mut stmt = self.conn.prepare_cached(
            "SELECT id, text_hash, dims, vector FROM embedded WHERE model = ?1 AND id > ?2 ORDER BY id",
        )?;
        let mut rows = stmt.query(params![key, base.through])?;
        while let Some(row) = rows.next()? {
            loaded.through = row.get(0)?;
            loaded.rows_read += 1;
            let hash: Vec<u8> = row.get(1)?;
            let dims = usize::try_from(row.get::<_, i64>(2)?).unwrap_or(0);
            let bytes: Vec<u8> = row.get(3)?;
            if loaded.dims == 0 {
                loaded.dims = dims;
            }
            if dims != loaded.dims || bytes.len() != dims * 4 {
                return Err(StoreError::Invalid(format!(
                    "a stored vector under {key} has {} bytes, not {}",
                    bytes.len(),
                    loaded.dims * 4
                )));
            }
            let floats = floats(&bytes);
            let largest = floats.iter().fold(0f32, |m, x| m.max(x.abs()));
            let scale = if largest > 0.0 { largest / 127.0 } else { 1.0 };
            let text = run.text(hash.into_boxed_slice());
            run.owner.push(text);
            run.scale.push(scale);
            run.values.extend(
                floats
                    .iter()
                    .map(|x| (x / scale).round().clamp(-127.0, 127.0) as i8),
            );
        }
        if !run.owner.is_empty() {
            loaded.push(run);
        }
        Ok(loaded)
    }

    pub fn status(&self) -> Result<VectorStatus> {
        let _read = self.read_transaction()?;
        let mut stmt = self.conn.prepare(
            "SELECT model, max(dims), count(DISTINCT text_hash), count(*),
                 (SELECT count(*) FROM embed_failure f WHERE f.model = e.model)
             FROM embedded e GROUP BY model ORDER BY model",
        )?;
        let keys = stmt
            .query_map([], |r| {
                Ok(KeyStatus {
                    key: r.get(0)?,
                    dims: r.get::<_, i64>(1)? as u32,
                    texts: r.get::<_, i64>(2)? as u64,
                    vectors: r.get::<_, i64>(3)? as u64,
                    failures: r.get::<_, i64>(4)? as u64,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(VectorStatus { keys })
    }
}

/// One prose text of the index, for embedding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProseText {
    pub id: i64,
    pub hash: Vec<u8>,
    pub text: String,
}

impl Reader<'_> {
    /// Prose texts (text and thinking blocks of human and model messages)
    /// with ids above `after`, in id order, at most `limit`. With
    /// `sessions`, only texts of those sessions' blocks.
    pub fn prose_texts(
        &self,
        after: i64,
        limit: u64,
        sessions: Option<&[String]>,
    ) -> Result<Vec<ProseText>> {
        let mut out = Vec::new();
        let rows: Vec<(i64, Vec<u8>, Vec<u8>)> = match sessions {
            None => self
                .conn()?
                .prepare_cached(
                    "SELECT t.id, t.hash, t.body FROM text t WHERE t.indexed & 1 AND t.id > ?1
                     AND EXISTS (SELECT 1 FROM block b JOIN message m ON m.pk=b.message_pk WHERE b.text_id=t.id)
                     ORDER BY t.id LIMIT ?2",
                )?
                .query_map(params![after, limit as i64], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?))
                })?
                .collect::<rusqlite::Result<_>>()?,
            Some(ids) => {
                let list = serde_json::to_string(ids)?;
                self.conn()?
                    .prepare_cached(
                        "SELECT t.id, t.hash, t.body FROM text t WHERE t.indexed & 1 AND t.id > ?1
                         AND t.id IN (SELECT b.text_id FROM block b JOIN message m ON m.pk=b.message_pk JOIN session s ON s.pk = b.session_pk
                                      WHERE s.id IN (SELECT value FROM json_each(?3)))
                         ORDER BY t.id LIMIT ?2",
                    )?
                    .query_map(params![after, limit as i64, list], |r| {
                        Ok((r.get(0)?, r.get(1)?, r.get(2)?))
                    })?
                    .collect::<rusqlite::Result<_>>()?
            }
        };
        for (id, hash, body) in rows {
            out.push(ProseText {
                id,
                hash,
                text: crate::encoding::unpack_text(&body)?,
            });
        }
        Ok(out)
    }
}

/// Little-endian f32 values.
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// A query text's key in `query_vector`: the BLAKE3 hash of its NFC bytes.
fn query_text_hash(text: &str) -> Vec<u8> {
    blake3::hash(text.as_bytes()).as_bytes().to_vec()
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}

/// Every vector of one embedding key, in memory, as runs in stored order.
/// An append adds a run and merges it with the runs before it while they
/// are no more than twice its size, so the set stays a few runs and each
/// vector is copied a logarithmic number of times over the set's life.
#[derive(Clone)]
pub struct Loaded {
    pub key: String,
    /// Model revision in the embeddings snapshot that supplied these rows.
    pub revision: i64,
    pub dims: usize,
    /// The last stored row held.
    through: i64,
    /// Rows the load or extension that made this set read from the store.
    pub rows_read: usize,
    runs: Vec<Arc<Run>>,
}

/// Consecutive stored vectors with their texts.
#[derive(Default)]
struct Run {
    /// Text hash to text number in this run, and back.
    texts: HashMap<Box<[u8]>, u32>,
    hashes: Vec<Box<[u8]>>,
    /// Per vector: its text, and the factor its bytes are scaled by.
    owner: Vec<u32>,
    scale: Vec<f32>,
    values: Vec<i8>,
    /// Vector numbers grouped by text: text `t`'s are
    /// `by_text[starts[t]..starts[t + 1]]`. Built once a run is complete.
    starts: Vec<u32>,
    by_text: Vec<u32>,
}

impl Run {
    /// Groups the vectors by text, for scoring one text at a time.
    fn indexed(mut self) -> Self {
        let mut starts = vec![0u32; self.hashes.len() + 1];
        for text in &self.owner {
            starts[*text as usize + 1] += 1;
        }
        for t in 1..starts.len() {
            starts[t] += starts[t - 1];
        }
        let mut next = starts.clone();
        let mut by_text = vec![0u32; self.owner.len()];
        for (v, text) in self.owner.iter().enumerate() {
            let at = &mut next[*text as usize];
            by_text[*at as usize] = u32::try_from(v).unwrap_or(u32::MAX);
            *at += 1;
        }
        self.starts = starts;
        self.by_text = by_text;
        self
    }

    fn vectors_of(&self, text: u32) -> &[u32] {
        let text = text as usize;
        &self.by_text[self.starts[text] as usize..self.starts[text + 1] as usize]
    }

    fn text(&mut self, hash: Box<[u8]>) -> u32 {
        if let Some(text) = self.texts.get(&hash) {
            return *text;
        }
        let next = u32::try_from(self.hashes.len()).unwrap_or(u32::MAX);
        self.hashes.push(hash.clone());
        self.texts.insert(hash, next);
        next
    }

    /// `self` followed by `later`.
    fn joined(&self, later: &Run) -> Run {
        let mut out = Run {
            texts: self.texts.clone(),
            hashes: self.hashes.clone(),
            owner: self.owner.clone(),
            scale: self.scale.clone(),
            values: self.values.clone(),
            starts: Vec::new(),
            by_text: Vec::new(),
        };
        let renumbered: Vec<u32> = later
            .hashes
            .iter()
            .map(|hash| out.text(hash.clone()))
            .collect();
        out.owner
            .extend(later.owner.iter().map(|text| renumbered[*text as usize]));
        out.scale.extend_from_slice(&later.scale);
        out.values.extend_from_slice(&later.values);
        out.texts.shrink_to_fit();
        out.indexed()
    }
}

impl Loaded {
    fn empty(key: &str) -> Self {
        Self {
            key: key.to_owned(),
            revision: 0,
            dims: 0,
            through: 0,
            rows_read: 0,
            runs: Vec::new(),
        }
    }

    fn push(&mut self, run: Run) {
        let mut run = Arc::new(run.indexed());
        while let Some(last) = self.runs.last()
            && last.owner.len() <= run.owner.len() * 2
        {
            let earlier = self.runs.pop().expect("a last run");
            run = Arc::new(earlier.joined(&run));
        }
        self.runs.push(run);
    }

    pub fn vectors(&self) -> usize {
        self.runs.iter().map(|run| run.owner.len()).sum()
    }

    /// The similarity of texts to `query`, a unit vector of the same
    /// dimension, scored as a query asks for them: nothing is scored here.
    pub fn similarity(self: &Arc<Self>, query: &[f32]) -> Result<Similarity> {
        if query.len() != self.dims && self.vectors() > 0 {
            return Err(StoreError::Invalid(format!(
                "the query vector has {} values; vectors under {} have {}",
                query.len(),
                self.key,
                self.dims
            )));
        }
        Ok(Similarity {
            loaded: Arc::clone(self),
            query: query.to_vec(),
            scores: Mutex::default(),
            vectors_scored: AtomicUsize::new(0),
            rescored: AtomicBool::new(false),
        })
    }

    /// A text's best chunk similarity to `query` from the int8 copies, and
    /// the vectors scored for it.
    fn score(&self, hash: &[u8], query: &[f32]) -> (Option<f32>, usize) {
        let dims = self.dims.max(1);
        let mut best: Option<f32> = None;
        let mut scored = 0;
        for run in &self.runs {
            let Some(&text) = run.texts.get(hash) else {
                continue;
            };
            for &v in run.vectors_of(text) {
                let v = v as usize;
                let row = &run.values[v * dims..(v + 1) * dims];
                let dot: f32 = row.iter().zip(query).map(|(a, q)| f32::from(*a) * q).sum();
                let score = dot * run.scale[v];
                best = Some(best.map_or(score, |best| best.max(score)));
                scored += 1;
            }
        }
        (best, scored)
    }
}

/// Texts re-scored from their stored f32 vectors after the int8 pass: the
/// best this many of the selected texts.
pub const RESCORED: usize = 200;

/// Texts' similarities to one query, looked up by text hash while a query
/// runs (`pl_similarity`, [`Reader::with_similarity`]). A text is scored
/// from the int8 copies when a query first asks for it, so the work is the
/// selected rows' texts, not every loaded vector; [`Reader::rank_selection`]
/// then re-scores the best [`RESCORED`] of them from their stored f32
/// vectors, which fixes the order where quantization tied or misordered.
pub struct Similarity {
    loaded: Arc<Loaded>,
    query: Vec<f32>,
    /// Each text asked for and its best chunk's score; `None`: no vector.
    scores: Mutex<HashMap<Box<[u8]>, Option<f32>>>,
    vectors_scored: AtomicUsize,
    rescored: AtomicBool,
}

impl Similarity {
    /// Exact model snapshot and seed supplying this query's scores.
    pub(crate) fn count_basis(&self) -> (&str, i64, usize, blake3::Hash) {
        let mut query = blake3::Hasher::new();
        for component in &self.query {
            query.update(&component.to_bits().to_le_bytes());
        }
        (
            &self.loaded.key,
            self.loaded.revision,
            self.loaded.dims,
            query.finalize(),
        )
    }

    /// No text's similarity is known: a `similar:` ranking whose query
    /// vector could not be had keeps every row, unranked.
    pub fn unknown() -> Self {
        Self {
            loaded: Arc::new(Loaded::empty("")),
            query: Vec::new(),
            scores: Mutex::default(),
            vectors_scored: AtomicUsize::new(0),
            rescored: AtomicBool::new(false),
        }
    }

    /// The best similarity of a text's chunks, which may sit in several runs.
    pub fn of(&self, hash: &[u8]) -> Option<f32> {
        let mut scores = self
            .scores
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(score) = scores.get(hash) {
            return *score;
        }
        let (score, scored) = self.loaded.score(hash, &self.query);
        self.vectors_scored.fetch_add(scored, Ordering::Relaxed);
        scores.insert(hash.into(), score);
        score
    }

    /// Int8 vectors scored so far.
    pub fn vectors_scored(&self) -> usize {
        self.vectors_scored.load(Ordering::Relaxed)
    }

    /// Re-scores the best [`RESCORED`] texts scored so far from their
    /// stored f32 vectors in `store`, once.
    fn rescore(&self, store: &Vectors) -> Result<()> {
        if self.rescored.load(Ordering::Acquire) || self.loaded.vectors() == 0 {
            return Ok(());
        }
        let mut best: Vec<(Box<[u8]>, f32)> = self
            .scores
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .filter_map(|(hash, score)| Some((hash.clone(), (*score)?)))
            .collect();
        best.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        best.truncate(RESCORED);
        let hashes: Vec<&[u8]> = best.iter().map(|(hash, _)| &**hash).collect();
        let stored = store.stored(&self.loaded.key, &hashes)?;
        let mut scores = self
            .scores
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (hash, chunks) in stored {
            let score = chunks
                .iter()
                .map(|values| values.iter().zip(&self.query).map(|(a, q)| a * q).sum())
                .reduce(f32::max);
            if let Some(score) = score {
                scores.insert(hash.into_boxed_slice(), Some(score));
            }
        }
        self.rescored.store(true, Ordering::Release);
        Ok(())
    }
}

impl Reader<'_> {
    /// Scores the texts of the rows `members` selects (a statement whose
    /// `_near` column asks `pl_similarity` for each selected row's texts)
    /// from the int8 copies, then re-scores the best [`RESCORED`] from their
    /// stored f32 vectors. Rows the query reads afterwards rank by these
    /// scores.
    pub(crate) fn rank_selection(
        &self,
        similarity: &Arc<Similarity>,
        members: &str,
        params: &[crate::SqlValue],
    ) -> Result<()> {
        if similarity.loaded.vectors() == 0 || similarity.rescored.load(Ordering::Acquire) {
            return Ok(());
        }
        crate::sql_functions::with_similarity(Arc::clone(similarity), || {
            self.query_counts(
                &format!("SELECT count(*), count(_near) FROM ({members})"),
                params,
            )
        })?;
        self.check()?;
        similarity.rescore(&*self.vectors()?)
    }

    /// Runs `f` with `pl_similarity(text_hash)` answering from
    /// `similarity`, on this thread, where the reader's queries run.
    pub fn with_similarity<T>(&self, similarity: Arc<Similarity>, f: impl FnOnce() -> T) -> T {
        crate::sql_functions::with_similarity(similarity, f)
    }
}

#[cfg(test)]
mod tests;
