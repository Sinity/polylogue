//! `embeddings.db`: purchased vectors keyed by embedding key and NFC text
//! hash, never destroyed by a reindex.
//! The texts come from `index.db`'s `text` table; a vector outlives every
//! index generation because the hash, not the row, names its text.
//!
//! Search is a scan of every vector of one key held in memory ([`Loaded`]):
//! a dot product against a unit query vector, the best chunk per text. The
//! daemon keeps loaded vectors by model revision read from the same SQLite
//! snapshot as their rows. [`writes`] is a coarse process-local invalidation
//! counter for query continuations, not the identity of a loaded snapshot.

use crate::archive::{Archive, open_read, open_write};
use crate::read::Reader;
use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Commits that stored vectors in this process. Only the daemon
/// writes `embeddings.db` (its embedding build), so a copy
/// loaded at the same count holds what the file holds.
static WRITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn writes() -> u64 {
    WRITES.load(std::sync::atomic::Ordering::Acquire)
}

pub(crate) fn wrote() {
    WRITES.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
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

    /// Stores every purchased vector, keyed by its content.
    pub fn put(&mut self, key: &str, vectors: &[Vector]) -> Result<()> {
        Self::insert(&mut self.conn, key, vectors)
    }

    fn insert(conn: &mut Connection, key: &str, vectors: &[Vector]) -> Result<()> {
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO embedded(text_hash, model, chunk, dims, vector, embedded_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(text_hash, model, chunk) DO UPDATE SET
                     dims = excluded.dims, vector = excluded.vector,
                     embedded_at = excluded.embedded_at",
            )?;
            let now = now_ms();
            for v in vectors {
                let bytes: Vec<u8> = v.values.iter().flat_map(|f| f.to_le_bytes()).collect();
                stmt.execute(params![
                    v.text_hash,
                    key,
                    v.chunk,
                    v.values.len() as i64,
                    bytes,
                    now
                ])?;
            }
        }
        tx.commit()?;
        wrote();
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

    /// Every vector of `key` in memory, for `near:`. Each value is kept as
    /// one signed byte scaled by its vector's largest value: a quarter of
    /// the size, and similarities within about 1% of the stored ones,
    /// which keeps their order for ranking.
    pub fn load(&self, key: &str) -> Result<Loaded> {
        let _read = self.read_transaction()?;
        let mut loaded = Loaded {
            key: key.to_owned(),
            revision: self.revision(key)?,
            dims: 0,
            texts: HashMap::new(),
            owner: Vec::new(),
            scale: Vec::new(),
            values: Vec::new(),
        };
        let mut stmt = self
            .conn
            .prepare("SELECT text_hash, dims, vector FROM embedded WHERE model = ?1")?;
        let mut rows = stmt.query([key])?;
        while let Some(row) = rows.next()? {
            let hash: Vec<u8> = row.get(0)?;
            let dims = usize::try_from(row.get::<_, i64>(1)?).unwrap_or(0);
            let bytes: Vec<u8> = row.get(2)?;
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
            let next = u32::try_from(loaded.texts.len()).unwrap_or(u32::MAX);
            let text = *loaded.texts.entry(hash.into_boxed_slice()).or_insert(next);
            let floats: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            let largest = floats.iter().fold(0f32, |m, x| m.max(x.abs()));
            let scale = if largest > 0.0 { largest / 127.0 } else { 1.0 };
            loaded.owner.push(text);
            loaded.scale.push(scale);
            loaded.values.extend(
                floats
                    .iter()
                    .map(|x| (x / scale).round().clamp(-127.0, 127.0) as i8),
            );
        }
        loaded.texts.shrink_to_fit();
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

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}

/// Every vector of one embedding key, in memory.
pub struct Loaded {
    pub key: String,
    /// Model revision in the embeddings snapshot that supplied these rows.
    pub revision: i64,
    pub dims: usize,
    /// Text hash to text number.
    texts: HashMap<Box<[u8]>, u32>,
    /// Per vector: its text, and the factor its bytes are scaled by.
    owner: Vec<u32>,
    scale: Vec<f32>,
    values: Vec<i8>,
}

impl Loaded {
    pub fn vectors(&self) -> usize {
        self.owner.len()
    }

    /// The similarity of every text to `query`, a unit vector of the same
    /// dimension: its best chunk's.
    pub fn similarity(self: &Arc<Self>, query: &[f32]) -> Result<Similarity> {
        if query.len() != self.dims && !self.owner.is_empty() {
            return Err(StoreError::Invalid(format!(
                "the query vector has {} values; vectors under {} have {}",
                query.len(),
                self.key,
                self.dims
            )));
        }
        let dims = self.dims.max(1);
        let n = self.owner.len();
        let mut scores = vec![0f32; n];
        let workers = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(8);
        let per = n.div_ceil(workers).max(1);
        let limit = crate::read::limit::ReadLimit::current();
        std::thread::scope(|scope| {
            for (part, out) in scores.chunks_mut(per).enumerate() {
                let limit = limit.clone();
                let first = part * per;
                scope.spawn(move || {
                    for (i, score) in out.iter_mut().enumerate() {
                        if i % 256 == 0
                            && limit.as_ref().is_some_and(|limit| limit.check().is_err())
                        {
                            break;
                        }
                        let v = first + i;
                        let row = &self.values[v * dims..(v + 1) * dims];
                        let dot: f32 = row.iter().zip(query).map(|(a, q)| f32::from(*a) * q).sum();
                        *score = dot * self.scale[v];
                    }
                });
            }
        });
        if let Some(limit) = &limit {
            limit.check()?;
        }
        let mut best = vec![f32::MIN; self.texts.len()];
        for (v, score) in scores.into_iter().enumerate() {
            if v % 256 == 0
                && let Some(limit) = &limit
            {
                limit.check()?;
            }
            let text = self.owner[v] as usize;
            if score > best[text] {
                best[text] = score;
            }
        }
        Ok(Similarity {
            loaded: Arc::clone(self),
            best,
        })
    }
}

/// Every text's similarity to one query, looked up by text hash while a
/// query runs (`pl_similarity`, [`Reader::with_similarity`]).
pub struct Similarity {
    loaded: Arc<Loaded>,
    best: Vec<f32>,
}

impl Similarity {
    pub fn of(&self, hash: &[u8]) -> Option<f32> {
        let text = *self.loaded.texts.get(hash)?;
        self.best.get(text as usize).copied()
    }
}

impl Reader<'_> {
    /// Runs `f` with `pl_similarity(text_hash)` answering from
    /// `similarity`, on this thread, where the reader's queries run.
    pub fn with_similarity<T>(&self, similarity: Arc<Similarity>, f: impl FnOnce() -> T) -> T {
        crate::sql_functions::with_similarity(similarity, f)
    }
}

#[cfg(test)]
mod tests;
