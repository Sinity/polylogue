//! `embeddings.db`: purchased vectors keyed by embedding key and NFC text
//! hash, never destroyed by a reindex.
//! The texts come from `index.db`'s `text` table; a vector outlives every
//! index generation because the hash, not the row, names its text.
//!
//! Search is a scan of every vector of one key held in memory ([`Loaded`]):
//! a dot product against a unit query vector, the best chunk per text. The
//! daemon keeps the loaded vectors and reloads them after [`writes`]
//! changes.

use crate::archive::{Archive, open_read, open_write};
use crate::read::Reader;
use crate::{Result, StoreError};
use rusqlite::{Connection, params};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Commits that stored or removed vectors in this process. Only the daemon
/// writes `embeddings.db` (its embedding build and excision), so a copy
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
    index: std::path::PathBuf,
}

impl Vectors {
    pub(crate) fn for_reader(
        archive: &Archive,
        cache_kib: i64,
        writable: bool,
        limit: Option<&crate::read::limit::ReadLimit>,
    ) -> Result<Self> {
        let conn = if writable {
            open_write(&archive.embeddings_db(), crate::archive::cache::SETUP)?
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
        Ok(Self {
            conn,
            writable,
            index: archive.index_db(),
        })
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

    /// Text hashes that already have a vector or a permanent failure for
    /// `key`: the build skips them.
    pub fn done(&self, key: &str) -> Result<HashSet<Vec<u8>>> {
        let _read = self.read_transaction()?;
        let mut out = HashSet::new();
        for sql in [
            "SELECT DISTINCT text_hash FROM embedded WHERE model = ?1",
            "SELECT text_hash FROM embed_failure WHERE model = ?1",
        ] {
            let mut stmt = self.conn.prepare(sql)?;
            for h in stmt.query_map([key], |r| r.get::<_, Vec<u8>>(0))? {
                out.insert(h?);
            }
        }
        Ok(out)
    }

    /// Runs `write` under the archive's write lock, which an excision holds
    /// from its embeddings commit to its index commit, with the text hashes
    /// the current index still holds, and refuses while the index misses an
    /// excision: a build that read a text before it was excised cannot store
    /// a vector for it afterwards.
    fn while_texts_exist<T>(
        &self,
        hashes: impl Iterator<Item = Vec<u8>>,
        write: impl FnOnce(&HashSet<Vec<u8>>) -> Result<T>,
    ) -> Result<T> {
        Self::while_texts_exist_at(&self.index, hashes, write)
    }

    fn while_texts_exist_at<T>(
        index: &std::path::Path,
        hashes: impl Iterator<Item = Vec<u8>>,
        write: impl FnOnce(&HashSet<Vec<u8>>) -> Result<T>,
    ) -> Result<T> {
        let archive = open_read(&index.with_file_name("archive.db"))?;
        let index = open_read(index)?;
        let _lock = crate::write_lock::hold(&index)?;
        if crate::excise::index_behind(&archive, &index)? {
            return Err(StoreError::Invalid(
                "an excision has not reached the index yet; reopen the archive".into(),
            ));
        }
        let mut exists = index.prepare("SELECT EXISTS(SELECT 1 FROM text WHERE hash = ?1)")?;
        let mut present = HashSet::new();
        for hash in hashes {
            if exists.query_row([&hash], |r| r.get::<_, bool>(0))? {
                present.insert(hash);
            }
        }
        write(&present)
    }

    /// Stores vectors in one transaction, except for texts no longer in the
    /// index.
    pub fn put(&mut self, key: &str, vectors: &[Vector]) -> Result<()> {
        let conn = &mut self.conn;
        let index = self.index.clone();
        Self::while_texts_exist_at(
            &index,
            vectors.iter().map(|v| v.text_hash.clone()),
            |present| Self::insert(conn, key, vectors, present),
        )
    }

    fn insert(
        conn: &mut Connection,
        key: &str,
        vectors: &[Vector],
        present: &HashSet<Vec<u8>>,
    ) -> Result<()> {
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
            for v in vectors.iter().filter(|v| present.contains(&v.text_hash)) {
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

    /// A text the provider permanently rejected, with its error as given.
    pub fn fail(&self, key: &str, text_hash: &[u8], error: &str) -> Result<()> {
        self.while_texts_exist(std::iter::once(text_hash.to_vec()), |present| {
            if present.is_empty() {
                return Ok(());
            }
            self.record_failure(key, text_hash, error)
        })
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

    /// Every vector of `key` in memory, for `near:`. Each value is kept as
    /// one signed byte scaled by its vector's largest value: a quarter of
    /// the size, and similarities within about 1% of the stored ones,
    /// which keeps their order for ranking.
    pub fn load(&self, key: &str) -> Result<Loaded> {
        let _read = self.read_transaction()?;
        let mut loaded = Loaded {
            key: key.to_owned(),
            writes: writes(),
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
    /// [`writes`] when loaded: the copy is current while it is unchanged.
    pub writes: u64,
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
