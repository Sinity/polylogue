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
//! only appended rows when the revision shows an append-only change
//! ([`Vectors::append`]), and reloading the snapshot after a replacement,
//! when a request needs that revision. [`writes`] is a coarse process-local invalidation
//! counter for query continuations, not the identity of a loaded snapshot.
//! Query vectors the provider answered are stored too
//! ([`Vectors::query_vector`]), so a query text is paid for once and ranks
//! with model calls off.

use crate::ConnExt as _;
use crate::archive::{Archive, open_read, open_write};
use crate::read::Reader;
use crate::row::{Rows as _, Statement, TextKey, row};
use crate::{Result, StoreError};
use polylogue_model::LockExt;
use rusqlite::{Connection, params};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Commits that stored vectors in this process. Only the daemon
/// writes `embeddings.db` (its embedding build), so a copy
/// loaded at the same count holds what the file holds.
static WRITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Characters per chunk: a text longer than this is embedded in chunks of
/// this many characters of its NFC form, and its best chunk ranks it.
pub const CHUNK_CHARS: usize = 6000;

/// A vector bought for another store, as it was stored there.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredVector {
    pub text_hash: Vec<u8>,
    pub model: String,
    pub chunk: i64,
    pub dims: i64,
    /// `dims` little-endian f32 values, as stored.
    pub vector: Vec<u8>,
    pub embedded_at: i64,
}

/// What became of one vector given to [`Vectors::restore`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Restored {
    /// Its row was written.
    Stored,
    /// No write is needed: the same values are present, or the address
    /// already has a newer purchase.
    Present,
    /// Not written, for the reason given.
    Refused(String),
}

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
#[derive(
    Debug, Clone, Default, PartialEq, Eq, Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct VectorStatus {
    pub keys: Vec<KeyStatus>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KeyStatus {
    pub key: String,
    /// The values in each vector under the key; absent while the key holds
    /// no vector, only refused texts.
    pub dims: Option<u32>,
    pub texts: u64,
    pub vectors: u64,
    pub failures: u64,
}

row! {
    struct VectorBytes {
        vector: Vec<u8>,
    }
    statements {
        const QUERY = "SELECT vector FROM query_vector
             WHERE substr(key, 1, length(?1)) = ?1 AND text_hash = ?2
             ORDER BY key LIMIT 1";
        const AT_ADDRESS = "SELECT vector FROM embedded WHERE text_hash = ?1 AND model = ?2 AND chunk = ?3";
    }
}

row! {
    struct TextVector {
        text_hash: Vec<u8>,
        vector: Vec<u8>,
    }
    statements {
        const OF_TEXTS = "SELECT text_hash, vector FROM embedded WHERE model = ?1 AND text_hash IN (?2)";
    }
}

row! {
    struct TextChunk {
        text_hash: Vec<u8>,
        chunk: u32,
    }
    statements {
        const OF_TEXTS = "SELECT text_hash,chunk FROM embedded WHERE model=?1 AND text_hash IN (?2)";
    }
}

row! {
    struct RefusedText {
        text_hash: Vec<u8>,
    }
    statements {
        const OF_MODEL = "SELECT text_hash FROM embed_failure WHERE model=?1";
    }
}

row! {
    struct PurchasedVector {
        id: i64,
        dims: i64,
        vector: Vec<u8>,
        embedded_at: i64,
    }
    statements {
        const AT_ADDRESS = "SELECT id,dims,vector,embedded_at FROM embedded WHERE text_hash = ?1 AND model = ?2 AND chunk = ?3";
    }
}

row! {
    struct ModelVector {
        id: i64,
        text_hash: Vec<u8>,
        dims: i64,
        vector: Vec<u8>,
    }
    statements {
        const AFTER_ID = "SELECT id, text_hash, dims, vector FROM embedded WHERE model = ?1 AND id > ?2 ORDER BY id";
    }
}

row! {
    struct ModelCounts {
        model: String,
        dims: Option<i64>,
        texts: i64,
        vectors: i64,
        failures: i64,
    }
    statements {
        const ALL = "WITH stored AS (
                 SELECT model, max(dims) AS dims, count(DISTINCT text_hash) AS texts,
                        count(*) AS vectors
                 FROM embedded GROUP BY model),
               refused AS (SELECT model, count(*) AS failures FROM embed_failure GROUP BY model)
             SELECT key.model, stored.dims, coalesce(stored.texts, 0) AS texts,
                    coalesce(stored.vectors, 0) AS vectors, coalesce(refused.failures, 0) AS failures
             FROM (SELECT model FROM stored UNION SELECT model FROM refused) key
             LEFT JOIN stored ON stored.model = key.model
             LEFT JOIN refused ON refused.model = key.model
             ORDER BY key.model";
    }
}

row! {
    struct ProseRow {
        id: TextKey,
        hash: Vec<u8>,
        body: Option<Vec<u8>>,
        start: i64,
        length: i64,
    }
    statements {
        const BY_NORMALIZED_HASH = "SELECT t.id,t.hash,g.body,t.start,t.length FROM text t JOIN content_group g ON g.id=t.group_id WHERE t.normalized_hash=?1 AND t.id>?2 AND t.indexed & 1
             AND EXISTS(SELECT 1 FROM block b JOIN message m ON m.pk=b.message_pk WHERE b.text_id=t.id)
             ORDER BY t.id LIMIT ?3";
        const AFTER_ID = "SELECT t.id, t.hash, g.body, t.start, t.length FROM text t JOIN content_group g ON g.id=t.group_id
                     WHERE t.indexed & 1 AND t.id > ?1
                     AND EXISTS (SELECT 1 FROM block b JOIN message m ON m.pk=b.message_pk WHERE b.text_id=t.id)
                     ORDER BY t.id LIMIT ?2";
        const OF_SESSIONS = "SELECT t.id, t.hash, g.body, t.start, t.length FROM text t JOIN content_group g ON g.id=t.group_id
                         WHERE t.indexed & 1 AND t.id > ?1
                         AND t.id IN (SELECT b.text_id FROM block b JOIN message m ON m.pk=b.message_pk JOIN session s ON s.pk = b.session_pk
                                      WHERE s.id IN (SELECT value FROM json_each(?3)))
                         ORDER BY t.id LIMIT ?2";
    }
}

row! {
    struct SessionMember {
        id: String,
    }
    statements {
        const OF_ROOTS = "SELECT s.id FROM session s WHERE (?2) IN
             (SELECT r.pk FROM session r WHERE r.id IN (SELECT value FROM json_each(?1)))
             ORDER BY s.id";
    }
}

impl ProseRow {
    fn text(&self) -> Result<String> {
        let body = self
            .body
            .as_deref()
            .ok_or_else(|| StoreError::stored(2, "a text row has no group"))?;
        crate::text_rows::slice(body, self.start, self.length).map_err(|error| match error {
            StoreError::StoredValue { reason, .. } => StoreError::stored(2, reason),
            other => other,
        })
    }
}

#[cfg(test)]
pub(crate) fn declared_rows(
    index: &mut crate::row::Check<'_>,
    embeddings: &mut crate::row::Check<'_>,
) {
    index.row::<ProseRow>();
    index.row::<SessionMember>();
    embeddings.row::<VectorBytes>();
    embeddings.row::<TextVector>();
    embeddings.row::<TextChunk>();
    embeddings.row::<RefusedText>();
    embeddings.row::<PurchasedVector>();
    embeddings.row::<ModelVector>();
    embeddings.row::<ModelCounts>();
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
    /// for query-vector lookups outside a reader.
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
        let found = self
            .conn
            .maybe(&VectorBytes::QUERY, params![prefix, query_text_hash(text)])?;
        Ok(found.map(|row| floats(&row.vector)))
    }

    /// Stores the provider's vector for a query text (NFC) under its
    /// embedding key. A text already stored under the key keeps the vector
    /// stored first, the one every earlier ranking used.
    pub fn put_query_vector(
        &self,
        key: &str,
        text: &str,
        values: &[f32],
        clock: &polylogue_model::WallClock,
    ) -> Result<()> {
        let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
        let tx = rusqlite::Transaction::new_unchecked(
            &self.conn,
            rusqlite::TransactionBehavior::Immediate,
        )?;
        tx.execute(
            "INSERT INTO query_vector(key, text_hash, vector, embedded_at) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(key, text_hash) DO NOTHING",
            params![key, query_text_hash(text), bytes, clock.now_ms()],
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
        let select = TextVector::OF_TEXTS;
        let statement = Statement::<TextVector>::built(select.sql().replacen(
            "IN (?2)",
            &format!("IN ({placeholders})"),
            1,
        ));
        let _ = self
            .conn
            .each(&statement, rusqlite::params_from_iter(arguments), |row| {
                out.entry(row.text_hash)
                    .or_default()
                    .push(floats(&row.vector));
                Ok(ControlFlow::Continue(()))
            })?;
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
                            StoreError::internal("embedding chunk identity is out of range")
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
        let select = TextChunk::OF_TEXTS;
        let statement = Statement::<TextChunk>::built(select.sql().replacen(
            "IN (?2)",
            &format!("IN ({placeholders})"),
            1,
        ));
        let _ = self
            .conn
            .each(&statement, rusqlite::params_from_iter(arguments), |row| {
                if let Some(chunks) = missing.get_mut(&row.text_hash) {
                    chunks.remove(&row.chunk);
                }
                Ok(ControlFlow::Continue(()))
            })?;
        Ok(missing)
    }

    /// Whole texts permanently refused for this exact embedding key.
    pub fn rejected(&self, key: &str) -> Result<HashSet<Vec<u8>>> {
        let _read = self.read_transaction()?;
        let mut rejected = HashSet::new();
        let _ = self.conn.each(&RefusedText::OF_MODEL, [key], |row| {
            rejected.insert(row.text_hash);
            Ok(ControlFlow::Continue(()))
        })?;
        Ok(rejected)
    }

    /// Stores every purchased vector, keyed by its content. A vector is
    /// written once: the same values again are already stored, and different
    /// values for the same text, key and chunk are refused, since the stored
    /// one was paid for. Nothing of a refused call is stored.
    pub fn put(
        &mut self,
        key: &str,
        vectors: &[Vector],
        clock: &polylogue_model::WallClock,
    ) -> Result<()> {
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let mut stored = false;
        {
            let mut insert = tx.prepare_cached(
                "INSERT INTO embedded(text_hash, model, chunk, dims, vector, embedded_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            let now = clock.now_ms();
            for v in vectors {
                let bytes: Vec<u8> = v.values.iter().flat_map(|f| f.to_le_bytes()).collect();
                let known =
                    tx.maybe(&VectorBytes::AT_ADDRESS, params![v.text_hash, key, v.chunk])?;
                match known {
                    Some(known) if known.vector == bytes => {}
                    Some(_) => {
                        return Err(StoreError::internal(format!(
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
    /// same text hash, key, chunk, values and known purchase time. At one
    /// address the newest known purchase wins, including a newer time for
    /// identical values. Older imports leave that newest value in
    /// place. Conflicting dimensions are refused; equal-time conflicting
    /// values cannot be ordered and are also refused.
    /// Original source stores keep the earlier paid values.
    pub fn restore(&mut self, vectors: &[StoredVector]) -> Result<Vec<Restored>> {
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let mut out = Vec::with_capacity(vectors.len());
        let mut stored = false;
        {
            let mut insert = tx.prepare_cached(
                "INSERT INTO embedded(text_hash, model, chunk, dims, vector, embedded_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            let mut update =
                tx.prepare_cached("UPDATE embedded SET vector=?2,embedded_at=?3 WHERE id=?1")?;
            for v in vectors {
                if v.dims.checked_mul(4).filter(|_| v.dims > 0)
                    != i64::try_from(v.vector.len()).ok()
                {
                    out.push(Restored::Refused(format!(
                        "it holds {} bytes, not {} four-byte values",
                        v.vector.len(),
                        v.dims
                    )));
                    continue;
                }
                let known = tx.maybe(
                    &PurchasedVector::AT_ADDRESS,
                    params![v.text_hash, v.model, v.chunk],
                )?;
                out.push(match known {
                    Some(row) if row.dims != v.dims => Restored::Refused(
                        "this address has another vector dimension under the same embedding key".into(),
                    ),
                    Some(row) if row.embedded_at > v.embedded_at => Restored::Present,
                    Some(row) if row.embedded_at == v.embedded_at => {
                        if row.vector == v.vector {
                            Restored::Present
                        } else {
                            Restored::Refused("this address holds different vector values with the same purchase time".into())
                        }
                    }
                    Some(row) => {
                        update.execute(params![row.id,v.vector,v.embedded_at])?;
                        stored = true;
                        Restored::Stored
                    },
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

    /// A text the provider permanently rejected, with its error as given.
    pub fn fail(
        &self,
        key: &str,
        text_hash: &[u8],
        error: &str,
        clock: &polylogue_model::WallClock,
    ) -> Result<()> {
        self.record_failure(key, text_hash, error, clock)
    }

    fn record_failure(
        &self,
        key: &str,
        text_hash: &[u8],
        error: &str,
        clock: &polylogue_model::WallClock,
    ) -> Result<()> {
        let tx = rusqlite::Transaction::new_unchecked(
            &self.conn,
            rusqlite::TransactionBehavior::Immediate,
        )?;
        tx.execute(
            "INSERT INTO embed_failure(text_hash, model, error, failed_at) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(text_hash, model) DO UPDATE SET error = excluded.error,
                 failed_at = excluded.failed_at",
            params![text_hash, key, error, clock.now_ms()],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// An embedding key starting with `prefix` that has vectors, without
    /// counting them.
    pub fn key_with_prefix(&self, prefix: &str) -> Result<Option<String>> {
        let _read = self.read_transaction()?;
        self.conn.scalar_optional(
            "SELECT model FROM embedded WHERE substr(model, 1, length(?1)) = ?1 LIMIT 1",
            [prefix],
        )
    }

    /// Revision of a model in this handle's SQLite read snapshot.
    pub fn revision(&self, key: &str) -> Result<i64> {
        Ok(self
            .conn
            .scalar_optional(
                "SELECT revision FROM embedding_revision WHERE model = ?1",
                [key],
            )?
            .unwrap_or(0))
    }

    /// Every vector of `key` in memory, for `similar:`. Each value is kept as
    /// one signed byte scaled by its vector's largest value: a quarter of
    /// the size, and similarities within about 1% of the stored ones,
    /// which keeps their order for ranking.
    pub fn load(&self, key: &str) -> Result<Loaded> {
        self.append(&Loaded::empty(key))
    }

    /// Refreshes `base` from this handle's snapshot. Row-level revision
    /// triggers increment once per changed row. When the revision moved by
    /// exactly the number of appended rows, sharing the earlier runs is
    /// valid. Any other change, including replacement or paid-time update,
    /// reloads the snapshot instead of keeping stale values below `through`.
    pub fn append(&self, base: &Loaded) -> Result<Loaded> {
        let _read = self.read_transaction()?;
        let revision = self.revision(&base.key)?;
        let loaded = self.read_after(base, revision)?;
        if base.through > 0
            && revision.checked_sub(base.revision) != i64::try_from(loaded.rows_read).ok()
        {
            let mut full = self.read_after(&Loaded::empty(&base.key), revision)?;
            full.rows_read += loaded.rows_read;
            return Ok(full);
        }
        Ok(loaded)
    }

    fn read_after(&self, base: &Loaded, revision: i64) -> Result<Loaded> {
        let key = base.key.as_str();
        crate::fault::check("vectors.read", key)?;
        let mut loaded = Loaded {
            key: key.to_owned(),
            revision,
            dims: base.dims,
            through: base.through,
            rows_read: 0,
            runs: base.runs.clone(),
        };
        let mut run = Run::default();
        let _ = self
            .conn
            .each(&ModelVector::AFTER_ID, params![key, base.through], |row| {
                loaded.through = row.id;
                loaded.rows_read += 1;
                let dims = usize::try_from(row.dims).unwrap_or(0);
                let bytes = row.vector;
                if loaded.dims == 0 {
                    loaded.dims = dims;
                }
                if dims != loaded.dims {
                    return Err(StoreError::stored_data(format!(
                        "a stored vector under {key} has {dims} values, not {}",
                        loaded.dims
                    )));
                }
                if bytes.len() != dims * 4 {
                    return Err(StoreError::stored_data(format!(
                        "a stored vector under {key} has {} bytes, not {}",
                        bytes.len(),
                        dims * 4
                    )));
                }
                let floats = floats(&bytes);
                let largest = floats.iter().fold(0f32, |m, x| m.max(x.abs()));
                let scale = if largest > 0.0 { largest / 127.0 } else { 1.0 };
                let text = run.text(row.text_hash.into_boxed_slice());
                run.text.push(text);
                run.scale.push(scale);
                run.values.extend(
                    floats
                        .iter()
                        .map(|x| (x / scale).round().clamp(-127.0, 127.0) as i8),
                );
                Ok(ControlFlow::Continue(()))
            })?;
        if !run.text.is_empty() {
            loaded.push(run);
        }
        Ok(loaded)
    }

    /// Every key that holds a vector or a refused text. A key counts its
    /// refusals whether or not any of its texts has a vector: the refusal
    /// recorded before a provider's first vector tells the dimensions is
    /// under a key of its own.
    pub fn status(&self) -> Result<VectorStatus> {
        let _read = self.read_transaction()?;
        let keys = self
            .conn
            .all(&ModelCounts::ALL, [])?
            .into_iter()
            .map(|row| KeyStatus {
                key: row.model,
                dims: row.dims.map(|dims| dims as u32),
                texts: row.texts as u64,
                vectors: row.vectors as u64,
                failures: row.failures as u64,
            })
            .collect();
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
    /// Current prose spellings of one normalized request, bounded by the
    /// request index before body decoding. Raw hashes remain vector keys.
    pub fn prose_texts_by_normalized_hash(
        &self,
        hash: &[u8; 32],
        after: i64,
        limit: u64,
    ) -> Result<Vec<ProseText>> {
        let conn = self.conn()?;
        let mut out = Vec::new();
        let _ = conn.each(
            &ProseRow::BY_NORMALIZED_HASH,
            params![hash.as_slice(), TextKey(after), limit as i64],
            |row| {
                let text = row.text()?;
                let raw = row.hash;
                if raw.as_slice() != blake3::hash(text.as_bytes()).as_bytes()
                    || crate::encoding::normalized_text_hash(&text) != *hash
                {
                    return Err(StoreError::internal(
                        "stored text does not match its request identity",
                    ));
                }
                out.push(ProseText {
                    id: row.id.0,
                    hash: raw,
                    text,
                });
                Ok(ControlFlow::Continue(()))
            },
        )?;
        Ok(out)
    }

    /// Physical members of the selected logical sessions, using the same
    /// known-root predicate as session query folding. Unresolved ancestry
    /// belongs to the session itself, not to an assumed ancestor.
    pub fn logical_session_members(&self, roots: &[String]) -> Result<Vec<String>> {
        if roots.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn()?;
        let statement = Statement::<SessionMember>::built(SessionMember::OF_ROOTS.sql().replacen(
            "(?2)",
            &crate::query::fields::logical_root("s"),
            1,
        ));
        let roots = serde_json::to_string(roots)?;
        Ok(conn
            .all(&statement, [roots])?
            .into_iter()
            .map(|row| row.id)
            .collect())
    }

    /// Indexed prose texts with current message membership, with IDs above
    /// `after`, in ID order, at most `limit`. With
    /// `sessions`, only texts of those sessions' blocks.
    pub fn prose_texts(
        &self,
        after: i64,
        limit: u64,
        sessions: Option<&[String]>,
    ) -> Result<Vec<ProseText>> {
        let mut out = Vec::new();
        let conn = self.conn()?;
        let limit = limit as i64;
        let after = TextKey(after);
        let mut visit = |row: ProseRow| {
            let text = row.text()?;
            out.push(ProseText {
                id: row.id.0,
                hash: row.hash,
                text,
            });
            Ok(ControlFlow::Continue(()))
        };
        let _ = match sessions {
            None => conn.each(&ProseRow::AFTER_ID, params![after, limit], &mut visit)?,
            Some(ids) => conn.each(
                &ProseRow::OF_SESSIONS,
                params![after, limit, serde_json::to_string(ids)?],
                &mut visit,
            )?,
        };
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
    text: Vec<u32>,
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
        for text in &self.text {
            starts[*text as usize + 1] += 1;
        }
        for t in 1..starts.len() {
            starts[t] += starts[t - 1];
        }
        let mut next = starts.clone();
        let mut by_text = vec![0u32; self.text.len()];
        for (v, text) in self.text.iter().enumerate() {
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
            text: self.text.clone(),
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
        out.text
            .extend(later.text.iter().map(|text| renumbered[*text as usize]));
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
            && last.text.len() <= run.text.len() * 2
        {
            let earlier = self.runs.pop().expect("a last run");
            run = Arc::new(earlier.joined(&run));
        }
        self.runs.push(run);
    }

    pub fn vectors(&self) -> usize {
        self.runs.iter().map(|run| run.text.len()).sum()
    }

    /// The similarity of texts to `query`, a unit vector of the same
    /// dimension, scored as a query asks for them: nothing is scored here.
    pub fn similarity(self: &Arc<Self>, query: &[f32]) -> Result<Similarity> {
        if query.len() != self.dims && self.vectors() > 0 {
            return Err(StoreError::internal(format!(
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
    pub(crate) fn count_inputs(&self) -> (&str, i64, usize, blake3::Hash) {
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
        let mut scores = self.scores.locked();
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
            .locked()
            .iter()
            .filter_map(|(hash, score)| Some((hash.clone(), (*score)?)))
            .collect();
        best.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        best.truncate(RESCORED);
        let hashes: Vec<&[u8]> = best.iter().map(|(hash, _)| &**hash).collect();
        let stored = store.stored(&self.loaded.key, &hashes)?;
        let mut scores = self.scores.locked();
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
