//! Read connections. The daemon keeps a pool; every read in one request
//! sees one consistent read transaction on `index.db` with `archive.db`
//! attached.

use crate::archive::{Archive, open_read};
use crate::blobs::FileFacts;
use crate::encoding::{restore_text, unpack_block, unpack_text};
use crate::{Result, StoreError};
use polylogue_model::{
    Block, Message, MessageId, PromptKind, Role, Session, SessionId, Timestamp, Usage,
};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::os::unix::fs::MetadataExt;
use std::sync::{Condvar, Mutex};
use std::thread::ThreadId;

mod behind_index;
mod captured;
mod parents;
mod tool_outcomes;
pub use behind_index::{BehindChain, ImportedAs};
pub use captured::CapturedSession;
mod rows;

pub use parents::{MessageSelection, OutlineEntry, ParentIssue, Parents, VisitStart};

/// Read-only connections to one archive, reused across requests.
pub struct ReadPool {
    archive: Archive,
    state: Mutex<PoolState>,
    available: Condvar,
    max_readers: usize,
    cache_kib: i64,
    #[cfg(test)]
    changed: Condvar,
}

#[derive(Default)]
struct PoolState {
    idle: Vec<(Connection, IndexIdentity)>,
    closed: bool,
    /// Includes idle, checked-out and still-opening connections.
    live: usize,
    owners: HashMap<ThreadId, usize>,
    #[cfg(test)]
    waiters: usize,
}

/// One consistent read: a read transaction on `index.db` with `archive.db`
/// attached as `a`. Acquire, use and drop it on the same thread. Returned to
/// the pool when dropped. Holding a reader slot takes no read lock; the first
/// database access starts the read on both databases and checks them.
pub struct Reader<'a> {
    pool: &'a ReadPool,
    conn: Option<Connection>,
    index_identity: Option<IndexIdentity>,
    state: std::cell::Cell<ReadState>,
    owner: ThreadId,
    // ThreadId ownership detects self-exhaustion only if live readers stay
    // on their checkout thread. The marker makes that invariant enforceable.
    thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
    vectors: std::cell::RefCell<Option<crate::vectors::Vectors>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReadState {
    Deferred,
    Ready,
    IndexReplaced,
}
type IndexIdentity = (u64, u64);
fn index_identity(archive: &Archive) -> Result<IndexIdentity> {
    let metadata = std::fs::metadata(archive.index_db())?;
    Ok((metadata.dev(), metadata.ino()))
}

/// What the loop knows about a path from its last look, for comparing
/// with the file now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnownInput {
    pub path: String,
    pub input_id: String,
    pub format: String,
    /// The file's facts at the last observation; `None` once it was seen
    /// gone.
    pub facts: Option<FileFacts>,
    /// The current revision chain and how many of its bytes are stored.
    pub chain: Option<i64>,
    pub stored: u64,
    /// Offset after the last complete record parsed in that chain, and
    /// whether that chain was parsed at all.
    pub parsed_upto: u64,
    pub parsed: bool,
    /// A successful parse reached source EOF with only an incomplete suffix.
    pub waiting: bool,
}

/// A revision chain's stored length and parser progress, independent of a
/// source path. Imports use this to retry parsing without storing bytes twice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainProgress {
    pub chain: i64,
    pub arrival_rank: u8,
    pub waiting: bool,
    pub length: u64,
    pub parsed_upto: u64,
    pub parsed: bool,
}

/// Totals for `status`. Unknown is never shown as zero: a count that could
/// not be read is an error, not a 0.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Status {
    /// Inputs whose bytes are still being written.
    pub pending_inputs: Vec<crate::staging::PendingBlob>,
    pub sessions: u64,
    pub messages: u64,
    pub documents: u64,
    pub tool_calls: u64,
    pub inputs: u64,
    pub revisions: u64,
    /// Bytes of input content stored (uncompressed).
    pub stored_bytes: u64,
    /// Input chains by parse status (`done`, `parsing`, `waiting` for growth).
    pub parse_states: BTreeMap<String, u64>,
    /// The most frequent record kinds and fields parsers did not read.
    pub unused: Vec<(String, u64)>,
    /// Texts stored but not yet in the search index: how far search lags
    /// the writer.
    pub search_pending: u64,
}

impl ReadPool {
    /// Standalone callers use a bounded connection and page-cache allocation.
    pub fn open(archive: &Archive) -> Result<Self> {
        Self::with_budget(archive, 4, 128 << 20)
    }

    /// `memory_bytes` is the combined SQLite page-cache target of all pool
    /// reader slots, including index, attached archive and optional embeddings. Mmap is disabled
    /// so mapped pages cannot multiply outside that allocation.
    /// This is not a process RSS limit: result values and SQLite work space vary.
    pub fn with_budget(archive: &Archive, max_readers: usize, memory_bytes: u64) -> Result<Self> {
        let divisor = u64::try_from(max_readers)
            .ok()
            .and_then(|n| n.checked_mul(3 * 1024))
            .filter(|n| *n != 0)
            .ok_or_else(|| {
                StoreError::Invalid("read pool needs a positive connection limit".into())
            })?;
        let cache_kib = i64::try_from(memory_bytes / divisor)
            .ok()
            .filter(|n| *n > 0)
            .ok_or_else(|| {
                StoreError::Invalid(
                    "read pool budget needs at least 1 KiB per database per connection".into(),
                )
            })?;
        Ok(Self {
            archive: archive.clone(),
            state: Mutex::new(PoolState::default()),
            available: Condvar::new(),
            max_readers,
            cache_kib,
            #[cfg(test)]
            changed: Condvar::new(),
        })
    }

    fn connect(&self) -> Result<Connection> {
        let conn = open_read(&self.archive.index_db())?;
        let uri = format!("file:{}?mode=ro", self.archive.archive_db().display());
        conn.execute("ATTACH DATABASE ?1 AS a", [uri])?;
        crate::sql_views::create(&conn)?;
        for schema in ["main", "a"] {
            conn.pragma_update(Some(schema), "cache_size", -self.cache_kib)?;
            conn.pragma_update(Some(schema), "mmap_size", 0)?;
        }
        Ok(conn)
    }

    /// Waits for a connection while the pool is full. Reuse the existing Reader
    /// within a request. Nested checkouts are allowed while capacity remains;
    /// waiting while this thread already holds a slot refuses to avoid deadlock.
    pub fn reader(&self) -> Result<Reader<'_>> {
        self.acquire(|| self.connect())
    }

    fn acquire(&self, connect: impl FnOnce() -> Result<Connection>) -> Result<Reader<'_>> {
        let owner = std::thread::current().id();
        let idle = {
            let mut state = self.state.lock().map_err(|_| pool_poisoned())?;
            loop {
                if state.closed {
                    return Err(StoreError::ReadPoolClosed);
                }
                if let Some(conn) = state.idle.pop() {
                    *state.owners.entry(owner).or_default() += 1;
                    break Some(conn);
                }
                if state.live < self.max_readers {
                    state.live += 1;
                    *state.owners.entry(owner).or_default() += 1;
                    break None;
                }
                if state.owners.contains_key(&owner) {
                    return Err(StoreError::ReadPoolNested);
                }
                #[cfg(test)]
                {
                    state.waiters += 1;
                    self.changed.notify_all();
                }
                state = self.available.wait(state).map_err(|_| pool_poisoned())?;
                #[cfg(test)]
                {
                    state.waiters -= 1;
                }
            }
        };
        // Hold the slot before the open and BEGIN, which can fail. Every
        // failure releases the slot and wakes a waiter, including a racing close.
        let mut reader = Reader {
            pool: self,
            conn: None,
            index_identity: None,
            state: std::cell::Cell::new(ReadState::Deferred),
            owner,
            thread_bound: std::marker::PhantomData,
            vectors: std::cell::RefCell::new(None),
        };
        // A replaced index is a new file, never an old inode reused. Compare
        // around the open without starting a read or delaying an unused slot.
        let current = index_identity(&self.archive);
        let idle = idle.filter(|(_, identity)| current.as_ref().is_ok_and(|now| now == identity));
        let (conn, identity) = match idle {
            Some(connection) => connection,
            None => {
                let conn = connect()?;
                let before = current?;
                let after = index_identity(&self.archive)?;
                if before != after {
                    return Err(StoreError::ReadIndexReplaced);
                }
                (conn, after)
            }
        };
        conn.execute_batch("BEGIN")?;
        let state = self.state.lock().map_err(|_| pool_poisoned())?;
        if state.closed {
            return Err(StoreError::ReadPoolClosed);
        }
        reader.conn = Some(conn);
        reader.index_identity = Some(identity);
        drop(state);
        Ok(reader)
    }

    /// Refuses new reads, wakes waiting checkouts and releases idle connections.
    /// Readers that have started keep their read; a slot not yet used must
    /// pass the closed check at its first read.
    pub fn close(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.closed = true;
        state.live -= state.idle.len();
        state.idle.clear();
        self.available.notify_all();
    }
}

fn pool_poisoned() -> StoreError {
    StoreError::Invalid("read pool lock poisoned".into())
}

impl Drop for Reader<'_> {
    fn drop(&mut self) {
        self.vectors.get_mut().take();
        let conn = self.conn.take().filter(|conn| {
            matches!(self.state.get(), ReadState::Deferred | ReadState::Ready)
                && conn.execute_batch("COMMIT").is_ok()
        });
        let mut state = self
            .pool
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(held) = state.owners.get_mut(&self.owner) {
            *held -= 1;
            if *held == 0 {
                state.owners.remove(&self.owner);
            }
        }
        match conn {
            Some(conn) if !state.closed => state.idle.push((
                conn,
                self.index_identity
                    .expect("opened connection has an index identity"),
            )),
            _ => state.live -= 1,
        }
        self.pool.available.notify_one();
    }
}

/// A parameter of read-only SQL, so callers outside the store need no
/// SQLite types.
#[derive(Debug, Clone, PartialEq)]
pub enum SqlValue {
    Int(i64),
    Real(f64),
    Text(String),
    Null,
}

impl rusqlite::ToSql for SqlValue {
    fn to_sql(&self) -> rusqlite::Result<rusqlite::types::ToSqlOutput<'_>> {
        use rusqlite::types::{ToSqlOutput, ValueRef};
        Ok(match self {
            SqlValue::Int(n) => ToSqlOutput::Borrowed(ValueRef::Integer(*n)),
            SqlValue::Real(f) => ToSqlOutput::Borrowed(ValueRef::Real(*f)),
            SqlValue::Text(s) => ToSqlOutput::Borrowed(ValueRef::Text(s.as_bytes())),
            SqlValue::Null => ToSqlOutput::Borrowed(ValueRef::Null),
        })
    }
}

pub(crate) fn parse_name<T: serde::de::DeserializeOwned>(s: &str) -> Option<T> {
    serde_json::from_value(serde_json::Value::String(s.to_owned())).ok()
}

pub(crate) fn ts(ms: Option<i64>) -> Option<Timestamp> {
    ms.and_then(Timestamp::from_unix_ms)
}

struct MessageContent<'a> {
    structured: bool,
    visibility: Option<&'a parents::Visibility>,
}

impl Reader<'_> {
    /// A row dump names its native source independently of its observations.
    pub fn tool_database_source(
        &self,
        input: &str,
        blobs: &hashed_blobs::Store,
    ) -> Result<Option<(String, String)>> {
        let Some(progress) = self.input_progress(input)? else {
            return Ok(None);
        };
        let chain = self.stored_chain(input, progress.chain, progress.length)?;
        let mut bytes = Vec::new();
        let mut offset = 0;
        while offset < progress.length {
            let end = progress.length.min(offset + 4096);
            bytes.extend(chain.read_range(blobs, offset, end)?);
            if let Some(end) = bytes.iter().position(|byte| *byte == b'\n') {
                let Ok(header) = serde_json::from_slice::<serde_json::Value>(&bytes[..end]) else {
                    return Ok(None);
                };
                if !header
                    .get("schema")
                    .is_some_and(serde_json::Value::is_object)
                {
                    return Ok(None);
                }
                return Ok(header.get("source").and_then(|source| {
                    Some((
                        source.get("machine")?.as_str()?.to_owned(),
                        source.get("path")?.as_str()?.to_owned(),
                    ))
                }));
            }
            offset = end;
        }
        Ok(None)
    }

    /// The optional embeddings read connection shares this reader's slot and
    /// cannot outlive it. Only one vector guard can be held.
    pub fn vectors(&self) -> Result<std::cell::RefMut<'_, crate::vectors::Vectors>> {
        self.vectors_access(false)
    }

    /// An embedding build uses the same connection and cache share.
    pub fn vectors_writer(&self) -> Result<std::cell::RefMut<'_, crate::vectors::Vectors>> {
        self.vectors_access(true)
    }

    fn vectors_access(
        &self,
        writable: bool,
    ) -> Result<std::cell::RefMut<'_, crate::vectors::Vectors>> {
        self.conn()?;
        let mut vectors = self.vectors.try_borrow_mut().map_err(|_| {
            StoreError::Invalid("this reader already has an embeddings database guard".into())
        })?;
        if vectors.as_ref().is_some_and(|v| writable && !v.writable()) {
            // Close before reopening, so a slot never owns two auxiliary connections.
            vectors.take();
        }
        if vectors.is_none() {
            *vectors = Some(crate::vectors::Vectors::for_reader(
                &self.pool.archive,
                self.pool.cache_kib,
                writable,
            )?);
        }
        Ok(std::cell::RefMut::map(vectors, |v| {
            v.as_mut().expect("initialized vectors")
        }))
    }

    pub(crate) fn conn(&self) -> Result<&Connection> {
        let conn = self
            .conn
            .as_ref()
            .expect("reader holds a connection until dropped");
        match self.state.get() {
            ReadState::Ready => {}
            ReadState::IndexReplaced => return Err(StoreError::ReadIndexReplaced),
            ReadState::Deferred => {
                // An unused slot holds no read and no lock. Start the read on
                // both databases and check them at the first actual access.
                if self.pool.state.lock().map_err(|_| pool_poisoned())?.closed {
                    return Err(StoreError::ReadPoolClosed);
                }
                let waiting = std::time::Instant::now();
                let _shared = crate::write_lock::wait_shared(&self.pool.archive, || {
                    if conn.is_interrupted() {
                        return Err(rusqlite::Error::SqliteFailure(
                            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_INTERRUPT),
                            None,
                        )
                        .into());
                    }
                    let state = self.pool.state.lock().map_err(|_| pool_poisoned())?;
                    if state.closed {
                        return Err(StoreError::ReadPoolClosed);
                    }
                    if waiting.elapsed() >= std::time::Duration::from_secs(30) {
                        return Err(rusqlite::Error::SqliteFailure(
                            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
                            None,
                        )
                        .into());
                    }
                    drop(
                        self.pool
                            .available
                            .wait_timeout(state, std::time::Duration::from_millis(5))
                            .map_err(|_| pool_poisoned())?,
                    );
                    Ok(())
                })?;
                // This closed check under the lock is the read's start. Closing
                // before it refuses; a read that started here may finish afterward.
                if self.pool.state.lock().map_err(|_| pool_poisoned())?.closed {
                    return Err(StoreError::ReadPoolClosed);
                }
                if Some(index_identity(&self.pool.archive)?) != self.index_identity {
                    self.state.set(ReadState::IndexReplaced);
                    return Err(StoreError::ReadIndexReplaced);
                }
                // One statement starts the read on `index.db` (`main`), then on
                // the register (`a`). The shared lock keeps register commits
                // out of the gap between the two, so they see a committed pair.
                conn.query_row(
                    "SELECT (SELECT count(*) FROM main.meta), (SELECT count(*) FROM a.meta)",
                    [],
                    |_| Ok(()),
                )?;
                self.state.set(ReadState::Ready);
            }
        }
        Ok(conn)
    }

    /// The index file this reader's read transaction started on.
    pub fn index_identity(&self) -> Result<(u64, u64)> {
        self.conn()?;
        self.index_identity.ok_or(StoreError::ReadIndexReplaced)
    }

    /// The key read cursors are sealed with: a secret of this index
    /// generation that no answer carries, so a cursor a client changed or
    /// made up is refused.
    pub fn cursor_key(&self) -> Result<[u8; 32]> {
        let secret: String = self.conn()?.query_row(
            "SELECT value FROM main.meta WHERE key = 'cursor_key'",
            [],
            |r| r.get(0),
        )?;
        Ok(*blake3::hash(secret.as_bytes()).as_bytes())
    }

    /// Both counters and the index identity from this reader's read.
    pub fn change_position(&self) -> Result<polylogue_model::ChangePosition> {
        Ok(self.conn()?.query_row(
            "SELECT (SELECT value FROM main.meta WHERE key='generation'),
             COALESCE((SELECT CAST(value AS INTEGER) FROM main.meta WHERE key='write_counter'),0),
             COALESCE((SELECT CAST(value AS INTEGER) FROM a.meta WHERE key='write_counter'),0)",
            [],
            |r| {
                Ok(polylogue_model::ChangePosition {
                    generation: r.get(0)?,
                    index: r.get(1)?,
                    archive: r.get(2)?,
                })
            },
        )?)
    }

    pub fn session(&self, id: &SessionId) -> Result<Option<Session>> {
        Ok(self
            .conn()?
            .prepare_cached(&format!(
                "SELECT {} FROM session WHERE id = ?1",
                crate::session_row::COLUMNS
            ))?
            .query_row([id.to_string()], |r| {
                crate::session_row::read(id.clone(), r)
            })
            .optional()?)
    }

    /// The stored title selected for display, separate from source metadata.
    /// A recorded empty string stays present; a missing title stays unknown.
    pub fn display_title(&self, id: &SessionId) -> Result<Option<String>> {
        Ok(self
            .conn()?
            .query_row(
                "SELECT coalesce(title, derived_title) FROM session WHERE id = ?1",
                [id.to_string()],
                |row| row.get(0),
            )
            .optional()?
            .flatten())
    }

    /// The session's messages in order, the shown version of each. With
    /// `with_parents`, its parent sessions' messages come first: a fork's parent prefix up
    /// to the branch point, and messages its file copied from another
    /// session, which are stored once, in the session they came from.
    pub fn messages(&self, id: &SessionId, with_parents: bool) -> Result<Vec<Message>> {
        let mut out = Vec::new();
        self.visit_messages(id, with_parents, |message| {
            out.push(message);
            std::ops::ControlFlow::Continue(())
        })?;
        Ok(out)
    }

    /// Visit messages in reading order, decoding only the current message's
    /// bodies. Breaking stops before later bodies are loaded. With parents reads
    /// keep inherited message identifiers for ordering and deduplication.
    pub fn visit_messages(
        &self,
        id: &SessionId,
        with_parents: bool,
        mut visit: impl FnMut(Message) -> std::ops::ControlFlow<()>,
    ) -> Result<()> {
        let selected = self.select_messages(id, with_parents)?;
        selected.require_complete()?;
        selected.visit(&mut visit)
    }

    /// Fingerprint stored message facts without decompressing bodies. A strict
    /// caller cannot mistake incomplete parents for a complete selection.
    #[cfg(test)]
    pub fn message_fingerprint(&self, id: &SessionId, with_parents: bool) -> Result<String> {
        let selected = self.select_messages(id, with_parents)?;
        selected.require_complete()?;
        selected.fingerprint()
    }

    /// The keys of the messages a session shows with its parents, each once.
    /// Totals with parents refuse a missing parent rather than count what is
    /// stored as complete.
    pub fn message_keys_with_parents(&self, id: &SessionId) -> Result<Vec<i64>> {
        let selected = self.select_messages(id, true)?;
        selected.require_complete()?;
        selected.keys()
    }

    fn session_pk(&self, id: &SessionId) -> Result<Option<i64>> {
        Ok(self
            .conn()?
            .prepare_cached("SELECT pk FROM session WHERE id = ?1")?
            .query_row([id.to_string()], |r| r.get(0))
            .optional()?)
    }

    /// A session's own message keys and IDs in reading order.
    fn own_keys(&self, session_pk: i64) -> Result<Vec<(i64, String)>> {
        Ok(self
            .conn()?
            .prepare_cached(
                "SELECT pk, id FROM message WHERE session_pk = ?1 AND duplicate_of IS NULL
                 ORDER BY order_input_id, seq, record_position, id",
            )?
            .query_map([session_pk], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// A message's blocks; without `structured`, tool results lack their
    /// structured data (`block.data`), which is then neither read nor decoded.
    fn message_blocks(&self, message_pk: i64, structured: bool) -> Result<Vec<Block>> {
        let mut stmt = self.conn()?.prepare_cached(if structured {
            "SELECT b.body, t.body, b.data FROM block b LEFT JOIN text t ON t.id=b.text_id
             WHERE b.message_pk=?1 ORDER BY b.idx"
        } else {
            "SELECT b.body, t.body, NULL FROM block b LEFT JOIN text t ON t.id=b.text_id
             WHERE b.message_pk=?1 ORDER BY b.idx"
        })?;
        let rows = stmt.query_map([message_pk], |r| {
            Ok((
                r.get::<_, Vec<u8>>(0)?,
                r.get::<_, Option<Vec<u8>>>(1)?,
                r.get::<_, Option<Vec<u8>>>(2)?,
            ))
        })?;
        let mut blocks = Vec::new();
        for row in rows {
            let (body, text, data) = row?;
            let mut block = unpack_block(&body)?;
            if let Some(text) = text {
                restore_text(&mut block, unpack_text(&text)?);
            }
            if let Some(data) = data {
                crate::encoding::restore_data(&mut block, &data)?;
            }
            blocks.push(block);
        }
        Ok(blocks)
    }

    /// Visits the session's messages in order, from message `from` (a key)
    /// when given, without reading the messages before it.
    fn visit_messages_where(
        &self,
        session_pk: i64,
        session: &SessionId,
        from: Option<i64>,
        content: MessageContent<'_>,
        mut keep: impl FnMut(i64) -> bool,
        visit: &mut impl FnMut(i64, Message) -> std::ops::ControlFlow<()>,
    ) -> Result<std::ops::ControlFlow<()>> {
        const COLUMNS: &str = "m.pk, m.id, m.role, m.prompt_kind, m.model, m.at, m.parent_id,
            m.input_tokens, m.cache_read_tokens, m.cache_write_tokens, m.output_tokens,
            m.reasoning_tokens, m.service_tier, m.stop_reason, m.duration_ms, m.attrs_json,
            m.cache_write_1h_tokens, m.speed, m.pem";
        const ORDER: &str = "ORDER BY m.order_input_id, m.seq, m.record_position, m.id";
        let start: Option<(String, i64, i64, String)> = match from {
            Some(pk) => self
                .conn()?
                .prepare_cached(
                    "SELECT order_input_id, seq, record_position, id FROM message
                     WHERE pk = ?1 AND session_pk = ?2",
                )?
                .query_row(rusqlite::params![pk, session_pk], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
                })
                .optional()?,
            None => None,
        };
        if from.is_some() && start.is_none() {
            return Err(StoreError::Invalid(
                "the read starts at a message the session does not hold".into(),
            ));
        }
        let mut stmt = match &start {
            None => self.conn()?.prepare_cached(&format!(
                "SELECT {COLUMNS} FROM message m
                 WHERE m.session_pk = ?1 AND m.duplicate_of IS NULL {ORDER}"
            ))?,
            // Index order from the start message: its key or a later one.
            Some(_) => self.conn()?.prepare_cached(&format!(
                "SELECT {COLUMNS} FROM message m
                 WHERE m.session_pk = ?1 AND m.duplicate_of IS NULL
                   AND (m.order_input_id, m.seq, m.record_position) >= (?2, ?3, ?4)
                   AND NOT ((m.order_input_id, m.seq, m.record_position) = (?2, ?3, ?4)
                            AND m.id < ?5)
                 {ORDER}"
            ))?,
        };
        let parameters: Vec<rusqlite::types::Value> = match start {
            None => vec![session_pk.into()],
            Some((input, seq, position, id)) => vec![
                session_pk.into(),
                input.into(),
                seq.into(),
                position.into(),
                id.into(),
            ],
        };
        let rows = stmt.query_map(rusqlite::params_from_iter(parameters), |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, Option<String>>(3)?,
                r.get::<_, Option<String>>(4)?,
                r.get::<_, Option<i64>>(5)?,
                r.get::<_, Option<String>>(6)?,
                [
                    r.get::<_, Option<i64>>(7)?,
                    r.get::<_, Option<i64>>(8)?,
                    r.get::<_, Option<i64>>(9)?,
                    r.get::<_, Option<i64>>(10)?,
                    r.get::<_, Option<i64>>(11)?,
                ],
                r.get::<_, Option<String>>(12)?,
                r.get::<_, Option<String>>(13)?,
                r.get::<_, Option<i64>>(14)?,
                r.get::<_, Option<String>>(15)?,
                (
                    r.get::<_, Option<i64>>(16)?,
                    r.get::<_, Option<String>>(17)?,
                ),
                r.get::<_, i64>(18)?,
            ))
        })?;
        for row in rows {
            let (
                pk,
                id,
                role,
                kind,
                model,
                at,
                parent,
                tokens,
                tier,
                stop,
                duration,
                attrs,
                (w1h, speed),
                pem,
            ) = row?;
            let Ok(id) = MessageId::parse(&id) else {
                continue;
            };
            if !keep(pk) {
                continue;
            }
            let n = |x: Option<i64>| x.map(|x| x as u64);
            let attrs: Option<serde_json::Value> =
                attrs.and_then(|a| serde_json::from_str(&a).ok());
            let visible = if pem != 0 {
                true
            } else {
                match content.visibility {
                    None => true,
                    Some(filter)
                        if filter.prompts
                            && (role != "user"
                                || matches!(kind.as_deref(), Some("harness" | "model"))) =>
                    {
                        false
                    }
                    Some(filter) if filter.kinds.is_empty() => false,
                    Some(filter) => {
                        let sql = format!(
                            "SELECT EXISTS(SELECT 1 FROM block WHERE message_pk=?1 AND kind IN ({}))",
                            filter.kinds
                        );
                        self.conn()?
                            .prepare_cached(&sql)?
                            .query_row([pk], |row| row.get::<_, bool>(0))?
                    }
                }
            };
            let blocks = if visible {
                self.message_blocks(pk, content.structured)?
            } else {
                Vec::new()
            };
            let message = Message {
                id,
                session: session.clone(),
                parent: parent.and_then(|p| MessageId::parse(&p).ok()),
                role: parse_name(&role).unwrap_or(Role::User),
                at: ts(at),
                model,
                prompt_kind: kind.as_deref().and_then(parse_name::<PromptKind>),
                blocks,
                usage: Usage {
                    input: n(tokens[0]),
                    cached_input: n(tokens[1]),
                    cache_write: n(tokens[2]),
                    output: n(tokens[3]),
                    reasoning: n(tokens[4]),
                    cache_write_1h: n(w1h),
                    service_tier: tier,
                    speed,
                },
                stop_reason: stop,
                reported_cost_usd: attrs
                    .as_ref()
                    .and_then(|a| a.get("reported_cost_usd"))
                    .and_then(serde_json::Value::as_number)
                    .cloned(),
                duration_ms: n(duration),
                production: attrs
                    .as_ref()
                    .and_then(|a| a.get("production"))
                    .and_then(|p| serde_json::from_value(p.clone()).ok())
                    .unwrap_or_default(),
            };
            if visit(pk, message).is_break() {
                return Ok(std::ops::ControlFlow::Break(()));
            }
        }
        Ok(std::ops::ControlFlow::Continue(()))
    }

    pub fn status(&self) -> Result<Status> {
        let c = self.conn()?;
        let count =
            |sql: &str| -> Result<u64> { Ok(c.query_row(sql, [], |r| r.get::<_, i64>(0))? as u64) };
        let mut parse_states = BTreeMap::new();
        let mut stmt =
            c.prepare_cached("SELECT status, COUNT(*) FROM parse_state GROUP BY status")?;
        for row in stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))? {
            let (s, n) = row?;
            parse_states.insert(s, n as u64);
        }
        let mut unused = Vec::new();
        let mut stmt = c.prepare_cached(
            "SELECT what, SUM(count) AS n FROM unused GROUP BY what ORDER BY n DESC LIMIT 20",
        )?;
        for row in stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))? {
            let (w, n) = row?;
            unused.push((w, n as u64));
        }
        Ok(Status {
            pending_inputs: crate::staging::pending_blobs(c, "a.")?,
            sessions: count("SELECT n FROM main.row_count WHERE name = 'session'")?,
            messages: count("SELECT n FROM main.row_count WHERE name = 'message'")?,
            documents: count("SELECT n FROM main.row_count WHERE name = 'document'")?,
            tool_calls: count("SELECT n FROM main.row_count WHERE name = 'tool_call'")?,
            inputs: count("SELECT n FROM a.row_count WHERE name = 'input'")?,
            revisions: count("SELECT n FROM a.row_count WHERE name = 'revision'")?,
            stored_bytes: count("SELECT n FROM a.row_count WHERE name = 'stored_bytes'")?,
            parse_states,
            unused,
            search_pending: count("SELECT COUNT(*) FROM search_pending")?,
        })
    }

    /// Every path this machine observed, with its last facts and how far
    /// its current chain is stored and parsed. The loop compares these with
    /// the files now.
    /// Checks that every blob under a range of a chain is present, without
    /// loading the chain's extents.
    pub fn check_bytes_present(
        &self,
        input: &str,
        chain: i64,
        range: std::ops::Range<u64>,
    ) -> Result<()> {
        crate::staging::require_range(self.conn()?, "a.", input, chain, range)
    }

    /// The stored bytes of an input's revision chain, for parsing them
    /// again without the file.
    pub fn stored_chain(
        &self,
        input: &str,
        chain: i64,
        length: u64,
    ) -> Result<crate::blobs::StoredChain> {
        self.check_bytes_present(input, chain, 0..length)?;
        crate::blobs::StoredChain::load(self.conn()?, "a.", input, chain, length)
    }

    /// Reads only the extents a range covers, with their integrity checks.
    pub fn stored_range(
        &self,
        input: &str,
        chain: i64,
        start: u64,
        end: u64,
        store: &hashed_blobs::Store,
    ) -> Result<Vec<u8>> {
        self.check_bytes_present(input, chain, start..end)?;
        crate::blobs::StoredChain::load_range(self.conn()?, "a.", input, chain, start, end)?
            .read_range(store, start, end)
    }

    fn current_attachment(&self, asset: &crate::assets::AssetMatch) -> Result<()> {
        let current = self
            .asset_source(&asset.input)?
            .ok_or_else(|| StoreError::NotFound("attachment input".into()))?;
        if current.chain != asset.chain
            || current.length != asset.size
            || current.version != asset.source_version
            || current.blob != asset.blob
        {
            return Err(StoreError::Conflict(
                "the attachment's stored bytes changed; select it again".into(),
            ));
        }
        Ok(())
    }

    /// The content store the archive's bytes are read from.
    pub fn blob_store(&self) -> Result<hashed_blobs::Store> {
        self.pool.archive.blobs()
    }

    /// Existing content, with no writable handle or directory setup.
    pub fn blob_reader(&self) -> hashed_blobs::Reader {
        self.pool.archive.blob_reader()
    }

    /// Reads a range of the selected stored attachment in this reader's read.
    pub fn attachment_range(
        &self,
        asset: &crate::assets::AssetMatch,
        range: std::ops::Range<u64>,
    ) -> Result<Vec<u8>> {
        self.current_attachment(asset)?;
        if range.start > range.end || range.end > asset.size {
            return Err(StoreError::Invalid(
                "attachment range is outside the stored bytes".into(),
            ));
        }
        let store = self.blob_store()?;
        self.stored_range(&asset.input, asset.chain, range.start, range.end, &store)
    }

    /// A current full hash may span several stored new blobs. Compute
    /// it once for an explicitly requested exact delivery, streaming the bytes.
    pub fn attachment_hash(
        &self,
        asset: &crate::assets::AssetMatch,
    ) -> Result<polylogue_model::BlobHash> {
        self.current_attachment(asset)?;
        use std::io::Read;
        let store = std::sync::Arc::new(self.blob_store()?);
        let mut source = self
            .stored_chain(&asset.input, asset.chain, asset.size)?
            .reader(store);
        let mut hasher = blake3::Hasher::new();
        let mut buffer = [0; 64 * 1024];
        let mut length = 0u64;
        loop {
            let count = source.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
            length += count as u64;
        }
        if length != asset.size {
            return Err(StoreError::Invalid(
                "the stored attachment length differs".into(),
            ));
        }
        Ok(polylogue_model::BlobHash(*hasher.finalize().as_bytes()))
    }

    /// Source file facts come from provider observations; an excision event
    /// records archive housekeeping, never a fresh observation of the file.
    #[cfg(any(test, feature = "fixture"))]
    pub fn known_inputs(&self, machine: &str) -> Result<Vec<KnownInput>> {
        self.known_inputs_at(machine, None, true)
    }

    /// Committed state of one source path after a rejected write.
    #[cfg(any(test, feature = "fixture"))]
    pub fn known_input_at(&self, machine: &str, path: &str) -> Result<Option<KnownInput>> {
        Ok(self.known_inputs_at(machine, Some(path), true)?.pop())
    }

    /// Mutable source observations used by discovery, excluding manual imports.
    pub fn known_watched_inputs(&self, machine: &str) -> Result<Vec<KnownInput>> {
        crate::fault::check("read.watched_inputs", machine)?;
        self.known_inputs_at(machine, None, false)
    }
    pub fn known_watched_input_at(&self, machine: &str, path: &str) -> Result<Option<KnownInput>> {
        Ok(self.known_inputs_at(machine, Some(path), false)?.pop())
    }

    fn known_inputs_at(
        &self,
        machine: &str,
        path: Option<&str>,
        include_imported: bool,
    ) -> Result<Vec<KnownInput>> {
        let mut stmt = self.conn()?.prepare_cached(
            "WITH latest AS (
                 SELECT path, max(rowid) AS row_id FROM a.observation
                 WHERE machine = ?1 AND (?2 IS NULL OR path = ?2)
                   AND class <> 'excised'
                   AND (?3 OR input NOT IN (
                       SELECT input FROM a.revision WHERE produced = 'imported'))
                 GROUP BY path
             )
             SELECT o.path, o.input, i.format, o.class, o.device, o.inode, o.size, o.mtime_ns,
                    o.ctime_ns,
                    (SELECT chain FROM a.revision r WHERE r.input = o.input ORDER BY number DESC LIMIT 1),
                    (SELECT length FROM a.revision r WHERE r.input = o.input ORDER BY number DESC LIMIT 1)
             FROM latest l JOIN a.observation o ON o.rowid = l.row_id
             JOIN a.input i ON i.id = o.input",
        )?;
        let rows = stmt.query_map(params![machine, path, include_imported], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                [
                    r.get::<_, Option<i64>>(4)?,
                    r.get::<_, Option<i64>>(5)?,
                    r.get::<_, Option<i64>>(6)?,
                    r.get::<_, Option<i64>>(7)?,
                    r.get::<_, Option<i64>>(8)?,
                ],
                r.get::<_, Option<i64>>(9)?,
                r.get::<_, Option<i64>>(10)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (path, input_id, format, class, f, chain, length) = row?;
            let facts = match (class.as_str(), f) {
                ("deleted", _) => None,
                (_, [Some(d), Some(i), Some(s), Some(m), Some(c)]) => Some(FileFacts {
                    device: d as u64,
                    inode: i as u64,
                    size: s as u64,
                    mtime_ns: i128::from(m),
                    ctime_ns: i128::from(c),
                }),
                _ => None,
            };
            out.push(KnownInput {
                path,
                input_id,
                format,
                facts,
                chain,
                stored: length.unwrap_or(0) as u64,
                parsed_upto: 0,
                parsed: false,
                waiting: false,
            });
        }
        let mut stmt = self.conn()?.prepare_cached(
            "SELECT p.parsed_upto, p.status, p.detail FROM parse_state p
             JOIN input_ref r ON r.pk = p.input_pk WHERE r.id = ?1 AND p.chain = ?2",
        )?;
        for k in &mut out {
            if let Some(chain) = k.chain
                && let Some((upto, status, detail)) = stmt
                    .query_row(params![k.input_id, chain], |r| {
                        Ok((
                            r.get::<_, i64>(0)?,
                            r.get::<_, String>(1)?,
                            r.get::<_, Option<String>>(2)?,
                        ))
                    })
                    .optional()?
            {
                k.parsed_upto = upto as u64;
                k.parsed = true;
                k.waiting = status == "waiting"
                    && detail.and_then(|end| end.parse::<u64>().ok()) == Some(k.stored);
            }
        }
        Ok(out)
    }

    /// The input's latest revision and its chain's parser progress.
    pub fn input_progress(&self, id: &str) -> Result<Option<ChainProgress>> {
        let row = self
            .conn()?
            .prepare_cached(
                "SELECT r.length, COALESCE(p.parsed_upto, 0), p.input_pk IS NOT NULL, r.produced, COALESCE(p.status = 'waiting', 0), r.chain
             FROM a.revision r
             LEFT JOIN input_ref i ON i.id = r.input
             LEFT JOIN parse_state p ON p.input_pk = i.pk AND p.chain = r.chain
             WHERE r.input = ?1 ORDER BY r.number DESC LIMIT 1",
            )?
            .query_row([id], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, bool>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, bool>(4)?,
                    r.get::<_, i64>(5)?,
                ))
            })
            .optional()?;
        let Some((length, parsed_upto, parsed, produced, waiting, chain)) = row else {
            return Ok(None);
        };
        let arrival_rank = match produced.as_str() {
            "imported" => 2,
            "first" | "appended" | "rewritten" => 0,
            _ => {
                return Err(StoreError::Invalid(format!(
                    "unknown revision source: {produced}"
                )));
            }
        };
        Ok(Some(ChainProgress {
            chain,
            arrival_rank,
            waiting,
            length: length as u64,
            parsed_upto: parsed_upto as u64,
            parsed,
        }))
    }
}

#[cfg(test)]
mod tests;
