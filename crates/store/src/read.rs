//! Read connections. The daemon keeps a pool; every read in one request
//! sees one consistent snapshot of `index.db` with `archive.db` attached.

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

mod rows;

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
/// the pool when dropped. The first database access pins and validates both
/// snapshots; reserving an unused slot holds no read lock.
pub struct Reader<'a> {
    pool: &'a ReadPool,
    conn: Option<Connection>,
    index_identity: Option<IndexIdentity>,
    snapshot: std::cell::Cell<SnapshotState>,
    owner: ThreadId,
    // ThreadId ownership detects self-exhaustion only if live readers stay
    // on their checkout thread. The marker makes that invariant enforceable.
    thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
    vectors: std::cell::RefCell<Option<crate::vectors::Vectors>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SnapshotState {
    Deferred,
    Ready,
    Refused,
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
    /// Offset after the last complete record parsed in that chain, and the
    /// parser version that parsed it.
    pub parsed_upto: u64,
    pub parser_version: Option<u32>,
    /// A successful parse reached source EOF with only an incomplete suffix.
    pub waiting: bool,
}

/// The current retained revision and its parser progress, independent of a
/// source path. Imports use this to retry parsing without retaining bytes twice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetainedInput {
    pub chain: i64,
    pub route_rank: u8,
    pub waiting: bool,
    pub length: u64,
    pub parsed_upto: u64,
    pub parser_version: Option<u32>,
}

/// Totals for `status`. Unknown is never shown as zero: a count that could
/// not be read is an error, not a 0.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Status {
    pub sessions: u64,
    pub messages: u64,
    pub tool_calls: u64,
    pub inputs: u64,
    pub revisions: u64,
    /// Bytes of input content stored (uncompressed).
    pub stored_bytes: u64,
    /// Input chains by parse status (`done`, `parsing`, `waiting` for growth).
    pub parse_states: BTreeMap<String, u64>,
    /// The most frequent record kinds and fields parsers did not read.
    pub unrecognized: Vec<(String, u64)>,
    /// Texts stored but not yet in the search index: how far search lags
    /// the writer (architecture 5.5c).
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
        // Own the reservation before fallible open/BEGIN work. Every failure
        // releases its slot and wakes a waiter, including a racing close.
        let mut reader = Reader {
            pool: self,
            conn: None,
            index_identity: None,
            snapshot: std::cell::Cell::new(SnapshotState::Deferred),
            owner,
            thread_bound: std::marker::PhantomData,
            vectors: std::cell::RefCell::new(None),
        };
        // Store replacements retire old files, never reinstall their inodes.
        // Compare around open without pinning SQLite or delaying an unused slot.
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
    /// Pinned readers keep their snapshot; unused reservations must enter
    /// first-read admission before closure.
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
            matches!(
                self.snapshot.get(),
                SnapshotState::Deferred | SnapshotState::Ready
            ) && conn.execute_batch("COMMIT").is_ok()
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

impl Reader<'_> {
    /// The optional embeddings read connection shares this reader's allocation
    /// and cannot outlive its admission. Only one vector guard can be held.
    pub fn vectors(&self) -> Result<std::cell::RefMut<'_, crate::vectors::Vectors>> {
        self.vectors_access(false)
    }

    /// An embedding build uses the same reserved connection/cache share.
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
                crate::excise::recovery::current(self.conn()?)?,
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
        match self.snapshot.get() {
            SnapshotState::Ready => {}
            SnapshotState::Refused => return Err(StoreError::ExcisionPending),
            SnapshotState::IndexReplaced => return Err(StoreError::ReadIndexReplaced),
            SnapshotState::Deferred => {
                // Unused reservations hold no snapshot or publication lock.
                // Pin and validate both databases at the first actual read.
                if self.pool.state.lock().map_err(|_| pool_poisoned())?.closed {
                    return Err(StoreError::ReadPoolClosed);
                }
                let waiting = std::time::Instant::now();
                let _admission = crate::publication::admit(&self.pool.archive, || {
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
                // The guarded state check is first-read admission. Closing
                // before it refuses; a read admitted here may finish afterward.
                if self.pool.state.lock().map_err(|_| pool_poisoned())?.closed {
                    return Err(StoreError::ReadPoolClosed);
                }
                self.snapshot.set(SnapshotState::Refused);
                if Some(index_identity(&self.pool.archive)?) != self.index_identity {
                    self.snapshot.set(SnapshotState::IndexReplaced);
                    return Err(StoreError::ReadIndexReplaced);
                }
                crate::excise::recovery::check_snapshot(conn)?;
                self.snapshot.set(SnapshotState::Ready);
            }
        }
        Ok(conn)
    }

    /// Both counters and the index identity from this reader's pinned view.
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

    /// The session's messages in order, the preferred version of each. With
    /// `logical`, messages it inherited come first: a fork's parent prefix up
    /// to the branch point, and messages its file copied from another
    /// session, which are stored once, in the session they came from.
    pub fn messages(&self, id: &SessionId, logical: bool) -> Result<Vec<Message>> {
        let Some(pk) = self.session_pk(id)? else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        if logical {
            // Inherited messages come in runs, one per ancestor session.
            let mut runs: Vec<(i64, SessionId, HashSet<i64>)> = Vec::new();
            for (session_pk, session, message_pk) in self.inherited_keys(pk, id)? {
                match runs.last_mut() {
                    Some((s, _, set)) if *s == session_pk => {
                        set.insert(message_pk);
                    }
                    _ => runs.push((session_pk, session, HashSet::from([message_pk]))),
                }
            }
            for (session_pk, session, set) in runs {
                out.extend(self.messages_where(session_pk, &session, |m| set.contains(&m))?);
            }
        }
        out.extend(self.own_messages(pk, id)?);
        Ok(out)
    }

    /// The message keys a session shows: what it inherits through its
    /// fork and resume links, in reading order, then its own. Each message
    /// once.
    pub fn logical_message_keys(&self, id: &SessionId) -> Result<Vec<i64>> {
        let Some(pk) = self.session_pk(id)? else {
            return Ok(Vec::new());
        };
        let mut keys: Vec<i64> = self
            .inherited_keys(pk, id)?
            .into_iter()
            .map(|(_, _, m)| m)
            .collect();
        keys.extend(self.own_keys(pk)?.into_iter().map(|(m, _)| m));
        Ok(keys)
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
                "SELECT pk, id FROM current_message WHERE session_pk = ?1 ORDER BY order_input_id, seq",
            )?
            .query_map([session_pk], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// What a session inherits: `(session key, session, message key)` in
    /// reading order. A fork shows its parent's view up to the branch
    /// point; a resume shows the messages its own file copied.
    fn inherited_keys(
        &self,
        child_pk: i64,
        child: &SessionId,
    ) -> Result<Vec<(i64, SessionId, i64)>> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        self.inherited(child_pk, child, &mut out, &mut seen, 0)?;
        Ok(out)
    }

    fn inherited(
        &self,
        child_pk: i64,
        child: &SessionId,
        out: &mut Vec<(i64, SessionId, i64)>,
        seen: &mut HashSet<i64>,
        depth: usize,
    ) -> Result<()> {
        if depth > 64 {
            return Ok(());
        }
        let links: Vec<(String, String, Option<String>)> = self
            .conn()?
            .prepare_cached(
                "SELECT parent_id, how, branch_point FROM current_link
                 WHERE child_id = ?1 AND kind = 'continuation' AND how IN ('fork', 'resume')
                 ORDER BY pk",
            )?
            .query_map([child.to_string()], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })?
            .collect::<rusqlite::Result<_>>()?;
        for (parent, kind, at) in links {
            let Ok(parent) = SessionId::parse(&parent) else {
                continue;
            };
            let Some(parent_pk) = self.session_pk(&parent)? else {
                continue;
            };
            let keys: Vec<i64> = if kind == "fork" {
                self.inherited(parent_pk, &parent, out, seen, depth + 1)?;
                let mut own = self.own_keys(parent_pk)?;
                if let Some(at) = at
                    && let Some(i) = own.iter().position(|(_, id)| *id == at)
                {
                    own.truncate(i + 1);
                }
                own.into_iter().map(|(m, _)| m).collect()
            } else {
                self.conn()?
                    .prepare_cached(
                        "SELECT DISTINCT m.pk, m.order_input_id, m.seq FROM current_message m
                         JOIN message_source s ON s.message_pk = m.pk
                         JOIN parsed p ON p.input_pk = s.input_pk AND p.session_pk = ?2
                         WHERE m.session_pk = ?1 AND s.current=1 ORDER BY m.order_input_id, m.seq",
                    )?
                    .query_map(params![parent_pk, child_pk], |r| r.get(0))?
                    .collect::<rusqlite::Result<_>>()?
            };
            for m in keys {
                if seen.insert(m) {
                    out.push((parent_pk, parent.clone(), m));
                }
            }
        }
        Ok(())
    }

    fn own_messages(&self, session_pk: i64, session: &SessionId) -> Result<Vec<Message>> {
        self.messages_where(session_pk, session, |_| true)
    }

    fn messages_where(
        &self,
        session_pk: i64,
        session: &SessionId,
        keep: impl Fn(i64) -> bool,
    ) -> Result<Vec<Message>> {
        let mut blocks: HashMap<i64, Vec<Block>> = HashMap::new();
        {
            let mut stmt = self.conn()?.prepare_cached(
                "SELECT b.message_pk, b.body, t.body FROM block b
                 JOIN current_message cm ON cm.pk = b.message_pk
                 LEFT JOIN text t ON t.id = b.text_id
                 WHERE b.session_pk = ?1 ORDER BY b.message_pk, b.idx",
            )?;
            let rows = stmt.query_map([session_pk], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, Vec<u8>>(1)?,
                    r.get::<_, Option<Vec<u8>>>(2)?,
                ))
            })?;
            for row in rows {
                let (pk, body, text) = row?;
                let mut block: Block = unpack_block(&body)?;
                if let Some(packed) = text {
                    restore_text(&mut block, unpack_text(&packed)?);
                }
                blocks.entry(pk).or_default().push(block);
            }
        }
        let mut stmt = self.conn()?.prepare_cached(
            "SELECT pk, id, role, prompt_kind, model, at, parent_id, input_tokens, cache_read_tokens,
                    cache_write_tokens, output_tokens, reasoning_tokens, service_tier, stop_reason,
                    duration_ms, attrs_json, cache_write_1h_tokens, speed
             FROM current_message WHERE session_pk = ?1 ORDER BY order_input_id, seq",
        )?;
        let rows = stmt.query_map([session_pk], |r| {
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
            ))
        })?;
        let mut out = Vec::new();
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
            ) = row?;
            if !keep(pk) {
                continue;
            }
            let Ok(id) = MessageId::parse(&id) else {
                continue;
            };
            let n = |x: Option<i64>| x.map(|x| x as u64);
            let attrs: Option<serde_json::Value> =
                attrs.and_then(|a| serde_json::from_str(&a).ok());
            out.push(Message {
                id,
                session: session.clone(),
                parent: parent.and_then(|p| MessageId::parse(&p).ok()),
                role: parse_name(&role).unwrap_or(Role::User),
                at: ts(at),
                model,
                prompt_kind: kind.as_deref().and_then(parse_name::<PromptKind>),
                blocks: blocks.remove(&pk).unwrap_or_default(),
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
            });
        }
        Ok(out)
    }

    pub fn status(&self) -> Result<Status> {
        let c = self.conn()?;
        let count =
            |sql: &str| -> Result<u64> { Ok(c.query_row(sql, [], |r| r.get::<_, i64>(0))? as u64) };
        let mut parse_states = BTreeMap::new();
        let mut stmt =
            c.prepare_cached("SELECT CASE WHEN p.parser_version<COALESCE(r.version,p.parser_version) OR p.parser_version>p.retired_version THEN 'reparsing' ELSE p.status END AS state,COUNT(*)
                FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk LEFT JOIN required_parser r ON r.format=i.format GROUP BY state")?;
        for row in stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))? {
            let (s, n) = row?;
            parse_states.insert(s, n as u64);
        }
        let mut unrecognized = Vec::new();
        let mut stmt = c.prepare_cached(
            "SELECT what, SUM(count) AS n FROM unrecognized GROUP BY what ORDER BY n DESC LIMIT 20",
        )?;
        for row in stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))? {
            let (w, n) = row?;
            unrecognized.push((w, n as u64));
        }
        Ok(Status {
            sessions: count("SELECT COUNT(*) FROM session")?,
            messages: count("SELECT COUNT(*) FROM current_message")?,
            tool_calls: count(
                "SELECT COUNT(*) FROM tool_call t JOIN block b ON b.pk=t.call_block_pk JOIN current_message m ON m.pk=b.message_pk",
            )?,
            inputs: count("SELECT COUNT(*) FROM a.input")?,
            revisions: count("SELECT COUNT(*) FROM a.revision")?,
            stored_bytes: count("SELECT COALESCE(SUM(size), 0) FROM a.blob")?,
            parse_states,
            unrecognized,
            search_pending: count("SELECT COUNT(*) FROM search_pending")?,
        })
    }

    /// Every path this machine observed, with its last facts and how far
    /// its current chain is stored and parsed. The loop compares these with
    /// the files now.
    /// The stored bytes of an input's revision chain, for parsing them
    /// again without the file.
    pub fn stored_chain(
        &self,
        input: &str,
        chain: i64,
        length: u64,
    ) -> Result<crate::blobs::StoredChain> {
        crate::blobs::StoredChain::load(self.conn()?, "a.", input, chain, length)
    }

    /// Read only extents intersecting a retained range, including its integrity checks.
    pub fn stored_range(
        &self,
        input: &str,
        chain: i64,
        start: u64,
        end: u64,
        store: &hashed_blobs::Store,
    ) -> Result<Vec<u8>> {
        crate::blobs::StoredChain::load_range(self.conn()?, "a.", input, chain, start, end)?
            .read_range(store, start, end)
    }

    /// Source file facts come from provider observations; an excision event
    /// records archive housekeeping, never a fresh observation of the file.
    pub fn known_inputs(&self, machine: &str) -> Result<Vec<KnownInput>> {
        self.known_inputs_at(machine, None)
    }

    /// Committed state of one source path after a rejected write.
    pub fn known_input_at(&self, machine: &str, path: &str) -> Result<Option<KnownInput>> {
        Ok(self.known_inputs_at(machine, Some(path))?.pop())
    }

    fn known_inputs_at(&self, machine: &str, path: Option<&str>) -> Result<Vec<KnownInput>> {
        let mut stmt = self.conn()?.prepare_cached(
            "SELECT o.path, o.input, i.format, o.class, o.device, o.inode, o.size, o.mtime_ns,
                    o.ctime_ns,
                    (SELECT chain FROM a.revision r WHERE r.input = o.input ORDER BY number DESC LIMIT 1),
                    (SELECT length FROM a.revision r WHERE r.input = o.input ORDER BY number DESC LIMIT 1)
             FROM a.observation o JOIN a.input i ON i.id = o.input
             WHERE o.machine = ?1 AND (?2 IS NULL OR o.path = ?2) AND o.rowid = (
                 SELECT MAX(rowid) FROM a.observation x WHERE x.machine = o.machine AND x.path = o.path
                 AND x.class <> 'excised')",
        )?;
        let rows = stmt.query_map(params![machine, path], |r| {
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
                parser_version: None,
                waiting: false,
            });
        }
        let mut stmt = self.conn()?.prepare_cached(
            "SELECT p.parsed_upto, p.parser_version, p.status, p.detail FROM parse_state p
             JOIN input_ref r ON r.pk = p.input_pk WHERE r.id = ?1 AND p.chain = ?2",
        )?;
        for k in &mut out {
            if let Some(chain) = k.chain
                && let Some((upto, version, status, detail)) = stmt
                    .query_row(params![k.input_id, chain], |r| {
                        Ok((
                            r.get::<_, i64>(0)?,
                            r.get::<_, i64>(1)?,
                            r.get::<_, String>(2)?,
                            r.get::<_, Option<String>>(3)?,
                        ))
                    })
                    .optional()?
            {
                k.parsed_upto = upto as u64;
                k.parser_version = Some(version as u32);
                k.waiting = status == "waiting"
                    && detail.and_then(|end| end.parse::<u64>().ok()) == Some(k.stored);
            }
        }
        Ok(out)
    }

    /// The latest retained revision and any committed parse progress.
    /// Every retained physical chain, with rebuildable parser progress.
    pub fn retained_chains(&self, id: &str) -> Result<Vec<(i64, RetainedInput)>> {
        let mut stmt=self.conn()?.prepare_cached("SELECT r.chain,max(r.length),COALESCE(p.parsed_upto,0),p.parser_version,COALESCE(p.status='waiting',0),max(r.produced='imported') FROM a.revision r LEFT JOIN input_ref i ON i.id=r.input LEFT JOIN parse_state p ON p.input_pk=i.pk AND p.chain=r.chain WHERE r.input=?1 GROUP BY r.chain ORDER BY r.chain")?;
        Ok(stmt
            .query_map([id], |r| {
                Ok((
                    r.get(0)?,
                    RetainedInput {
                        chain: r.get(0)?,
                        length: r.get::<_, i64>(1)? as u64,
                        parsed_upto: r.get::<_, i64>(2)? as u64,
                        parser_version: r.get::<_, Option<i64>>(3)?.map(|v| v as u32),
                        waiting: r.get(4)?,
                        route_rank: if r.get::<_, bool>(5)? { 2 } else { 0 },
                    },
                ))
            })?
            .collect::<rusqlite::Result<_>>()?)
    }

    pub fn retained_input(&self, id: &str) -> Result<Option<RetainedInput>> {
        let row = self
            .conn()?
            .prepare_cached(
                "SELECT r.length, COALESCE(p.parsed_upto, 0), p.parser_version, r.produced, COALESCE(p.status = 'waiting', 0), r.chain
             FROM a.revision r
             LEFT JOIN input_ref i ON i.id = r.input
             LEFT JOIN parse_state p ON p.input_pk = i.pk AND p.chain = r.chain
             WHERE r.input = ?1 ORDER BY r.number DESC LIMIT 1",
            )?
            .query_row([id], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, Option<i64>>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, bool>(4)?,
                    r.get::<_, i64>(5)?,
                ))
            })
            .optional()?;
        let Some((length, parsed_upto, parser_version, produced, waiting, chain)) = row else {
            return Ok(None);
        };
        let route_rank = match produced.as_str() {
            "imported" => 2,
            "first" | "appended" | "head_rewritten" | "rewritten" | "snapshot" => 0,
            _ => {
                return Err(StoreError::Invalid(format!(
                    "unknown revision source: {produced}"
                )));
            }
        };
        Ok(Some(RetainedInput {
            chain,
            route_rank,
            waiting,
            length: length as u64,
            parsed_upto: parsed_upto as u64,
            parser_version: parser_version.map(|v| v as u32),
        }))
    }
}

/// A normalized version at its exact retained source range.
#[derive(Debug, Clone, PartialEq)]
pub struct MessageVersion {
    pub input: String,
    pub chain: u64,
    pub snapshot: Option<u64>,
    pub start: u64,
    pub end: u64,
    pub current: bool,
    pub message: Message,
}

impl Reader<'_> {
    pub fn message_versions(&self, id: &polylogue_model::MessageId) -> Result<Vec<MessageVersion>> {
        let mut stmt = self.conn()?.prepare("SELECT r.id,s.chain,s.source_snapshot,s.source_start,s.source_end,s.current,s.version_hash FROM message_source s JOIN message m ON m.pk=s.message_pk JOIN input_ref r ON r.pk=s.input_pk JOIN session placement ON placement.pk=s.session_pk WHERE m.id=?1 ORDER BY r.id,s.chain,s.source_snapshot,s.source_start,placement.id")?;
        let rows = stmt.query_map([id.to_string()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, Vec<u8>>(6)?,
            ))
        })?;
        let mut versions = Vec::new();
        for row in rows {
            let (input, chain, snapshot, start, end, current, hash) = row?;
            versions.push(MessageVersion {
                input,
                chain: chain as u64,
                snapshot: (snapshot >= 0).then_some(snapshot as u64),
                start: start as u64,
                end: end as u64,
                current: current != 0,
                message: crate::versions::load(self.conn()?, &hash)?.1,
            });
        }
        Ok(versions)
    }
}

#[cfg(test)]
mod tests;
