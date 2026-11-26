//! Read connections. The daemon keeps a pool; every read in one request
//! sees one consistent read transaction on `index.db` with `archive.db`
//! attached.

use crate::ConnExt as _;
use crate::archive::Archive;
use crate::archive::index_build::Reads;
use crate::blobs::FileFacts;
use crate::encoding::Columns;
use crate::{Result, StoreError};
use polylogue_model::LockExt;
use polylogue_model::{
    Block, Message, MessageId, PromptKind, Role, Session, SessionId, Usage, bundle::ShownBlocks,
};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::ThreadId;

pub mod limit;
pub mod scope;

pub mod authored_read;
mod object;
pub use object::{ObjectRead, ObjectRecord};
mod behind_index;
mod completeness;
pub use completeness::ReadingContract;
mod captured;
mod cursor_generation;
pub use cursor_generation::open_cursor;
mod hook_fields;
mod parser_checkpoint;
pub use parser_checkpoint::{StoredParserCheckpoint, StoredParserProgress};
mod import_counts;
pub use import_counts::ImportCounts;
mod parents;
mod revisions;
mod trace;
pub use revisions::RevisionPoint;
mod tool_outcomes;
mod tool_pairs;
pub use behind_index::{BehindChain, RevisionChains, StoppedChains, StoredSource};
pub use captured::{CapturedSession, StoredFile};
mod order;
mod record_seek;
pub use record_seek::PhysicalRecordCounts;
mod rows;
mod signals;
mod source_commit;
mod span;
pub use span::SpanSource;
mod source_snapshot;
use order::RecordOrder;

pub(crate) use parents::compare_session_messages;
pub use parents::{MessagePosition, MessageSelection, ParentIssue, Parents};

/// Where a visit starts and which way it walks.
/// First, At and After walk forward; Before and Last walk backward.
pub enum VisitFrom<P> {
    First,
    At(P),
    After(P),
    Before(P),
    Last,
}

/// What a sourced message visit adds to each message.
pub struct MessageParse {
    pub blocks: ShownBlocks,
    pub committed: Option<u64>,
    pub ordinal: u64,
}

struct MessageArrival {
    order: Option<i64>,
    fact: Option<i64>,
    source: Option<(String, i64, i64)>,
}

impl MessageArrival {
    fn record(self, conn: &Connection) -> Result<(Option<u64>, u64)> {
        let order =
            order::checked(self.order.ok_or_else(|| {
                StoreError::stored(0, "message snapshot ordinal was not selected")
            })?)?;
        Ok((self.resolve(conn)?, order))
    }

    fn resolve(self, conn: &Connection) -> Result<Option<u64>> {
        if let Some(fact) = self.fact {
            return u64::try_from(fact)
                .map(Some)
                .map_err(|_| StoreError::stored(0, "negative message commit"));
        }
        let Some((input, chain, end)) = self.source else {
            return Ok(None);
        };
        let end = u64::try_from(end)
            .map_err(|_| StoreError::stored(0, "negative message source endpoint"))?;
        crate::source::Source::committed_through(conn, &input, chain, end)
    }
}

/// Read-only connections to one archive, reused across requests.
/// `max_readers` connections serve every read; one more is kept for reads
/// that run inside [`reserved_reads`], so status and completion answers
/// never wait behind a full pool.
pub struct ReadPool {
    archive: Archive,
    /// Which index file the pool reads.
    reads: Reads,
    state: Arc<Mutex<PoolState>>,
    available: Arc<Condvar>,
    max_readers: usize,
    cache_kib: i64,
    counts: Mutex<crate::query::execute::CountCache>,
    /// Wakes waiting checkouts when the access scope changes.
    _scope_wake: Arc<scope::ResetWake>,
    /// A pipeline pool closes its idle connections before a build file is
    /// installed ([`crate::archive::index_build`]).
    _release: Option<Arc<dyn Fn() + Send + Sync>>,
    #[cfg(test)]
    changed: Condvar,
}

thread_local! {
    static RESERVED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `work` with access to the pool's reserved connection, for the
/// interactive answers that must keep their budget while bulk reads hold
/// every other connection.
pub fn reserved_reads<T>(work: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            RESERVED.with(|reserved| reserved.set(self.0));
        }
    }
    let _restore = Restore(RESERVED.with(|reserved| reserved.replace(true)));
    work()
}

#[derive(Default)]
struct PoolState {
    idle: Vec<(Connection, IndexIdentity)>,
    closed: bool,
    /// Includes idle, checked-out and still-opening connections.
    live: usize,
    /// Checked-out and still-opening connections.
    in_use: usize,
    holders: HashMap<ThreadId, usize>,
    #[cfg(test)]
    waiters: usize,
}

/// One consistent read: a read transaction on `index.db` with `archive.db`
/// attached as `a`. Acquire, use and drop it on the same thread. Returned to
/// the pool when dropped. Holding a reader slot takes no read lock; the first
/// database access starts the read on both databases and checks them.
/// Current annotation revisions are stored (`revision_state`), so no read builds it.
pub struct Reader<'a> {
    pool: &'a ReadPool,
    conn: Option<Connection>,
    index_identity: Option<IndexIdentity>,
    state: std::cell::Cell<ReadState>,
    holder: ThreadId,
    limit: Option<limit::ReadLimit>,
    scope: scope::ReadScopeToken,
    // Counting holders by ThreadId detects self-exhaustion only if live readers stay
    // on their checkout thread. The marker makes that invariant enforceable.
    thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
    vectors: std::cell::RefCell<Option<crate::vectors::Vectors>>,
    calendar: crate::sql_functions::CalendarContext,
    annotation_history_ready: std::cell::RefCell<std::collections::BTreeSet<i64>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReadState {
    Deferred,
    Ready,
    IndexReplaced,
}
/// The path a connection was opened through and the file it named: a build
/// file renamed into place keeps its inode, and a connection opened through
/// its old path must not be reused.
type IndexIdentity = (PathBuf, u64, u64);

/// What the loop knows about a path from its last look, for comparing
/// with the file now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnownInput {
    pub path: String,
    /// None means an older record has no path bytes; never a UTF-8 fallback.
    pub path_bytes: Option<Vec<u8>>,
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
    /// An incomplete suffix, refusal or failure waits at this stored length.
    pub waiting: bool,
}

/// A revision chain's stored length and parser progress, independent of a
/// source path. Imports use this to retry parsing without storing bytes twice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainProgress {
    pub chain: i64,
    /// The chain's bytes came in through `polylogue import`.
    pub imported: bool,
    pub waiting: bool,
    pub length: u64,
    pub parsed_upto: u64,
    pub parsed: bool,
}

/// Totals for `status`. Unknown is never shown as zero: a count that could
/// not be read is an error, not a 0.
#[derive(
    Debug, Clone, Default, PartialEq, Eq, Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct Status {
    /// Inputs whose bytes are still being written.
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
    /// The latest confirmed full discovery walk from each source machine.
    pub capture_coverage: Vec<polylogue_model::capture::Coverage>,
    /// Set when this index replaced one written for another schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_rebuild: Option<crate::archive::index_rebuild::IndexRebuild>,
}

impl ReadPool {
    /// A pipeline pool with a bounded connection and page-cache allocation.
    pub fn open(archive: &Archive) -> Result<Self> {
        Self::with_budget(archive, 4, 128 << 20)
    }

    /// The pool client-facing reads use. While an index is being built and
    /// no previous index can be served, a checkout answers
    /// [`StoreError::IndexBuilding`] with the build's progress.
    pub fn serving(archive: &Archive, max_readers: usize, memory_bytes: u64) -> Result<Self> {
        Self::create(archive, Reads::Serving, max_readers, memory_bytes)
    }

    /// `memory_bytes` is the combined SQLite page-cache target of the
    /// `max_readers` reader slots, including index, attached archive and
    /// optional embeddings; the reserved connection takes one slot's share
    /// on top. Mmap is disabled
    /// so mapped pages cannot multiply outside that allocation.
    /// This is not a process RSS limit: result values and SQLite work space vary.
    /// The pool reads what the build pipeline writes; only the build pipeline
    /// uses one.
    pub fn with_budget(archive: &Archive, max_readers: usize, memory_bytes: u64) -> Result<Self> {
        Self::create(archive, Reads::Pipeline, max_readers, memory_bytes)
    }

    fn create(
        archive: &Archive,
        reads: Reads,
        max_readers: usize,
        memory_bytes: u64,
    ) -> Result<Self> {
        let divisor = u64::try_from(max_readers)
            .ok()
            .and_then(|n| n.checked_mul(3 * 1024))
            .filter(|n| *n != 0)
            .ok_or_else(|| polylogue_model::Reason::NotPositive {
                field: "the read pool's connection limit".into(),
            })?;
        let cache_kib = i64::try_from(memory_bytes / divisor)
            .ok()
            .filter(|n| *n > 0)
            .ok_or_else(|| polylogue_model::Reason::BadConfiguration {
                problem: "the read pool budget needs at least 1 KiB per database per connection"
                    .into(),
            })?;
        let state = Arc::new(Mutex::new(PoolState::default()));
        let available = Arc::new(Condvar::new());
        let scope_wake: Arc<scope::ResetWake> = {
            let (state, available) = (Arc::clone(&state), Arc::clone(&available));
            Arc::new(move || wake_all(&state, &available))
        };
        archive.read_scope.on_reset(&scope_wake);
        let release = (reads == Reads::Pipeline).then(|| {
            let state = Arc::clone(&state);
            let release: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
                let mut state = state.locked();
                state.live -= state.idle.len();
                state.idle.clear();
            });
            archive.index_build.on_release(Arc::clone(&release));
            release
        });
        Ok(Self {
            archive: archive.clone(),
            reads,
            _release: release,
            state,
            available,
            max_readers,
            cache_kib,
            counts: Mutex::default(),
            _scope_wake: scope_wake,
            #[cfg(test)]
            changed: Condvar::new(),
        })
    }

    /// The connections reads may hold at once, besides the reserved one.
    #[cfg(any(test, feature = "fixture"))]
    pub fn max_readers(&self) -> usize {
        self.max_readers
    }

    #[cfg(any(test, feature = "fixture"))]
    pub fn checked_out(&self) -> usize {
        self.state.lock().unwrap().holders.values().sum()
    }

    #[cfg(any(test, feature = "fixture"))]
    pub fn checked_out_by(&self, thread: ThreadId) -> usize {
        self.state
            .lock()
            .unwrap()
            .holders
            .get(&thread)
            .copied()
            .unwrap_or(0)
    }

    /// The index file this pool reads now.
    fn index_path(&self) -> Result<PathBuf> {
        match self.reads {
            Reads::Pipeline => Ok(self.archive.index_db()),
            Reads::Serving => self.archive.served_index(),
        }
    }

    fn index_identity(&self) -> Result<IndexIdentity> {
        self.archive.with_index_path(self.reads, |path| {
            let metadata = std::fs::metadata(path)?;
            Ok((path.to_path_buf(), metadata.dev(), metadata.ino()))
        })
    }

    /// Whether this pool reads a build in progress.
    pub(crate) fn reads_build(&self) -> bool {
        self.reads == Reads::Pipeline && self.archive.index_build.building()
    }

    fn connect(&self, path: &Path) -> Result<Connection> {
        let conn = self.archive.open_index_connection(path, self.reads)?;
        // Set temp storage before creating views: changing it drops TEMP objects.
        conn.pragma_update(None, "temp_store", "file")?;
        conn.pragma_update(Some("temp"), "cache_size", -1024)?;
        let uri = format!("file:{}?mode=ro", self.archive.archive_db().display());
        conn.execute("ATTACH DATABASE ?1 AS a", [uri])?;
        // Each view is a write to the temporary schema: one transaction per
        // connection instead of one per view.
        conn.execute_batch("BEGIN")?;
        let created = (|| {
            crate::source::views(&conn)?;
            crate::sql_views::create(&conn)?;
            Ok::<_, StoreError>(())
        })();
        match created {
            Ok(()) => conn.execute_batch("COMMIT")?,
            Err(error) => {
                let _ = conn.execute_batch("ROLLBACK");
                return Err(error);
            }
        }
        for schema in ["main", "a"] {
            conn.pragma_update(Some(schema), "cache_size", -self.cache_kib)?;
            conn.pragma_update(Some(schema), "mmap_size", 0)?;
        }
        // The full-text files share what one index file's entries had.
        let files = if crate::search::files::attached(&conn)? {
            crate::search::files::FILES
        } else {
            0
        };
        for n in 0..files {
            let schema = crate::search::files::schema(n);
            conn.pragma_update(
                Some(schema.as_str()),
                "cache_size",
                -(self.cache_kib / crate::search::files::FILES as i64).max(1),
            )?;
            conn.pragma_update(Some(schema.as_str()), "mmap_size", 0)?;
        }
        Ok(conn)
    }

    /// Pins the commit scope without acquiring a SQL reader.
    pub fn scope_token(&self) -> Result<scope::ReadScopeToken> {
        self.archive.read_scope.token()
    }

    /// Waits for a connection while the pool is full. Reuse the existing Reader
    /// within a request. Nested checkouts are allowed while capacity remains;
    /// waiting while this thread already holds a slot refuses to avoid deadlock.
    pub fn reader(&self) -> Result<Reader<'_>> {
        self.acquire(|path| self.connect(path))
    }

    /// Runs `read` on a new reader, and again on another when a build's
    /// install replaced the index under the first
    /// ([`StoreError::ReadIndexReplaced`]): the replacement is the index to
    /// read, so the error is never the answer.
    pub fn read_current<T>(&self, read: impl Fn(&Reader<'_>) -> Result<T>) -> Result<T> {
        loop {
            match self.reader().and_then(|reader| read(&reader)) {
                Err(StoreError::ReadIndexReplaced) => continue,
                result => return result,
            }
        }
    }

    fn acquire(&self, connect: impl FnOnce(&Path) -> Result<Connection>) -> Result<Reader<'_>> {
        // Refused before a slot is taken: nothing can be read until the build ends.
        self.index_path()?;
        let scope = self.archive.read_scope.token()?;
        let limit = limit::ReadLimit::current();
        if let Some(limit) = &limit {
            limit.bind_scope(scope.clone())?;
        }
        let holder = std::thread::current().id();
        let capacity = self.max_readers + usize::from(RESERVED.with(std::cell::Cell::get));
        // The process's stop and a cancelled request wake a waiting checkout;
        // the deadline bounds the wait. Nothing polls.
        let stop = self.archive.stop();
        let wake = || {
            let (state, available) = (Arc::clone(&self.state), Arc::clone(&self.available));
            move || wake_all(&state, &available)
        };
        let _stop_wake = stop.on_cancel(wake());
        let _cancel_wake = limit
            .as_ref()
            .map(|limit| limit.cancellation.on_cancel(wake()));
        let idle = {
            let mut state = self.state.lock().map_err(|_| pool_poisoned())?;
            loop {
                stop.check()?;
                scope.check()?;
                if let Some(limit) = &limit {
                    limit.check()?;
                }
                if state.closed {
                    return Err(StoreError::ReadPoolClosed);
                }
                if state.in_use < capacity {
                    state.in_use += 1;
                    *state.holders.entry(holder).or_default() += 1;
                    match state.idle.pop() {
                        Some(conn) => break Some(conn),
                        None => {
                            state.live += 1;
                            break None;
                        }
                    }
                }
                if state.holders.contains_key(&holder) {
                    return Err(StoreError::ReadPoolNested);
                }
                #[cfg(test)]
                {
                    state.waiters += 1;
                    self.changed.notify_all();
                }
                state = match limit.as_ref().and_then(limit::ReadLimit::remaining) {
                    Some(left) => {
                        self.available
                            .wait_timeout(state, left)
                            .map_err(|_| pool_poisoned())?
                            .0
                    }
                    None => self.available.wait(state).map_err(|_| pool_poisoned())?,
                };
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
            holder,
            limit,
            scope,
            thread_bound: std::marker::PhantomData,
            vectors: std::cell::RefCell::new(None),
            calendar: crate::sql_functions::CalendarContext::new(),
            annotation_history_ready: std::cell::RefCell::new(Default::default()),
        };
        // A replaced index is a new file, never an old inode reused. Compare
        // around the open without starting a read or delaying an unused slot.
        let current = self.index_identity();
        let idle = idle.filter(|(_, identity)| current.as_ref().is_ok_and(|now| now == identity));
        let (conn, identity) = match idle {
            Some(connection) => connection,
            None => {
                let before = current?;
                let conn = match &reader.limit {
                    Some(limit) => limit.scope(|| connect(&before.0)),
                    None => connect(&before.0),
                };
                let after = self.index_identity()?;
                // An open that raced a build's install fails on the moved file.
                let conn = match conn {
                    Err(_) if before != after => return Err(StoreError::ReadIndexReplaced),
                    conn => conn?,
                };
                if before != after {
                    return Err(StoreError::ReadIndexReplaced);
                }
                (conn, after)
            }
        };
        // Every reader observes the process's stop and the access scope; a
        // request's reader also its own deadline and cancellation.
        if let Some(limit) = &reader.limit {
            limit.install(&conn)?;
        }
        let (scope, stop, limit) = (reader.scope.clone(), stop.clone(), reader.limit.clone());
        crate::cancellation::interrupt_on(&conn, move || {
            stop.check()?;
            scope.check()?;
            limit
                .as_ref()
                .map_or(Ok(()), limit::ReadLimit::count_and_check)
        })?;
        reader.check()?;
        crate::sql_functions::register_calendar(&conn, &reader.calendar)?;
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
        let mut state = self.state.locked();
        state.closed = true;
        state.live -= state.idle.len();
        state.idle.clear();
        self.available.notify_all();
    }
}

fn wake_all(state: &Mutex<PoolState>, available: &Condvar) {
    // Taking the lock orders the wake after a waiter's last check.
    let _held = state.locked();
    available.notify_all();
}

fn pool_poisoned() -> StoreError {
    StoreError::internal("read pool lock poisoned")
}

impl Drop for Reader<'_> {
    fn drop(&mut self) {
        self.vectors.get_mut().take();
        let conn = self.conn.take().filter(|conn| {
            if conn.progress_handler(0, None::<fn() -> bool>).is_err()
                || conn
                    .busy_timeout(std::time::Duration::from_secs(30))
                    .is_err()
            {
                return false;
            }
            matches!(self.state.get(), ReadState::Deferred | ReadState::Ready)
                && conn.execute_batch("ROLLBACK").is_ok()
        });
        let mut state = self.pool.state.locked();
        if let Some(held) = state.holders.get_mut(&self.holder) {
            *held -= 1;
            if *held == 0 {
                state.holders.remove(&self.holder);
            }
        }
        state.in_use -= 1;
        match conn {
            Some(conn) if !state.closed => state.idle.push((
                conn,
                self.index_identity
                    .take()
                    .expect("opened connection has an index identity"),
            )),
            _ => state.live -= 1,
        }
        // A freed slot may suit only a reserved waiter, so every waiter looks.
        self.pool.available.notify_all();
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

struct MessageContent<'a> {
    order: RecordOrder,
    structured: bool,
    visibility: Option<&'a parents::Visibility>,
}

/// A chain's parse standing from its stored status word, waiting length and
/// reason.
pub(crate) fn stored_parse_progress(
    status: &str,
    at_length: Option<i64>,
    reason: Option<String>,
) -> Result<polylogue_model::batch::ParseProgress> {
    let at_length = at_length
        .map(u64::try_from)
        .transpose()
        .map_err(|_| StoreError::stored(0, "a stored waiting length is negative"))?;
    polylogue_model::batch::ParseProgress::from_stored(status, at_length, reason)
        .map_err(|error| StoreError::stored(0, error))
}

impl Reader<'_> {
    /// Pages this reader's connection had to read from its database files
    /// (`SQLITE_DBSTATUS_CACHE_MISS`, every attached database) since the last
    /// call, which resets the count. Tests bound a request's reads with it.
    #[cfg(any(test, feature = "fixture"))]
    #[allow(unsafe_code)]
    #[cfg(any(test, feature = "fixture"))]
    pub fn take_page_reads(&self) -> Result<u64> {
        let conn = self.conn()?;
        let (mut current, mut highest) = (0, 0);
        // SAFETY: the handle belongs to this reader's open connection, which
        // outlives the call; the status call only reads counters and resets one.
        let code = unsafe {
            rusqlite::ffi::sqlite3_db_status(
                conn.handle(),
                rusqlite::ffi::SQLITE_DBSTATUS_CACHE_MISS,
                &mut current,
                &mut highest,
                1,
            )
        };
        if code != rusqlite::ffi::SQLITE_OK {
            return Err(StoreError::internal(format!(
                "page read count is unavailable (SQLite code {code})"
            )));
        }
        Ok(u64::try_from(current).unwrap_or(0))
    }

    /// As-of annotation reads need the current revisions among the records written
    /// before `cutoff`, computed once per cutoff in this read transaction. Present
    /// current revisions are stored and needs no preparation.
    pub(crate) fn prepare_annotation_history(&self, cutoff: i64) -> Result<()> {
        let conn = self.conn()?;
        if !self.annotation_history_ready.borrow().contains(&cutoff) {
            crate::annotations::current_revisions::prepare_history(conn, cutoff)?;
            self.check()?;
            self.annotation_history_ready.borrow_mut().insert(cutoff);
        }
        Ok(())
    }
    /// Retain this token through prepared or streamed delivery.
    pub fn scope_token(&self) -> scope::ReadScopeToken {
        self.scope.clone()
    }

    pub fn check(&self) -> Result<()> {
        self.pool.archive.stop().check()?;
        self.scope.check()?;
        self.limit.as_ref().map_or(Ok(()), limit::ReadLimit::check)
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
            StoreError::internal("this reader already has an embeddings database guard")
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
                self.limit.as_ref(),
            )?);
        }
        Ok(std::cell::RefMut::map(vectors, |v| {
            v.as_mut().expect("initialized vectors")
        }))
    }

    pub(crate) fn count_cache(&self) -> &Mutex<crate::query::execute::CountCache> {
        &self.pool.counts
    }

    pub(crate) fn read_limit(&self) -> Option<&limit::ReadLimit> {
        self.limit.as_ref()
    }

    pub(crate) fn conn(&self) -> Result<&Connection> {
        self.check()?;
        let conn = self
            .conn
            .as_ref()
            .expect("reader holds a connection until dropped");
        match self.state.get() {
            ReadState::Ready => {}
            ReadState::IndexReplaced => return Err(StoreError::ReadIndexReplaced),
            ReadState::Deferred => {
                // An unused slot holds no read and no lock. Start the read on
                // both databases and check them at the first database access.
                if self.pool.state.lock().map_err(|_| pool_poisoned())?.closed {
                    return Err(StoreError::ReadPoolClosed);
                }
                let waiting = std::time::Instant::now();
                let _shared = crate::write_lock::wait_shared(&self.pool.archive, || {
                    self.pool.archive.stop().check()?;
                    self.scope.check()?;
                    if conn.is_interrupted() {
                        return Err(rusqlite::Error::SqliteFailure(
                            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_INTERRUPT),
                            None,
                        )
                        .into());
                    }
                    if let Some(limit) = &self.limit {
                        limit.check()?;
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
                if Some(self.pool.index_identity()?) != self.index_identity {
                    self.state.set(ReadState::IndexReplaced);
                    return Err(StoreError::ReadIndexReplaced);
                }
                // One statement starts the read on `index.db` (`main`), then on
                // the register (`a`). The shared lock keeps register commits
                // out of the gap between the two, so they see a committed pair.
                let started = conn.query_row(
                    "SELECT (SELECT count(*) FROM main.meta), (SELECT count(*) FROM source_meta)",
                    [],
                    |_| Ok(()),
                );
                // A build file installed between the check and this start
                // (`crate::archive::index_build`) changes the path identity. An
                // already-started read keeps its open file and snapshot.
                if let Err(error) = started {
                    if Some(self.pool.index_identity()?) != self.index_identity {
                        self.state.set(ReadState::IndexReplaced);
                        return Err(StoreError::ReadIndexReplaced);
                    }
                    return Err(error.into());
                }
                self.state.set(ReadState::Ready);
            }
        }
        Ok(conn)
    }

    /// The index file this reader's read transaction started on.
    pub fn index_identity(&self) -> Result<(u64, u64)> {
        self.conn()?;
        self.index_identity
            .as_ref()
            .map(|(_, device, inode)| (*device, *inode))
            .ok_or(StoreError::ReadIndexReplaced)
    }

    /// The key read cursors are sealed with: a secret of this index
    /// generation that no answer carries, so a cursor a client changed or
    /// made up is refused.
    pub fn cursor_key(&self) -> Result<[u8; 32]> {
        let secret: String = self
            .conn()?
            .scalar("SELECT value FROM main.meta WHERE key = 'cursor_key'", [])?;
        Ok(*blake3::hash(secret.as_bytes()).as_bytes())
    }

    /// Opens a read cursor sealed with this index generation's key; see
    /// [`open_cursor`] for how a cursor from an earlier generation is refused.
    pub fn open_cursor(
        &self,
        cursor: &query_language::cursor::SealedCursor,
        question: query_language::identity::QuestionId,
        resolve_zone: impl FnOnce(
            &str,
        )
            -> std::result::Result<jiff::tz::TimeZone, query_language::Error>,
        available: impl FnOnce(&query_language::cursor::CursorContext) -> bool,
    ) -> std::result::Result<query_language::cursor::Continuation, crate::selection::SelectionError>
    {
        let seal = query_language::cursor::CursorSeal::new(self.cursor_key()?);
        let generation = self.change_position()?.generation;
        Ok(open_cursor(
            &seal,
            &generation,
            cursor,
            question,
            resolve_zone,
            available,
        )?)
    }

    /// Both counters and the index identity from this reader's read.
    pub fn change_position(&self) -> Result<polylogue_model::ChangePosition> {
        Ok(self.conn()?.query_row(
            "SELECT (SELECT value FROM main.meta WHERE key='generation'),
             COALESCE((SELECT CAST(value AS INTEGER) FROM main.meta WHERE key='write_counter'),0),
             COALESCE((SELECT CAST(value AS INTEGER) FROM source_meta WHERE key='write_counter'),0)",
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

    /// The title the session is shown with ([`crate::session_sql::display_title`]).
    /// A recorded empty string stays present; a missing title stays unknown.
    pub fn display_title(&self, id: &SessionId) -> Result<Option<String>> {
        Ok(self
            .conn()?
            .scalar_optional(
                &format!(
                    "SELECT {} FROM session WHERE id = ?1",
                    crate::session_sql::display_title("session")
                ),
                [id.to_string()],
            )?
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

    /// The shown reading and its original block positions in this snapshot.
    pub fn message_with_block_selection(
        &self,
        id: &MessageId,
    ) -> Result<Option<(Message, ShownBlocks)>> {
        let conn = self.conn()?;
        let id = crate::chain_names::resolved_message(conn, id)?;
        let shown: Option<(i64, String, Option<i64>)> = conn
            .query_row(
                "SELECT m.pk,i.id,m.source_chain FROM message m JOIN input_ref i ON i.pk=m.input_pk
                 WHERE m.id=?1",
                [id.to_string()],
                |row| <(i64, String, Option<i64>)>::try_from(row),
            )
            .optional()?;
        let Some((pk, input, chain)) = shown else {
            return Ok(None);
        };
        let (mut message, selection) = crate::writer::shown_message(conn, pk)?
            .ok_or_else(|| StoreError::stored(0, "the message has no source reading"))?;
        // A message read whole shows what its reading left in the stored
        // bytes. Reading a message does not depend on those bytes being at
        // hand (docs/archive.md, text): without them the value keeps its
        // place, which still reads as encrypted reasoning or restated history.
        if let Some(chain) = chain
            && let Err(error) = self.resolve_stored_blocks(&input, chain, &mut message.blocks)
        {
            crate::log!(
                warning,
                "message {id}: a value left in the stored bytes is not shown: {error}"
            );
        }
        Ok(Some((message, selection)))
    }

    /// Commit of the shown stored rows, including all contributing sparse
    /// readings. A reading of a source file uses its source extent commit; an
    /// older record without one stays unknown.
    pub fn message_commit(&self, id: &MessageId) -> Result<Option<u64>> {
        let conn = self.conn()?;
        let id = crate::chain_names::resolved_message(conn, id)?;
        let commit = conn
            .query_row(
                "SELECT (SELECT max(r.committed) FROM message_parse r WHERE r.message_id=m.id AND r.hash=m.version_hash),i.id,m.source_chain,m.source_end
             FROM message m JOIN input_ref i ON i.pk=m.input_pk WHERE m.id=?1 AND m.version_hash IS NOT NULL",
                [id.to_string()],
                |row| {
                    Ok(MessageArrival {
                        order: None,
                        fact: row.get(0)?,
                        source: Some((row.get(1)?, row.get(2)?, row.get(3)?)),
                    })
                },
            )
            .optional()?;
        match commit {
            Some(commit) => commit.resolve(conn),
            None => Ok(None),
        }
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
        selected.visit(Default::default(), &mut visit).map(drop)
    }

    /// The keys of the messages a session shows with its parents, each once.
    /// Totals with parents refuse a missing parent rather than count what is
    /// stored as complete.
    pub fn message_keys_with_parents(&self, id: &SessionId) -> Result<Vec<i64>> {
        let selected = self.select_messages(id, true)?;
        selected.require_complete()?;
        selected.keys()
    }

    pub(crate) fn session_pk(&self, id: &SessionId) -> Result<Option<i64>> {
        Ok(self
            .conn()?
            .prepare_cached("SELECT pk FROM session WHERE id = ?1")?
            .query_row([id.to_string()], |r| r.get(0))
            .optional()?)
    }

    /// A session's own message keys and IDs in reading order.
    fn own_keys(&self, session_pk: i64) -> Result<Vec<(i64, String, i64)>> {
        parents::own_keys(self.conn()?, session_pk)
    }

    /// A message's blocks; without `structured`, tool results lack their
    /// structured data (`block.data`), which is then neither read nor decoded.
    fn message_blocks(&self, message_pk: i64, structured: bool) -> Result<Vec<Block>> {
        crate::ordered_blocks::of_message(self.conn()?, message_pk, structured)
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
        visit: &mut impl FnMut(
            i64,
            Message,
            Option<ShownBlocks>,
            MessageArrival,
        ) -> std::ops::ControlFlow<()>,
    ) -> Result<std::ops::ControlFlow<()>> {
        const COLUMNS: &str = "m.pk, m.id, m.role, m.prompt_kind, m.model, m.at_time, m.parent_id,
            m.input_tokens, m.cache_read_tokens, m.cache_write_tokens, m.output_tokens,
            m.reasoning_tokens, m.service_tier, m.stop_reason, m.duration_ms, m.attrs_json,
            m.cache_write_1h_tokens, m.speed, m.version_hash, m.layout,
            CASE WHEN m.layout IS NULL THEN
                (SELECT count(*) FROM block WHERE message_pk=m.pk) ELSE 0 END,
            (SELECT max(r.committed) FROM message_parse r WHERE r.message_id=m.id AND r.hash=m.version_hash),
            CASE WHEN m.version_hash IS NOT NULL THEN source_input.id END,
            CASE WHEN m.version_hash IS NOT NULL THEN m.source_chain END,
            CASE WHEN m.version_hash IS NOT NULL THEN m.source_end END";
        const SOURCE: &str =
            "FROM message m LEFT JOIN input_ref source_input ON source_input.pk=m.input_pk";
        const ORDER: &str = "ORDER BY m.session_order, m.id";
        let start: Option<(String, String)> = match from {
            Some(pk) => self
                .conn()?
                .prepare_cached(
                    "SELECT session_order, id FROM message
                     WHERE pk = ?1 AND session_pk = ?2",
                )?
                .query_row(rusqlite::params![pk, session_pk], |r| <(_, _)>::try_from(r))
                .optional()?,
            None => None,
        };
        if from.is_some() && start.is_none() {
            return Err(polylogue_model::Reason::ReadStartOutsideSession {}.into());
        }
        let (prefix, join, ordinal) = match content.order {
            RecordOrder::Skip => (String::new(), "", "NULL".to_owned()),
            RecordOrder::Known(value) => (String::new(), "", value.to_string()),
            RecordOrder::OwnSession => (
                format!(
                    "WITH record_ranks AS ({}) ",
                    order::session_record_rows("?1")
                ),
                " JOIN record_ranks ON record_ranks.pk=m.pk",
                "record_ranks.record_order".to_owned(),
            ),
        };
        let mut stmt = match &start {
            None => self.conn()?.prepare_cached(&format!(
                "{prefix}SELECT {COLUMNS},{ordinal} AS record_order {SOURCE}{join}
                 WHERE m.session_pk = ?1 AND m.duplicate_of IS NULL {ORDER}"
            ))?,
            // Index order from the start message: its key or a later one.
            Some(_) => self.conn()?.prepare_cached(&format!(
                "{prefix}SELECT {COLUMNS},{ordinal} AS record_order {SOURCE}{join}
                 WHERE m.session_pk = ?1 AND m.duplicate_of IS NULL
                   AND (m.session_order, m.id) >= (?2, ?3)
                 {ORDER}"
            ))?,
        };
        let parameters: Vec<rusqlite::types::Value> = match start {
            None => vec![session_pk.into()],
            Some((order, id)) => vec![session_pk.into(), order.into(), id.into()],
        };
        let rows = stmt.query_map(rusqlite::params_from_iter(parameters), |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.name::<Role>(2)?,
                r.optional_name::<PromptKind>(3)?,
                r.get::<_, Option<String>>(4)?,
                crate::encoding::read_time(r, 5)?,
                r.get::<_, Option<String>>(6)?,
                [
                    r.get::<_, Option<u64>>(7)?,
                    r.get::<_, Option<u64>>(8)?,
                    r.get::<_, Option<u64>>(9)?,
                    r.get::<_, Option<u64>>(10)?,
                    r.get::<_, Option<u64>>(11)?,
                ],
                r.get::<_, Option<String>>(12)?,
                r.get::<_, Option<String>>(13)?,
                r.get::<_, Option<u64>>(14)?,
                r.get::<_, Option<String>>(15)?,
                (
                    r.get::<_, Option<u64>>(16)?,
                    r.get::<_, Option<String>>(17)?,
                ),
                r.get::<_, Option<Vec<u8>>>(18)?,
                r.get::<_, Option<Vec<u8>>>(19)?,
                r.get::<_, i64>(20)?,
                r.get::<_, Option<i64>>(21)?,
                r.get::<_, Option<String>>(22)?,
                r.get::<_, Option<i64>>(23)?,
                r.get::<_, Option<i64>>(24)?,
                r.get::<_, Option<i64>>("record_order")?,
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
                hash,
                layout,
                stored_block_count,
                committed,
                source_input,
                source_chain,
                source_end,
                record_order,
            ) = row?;
            let id = MessageId::parse(&id).map_err(|_| {
                StoreError::stored(1, format!("message row {pk} has an invalid ID"))
            })?;
            if !keep(pk) {
                continue;
            }
            let named = id.to_string();
            let unreadable =
                |what: &str| StoreError::stored(0, format!("message {named} has {what}"));
            let attrs: Option<serde_json::Value> = attrs
                .map(|a| serde_json::from_str(&a))
                .transpose()
                .map_err(|_| unreadable("unparseable attributes"))?;
            let visible = match content.visibility {
                None => true,
                Some(filter) if filter.prompts && !polylogue_model::is_prompt(role, kind) => false,
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
            };
            let mut blocks = if visible {
                self.message_blocks(pk, content.structured)?
            } else {
                Vec::new()
            };
            // A read of whole values shows what the reading left in the
            // stored bytes.
            if content.structured
                && let (Some(input), Some(chain)) = (&source_input, source_chain)
            {
                self.resolve_stored_blocks(input, chain, &mut blocks)?;
            }
            let message = Message {
                id,
                session: session.clone(),
                parent: parent
                    .map(|p| MessageId::parse(&p))
                    .transpose()
                    .map_err(|_| unreadable("an invalid parent ID"))?,
                role,
                at: at.0,
                at_quality: at.1.quality,
                at_precision: at.1.precision,
                at_until: at.1.until,
                model,
                prompt_kind: kind,
                blocks,
                usage: Usage {
                    input: tokens[0],
                    cached_input: tokens[1],
                    cache_write: tokens[2],
                    output: tokens[3],
                    reasoning: tokens[4],
                    cache_write_1h: w1h,
                    service_tier: tier,
                    speed,
                },
                stop_reason: stop,
                reported_cost_usd: attrs
                    .as_ref()
                    .and_then(|a| a.get("reported_cost_usd"))
                    .and_then(serde_json::Value::as_number)
                    .cloned(),
                duration_ms: duration,
                attrs: match attrs.as_ref().and_then(|value| value.get("attrs")) {
                    None => Default::default(),
                    Some(serde_json::Value::Object(map)) => map.clone(),
                    Some(_) => return Err(unreadable("unparseable attributes")),
                },
                production: attrs
                    .as_ref()
                    .and_then(|a| a.get("production"))
                    .map(|p| serde_json::from_value(p.clone()))
                    .transpose()
                    .map_err(|_| unreadable("an unparseable production record"))?
                    .unwrap_or_default(),
            };
            let block_selection = hash
                .map(|hash| {
                    let mut block_selection = crate::encoding::message_block_selection(
                        &hash,
                        layout.as_deref(),
                        if visible {
                            message.blocks.len()
                        } else {
                            usize::try_from(stored_block_count).map_err(|_| {
                                StoreError::stored(0, "a message has an invalid stored block count")
                            })?
                        },
                    )?;
                    if visible {
                        block_selection.validate(message.blocks.len())?;
                    } else {
                        block_selection.block_ordinals.clear();
                    }
                    Ok::<_, StoreError>(block_selection)
                })
                .transpose()?;
            let commit = MessageArrival {
                order: record_order,
                fact: committed,
                source: source_input
                    .zip(source_chain)
                    .zip(source_end)
                    .map(|((input, chain), end)| (input, chain, end)),
            };
            if visit(pk, message, block_selection, commit).is_break() {
                return Ok(std::ops::ControlFlow::Break(()));
            }
        }
        Ok(std::ops::ControlFlow::Continue(()))
    }

    pub fn status(&self) -> Result<Status> {
        limit::ReadLimit::set_stage("status counts");
        let c = self.conn()?;
        let count = |sql: &str| -> Result<u64> { Ok(c.scalar::<i64>(sql, [])? as u64) };
        let mut parse_states = BTreeMap::new();
        let mut stmt = c.prepare_cached("SELECT status, n FROM parse_status_count WHERE n > 0")?;
        for row in stmt.query_map([], |r| <(String, i64)>::try_from(r))? {
            let (s, n) = row?;
            parse_states.insert(s, n as u64);
        }
        let mut unused = Vec::new();
        let mut stmt = c.prepare_cached(
            "SELECT what, n FROM unused_count WHERE n > 0 ORDER BY n DESC, what LIMIT 20",
        )?;
        for row in stmt.query_map([], |r| <(String, i64)>::try_from(r))? {
            let (w, n) = row?;
            unused.push((w, n as u64));
        }
        Ok(Status {
            sessions: count("SELECT n FROM main.row_count WHERE name = 'session'")?,
            messages: count("SELECT n FROM main.row_count WHERE name = 'message'")?,
            documents: count("SELECT n FROM main.row_count WHERE name = 'document'")?,
            tool_calls: count("SELECT n FROM main.row_count WHERE name = 'tool_call'")?,
            inputs: count("SELECT n FROM source_row_count WHERE name = 'input'")?,
            revisions: count("SELECT n FROM source_row_count WHERE name = 'revision'")?,
            stored_bytes: count("SELECT n FROM source_row_count WHERE name = 'stored_bytes'")?,
            parse_states,
            unused,
            search_pending: count("SELECT n FROM main.row_count WHERE name = 'search_pending'")?,
            capture_coverage: self.all_capture_coverage()?,
            index_rebuild: crate::archive::index_rebuild::status(c, &self.pool.archive)?,
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
        crate::staging::require_range(self.conn()?, "source_", input, chain, range)
    }

    fn chain_source(&self) -> crate::blobs::ChainSource {
        crate::blobs::ChainSource::of(&self.pool.archive)
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
        crate::blobs::StoredChain::load(&self.chain_source(), input, chain, length)
    }

    /// Puts back the values a reading of `input`'s chain `chain` left in the
    /// stored bytes (docs/archive.md, content): each block value that reads
    /// as a stored marker is read from the record the marker names.
    pub(crate) fn resolve_stored_blocks(
        &self,
        input: &str,
        chain: i64,
        blocks: &mut [Block],
    ) -> Result<()> {
        let mut records = StoredRecords::default();
        for block in blocks {
            if let Block::Thinking {
                signature: Some(signature),
                ..
            } = block
                && let Some((start, end, pointer)) = crate::content::stored_at(signature)
            {
                let value = records.value(self, input, chain, start, end, pointer)?;
                *signature = value
                    .as_str()
                    .ok_or_else(|| StoreError::stored_data("a stored signature is not a string"))?
                    .to_owned();
            }
        }
        Ok(())
    }

    /// [`Self::resolve_stored_blocks`] for any JSON value: every string in it
    /// that is a stored marker becomes the value it names.
    pub(crate) fn resolve_stored_value(
        &self,
        input: &str,
        chain: i64,
        value: &mut serde_json::Value,
    ) -> Result<()> {
        let mut records = StoredRecords::default();
        let mut pending = vec![value];
        while let Some(value) = pending.pop() {
            match value {
                serde_json::Value::String(text) => {
                    if let Some((start, end, pointer)) = crate::content::stored_at(text) {
                        let found = records.value(self, input, chain, start, end, pointer)?;
                        *value = found;
                    }
                }
                serde_json::Value::Array(items) => pending.extend(items.iter_mut()),
                serde_json::Value::Object(members) => pending.extend(members.values_mut()),
                _ => {}
            }
        }
        Ok(())
    }

    /// Reads only the extents a range covers, with their integrity checks.
    pub fn stored_range(
        &self,
        input: &str,
        chain: i64,
        start: u64,
        end: u64,
        store: &crate::blobs::Blobs,
    ) -> Result<Vec<u8>> {
        self.check_bytes_present(input, chain, start..end)?;
        crate::blobs::StoredChain::load_range(&self.chain_source(), input, chain, start, end)?
            .read_range(store, start, end)
    }

    fn current_attachment(&self, asset: &crate::assets::AssetMatch) -> Result<()> {
        let current = if asset.size == 0 {
            let latest: Option<(i64,i64)> = self.conn()?.query_row("SELECT chain,length FROM source_revision WHERE input=?1 ORDER BY number DESC LIMIT 1", [&asset.input], |row| <(_, _)>::try_from(row)).optional()?;
            if latest != Some((asset.chain,0)) {
                return Err(polylogue_model::Reason::AttachmentChanged {
                    input: asset.input.clone(),
                }
                .into());
            }
            crate::assets::empty::source(
                self.conn()?,
                self.conn()?,
                "source_",
                &asset.input,
                asset.chain,
                None,
                Some(&asset.source_version),
            )?
        } else {
            self.asset_source(&asset.input)?
        }
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
    pub fn blob_store(&self) -> Result<crate::blobs::Blobs> {
        self.blob_reader()
    }

    /// Use the byte catalog belonging to this reader's index snapshot.
    pub fn blob_reader(&self) -> Result<crate::blobs::Blobs> {
        self.conn()?;
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
            return Err(StoreError::stored_data(
                "attachment range is outside the stored bytes",
            ));
        }
        if asset.size == 0 {
            let source = crate::assets::AssetSource {
                chain: asset.chain,
                length: 0,
                version: asset.source_version.clone(),
                blob: asset.blob,
            };
            return crate::assets::empty::read(self, &source);
        }
        let store = self.blob_store()?;
        self.stored_range(&asset.input, asset.chain, range.start, range.end, &store)
    }

    /// A current full hash may span several stored new blobs. Compute
    /// it once for an explicitly requested delivery, streaming the bytes.
    pub fn attachment_hash(
        &self,
        asset: &crate::assets::AssetMatch,
    ) -> Result<polylogue_model::BlobHash> {
        if asset.size == 0 {
            return Ok(polylogue_model::BlobHash::of(
                &self.attachment_range(asset, 0..0)?,
            ));
        }
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
            self.check()?;
            let count = source.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
            length += count as u64;
        }
        if length != asset.size {
            return Err(StoreError::stored_data(
                "the stored attachment length differs",
            ));
        }
        Ok(polylogue_model::BlobHash(*hasher.finalize().as_bytes()))
    }

    /// Source file facts come from provider observations.
    #[cfg(any(test, feature = "fixture"))]
    pub fn known_inputs(&self, machine: &str) -> Result<Vec<KnownInput>> {
        self.known_inputs_at(machine, None, true)
    }

    /// Committed state of one source path after a rejected write.
    #[cfg(any(test, feature = "fixture"))]
    pub fn known_input_at(&self, machine: &str, path: &[u8]) -> Result<Option<KnownInput>> {
        Ok(self.known_inputs_at(machine, Some(path), true)?.pop())
    }

    /// Mutable source observations used by discovery, excluding manual imports.
    pub fn known_watched_inputs(&self, machine: &str) -> Result<Vec<KnownInput>> {
        crate::fault::check("read.watched_inputs", machine)?;
        self.known_inputs_at(machine, None, false)
    }
    pub fn known_watched_input_at(&self, machine: &str, path: &[u8]) -> Result<Option<KnownInput>> {
        Ok(self.known_inputs_at(machine, Some(path), false)?.pop())
    }

    fn known_inputs_at(
        &self,
        machine: &str,
        path: Option<&[u8]>,
        include_imported: bool,
    ) -> Result<Vec<KnownInput>> {
        // Whether an imported input counts is decided per input, not per observation. A point
        // lookup only visits inputs at its path; startup builds the set once.
        let (location, imported) = if path.is_some() {
            (
                "x.machine = ?1 AND x.path_bytes = ?2",
                "SELECT r.input FROM (
                 SELECT DISTINCT input FROM source_observation WHERE machine = ?1 AND path_bytes = ?2
             ) paths CROSS JOIN source_revision r ON r.input = paths.input
             WHERE r.produced = 'imported'",
            )
        } else {
            (
                "x.machine = ?1",
                "SELECT input FROM source_revision WHERE produced = 'imported'",
            )
        };
        let sql = format!(
            "SELECT o.path, o.input, i.format, o.class, o.device, o.inode, o.size, o.mtime_ns,
                    o.ctime_ns,
                    (SELECT chain FROM source_revision r WHERE r.input = o.input ORDER BY number DESC LIMIT 1),
                    (SELECT length FROM source_revision r WHERE r.input = o.input ORDER BY number DESC LIMIT 1),
                    o.path_bytes
             FROM (
                 SELECT MAX(x.rowid) AS latest FROM source_observation x
                 WHERE {location}
                   AND (?3 OR x.input NOT IN ({imported}))
                 GROUP BY x.path_bytes,CASE WHEN x.path_bytes IS NULL THEN x.path END
             ) latest
             JOIN source_observation o ON o.rowid = latest.latest
             JOIN source_input i ON i.id = o.input"
        );
        let mut stmt = self.conn()?.prepare_cached(&sql)?;
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
                r.get::<_, Option<Vec<u8>>>(11)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (path, input_id, format, class, f, chain, length, path_bytes) = row?;
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
                path_bytes,
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
            "SELECT p.parsed_upto, p.status, p.at_length, p.reason FROM parse_state p
             JOIN input_ref r ON r.pk = p.input_pk WHERE r.id = ?1 AND p.chain = ?2",
        )?;
        for k in &mut out {
            if let Some(chain) = k.chain
                && let Some((upto, status, at_length, reason)) = stmt
                    .query_row(params![k.input_id, chain], |r| {
                        <(i64, String, Option<i64>, Option<String>)>::try_from(r)
                    })
                    .optional()?
            {
                k.parsed_upto = upto as u64;
                k.parsed = true;
                k.waiting = stored_parse_progress(&status, at_length, reason)?.at_length()
                    == Some(k.stored);
            }
        }
        Ok(out)
    }

    /// The input's latest revision and its chain's parser progress.
    pub fn input_progress(&self, id: &str) -> Result<Option<ChainProgress>> {
        let database = crate::source::database_original("r.input");
        let row = self
            .conn()?
            .prepare_cached(
                &format!("SELECT r.length, COALESCE(p.parsed_upto, 0), p.input_pk IS NOT NULL, r.produced, COALESCE(p.status IN ('waiting','refused','failed') AND p.at_length = r.length, 0), r.chain
             FROM source_revision r
             LEFT JOIN input_ref i ON i.id = r.input
             LEFT JOIN parse_state p ON p.input_pk = i.pk AND p.chain = r.chain AND NOT {database}
             WHERE r.input = ?1 ORDER BY r.number DESC LIMIT 1"),
            )?
            .query_row([id], |r| <(i64, i64, bool, String, bool, i64)>::try_from(r))
            .optional()?;
        let Some((length, parsed_upto, parsed, produced, waiting, chain)) = row else {
            return Ok(None);
        };
        let imported = match produced.as_str() {
            "imported" => true,
            "first" | "appended" | "rewritten" => false,
            _ => {
                return Err(StoreError::stored(
                    3,
                    format!("unknown revision source: {produced}"),
                ));
            }
        };
        Ok(Some(ChainProgress {
            chain,
            imported,
            waiting,
            length: length as u64,
            parsed_upto: parsed_upto as u64,
            parsed,
        }))
    }
}

/// The record a stored marker names, read once for the markers that name it.
#[derive(Default)]
struct StoredRecords {
    last: Option<((u64, u64), serde_json::Value)>,
}

impl StoredRecords {
    fn value(
        &mut self,
        reader: &Reader<'_>,
        input: &str,
        chain: i64,
        start: u64,
        end: u64,
        pointer: &str,
    ) -> Result<serde_json::Value> {
        if self
            .last
            .as_ref()
            .is_none_or(|(range, _)| *range != (start, end))
        {
            let store = reader.blob_store()?;
            let bytes = reader.stored_range(input, chain, start, end, &store)?;
            let record: serde_json::Value = serde_json::from_slice(bytes.trim_ascii())?;
            self.last = Some(((start, end), record));
        }
        let (_, record) = self.last.as_ref().expect("just read");
        record.pointer(pointer).cloned().ok_or_else(|| {
            StoreError::stored_data(format!("the stored record holds no value at {pointer}"))
        })
    }
}

#[cfg(test)]
mod tests;

impl Reader<'_> {
    /// Execute against one caller-supplied calendar, restoring any enclosing
    /// query's calendar even when the callback unwinds.
    pub fn with_clock<T>(&self, clock: &query_language::time::Clock, f: impl FnOnce() -> T) -> T {
        self.calendar.with_clock(clock, f)
    }
}

/// The sessions whose authored titles changed after archive position
/// `after`: a title written, revised or withdrawn changes how its session is
/// shown, though no session row changed.
pub(crate) fn retitled_sessions(
    conn: &rusqlite::Connection,
    after: i64,
) -> Result<std::collections::BTreeSet<String>> {
    let mut sessions = std::collections::BTreeSet::new();
    let mut statement = conn.prepare_cached(
        "SELECT DISTINCT substr(l.ref, 9) FROM source_annotation a \
         JOIN source_annotation_link l ON l.record = a.id AND l.role = 'about' \
         WHERE a.root = ?1 AND a.kind = 'title' AND l.ref LIKE 'session:%'",
    )?;
    for root in crate::annotations::changed_roots(conn, after)? {
        for id in statement.query_map([&root], |r| r.get::<_, String>(0))? {
            sessions.insert(id?);
        }
    }
    Ok(sessions)
}
