//! Read connections. The daemon keeps a pool; every read in one request
//! sees one consistent read transaction on `index.db` with `archive.db`
//! attached.

use crate::ConnExt as _;
use crate::archive::Archive;
use crate::archive::index_build::Reads;
use crate::blobs::FileFacts;
use crate::row::{Json, MessagePk, Rows as _, SessionPk, Statement, row};
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

row! {
    /// Both write counters and the index generation.
    struct ChangePosition {
        generation: String,
        index: i64,
        archive: i64,
    }
    statements {
        const CURRENT = "SELECT (SELECT value FROM main.meta WHERE key='generation') AS generation,
            COALESCE((SELECT CAST(value AS INTEGER) FROM main.meta WHERE key='write_counter'),0) AS \"index\",
            COALESCE((SELECT CAST(value AS INTEGER) FROM source_meta WHERE key='write_counter'),0) AS archive";
    }
}

row! {
    /// The shown message and the input and chain its reading came from.
    struct ShownSource {
        pk: MessagePk,
        input: String,
        source_chain: Option<i64>,
    }
    statements {
        const OF_MESSAGE = "SELECT m.pk, i.id AS input, m.source_chain FROM message m
            JOIN input_ref i ON i.pk=m.input_pk WHERE m.id=?1";
    }
}

row! {
    /// The commit facts of a shown message's reading.
    struct MessageCommit {
        committed: Option<i64>,
        input: String,
        source_chain: i64,
        source_end: i64,
    }
    statements {
        const OF_MESSAGE = "SELECT (SELECT max(r.committed) FROM message_parse r WHERE r.message_id=m.id AND r.hash=m.version_hash) AS committed,
            i.id AS input, m.source_chain, m.source_end
            FROM message m JOIN input_ref i ON i.pk=m.input_pk WHERE m.id=?1 AND m.version_hash IS NOT NULL";
    }
}

row! {
    /// Where a visit from a message starts in its session's order.
    struct MessageStart {
        session_order: String,
        id: String,
    }
    statements {
        const OF_MESSAGE = "SELECT session_order, id FROM message WHERE pk = ?1 AND session_pk = ?2";
    }
}

/// The columns of [`MessageRow`] before `record_order`.
macro_rules! message_columns {
    () => {
        "SELECT m.pk, m.id, m.role, m.prompt_kind, m.model, m.at_time, m.parent_id,
            m.input_tokens, m.cache_read_tokens, m.cache_write_tokens, m.output_tokens,
            m.reasoning_tokens, m.service_tier, m.stop_reason, m.duration_ms, m.attrs_json,
            m.cache_write_1h_tokens, m.speed, m.version_hash, m.layout,
            CASE WHEN m.layout IS NULL THEN
                (SELECT count(*) FROM block WHERE message_pk=m.pk) ELSE 0 END AS stored_block_count,
            (SELECT max(r.committed) FROM message_parse r WHERE r.message_id=m.id AND r.hash=m.version_hash) AS committed,
            CASE WHEN m.version_hash IS NOT NULL THEN source_input.id END AS source_input,
            CASE WHEN m.version_hash IS NOT NULL THEN m.source_chain END AS source_chain,
            CASE WHEN m.version_hash IS NOT NULL THEN m.source_end END AS source_end, "
    };
}

row! {
    /// A shown message's stored columns, read in session order.
    struct MessageRow {
        pk: MessagePk,
        id: MessageId,
        role: Role,
        prompt_kind: Option<PromptKind>,
        model: Option<String>,
        at_time: Json<(Option<polylogue_model::Timestamp>, polylogue_model::TimeAccuracy)>,
        parent_id: Option<MessageId>,
        input_tokens: Option<u64>,
        cache_read_tokens: Option<u64>,
        cache_write_tokens: Option<u64>,
        output_tokens: Option<u64>,
        reasoning_tokens: Option<u64>,
        service_tier: Option<String>,
        stop_reason: Option<String>,
        duration_ms: Option<u64>,
        attrs_json: Option<String>,
        cache_write_1h_tokens: Option<u64>,
        speed: Option<String>,
        version_hash: Option<Vec<u8>>,
        layout: Option<Vec<u8>>,
        stored_block_count: i64,
        committed: Option<i64>,
        source_input: Option<String>,
        source_chain: Option<i64>,
        source_end: Option<i64>,
        record_order: Option<i64>,
    }
    statements {
        // Session key ?1; a start message's order and ID ?2 and ?3; a given
        // reading position the last parameter.
        const GIVEN_ORDER = concat!(
            message_columns!(),
            "?2 AS record_order
            FROM message m LEFT JOIN input_ref source_input ON source_input.pk=m.input_pk
            WHERE m.session_pk = ?1 AND m.duplicate_of IS NULL ORDER BY m.session_order, m.id"
        );
        const GIVEN_ORDER_FROM = concat!(
            message_columns!(),
            "?4 AS record_order
            FROM message m LEFT JOIN input_ref source_input ON source_input.pk=m.input_pk
            WHERE m.session_pk = ?1 AND m.duplicate_of IS NULL AND (m.session_order, m.id) >= (?2, ?3)
            ORDER BY m.session_order, m.id"
        );
        const OWN_ORDER = concat!(
            order::record_ranks!(),
            message_columns!(),
            "record_ranks.record_order
            FROM message m LEFT JOIN input_ref source_input ON source_input.pk=m.input_pk
            JOIN record_ranks ON record_ranks.pk=m.pk
            WHERE m.session_pk = ?1 AND m.duplicate_of IS NULL ORDER BY m.session_order, m.id"
        );
        const OWN_ORDER_FROM = concat!(
            order::record_ranks!(),
            message_columns!(),
            "record_ranks.record_order
            FROM message m LEFT JOIN input_ref source_input ON source_input.pk=m.input_pk
            JOIN record_ranks ON record_ranks.pk=m.pk
            WHERE m.session_pk = ?1 AND m.duplicate_of IS NULL AND (m.session_order, m.id) >= (?2, ?3)
            ORDER BY m.session_order, m.id"
        );
    }
}

row! {
    /// A count kept under a name.
    struct NamedCount {
        name: String,
        n: u64,
    }
    statements {
        const PARSE_STATES = "SELECT status AS name, n FROM parse_status_count WHERE n > 0";
        const UNUSED = "SELECT what AS name, n FROM unused_count WHERE n > 0 ORDER BY n DESC, what LIMIT 20";
    }
}

row! {
    /// An input's latest revision.
    struct LatestRevision {
        chain: i64,
        length: i64,
    }
    statements {
        const OF_INPUT = "SELECT chain, length FROM source_revision WHERE input=?1 ORDER BY number DESC LIMIT 1";
    }
}

row! {
    /// A path's latest observation, its input's origin and latest revision.
    struct Observation {
        input: String,
        origin: String,
        class: String,
        device: Option<i64>,
        inode: Option<i64>,
        size: Option<i64>,
        mtime_ns: Option<i64>,
        ctime_ns: Option<i64>,
        chain: Option<i64>,
        length: Option<i64>,
        path_bytes: Vec<u8>,
    }
}

row! {
    /// A chain's parse standing.
    struct ParseProgressRow {
        parsed_upto: i64,
        status: String,
        at_length: Option<i64>,
        reason: Option<String>,
    }
    statements {
        const OF_CHAIN = "SELECT p.parsed_upto, p.status, p.at_length, p.reason FROM parse_state p
            JOIN input_ref r ON r.pk = p.input_pk WHERE r.id = ?1 AND p.chain = ?2";
    }
}

row! {
    /// An input's latest revision and its chain's parse progress.
    struct InputProgress {
        length: i64,
        parsed_upto: i64,
        parsed: bool,
        produced: String,
        waiting: bool,
        chain: i64,
    }
}

/// Read-only connections to one archive, reused across requests.
/// `max_readers` connections serve every read; one more is kept for the
/// [`Lane::Interactive`] checkouts, so status, completion and brief answers
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

/// Which connections a checkout may take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    /// Answers that keep a budget while bulk reads hold every other
    /// connection: they may also take the reserved one.
    Interactive,
    Ordinary,
    /// Idle-priority work: one thread at a time holds such readers.
    Background,
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
    /// Readers held in the background lane, by thread.
    background: HashMap<ThreadId, usize>,
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
    background: bool,
    limit: Option<limit::ReadLimit>,
    scope: scope::ReadScopeToken,
    // Counting holders by ThreadId detects self-exhaustion only if live readers stay
    // on their checkout thread. The marker makes that invariant enforceable.
    thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
    vectors: std::cell::RefCell<Option<crate::vectors::Vectors>>,
    calendar: crate::sql_functions::CalendarContext,
    annotation_history_ready: std::cell::RefCell<std::collections::BTreeSet<i64>>,
    /// How many times the read was ended for a newer one ([`Self::restarts`]).
    restarts: std::cell::Cell<u64>,
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
    /// The location's bytes; `path` shows them as text.
    pub path_bytes: Vec<u8>,
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
    /// Bytes of input content stored, decoded: what the inputs hold, not
    /// what the segment log takes on disk.
    pub stored_bytes: u64,
    /// Input chains by parse status (`done`, `parsing`, `waiting` for growth).
    pub parse_states: BTreeMap<String, u64>,
    /// The 20 record kinds and fields parsers did not read most often.
    pub unused: Vec<(String, u64)>,
    /// How many record kinds and fields parsers did not read, all of them.
    pub unused_fields: u64,
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

    /// A request's reader: waits for a connection of its lane until the
    /// request's deadline or cancellation, and reads under `limit`. Reuse the
    /// existing Reader within a request. Nested checkouts are allowed while
    /// capacity remains; waiting while this thread already holds a slot
    /// refuses to avoid deadlock.
    pub fn checkout(&self, lane: Lane, limit: &limit::ReadLimit) -> Result<Reader<'_>> {
        self.acquire(lane, Some(limit.clone()), |path| self.connect(path))
    }

    /// A reader for work outside a request (the loop, workers): the ordinary
    /// lane, under the limit of the scope this thread runs in, if any.
    pub fn reader(&self) -> Result<Reader<'_>> {
        self.acquire(Lane::Ordinary, limit::ReadLimit::current(), |path| {
            self.connect(path)
        })
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

    fn acquire(
        &self,
        lane: Lane,
        limit: Option<limit::ReadLimit>,
        connect: impl FnOnce(&Path) -> Result<Connection>,
    ) -> Result<Reader<'_>> {
        // Refused before a slot is taken: nothing can be read until the build ends.
        self.index_path()?;
        let scope = self.archive.read_scope.token()?;
        if let Some(limit) = &limit {
            limit.bind_scope(scope.clone())?;
        }
        let holder = std::thread::current().id();
        let capacity = self.max_readers + usize::from(lane == Lane::Interactive);
        let background = lane == Lane::Background;
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
                let lane_free = !background
                    || state.background.is_empty()
                    || state.background.contains_key(&holder);
                if state.in_use < capacity && lane_free {
                    state.in_use += 1;
                    *state.holders.entry(holder).or_default() += 1;
                    if background {
                        *state.background.entry(holder).or_default() += 1;
                    }
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
            background,
            limit,
            scope,
            thread_bound: std::marker::PhantomData,
            vectors: std::cell::RefCell::new(None),
            calendar: crate::sql_functions::CalendarContext::new(),
            annotation_history_ready: std::cell::RefCell::new(Default::default()),
            restarts: std::cell::Cell::new(0),
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
        if self.background
            && let Some(held) = state.background.get_mut(&self.holder)
        {
            *held -= 1;
            if *held == 0 {
                state.background.remove(&self.holder);
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

    /// How many times this reader ended its read so that the next access
    /// starts a newer one: a transcript read waits for a due session's
    /// summary that way. Work that read several things before must read
    /// them again when this moved, so that all of it comes from one read.
    pub fn restarts(&self) -> u64 {
        self.restarts.get()
    }

    /// Ends the read in progress; the next access starts a new one.
    pub(crate) fn restart(&self) -> Result<()> {
        if self.state.get() == ReadState::Ready {
            self.conn
                .as_ref()
                .expect("reader holds a connection until dropped")
                .execute_batch("ROLLBACK; BEGIN")?;
            self.state.set(ReadState::Deferred);
            self.annotation_history_ready.borrow_mut().clear();
        }
        self.restarts.set(self.restarts.get() + 1);
        Ok(())
    }

    /// The request limit this reader reads under, if any.
    pub(crate) fn limit(&self) -> Option<&limit::ReadLimit> {
        self.limit.as_ref()
    }

    pub(crate) fn archive(&self) -> &Archive {
        &self.pool.archive
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
        let position = self.conn()?.one(&ChangePosition::CURRENT, [])?;
        Ok(polylogue_model::ChangePosition {
            generation: position.generation,
            index: position.index,
            archive: position.archive,
        })
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
        let Some(ShownSource {
            pk,
            input,
            source_chain: chain,
        }) = conn.maybe(&ShownSource::OF_MESSAGE, [id.to_string()])?
        else {
            return Ok(None);
        };
        let (mut message, selection) = crate::writer::shown_message(conn, pk.0)?
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
        match conn.maybe(&MessageCommit::OF_MESSAGE, [id.to_string()])? {
            Some(commit) => MessageArrival {
                order: None,
                fact: commit.committed,
                source: Some((commit.input, commit.source_chain, commit.source_end)),
            }
            .resolve(conn),
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
    pub fn message_keys_with_parents(&self, id: &SessionId) -> Result<Vec<MessagePk>> {
        let selected = self.select_messages(id, true)?;
        selected.require_complete()?;
        selected.keys()
    }

    pub(crate) fn session_pk(&self, id: &SessionId) -> Result<Option<SessionPk>> {
        self.conn()?
            .value("SELECT pk FROM session WHERE id = ?1", [id.to_string()])
    }

    /// A session's own message keys and IDs in reading order.
    fn own_keys(&self, session: SessionPk) -> Result<Vec<parents::OwnKey>> {
        parents::own_keys(self.conn()?, session)
    }

    /// A message's blocks; without `structured`, tool results lack their
    /// structured data (`block.data`), which is then neither read nor decoded.
    fn message_blocks(&self, message: MessagePk, structured: bool) -> Result<Vec<Block>> {
        crate::ordered_blocks::of_message(self.conn()?, message.0, structured)
    }

    /// Visits the session's messages in order, from message `from` (a key)
    /// when given, without reading the messages before it.
    fn visit_messages_where(
        &self,
        session_pk: SessionPk,
        session: &SessionId,
        from: Option<MessagePk>,
        content: MessageContent<'_>,
        mut keep: impl FnMut(MessagePk) -> bool,
        visit: &mut impl FnMut(
            MessagePk,
            Message,
            Option<ShownBlocks>,
            MessageArrival,
        ) -> std::ops::ControlFlow<()>,
    ) -> Result<std::ops::ControlFlow<()>> {
        use rusqlite::types::Value;
        let start = match from {
            Some(pk) => self
                .conn()?
                .maybe(&MessageStart::OF_MESSAGE, params![pk, session_pk])?,
            None => None,
        };
        if from.is_some() && start.is_none() {
            return Err(polylogue_model::Reason::ReadStartOutsideSession {}.into());
        }
        let ordinal = match content.order {
            RecordOrder::Skip => Some(Value::Null),
            RecordOrder::Known(value) => Some(Value::Integer(value)),
            RecordOrder::OwnSession => None,
        };
        let mut parameters = vec![Value::Integer(session_pk.0)];
        if let Some(start) = &start {
            parameters.push(Value::Text(start.session_order.clone()));
            parameters.push(Value::Text(start.id.clone()));
        }
        let statement = match (&start, ordinal) {
            (None, None) => &MessageRow::OWN_ORDER,
            (Some(_), None) => &MessageRow::OWN_ORDER_FROM,
            (None, Some(ordinal)) => {
                parameters.push(ordinal);
                &MessageRow::GIVEN_ORDER
            }
            (Some(_), Some(ordinal)) => {
                parameters.push(ordinal);
                &MessageRow::GIVEN_ORDER_FROM
            }
        };
        self.conn()?.each(statement, rusqlite::params_from_iter(parameters), |row| {
            let pk = row.pk;
            if !keep(pk) {
                return Ok(std::ops::ControlFlow::Continue(()));
            }
            let named = row.id.to_string();
            let unreadable =
                |what: &str| StoreError::stored(0, format!("message {named} has {what}"));
            let attrs: Option<serde_json::Value> = row
                .attrs_json
                .map(|attrs| serde_json::from_str(&attrs))
                .transpose()
                .map_err(|_| unreadable("unparseable attributes"))?;
            let (role, kind) = (row.role, row.prompt_kind);
            let visible = match content.visibility {
                None => true,
                Some(filter) if filter.prompts && !polylogue_model::is_prompt(role, kind) => false,
                Some(filter) if filter.kinds.is_empty() => false,
                Some(filter) => self
                    .conn()?
                    .value(
                        &format!(
                            "SELECT EXISTS(SELECT 1 FROM block WHERE message_pk=?1 AND kind IN ({}))",
                            filter.kinds
                        ),
                        [pk],
                    )?
                    .unwrap_or(false),
            };
            let mut blocks = if visible {
                self.message_blocks(pk, content.structured)?
            } else {
                Vec::new()
            };
            // A read of whole values shows what the reading left in the
            // stored bytes.
            if content.structured
                && let (Some(input), Some(chain)) = (&row.source_input, row.source_chain)
            {
                self.resolve_stored_blocks(input, chain, &mut blocks)?;
            }
            let Json((at, accuracy)) = row.at_time;
            let message = Message {
                id: row.id,
                session: session.clone(),
                parent: row.parent_id,
                role,
                at,
                at_quality: accuracy.quality,
                at_precision: accuracy.precision,
                at_until: accuracy.until,
                model: row.model,
                prompt_kind: kind,
                blocks,
                usage: Usage {
                    input: row.input_tokens,
                    cached_input: row.cache_read_tokens,
                    cache_write: row.cache_write_tokens,
                    output: row.output_tokens,
                    reasoning: row.reasoning_tokens,
                    cache_write_1h: row.cache_write_1h_tokens,
                    service_tier: row.service_tier,
                    speed: row.speed,
                },
                stop_reason: row.stop_reason,
                reported_cost_usd: attrs
                    .as_ref()
                    .and_then(|a| a.get("reported_cost_usd"))
                    .and_then(serde_json::Value::as_number)
                    .cloned(),
                duration_ms: row.duration_ms,
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
            let block_selection = row
                .version_hash
                .map(|hash| {
                    let mut block_selection = crate::encoding::message_block_selection(
                        &hash,
                        row.layout.as_deref(),
                        if visible {
                            message.blocks.len()
                        } else {
                            usize::try_from(row.stored_block_count).map_err(|_| {
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
                order: row.record_order,
                fact: row.committed,
                source: row
                    .source_input
                    .zip(row.source_chain)
                    .zip(row.source_end)
                    .map(|((input, chain), end)| (input, chain, end)),
            };
            Ok(visit(pk, message, block_selection, commit))
        })
    }

    pub fn status(&self) -> Result<Status> {
        limit::ReadLimit::set_stage("status counts");
        let c = self.conn()?;
        let count = |sql: &str| -> Result<u64> { Ok(c.scalar::<i64>(sql, [])? as u64) };
        let parse_states = c
            .all(&NamedCount::PARSE_STATES, [])?
            .into_iter()
            .map(|count| (count.name, count.n))
            .collect();
        let unused = c
            .all(&NamedCount::UNUSED, [])?
            .into_iter()
            .map(|count| (count.name, count.n))
            .collect();
        Ok(Status {
            sessions: count("SELECT n FROM main.row_count WHERE name = 'session'")?,
            messages: count("SELECT n FROM main.row_count WHERE name = 'message'")?,
            documents: count("SELECT n FROM main.row_count WHERE name = 'document'")?,
            tool_calls: count("SELECT n FROM main.row_count WHERE name = 'tool_call'")?,
            inputs: count("SELECT count FROM source_row_count WHERE name = 'input'")?,
            revisions: count("SELECT count FROM source_row_count WHERE name = 'revision'")?,
            stored_bytes: count("SELECT count FROM source_row_count WHERE name = 'stored_bytes'")?,
            parse_states,
            unused,
            unused_fields: count("SELECT count(*) FROM unused_count WHERE n > 0")?,
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
            let latest = self
                .conn()?
                .maybe(&LatestRevision::OF_INPUT, [&asset.input])?;
            if latest
                != Some(LatestRevision {
                    chain: asset.chain,
                    length: 0,
                })
            {
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
            "SELECT o.input, i.origin, o.class, o.device, o.inode,
                    CASE WHEN nr.commit_seq > o.commit_seq AND o.class <> 'deleted' THEN nr.length ELSE o.size END AS size,
                    CASE WHEN nr.commit_seq > o.commit_seq AND o.class <> 'deleted' THEN nr.mtime_ns ELSE o.mtime_ns END AS mtime_ns,
                    CASE WHEN nr.commit_seq > o.commit_seq AND o.class <> 'deleted' THEN nr.ctime_ns ELSE o.ctime_ns END AS ctime_ns,
                    (SELECT chain FROM source_revision r WHERE r.input = o.input ORDER BY number DESC LIMIT 1) AS chain,
                    (SELECT length FROM source_revision r WHERE r.input = o.input ORDER BY number DESC LIMIT 1) AS length,
                    o.path_bytes
             FROM (
                 SELECT MAX(x.id) AS latest FROM source_observation x
                 WHERE {location}
                   AND (?3 OR x.input NOT IN ({imported}))
                 GROUP BY x.path_bytes
             ) latest
             JOIN source_observation o ON o.id = latest.latest
             JOIN source_input i ON i.id = o.input
             LEFT JOIN source_revision nr ON nr.input = o.input
               AND nr.number = (SELECT max(number) FROM source_revision WHERE input = o.input)"
        );
        let observed = self.conn()?.all(
            &Statement::<Observation>::built(sql),
            params![machine, path, include_imported],
        )?;
        let mut out = Vec::new();
        for o in observed {
            let f = [o.device, o.inode, o.size, o.mtime_ns, o.ctime_ns];
            let (input_id, format, chain, length, path_bytes) =
                (o.input, o.origin, o.chain, o.length, o.path_bytes);
            let path = String::from_utf8_lossy(&path_bytes).into_owned();
            let gone = crate::input_rows::ObservationClass::from_name(&o.class)
                == Some(crate::input_rows::ObservationClass::Deleted);
            let facts = match (gone, f) {
                (true, _) => None,
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
        for k in &mut out {
            if let Some(chain) = k.chain
                && let Some(state) = self
                    .conn()?
                    .maybe(&ParseProgressRow::OF_CHAIN, params![k.input_id, chain])?
            {
                k.parsed_upto = state.parsed_upto as u64;
                k.parsed = true;
                k.waiting = stored_parse_progress(&state.status, state.at_length, state.reason)?
                    .at_length()
                    == Some(k.stored);
            }
        }
        Ok(out)
    }

    /// The input's latest revision and its chain's parser progress.
    pub fn input_progress(&self, id: &str) -> Result<Option<ChainProgress>> {
        let database = crate::source::database_original("r.input");
        let statement = Statement::<InputProgress>::built(format!(
            "SELECT r.length, COALESCE(p.parsed_upto, 0) AS parsed_upto, p.input_pk IS NOT NULL AS parsed,
                r.produced,
                COALESCE(p.status IN ('waiting','refused','failed') AND p.at_length = r.length, 0) AS waiting,
                r.chain
             FROM source_revision r
             LEFT JOIN input_ref i ON i.id = r.input
             LEFT JOIN parse_state p ON p.input_pk = i.pk AND p.chain = r.chain AND NOT {database}
             WHERE r.input = ?1 ORDER BY r.number DESC LIMIT 1"
        ));
        let Some(InputProgress {
            length,
            parsed_upto,
            parsed,
            produced,
            waiting,
            chain,
        }) = self.conn()?.maybe(&statement, [id])?
        else {
            return Ok(None);
        };
        let imported = crate::input_rows::Produced::from_name(&produced)
            .ok_or_else(|| StoreError::stored(3, format!("unknown revision source: {produced}")))?
            == crate::input_rows::Produced::Imported;
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

/// The rows this module declares statements for, for the every-statement test.
#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<ChangePosition>();
    check.row::<ShownSource>();
    check.row::<MessageCommit>();
    check.row::<MessageStart>();
    check.row::<MessageRow>();
    check.row::<NamedCount>();
    check.row::<LatestRevision>();
    check.row::<ParseProgressRow>();
    parents::declared_rows(check);
    trace::declared_rows(check);
    revisions::declared_rows(check);
    record_seek::declared_rows(check);
    tool_outcomes::declared_rows(check);
    captured::declared_rows(check);
    hook_fields::declared_rows(check);
    tool_pairs::declared_rows(check);
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
    for root in crate::annotations::changed_roots(conn, after)? {
        sessions.extend(conn.values::<String>(
            "SELECT DISTINCT substr(l.ref, 9) FROM source_annotation a \
             JOIN source_annotation_link l ON l.annotation = a.id AND l.role = 'about' \
             WHERE a.root = ?1 AND a.kind = 'title' AND l.ref LIKE 'session:%'",
            [&root],
        )?);
    }
    Ok(sessions)
}
