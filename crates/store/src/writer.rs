//! The one writer. It runs on its own thread, owns
//! the only writing connections to `archive.db` and `index.db`, and writes
//! batches from the parse pool in large transactions: first the bytes'
//! rows in `archive.db` (inputs, revisions, extents, observations, inline
//! blobs), then everything parsed from them in `index.db` (sessions,
//! messages and their versions, blocks, search entries, tool calls,
//! events, links) together with the parse position they cover.
//!
//! `archive.db` commits before `index.db`, so an index row never refers to
//! bytes that are not stored. A crash between the two leaves the index
//! behind its bytes; the loop parses them again, and every write here is an
//! upsert, so that changes nothing.

use crate::archive::{Archive, bump_counter, open_write};
use crate::blobs::{Change, FileFacts, NewBlob};
use crate::{Result, StoreError};
use polylogue_model::{
    BlobHash, Block, ErrorReason, Event, Json, Link, Message, MessageRow, Output, RecordCounts,
    Role, Session, SessionId, Timestamp, ToolAction, ToolOutcome,
};
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

mod attachments;
mod batch;
mod bytes;
mod chain;
mod confirmed;
mod primary;
mod primary_capture;
pub(crate) use confirmed::apply_confirmed_readings;
pub(crate) mod resolve;
mod rows;
mod search;
mod staging;
pub use chain::ChainReading;

#[cfg(test)]
use crate::encoding::pack_block;
use crate::encoding::{PreparedDocument, PreparedMessage};
use crate::encoding::{block_text, unpack_block};
pub use batch::{
    Batch, ImportMember, InContainer, InFlight, InputKey, Observation, ParseStep, WriteReport,
};
use bytes::*;
use rows::*;
use search::*;

enum Msg {
    StoreBytes(
        Box<crate::staging::PreparedBytes>,
        Box<Batch>,
        SyncSender<Result<crate::staging::StoredInput>>,
    ),
    StoreGroup(
        Vec<crate::staging::PreparedBatch>,
        std::sync::Arc<crate::blobs::Blobs>,
        SyncSender<Vec<Result<crate::staging::StoredInput>>>,
    ),
    Assets(
        Box<crate::export_items::ExportItems>,
        SyncSender<Result<()>>,
    ),
    Coverage(
        Box<polylogue_model::capture::Coverage>,
        SyncSender<Result<()>>,
    ),
    /// Bytes no input holds (a chat action's attachment), as one blob.
    StoreBlob(
        Box<(crate::NewBlob, Option<crate::staging::PreparedBytes>)>,
        SyncSender<Result<BlobHash>>,
    ),
    ReplaceChain(InputKey, i64, u64, ChainReading, SyncSender<Result<u64>>),
    Confirmed(Vec<crate::primary::Confirmed>, SyncSender<Result<i64>>),
    SynchronizePrimary(SyncSender<Result<i64>>),
    Stop,
    /// A user write is waiting in its own queue.
    Wake,
    Batch(Box<Batch>),
    Flush(SyncSender<WriteReport>, bool),
    /// Run the derive step over sessions whose derived rows are stale;
    /// answered with the number of sessions derived.
    Derive(SyncSender<Result<u64>>),
}

/// Handle to the writer thread. Dropping it stops the thread after the
/// batches already sent are written.
pub struct Writer {
    tx: Option<WriterQueue>,
    handle: Option<JoinHandle<()>>,
    /// The index was empty when the writer started: its first pass is a
    /// first build.
    started_empty: bool,
    resolver: Arc<resolve::Resolver>,
}

/// Orders new work against shutdown. Once the stop message is queued,
/// every remaining sender refuses new work rather than acknowledging a
/// message that the receiver will never process.
///
/// User writes have a queue of their own, which the writer serves before
/// its next piece of bulk work: a note, setting, delivery or browser
/// request waits for at most the transaction being written, not behind a
/// first build's queued batches.
#[derive(Clone)]
struct WriterQueue {
    bulk: Arc<Mutex<Option<SyncSender<Msg>>>>,
    users: Arc<Mutex<Option<SyncSender<UserMsg>>>>,
    resolver: Arc<resolve::Resolver>,
}

type UserMsg = (
    Box<polylogue_model::user::UserRequest>,
    SyncSender<Result<polylogue_model::user::UserOutcome>>,
);

/// User writes waiting before their senders block.
const USER_QUEUE: usize = 64;

impl WriterQueue {
    fn new(
        bulk: SyncSender<Msg>,
        users: SyncSender<UserMsg>,
        resolver: Arc<resolve::Resolver>,
    ) -> Self {
        Self {
            bulk: Arc::new(Mutex::new(Some(bulk))),
            users: Arc::new(Mutex::new(Some(users))),
            resolver,
        }
    }

    fn send(&self, message: Msg) -> std::result::Result<(), ()> {
        let counted = matches!(&message, Msg::StoreGroup(..) | Msg::Batch(..));
        if counted {
            crate::timing::queue(crate::timing::Queue::Writer, 1);
        }
        let result = (|| {
            self.bulk
                .lock()
                .map_err(|_| ())?
                .as_ref()
                .ok_or(())?
                .send(message)
                .map_err(|_| ())
        })();
        if result.is_err() && counted {
            crate::timing::queue(crate::timing::Queue::Writer, -1);
        }
        result
    }

    fn send_user(&self, message: UserMsg) -> std::result::Result<(), ()> {
        self.users
            .lock()
            .map_err(|_| ())?
            .as_ref()
            .ok_or(())?
            .send(message)
            .map_err(|_| ())?;
        // Wakes an idle writer. While a bulk sender holds the lock the queue
        // is full and the writer busy; it serves user writes before its next
        // bulk message anyway.
        if let Ok(bulk) = self.bulk.try_lock()
            && let Some(bulk) = bulk.as_ref()
        {
            let _ = bulk.try_send(Msg::Wake);
        }
        Ok(())
    }

    fn stop(&self) {
        if let Ok(mut users) = self.users.lock() {
            users.take();
        }
        if let Ok(mut accepting) = self.bulk.lock()
            && let Some(sender) = accepting.take()
        {
            let _ = sender.send(Msg::Stop);
        }
    }
}

/// Rows written per transaction before committing; a transaction also
/// commits whenever no batch is waiting.
const ROWS_PER_TRANSACTION: usize = 50_000;

/// Secondary indexes no statement of the writer uses: a first build drops
/// them and builds each once at its end ([`Inner::build_deferred`]).
/// Checked with `EXPLAIN QUERY PLAN` over the store's statements
/// (2026-10-19).
const READ_ONLY_INDEXES: &[&str] = &[
    "message_at",
    "message_session_at",
    "message_model_at",
    "message_session_model_at",
    "event_at",
    "session_activity",
    "session_started",
    "session_cwd",
    "session_repo",
    "session_live",
    "message_prose",
    "message_branch",
    "tool_call_tool",
    "tool_call_action",
    "tool_call_failed",
    "tool_call_path",
    "tool_call_repo_command",
    "link_child",
    "event_after",
    "event_kind",
    "commit_link_hash",
    "pr_link_number",
    "work_node_session",
    "work_node_verification",
    "removed_session_changed",
    "message_pem",
];

/// Unique indexes on natural keys, which the writer's statements use: a first
/// build drops them too, finds rows by key in memory meanwhile
/// ([`crate::first_build`]), and builds them first when it ends.
const KEY_INDEXES: &[&str] = &["message_id", "event_id", "text_hash"];

/// Batches waiting for the writer before senders block.
const QUEUE: usize = 4;

/// Input bytes a transaction covers before committing. A batch's bytes
/// stay counted against the parse pool's limit until its transaction
/// commits ([`InFlight`]); committing well before that limit lets the pool
/// keep parsing while the writer commits, instead of the two alternating.
const BYTES_PER_TRANSACTION: u64 = 128 << 20;

pub(crate) fn deferred_index_names() -> impl Iterator<Item = &'static str> {
    KEY_INDEXES.iter().chain(READ_ONLY_INDEXES).copied()
}

impl Writer {
    /// Starts the writer thread on `archive`, which the caller has locked.
    /// A first build defers search and secondary indexes. Each group commits
    /// durably, and an interrupted build resumes its saved rows and positions.
    pub fn start(archive: &Archive) -> Result<Self> {
        Self::start_with(archive, None)
    }

    /// Starts the writer on an archive Sinex holds: every register row is
    /// sent to `sinex` first and commits only once Sinex acknowledged it.
    pub fn start_on_sinex(
        archive: &Archive,
        sinex: std::sync::Arc<dyn crate::primary::Sinex>,
    ) -> Result<Self> {
        Self::start_with(archive, Some(sinex))
    }

    fn start_with(
        archive: &Archive,
        sinex: Option<std::sync::Arc<dyn crate::primary::Sinex>>,
    ) -> Result<Self> {
        if archive.is_primary() != sinex.is_some() {
            return Err(StoreError::Invalid(
                "the writer mode must match its archive".into(),
            ));
        }
        let index = open_write(&archive.index_db(), crate::archive::cache::INDEX_WRITER)?;
        if !archive.is_primary() {
            let uri = format!("file:{}?mode=ro", archive.archive_db().display());
            index.execute("ATTACH DATABASE ?1 AS a", [uri])?;
            crate::source::views(&index)?;
        }
        let empty: bool =
            index.query_row("SELECT NOT EXISTS (SELECT 1 FROM parse_state)", [], |r| {
                r.get(0)
            })?;
        let building = crate::schema::building(&index)?;
        if building {
            // Parsed rows and their positions survive a lost process or boot.
            index.pragma_update(None, "synchronous", "full")?;
            index.execute("INSERT INTO meta(key,value) VALUES ('building','1') ON CONFLICT(key) DO UPDATE SET value=excluded.value", [])?;
        }
        // A first build inserts rows without maintaining the indexes only
        // reads use, or the unique indexes on natural keys; each is built
        // once from a sorted scan when the build ends. The writer's other
        // indexes stay, and an index missing from these lists is kept, which
        // is the safe default.
        let deferred_indexes = if building {
            let expected = Connection::open_in_memory()?;
            expected.execute_batch(crate::schema::INDEX)?;
            expected.execute_batch(crate::schema::READ_INDEXES)?;
            let mut defs = Vec::new();
            for name in deferred_index_names() {
                let sql: Option<String> = expected
                    .query_row(
                        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?1",
                        [name],
                        |r| r.get(0),
                    )
                    .optional()?;
                if let Some(sql) = sql {
                    index.execute(&format!("DROP INDEX IF EXISTS \"{name}\""), [])?;
                    defs.push(sql);
                }
            }
            defs
        } else {
            Vec::new()
        };
        let archive_conn = if archive.is_primary() {
            crate::archive::open_read(&archive.index_db())?
        } else {
            crate::archive::open_archive_writer(archive)?
        };
        let first_build = building
            .then(|| crate::first_build::FirstBuild::load(&index).map(Arc::new))
            .transpose()?;
        let segment = if sinex.is_none() {
            Some(crate::blobs::open_writer(&archive_conn, &archive.root)?)
        } else {
            None
        };
        let resolver = resolve::Resolver::new(archive, &index, first_build.clone())?;
        let mut inner = Inner {
            resolver: Arc::clone(&resolver),
            segment,
            archive: archive_conn,
            index,
            report: WriteReport::default(),
            first_build,
            deferred_indexes,
            derive_due: true,
            derived_version: None,
            search: Search::start(archive.index_db(), !building)?,
            location: archive.clone(),
            sinex,
        };
        // Senders block once a few batches wait. The writer is the slowest
        // stage of a first build, and a parsed batch holds its rows and
        // packed bodies; a deep queue let finished batches pile up past
        // the parse pool's bytes-in-flight limit (8.4 GB on the real
        // corpus). With a short queue the pool keeps a job's bytes counted
        // until its batch is queued, so memory stays bounded by that limit.
        let (tx, rx) = sync_channel::<Msg>(QUEUE);
        let (users, users_rx) = sync_channel::<UserMsg>(USER_QUEUE);
        let handle = std::thread::Builder::new()
            .name("polylogue-writer".into())
            .spawn(move || inner.run(&rx, &users_rx))?;
        Ok(Self {
            tx: Some(WriterQueue::new(tx, users, Arc::clone(&resolver))),
            handle: Some(handle),
            started_empty: empty,
            resolver,
        })
    }

    pub fn apply_confirmed(&self, events: Vec<crate::primary::Confirmed>) -> Result<i64> {
        let (reply, wait) = sync_channel(1);
        self.tx
            .as_ref()
            .ok_or(StoreError::WriterGone)?
            .send(Msg::Confirmed(events, reply))
            .map_err(|_| StoreError::WriterGone)?;
        wait.recv().map_err(|_| StoreError::WriterGone)?
    }
    pub fn synchronize_primary(&self) -> Result<i64> {
        let (reply, wait) = sync_channel(1);
        self.tx
            .as_ref()
            .ok_or(StoreError::WriterGone)?
            .send(Msg::SynchronizePrimary(reply))
            .map_err(|_| StoreError::WriterGone)?;
        wait.recv().map_err(|_| StoreError::WriterGone)?
    }

    /// Whether the index was empty when the writer started.
    pub fn started_empty(&self) -> bool {
        self.started_empty
    }

    /// Queues one batch. Blocks while the writer is busy and its queue is
    /// full, which is what holds the parse pool back.
    pub fn send(&self, mut batch: Batch) -> Result<()> {
        if !self.resolver.location().is_primary() {
            self.resolver.batches(&mut [&mut batch])?;
        }
        self.tx
            .as_ref()
            .ok_or(StoreError::WriterGone)?
            .send(Msg::Batch(Box::new(batch)))
            .map_err(|_| StoreError::WriterGone)
    }

    /// A cheap handle the parse pool's threads send through.
    pub fn sender(&self) -> Result<WriterSender> {
        Ok(WriterSender(self.tx.clone().ok_or(StoreError::WriterGone)?))
    }

    /// Writes one user request (a note, delivery or setting) and waits for
    /// it to commit. It goes ahead of batches still queued.
    pub fn user(
        &self,
        request: polylogue_model::user::UserRequest,
    ) -> Result<polylogue_model::user::UserOutcome> {
        user_via(self.tx.as_ref().ok_or(StoreError::WriterGone)?, request)
    }

    /// Derives the products of sessions changed since they were last
    /// derived, after the batches already queued.
    pub fn derive(&self) -> Result<u64> {
        derive_via(self.tx.as_ref().ok_or(StoreError::WriterGone)?)
    }

    /// Stores one import's bytes ([`Inner::store_bytes`]). The batch
    /// carries the observation, never parsed rows.
    pub fn store_bytes(
        &self,
        prepared: crate::staging::PreparedBytes,
        batch: Batch,
    ) -> Result<crate::staging::StoredInput> {
        let (reply, wait) = sync_channel(1);
        self.tx
            .as_ref()
            .ok_or(StoreError::WriterGone)?
            .send(Msg::StoreBytes(Box::new(prepared), Box::new(batch), reply))
            .map_err(|_| StoreError::WriterGone)?;
        wait.recv().map_err(|_| StoreError::WriterGone)?
    }

    /// Stores bytes no input holds as one blob ([`Inner::store_blob`]);
    /// `file` is the private file [`crate::Archive::prepare_decoded`] wrote
    /// for a new blob too large to keep inline.
    pub fn store_blob(
        &self,
        new_blob: crate::NewBlob,
        file: Option<crate::staging::PreparedBytes>,
    ) -> Result<BlobHash> {
        let (reply, wait) = sync_channel(1);
        self.tx
            .as_ref()
            .ok_or(StoreError::WriterGone)?
            .send(Msg::StoreBlob(Box::new((new_blob, file)), reply))
            .map_err(|_| StoreError::WriterGone)?;
        wait.recv().map_err(|_| StoreError::WriterGone)?
    }

    /// Accepts one completed discovery walk. Primary visibility follows its
    /// confirmed product; standalone visibility follows the index transaction.
    pub fn coverage(&self, coverage: polylogue_model::capture::Coverage) -> Result<()> {
        coverage_via(self.tx.as_ref().ok_or(StoreError::WriterGone)?, coverage)
    }

    /// Installs export associations derived from committed source observations.
    pub fn import_assets(&self, staged: crate::export_items::ExportItems) -> Result<()> {
        let (reply, wait) = sync_channel(1);
        self.tx
            .as_ref()
            .ok_or(StoreError::WriterGone)?
            .send(Msg::Assets(Box::new(staged), reply))
            .map_err(|_| StoreError::WriterGone)?;
        wait.recv().map_err(|_| StoreError::WriterGone)?
    }

    /// Waits until earlier batches are committed and their memory charges
    /// are released, then returns what was written since the previous flush.
    /// Outside a first build what it reports is also derived and searchable;
    /// a first build derives and indexes once, when it ends.
    pub fn flush(&self) -> Result<WriteReport> {
        self.flush_with_derivation(true)
    }

    /// Commits queued rows; the engine derives once its round is complete.
    pub fn flush_rows(&self) -> Result<WriteReport> {
        self.flush_with_derivation(false)
    }

    fn flush_with_derivation(&self, derive: bool) -> Result<WriteReport> {
        let (reply, wait) = sync_channel(1);
        self.tx
            .as_ref()
            .ok_or(StoreError::WriterGone)?
            .send(Msg::Flush(reply, derive))
            .map_err(|_| StoreError::WriterGone)?;
        wait.recv().map_err(|_| StoreError::WriterGone)
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        if let Some(queue) = self.tx.take() {
            queue.stop();
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// A clonable sender into the writer, for worker threads.
#[derive(Clone)]
pub struct WriterSender(WriterQueue);

impl WriterSender {
    pub fn store_group(
        &self,
        group: Vec<crate::staging::PreparedBatch>,
        store: std::sync::Arc<crate::blobs::Blobs>,
    ) -> Result<Vec<Result<crate::staging::StoredInput>>> {
        self.enqueue_group(group, store)?
            .recv()
            .map_err(|_| StoreError::WriterGone)
    }

    /// Queue a group; parsing can continue while the writer commits it.
    pub fn enqueue_group(
        &self,
        mut group: Vec<crate::staging::PreparedBatch>,
        store: std::sync::Arc<crate::blobs::Blobs>,
    ) -> Result<Receiver<Vec<Result<crate::staging::StoredInput>>>> {
        let mut batches: Vec<_> = group.iter_mut().map(|item| &mut item.batch).collect();
        if !self.0.resolver.location().is_primary() {
            self.0.resolver.batches(&mut batches)?;
        }
        let (reply, wait) = sync_channel(1);
        self.0
            .send(Msg::StoreGroup(group, store, reply))
            .map_err(|_| StoreError::WriterGone)?;
        Ok(wait)
    }

    pub fn send(&self, mut batch: Batch) -> Result<()> {
        if !self.0.resolver.location().is_primary() {
            self.0.resolver.batches(&mut [&mut batch])?;
        }
        self.0
            .send(Msg::Batch(Box::new(batch)))
            .map_err(|_| StoreError::WriterGone)
    }

    /// Replaces one chain's parsed rows in one index transaction.
    pub fn replace_chain(
        &self,
        input: InputKey,
        chain: i64,
        length: u64,
        reading: ChainReading,
    ) -> Result<u64> {
        self.queue_chain(input, chain, length, reading)?
            .recv()
            .map_err(|_| StoreError::WriterGone)?
    }

    /// Queue a chain without waiting for its transaction to commit.
    pub fn queue_chain(
        &self,
        input: InputKey,
        chain: i64,
        length: u64,
        mut reading: ChainReading,
    ) -> Result<Receiver<Result<u64>>> {
        reading.finish()?;
        if !self.0.resolver.location().is_primary() {
            reading.resolve(&self.0.resolver, &input, chain, length)?;
        }
        let (reply, wait) = sync_channel(1);
        crate::timing::reading_commit_queue(1);
        if self
            .0
            .send(Msg::ReplaceChain(input, chain, length, reading, reply))
            .is_err()
        {
            crate::timing::reading_commit_queue(-1);
            return Err(StoreError::WriterGone);
        }
        Ok(wait)
    }

    /// As [`Writer::user`].
    pub fn user(
        &self,
        request: polylogue_model::user::UserRequest,
    ) -> Result<polylogue_model::user::UserOutcome> {
        user_via(&self.0, request)
    }

    /// As [`Writer::store_blob`].
    pub fn store_blob(
        &self,
        new_blob: crate::NewBlob,
        file: Option<crate::staging::PreparedBytes>,
    ) -> Result<BlobHash> {
        let (reply, wait) = sync_channel(1);
        self.0
            .send(Msg::StoreBlob(Box::new((new_blob, file)), reply))
            .map_err(|_| StoreError::WriterGone)?;
        wait.recv().map_err(|_| StoreError::WriterGone)?
    }

    /// As [`Writer::coverage`].
    pub fn coverage(&self, coverage: polylogue_model::capture::Coverage) -> Result<()> {
        coverage_via(&self.0, coverage)
    }

    /// As [`Writer::derive`].
    pub fn derive(&self) -> Result<u64> {
        derive_via(&self.0)
    }
}

fn coverage_via(tx: &WriterQueue, coverage: polylogue_model::capture::Coverage) -> Result<()> {
    let (reply, wait) = sync_channel(1);
    tx.send(Msg::Coverage(Box::new(coverage), reply))
        .map_err(|_| StoreError::WriterGone)?;
    wait.recv().map_err(|_| StoreError::WriterGone)?
}

fn derive_via(tx: &WriterQueue) -> Result<u64> {
    let (reply, wait) = sync_channel(1);
    tx.send(Msg::Derive(reply))
        .map_err(|_| StoreError::WriterGone)?;
    wait.recv().map_err(|_| StoreError::WriterGone)?
}

fn user_via(
    tx: &WriterQueue,
    request: polylogue_model::user::UserRequest,
) -> Result<polylogue_model::user::UserOutcome> {
    let (reply, wait) = sync_channel(1);
    tx.send_user((Box::new(request), reply))
        .map_err(|_| StoreError::WriterGone)?;
    wait.recv().map_err(|_| StoreError::WriterGone)?
}

struct Inner {
    resolver: Arc<resolve::Resolver>,
    segment: Option<hashed_blobs::SegmentWriter>,
    archive: Connection,
    index: Connection,
    report: WriteReport,
    /// Set while the index is being built from nothing, without syncs
    /// ([`Writer::start`]).
    first_build: Option<Arc<crate::first_build::FirstBuild>>,
    /// Secondary indexes dropped for a first build, recreated when it ends.
    deferred_indexes: Vec<String>,
    /// This writer wrote index rows since the last derive, or none has run
    /// yet (a rule version may have changed).
    derive_due: bool,
    /// `data_version` of the index when the last derive ended: it moves when
    /// another connection commits.
    derived_version: Option<i64>,
    /// The search thread ([`search_thread`]): woken after each commit, and
    /// allowed to write once the index is out of a first build.
    search: Search,
    location: Archive,
    /// In primary mode, where every register row goes before it commits.
    sinex: Option<std::sync::Arc<dyn crate::primary::Sinex>>,
}

impl Inner {
    /// Writes every waiting user request, each answered once committed.
    fn serve_users(&mut self, users: &Receiver<UserMsg>) {
        while let Ok((request, reply)) = users.try_recv() {
            let _ = reply.send(self.write_user(&request));
        }
    }

    fn run(&mut self, rx: &Receiver<Msg>, users: &Receiver<UserMsg>) {
        let _worker = crate::timing::worker();
        let mut flushes = Vec::new();
        let mut derives = Vec::new();
        let mut single_files = Vec::new();
        let mut file_groups = Vec::new();
        let mut assets = Vec::new();
        let mut coverage = Vec::new();
        let mut confirmations = Vec::new();
        let mut synchronizations = Vec::new();
        let mut replacements = Vec::new();
        let mut blobs = Vec::new();
        loop {
            let first = match rx.recv() {
                Ok(first) => first,
                Err(_) => break,
            };
            let _busy = crate::timing::writer_started();
            self.serve_users(users);
            let turn = Arc::clone(&self.search.turn);
            let _turn = turn
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut stopping = false;
            let mut group = Vec::new();
            let mut rows = 0;
            let mut bytes = 0;
            let mut push =
                |m: Msg, group: &mut Vec<Box<Batch>>, rows: &mut usize, bytes: &mut u64| match m {
                    Msg::StoreBytes(prepared, batch, reply) => {
                        single_files.push((prepared, batch, reply));
                        false
                    }
                    Msg::StoreGroup(items, store, reply) => {
                        crate::timing::queue(crate::timing::Queue::Writer, -1);
                        for item in &items {
                            *rows += weight(&item.batch);
                            *bytes += input_bytes(&item.batch);
                        }
                        file_groups.push((items, store, reply));
                        true
                    }
                    Msg::Assets(staged, reply) => {
                        assets.push((staged, reply));
                        false
                    }
                    Msg::Coverage(fact, reply) => {
                        coverage.push((fact, reply));
                        false
                    }
                    Msg::StoreBlob(blob, reply) => {
                        blobs.push((blob, reply));
                        false
                    }
                    Msg::ReplaceChain(input, chain, length, reading, reply) => {
                        replacements.push((input, chain, length, reading, reply));
                        false
                    }
                    Msg::Confirmed(events, reply) => {
                        confirmations.push((events, reply));
                        false
                    }
                    Msg::SynchronizePrimary(reply) => {
                        synchronizations.push(reply);
                        false
                    }
                    Msg::Stop => {
                        stopping = true;
                        false
                    }
                    Msg::Wake => false,
                    Msg::Batch(b) => {
                        crate::timing::queue(crate::timing::Queue::Writer, -1);
                        *rows += weight(&b);
                        *bytes += input_bytes(&b);
                        group.push(b);
                        true
                    }
                    Msg::Flush(reply, derive) => {
                        flushes.push((reply, derive));
                        false
                    }

                    Msg::Derive(reply) => {
                        derives.push(reply);
                        false
                    }
                };
            if push(first, &mut group, &mut rows, &mut bytes) {
                while rows < ROWS_PER_TRANSACTION && bytes < BYTES_PER_TRANSACTION {
                    match rx.try_recv() {
                        Ok(m) => {
                            if !push(m, &mut group, &mut rows, &mut bytes) {
                                break;
                            }
                        }
                        Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
                    }
                }
            }
            for (events, reply) in confirmations.drain(..) {
                let _ = reply.send(self.apply_confirmed(&events));
            }
            for reply in synchronizations.drain(..) {
                let _ = reply.send(self.synchronize_primary());
            }
            if !group.is_empty() {
                crate::timing::time(crate::timing::Stage::Rows, || self.write_group(&group));
            }
            // Later queue replies also observe release of the parsed batches'
            // memory charges, not only their committed database rows.
            drop(group);
            for (prepared, batch, reply) in single_files.drain(..) {
                let _ = reply.send(self.store_bytes(*prepared, *batch));
            }
            self.store_groups(std::mem::take(&mut file_groups));
            for (input, chain, length, reading, reply) in replacements.drain(..) {
                let _ = reply.send(crate::timing::time(crate::timing::Stage::Rows, || {
                    self.replace_chain(input, chain, length, reading)
                }));
                crate::timing::reading_commit_queue(-1);
                self.serve_users(users);
            }
            for (blob, reply) in blobs.drain(..) {
                let (new_blob, file) = *blob;
                let _ = reply.send(self.store_blob(new_blob, file));
            }
            self.serve_users(users);

            if !derives.is_empty() {
                // A first build's indexes exist before anything is derived
                // from its rows.
                let result = self.finish_build().and_then(|()| self.derive());
                for reply in derives.drain(..) {
                    let _ = reply.send(match &result {
                        Ok(n) => Ok(*n),
                        Err(e) => Err(StoreError::Invalid(format!("derive: {e}"))),
                    });
                }
            }
            // Whatever a flush reports written is also derived, except in a
            // first build, which a flush in the middle of a pass (each import
            // item, each run of the parse pool) must not end.
            let derived_on_flush =
                flushes.iter().any(|(_, derive)| *derive) && self.first_build.is_none();
            if derived_on_flush && let Err(e) = self.derive() {
                self.report.errors.push(format!("derive: {e}"));
            }
            for (staged, reply) in assets.drain(..) {
                let result = crate::assets::install(&mut self.index, &self.archive, *staged);
                self.derive_due |= result.is_ok();
                let _ = reply.send(result);
            }
            for (fact, reply) in coverage.drain(..) {
                let _ = reply.send(self.write_coverage(*fact));
            }
            for (reply, _) in flushes.drain(..) {
                let _ = reply.send(std::mem::take(&mut self.report));
            }

            if stopping {
                break;
            }
        }
        let _ = self.finish_build();
    }

    fn write_coverage(&mut self, coverage: polylogue_model::capture::Coverage) -> Result<()> {
        coverage
            .validate()
            .map_err(|error| StoreError::Invalid(error.into()))?;
        if let Some(sinex) = &self.sinex {
            let machine = coverage.machine.clone();
            sinex.prepare_products(&[crate::primary::ProductWrite {
                product: crate::primary::Product::Coverage(coverage),
                location: None,
                parents: Vec::new(),
                machine: Some(machine),
                output_index: 0,
            }])?;
            return Ok(());
        }
        let tx = self.index.transaction()?;
        bump_counter(&tx)?;
        crate::coverage::apply(&tx, &coverage)?;
        crate::write_lock::commit(tx)?;
        self.report.transactions += 1;
        Ok(())
    }

    /// Derives from the rows written since the last derive. With nothing
    /// written since, by this writer or another connection, nothing is
    /// stale, so the scan for stale sessions, which reads every session for
    /// every derivation, is skipped.
    fn derive(&mut self) -> Result<u64> {
        // Changes only when another connection commits.
        let version = |index: &Connection| -> Result<i64> {
            Ok(index.pragma_query_value(None, "data_version", |r| r.get(0))?)
        };
        if !self.derive_due && self.derived_version == Some(version(&self.index)?) {
            return Ok(0);
        }
        if self.first_build.is_none() {
            crate::text_rows::index_pending(&self.index, || ())?;
        }
        crate::versions::remove_unused(&self.index, self.first_build.is_some())?;
        // A full search batch's deletions get one merge at the end of a round.
        let deleted:i64=self.index.query_row("SELECT coalesce((SELECT CAST(value AS INTEGER) FROM meta WHERE key='reparse_deletions'),0)",[],|r|r.get(0))?;
        if self.first_build.is_none() && deleted >= 2_000 {
            for table in ["text_fts", "output_fts", "tool_call_trigram"] {
                self.index.execute(
                    &format!("INSERT INTO {table}({table}) VALUES ('optimize')"),
                    [],
                )?;
            }
            self.index
                .execute("DELETE FROM meta WHERE key='reparse_deletions'", [])?;
        }
        let derived = crate::timing::time(crate::timing::Stage::Derive, || {
            crate::derive::run(&mut self.index)
        })?;
        self.derive_due = false;
        self.derived_version = Some(version(&self.index)?);
        Ok(derived)
    }

    /// Clears the build marker only after every deferred index has committed.
    fn finish_build(&mut self) -> Result<()> {
        if self.first_build.is_none() {
            return Ok(());
        }
        self.build_deferred()?;
        self.index
            .execute("UPDATE meta SET value = '0' WHERE key = 'building'", [])?;
        self.index.pragma_update(None, "synchronous", "normal")?;
        self.first_build = None;
        self.search.ready();
        Ok(())
    }

    /// Ends a first build's bulk load: writes the search entries the build
    /// marked, rebuilds the trigram index from `tool_call`, and creates the
    /// indexes it dropped, the unique ones on natural keys first. Each from one ordered scan, so the
    /// indexes are written once instead of rewritten through the build.
    fn build_deferred(&mut self) -> Result<()> {
        use crate::timing::{Stage, add};
        // Bulk inserts are finished; release their cached programs before the search scan.
        self.index.flush_prepared_statement_cache();
        let started = std::time::Instant::now();
        crate::text_rows::index_marked(&self.index)?;
        add(Stage::SearchIndex, started.elapsed());
        let started = std::time::Instant::now();
        self.index.execute(
            "INSERT INTO tool_call_trigram(tool_call_trigram) VALUES ('rebuild')",
            [],
        )?;
        add(Stage::TrigramIndex, started.elapsed());
        let started = std::time::Instant::now();
        for sql in &self.deferred_indexes {
            let sql = sql
                .replacen(
                    "CREATE UNIQUE INDEX ",
                    "CREATE UNIQUE INDEX IF NOT EXISTS ",
                    1,
                )
                .replacen("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS ", 1);
            self.index.execute(&sql, [])?;
            crate::fault::check("index.deferred", &self.location.root.to_string_lossy())?;
        }
        self.deferred_indexes.clear();
        add(Stage::SecondaryIndexes, started.elapsed());
        Ok(())
    }

    /// Writes a group in one transaction per database. If it fails, each
    /// batch is retried alone, so one bad batch costs only itself.
    fn write_group(&mut self, group: &[Box<Batch>]) {
        if group.len() > 1 && self.write(group.iter().map(|b| &**b)).is_ok() {
            return;
        }
        if group.len() == 1 && self.write(std::iter::once(&*group[0])).is_ok() {
            return;
        }
        for b in group {
            if let Err(e) = self.write(std::iter::once(&**b)) {
                self.report
                    .errors
                    .push(format!("input {}: {e}", b.input.id));
            }
        }
    }

    /// Standalone bytes without an input, such as a chat attachment, follow
    /// the same append, sync and catalog commit order as captured inputs.
    fn store_blob(
        &mut self,
        new_blob: crate::NewBlob,
        file: Option<crate::staging::PreparedBytes>,
    ) -> Result<BlobHash> {
        if self.sinex.is_some() {
            return self.primary_blob(new_blob, file);
        }
        if crate::blobs::load_extent(&self.archive, "", &hashed_blobs::Hash(new_blob.hash.0))?
            .is_some()
        {
            return Ok(new_blob.hash);
        }
        let file =
            file.ok_or_else(|| StoreError::Invalid("new blob has no prepared bytes".into()))?;
        if file.owner != self.location.root
            || file.hash() != new_blob.hash
            || file.len() != new_blob.size
        {
            return Err(StoreError::Invalid(
                "prepared bytes do not match the new blob".into(),
            ));
        }
        let crate::staging::Prepared::Local { prepared } = file.file else {
            return Err(StoreError::Invalid(
                "Sinex bytes reached local segment writer".into(),
            ));
        };
        if self.segment.is_none() {
            self.segment = Some(crate::blobs::open_writer(
                &self.archive,
                &self.location.root,
            )?);
        }
        let writer = self.segment.as_mut().expect("opened segment writer");
        let pending = match writer.append_group(vec![*prepared], |_| Ok(None)) {
            Ok(pending) => pending,
            Err(error) => {
                self.segment.take();
                return Err(error.into());
            }
        };
        let synced = match pending.wait() {
            Ok(synced) => synced,
            Err(error) => {
                self.segment.take();
                return Err(error.into());
            }
        };
        let result = (|| {
            let tx = self.archive.transaction()?;
            for extent in &synced.extents {
                crate::blobs::insert_extent(&tx, extent)?;
            }
            tx.execute("INSERT INTO segment(name,committed_end) VALUES (?1,?2) ON CONFLICT(name) DO UPDATE SET committed_end=excluded.committed_end",
                params!["segments/current.hb",synced.end as i64])?;
            crate::write_lock::commit(tx)?;
            Ok(())
        })();
        if let Err(error) = result {
            self.segment.take();
            return Err(error);
        }
        self.report.segment_syncs += synced.metrics.syncs;
        self.report.segment_bytes_written += synced.metrics.appended_bytes;
        self.report.blobs_appended += synced.metrics.appended_materials;
        self.report.blobs_deduplicated += synced.metrics.deduplicated_materials;
        self.segment
            .as_mut()
            .expect("appended writer")
            .acknowledge(&synced)?;
        Ok(new_blob.hash)
    }

    fn write_user(
        &mut self,
        request: &polylogue_model::user::UserRequest,
    ) -> Result<polylogue_model::user::UserOutcome> {
        if self.sinex.is_some() {
            return self.primary_request(request);
        }
        let tx = self.archive.transaction()?;
        let known: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM client_request WHERE id=?1)",
            [&request.id],
            |row| row.get(0),
        )?;
        let seq = if known {
            tx.query_row(
                "SELECT CAST(value AS INTEGER) FROM meta WHERE key='write_counter'",
                [],
                |row| row.get(0),
            )
            .optional()?
            .unwrap_or(0)
        } else {
            bump_counter(&tx)?
        };
        let outcome = crate::user_writes::apply(&tx, request, seq)?;
        if known {
            tx.rollback()?;
        } else {
            crate::write_lock::commit(tx)?;
        }
        Ok(outcome)
    }

    fn write<'a>(&mut self, batches: impl Iterator<Item = &'a Batch> + Clone) -> Result<()> {
        if self.sinex.is_some() {
            return self.primary_products(batches);
        }
        for b in batches.clone() {
            if crate::container_members::container_of(b).is_some() {
                return Err(StoreError::Invalid(
                    "a container member is stored with its container's group".into(),
                ));
            }
            crate::fault::check("write.input", &b.input.id)?;
            for p in &b.parse {
                p.counts.check()?;
            }
        }
        let parsed_only = batches.clone().all(|batch| {
            matches!(batch.change, Change::Reparse { .. })
                && batch.new_blob.is_none()
                && batch.blobs.is_empty()
                && batch.observation.is_none()
        });
        let mut chains = Vec::new();
        if parsed_only {
            for batch in batches.clone() {
                let Change::Reparse { chain } = batch.change else {
                    unreachable!()
                };
                chains.push(Some(chain));
            }
        } else {
            let atx = self.archive.transaction()?;
            let seq = bump_counter(&atx)?;
            for batch in batches.clone() {
                let (chain, _) = write_bytes(&atx, batch, seq)?;
                chains.push(chain);
            }
            crate::fault::check("write.archive_commit", "")?;
            crate::write_lock::commit(atx)?;
            crate::fault::check("write.between_commits", "")?;
        }

        let indexed = batches.collect::<Vec<_>>();
        let itx = self
            .index
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let counter = bump_counter(&itx)?;
        let mut stats = (0u64, 0u64);
        for (b, chain) in indexed.iter().zip(chains) {
            if let Some(chain) = chain {
                for p in &b.parse {
                    let receipt = crate::source::Source::reading_receipt(
                        &itx,
                        &b.input.id,
                        chain,
                        p.parsed_upto,
                    )?;
                    stats.1 += write_rows(
                        &itx,
                        b,
                        p,
                        chain,
                        counter,
                        self.first_build.as_deref(),
                        Some(&receipt),
                    )?;
                }
                if b.parse.iter().any(|p| p.at_end) {
                    rows::prune_sessions(&itx, &b.input.id, chain)?;
                }
            }
            stats.0 += 1;
        }
        crate::fault::check("write.index_commit", "")?;
        itx.commit()?;
        self.derive_due = true;
        self.search.wake();
        self.report.batches += stats.0;
        self.report.messages += stats.1;
        self.report.transactions += 1;
        Ok(())
    }
}

/// The input bytes a batch was parsed from (or stored, when unparsed).
fn input_bytes(b: &Batch) -> u64 {
    match (b.parse.first(), b.parse.last()) {
        (Some(first), Some(last)) => last.parsed_upto.saturating_sub(first.start),
        _ => b.new_blob.as_ref().map_or(0, |p| p.size),
    }
}

fn weight(b: &Batch) -> usize {
    1 + b
        .parse
        .iter()
        .map(|p| {
            p.output.messages.len()
                + p.output.documents.len()
                + p.output.events.len()
                + p.output.links.len()
        })
        .sum::<usize>()
}

// ---------------------------------------------------------------- archive.db

// ------------------------------------------------------------------ index.db

#[cfg(test)]
mod tests;
