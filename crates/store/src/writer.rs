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
mod excluded;
mod rows;
mod search;
mod staging;
pub use chain::ChainReading;

#[cfg(test)]
use crate::encoding::pack_block;
use crate::encoding::{PreparedDocument, PreparedMessage};
use crate::encoding::{block_text, pack_event, unpack_block};
pub use batch::{
    Batch, ImportMember, InFlight, InZip, InputKey, Observation, ParseStep, WriteReport,
};
pub(crate) use bytes::observation_id;
use bytes::*;
use excluded::*;
pub(crate) use rows::retract_input;
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
    /// Bytes no input holds (a chat action's attachment), as one blob.
    StoreBlob(
        Box<(crate::NewBlob, Option<crate::staging::PreparedBytes>)>,
        SyncSender<Result<BlobHash>>,
    ),
    ReplaceChain(InputKey, i64, u64, ChainReading, SyncSender<Result<u64>>),
    Stop,
    /// A user write is waiting in its own queue.
    Wake,
    Batch(Box<Batch>),
    Flush(SyncSender<WriteReport>, bool),
    /// An excision, with the blob store whose files it changes.
    Excise(
        Box<crate::excise::ExciseRequest>,
        std::sync::Arc<crate::blobs::Blobs>,
        SyncSender<Result<crate::excise::Excised>>,
    ),
    /// Run the derive step over sessions whose derived rows are stale;
    /// answered with the number of sessions derived.
    Derive(SyncSender<Result<u64>>),
    /// An excision another archive recorded, brought in with its user
    /// state; answered with whether it was new here.
    ImportExcision(Box<crate::primary::ExcisionRows>, SyncSender<Result<bool>>),
}

/// Handle to the writer thread. Dropping it stops the thread after the
/// batches already sent are written.
pub struct Writer {
    tx: Option<WriterQueue>,
    handle: Option<JoinHandle<()>>,
    /// The index was empty when the writer started: its first pass is a
    /// first build.
    started_empty: bool,
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
}

type UserMsg = (
    Box<crate::user_writes::UserRequest>,
    SyncSender<Result<crate::user_writes::UserOutcome>>,
);

/// User writes waiting before their senders block.
const USER_QUEUE: usize = 64;

impl WriterQueue {
    fn new(bulk: SyncSender<Msg>, users: SyncSender<UserMsg>) -> Self {
        Self {
            bulk: Arc::new(Mutex::new(Some(bulk))),
            users: Arc::new(Mutex::new(Some(users))),
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
        let index = open_write(&archive.index_db(), crate::archive::cache::INDEX_WRITER)?;
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
        let archive_conn = crate::archive::open_archive_writer(archive)?;
        let excisions = crate::excise::any_excision(&archive_conn)?;
        let first_build = building
            .then(|| crate::first_build::FirstBuild::load(&index))
            .transpose()?;
        let segment = if sinex.is_none() {
            let end: u64 = archive_conn
                .query_row(
                    "SELECT committed_end FROM segment WHERE name=?1",
                    ["segments/current.hb"],
                    |r| r.get(0),
                )
                .optional()?
                .unwrap_or(0);
            Some(hashed_blobs::SegmentWriter::open(
                &archive.root,
                "segments/current.hb",
                end,
                &[],
                0o600,
            )?)
        } else {
            None
        };
        let mut inner = Inner {
            segment,
            archive: archive_conn,
            index,
            report: WriteReport::default(),
            first_build,
            deferred_indexes,
            derive_due: true,
            derived_version: None,
            logs_hold_excised: excisions,
            search: Search::start(archive.index_db(), !building)?,
            location: archive.clone(),
            excisions,
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
            tx: Some(WriterQueue::new(tx, users)),
            handle: Some(handle),
            started_empty: empty,
        })
    }

    /// Whether the index was empty when the writer started.
    pub fn started_empty(&self) -> bool {
        self.started_empty
    }

    /// Queues one batch. Blocks while the writer is busy and its queue is
    /// full, which is what holds the parse pool back.
    pub fn send(&self, batch: Batch) -> Result<()> {
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
        request: crate::user_writes::UserRequest,
    ) -> Result<crate::user_writes::UserOutcome> {
        user_via(self.tx.as_ref().ok_or(StoreError::WriterGone)?, request)
    }

    /// Runs an excision after the batches already queued and waits for it
    /// to commit ([`crate::excise`]).
    pub fn excise(
        &self,
        request: crate::excise::ExciseRequest,
        store: std::sync::Arc<crate::blobs::Blobs>,
    ) -> Result<crate::excise::Excised> {
        let (reply, wait) = sync_channel(1);
        self.tx
            .as_ref()
            .ok_or(StoreError::WriterGone)?
            .send(Msg::Excise(Box::new(request), store, reply))
            .map_err(|_| StoreError::WriterGone)?;
        wait.recv().map_err(|_| StoreError::WriterGone)?
    }

    /// Derives the products of sessions changed since they were last
    /// derived, after the batches already queued.
    pub fn derive(&self) -> Result<u64> {
        derive_via(self.tx.as_ref().ok_or(StoreError::WriterGone)?)
    }

    /// Records an excision another archive made, so what it removed is
    /// refused here too ([`Inner::import_excision`]). `false` when this
    /// archive already holds it.
    pub fn import_excision(&self, excision: crate::primary::ExcisionRows) -> Result<bool> {
        let (reply, wait) = sync_channel(1);
        self.tx
            .as_ref()
            .ok_or(StoreError::WriterGone)?
            .send(Msg::ImportExcision(Box::new(excision), reply))
            .map_err(|_| StoreError::WriterGone)?;
        wait.recv().map_err(|_| StoreError::WriterGone)?
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
        group: Vec<crate::staging::PreparedBatch>,
        store: std::sync::Arc<crate::blobs::Blobs>,
    ) -> Result<Receiver<Vec<Result<crate::staging::StoredInput>>>> {
        let (reply, wait) = sync_channel(1);
        self.0
            .send(Msg::StoreGroup(group, store, reply))
            .map_err(|_| StoreError::WriterGone)?;
        Ok(wait)
    }

    pub fn send(&self, batch: Batch) -> Result<()> {
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
        request: crate::user_writes::UserRequest,
    ) -> Result<crate::user_writes::UserOutcome> {
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

    /// As [`Writer::derive`].
    pub fn derive(&self) -> Result<u64> {
        derive_via(&self.0)
    }
}

fn derive_via(tx: &WriterQueue) -> Result<u64> {
    let (reply, wait) = sync_channel(1);
    tx.send(Msg::Derive(reply))
        .map_err(|_| StoreError::WriterGone)?;
    wait.recv().map_err(|_| StoreError::WriterGone)?
}

fn user_via(
    tx: &WriterQueue,
    request: crate::user_writes::UserRequest,
) -> Result<crate::user_writes::UserOutcome> {
    let (reply, wait) = sync_channel(1);
    tx.send_user((Box::new(request), reply))
        .map_err(|_| StoreError::WriterGone)?;
    wait.recv().map_err(|_| StoreError::WriterGone)?
}

struct Inner {
    segment: Option<hashed_blobs::SegmentWriter>,
    archive: Connection,
    index: Connection,
    report: WriteReport,
    /// Set while the index is being built from nothing, without syncs
    /// ([`Writer::start`]).
    first_build: Option<crate::first_build::FirstBuild>,
    /// Secondary indexes dropped for a first build, recreated when it ends.
    deferred_indexes: Vec<String>,
    /// This writer wrote index rows since the last derive, or none has run
    /// yet (a rule version may have changed).
    derive_due: bool,
    /// `data_version` of the index when the last derive ended: it moves when
    /// another connection commits.
    derived_version: Option<i64>,
    /// A write-ahead log may still hold text an excision removed
    /// ([`Self::empty_logs`]).
    logs_hold_excised: bool,
    /// The search thread ([`search_thread`]): woken after each commit, and
    /// allowed to write once the index is out of a first build.
    search: Search,
    location: Archive,
    /// Whether the archive holds an excision, whose sessions, messages and
    /// bytes are never written again.
    excisions: bool,
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
        let mut excisions = Vec::new();
        let mut imported_excisions = Vec::new();
        let mut derives = Vec::new();
        let mut single_files = Vec::new();
        let mut file_groups = Vec::new();
        let mut assets = Vec::new();
        let mut replacements = Vec::new();
        let mut blobs = Vec::new();
        loop {
            // While a log still holds excised text, the writer wakes to
            // empty it even when no work comes.
            let first = if self.logs_hold_excised {
                match rx.recv_timeout(std::time::Duration::from_secs(1)) {
                    Ok(first) => first,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        self.empty_logs_or_report();
                        continue;
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                }
            } else {
                match rx.recv() {
                    Ok(first) => first,
                    Err(_) => break,
                }
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
                    Msg::StoreBlob(blob, reply) => {
                        blobs.push((blob, reply));
                        false
                    }
                    Msg::ReplaceChain(input, chain, length, reading, reply) => {
                        replacements.push((input, chain, length, reading, reply));
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
                    Msg::Excise(request, store, reply) => {
                        excisions.push((request, store, reply));
                        false
                    }
                    Msg::Derive(reply) => {
                        derives.push(reply);
                        false
                    }
                    Msg::ImportExcision(excision, reply) => {
                        imported_excisions.push((excision, reply));
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
            for (request, store, reply) in excisions.drain(..) {
                let result = self.write_excision(&request, &store);
                let interrupted = matches!(&result, Err(StoreError::ExcisionInterrupted(_)));
                let _ = reply.send(result);
                if interrupted {
                    return;
                }
            }
            for (excision, reply) in imported_excisions.drain(..) {
                let _ = reply.send(self.import_excision(&excision));
            }
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
            for (reply, _) in flushes.drain(..) {
                let _ = reply.send(std::mem::take(&mut self.report));
            }
            if self.logs_hold_excised {
                self.empty_logs_or_report();
            }
            if stopping {
                break;
            }
        }
        let _ = self.finish_build();
    }

    fn empty_logs_or_report(&mut self) {
        if let Err(e) = self.empty_logs() {
            self.report
                .errors
                .push(format!("empty the write-ahead logs: {e}"));
        }
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

    /// An excision in one pass ([`crate::excise`]): plan, then one
    /// transaction per database, committed embeddings, archive, index; a dry
    /// run plans and rolls back. Deleting vectors first only costs a later
    /// purchase if a later commit fails. Once the archive commits, the
    /// excision is in force: an index that missed it applies the recorded
    /// identities when the archive is next opened, and the files of blobs marked for deletion go
    /// after the commits, or when the archive is next opened. Running a
    /// recorded request again finishes what is left.
    fn write_excision(
        &mut self,
        request: &crate::excise::ExciseRequest,
        store: &crate::blobs::Blobs,
    ) -> Result<crate::excise::Excised> {
        if request.dry_run {
            let _lock = crate::write_lock::hold(&self.archive)?;
            let plan = crate::excise::plan(&self.archive, &self.index, request)?;
            if plan.recorded() {
                return crate::excise::recorded(&self.archive, request);
            }
            return self.run_excision(request, &plan);
        }
        // An excision is written with WAL syncs even during a first build,
        // and the archive's rows are durable before blob files are deleted.
        let previous: i64 = self
            .archive
            .pragma_query_value(None, "synchronous", |r| r.get(0))?;
        self.archive
            .pragma_update(None, "synchronous", previous.max(2))?;
        let index_sync: i64 = self
            .index
            .pragma_query_value(None, "synchronous", |r| r.get(0))?;
        self.index.pragma_update(None, "synchronous", 2)?;
        let lock = crate::write_lock::hold(&self.archive)?;
        // Pages the excision frees are overwritten with zeros, so no removed
        // content stays readable in the database files.
        for conn in [&self.archive, &self.index] {
            conn.pragma_update(None, "secure_delete", true)?;
        }
        let result = crate::excise::plan(&self.archive, &self.index, request).and_then(|plan| {
            if plan.recorded() {
                crate::excise::recorded(&self.archive, request)
            } else {
                self.run_excision(request, &plan)
            }
        });
        drop(lock);
        // Then the blob files and rows go, and the database files are
        // scrubbed of every freed page.
        let result = result.and_then(|done| {
            let files = self.sinex.is_none().then_some(store);
            crate::blobs::delete_marked(&self.archive, files)
                .and_then(|_| {
                    let _lock = crate::write_lock::hold(&self.archive)?;
                    self.scrub_after_excision()
                })
                .map(|()| done)
                .map_err(|error| StoreError::ExcisionInterrupted(Box::new(error)))
        });
        for conn in [&self.archive, &self.index] {
            conn.pragma_update(None, "secure_delete", false)
                .map_err(|e| StoreError::ExcisionInterrupted(Box::new(e.into())))?;
        }
        self.archive
            .pragma_update(None, "synchronous", previous)
            .map_err(|e| StoreError::ExcisionInterrupted(Box::new(e.into())))?;
        self.index
            .pragma_update(None, "synchronous", index_sync)
            .map_err(|e| StoreError::ExcisionInterrupted(Box::new(e.into())))?;
        if result.is_ok() {
            self.excisions = true;
            self.derive_due = true;
        }
        result
    }

    /// After an excision commits: the search indexes are merged into one
    /// segment each, so deleted entries leave the old segments, and both
    /// write-ahead logs are emptied into the database files, whose freed pages
    /// `secure_delete` has zeroed ([`Self::empty_logs`]).
    fn scrub_after_excision(&mut self) -> Result<()> {
        for table in ["text_fts", "output_fts", "tool_call_trigram"] {
            self.index.execute(
                &format!("INSERT INTO {table}({table}) VALUES ('optimize')"),
                [],
            )?;
        }
        self.logs_hold_excised = true;
        self.empty_logs()
    }

    /// Empties both write-ahead logs into the database files and truncates
    /// them, so excised text leaves them. A read still open on an older
    /// snapshot keeps its frames; rather than wait for it, the writer tries
    /// again after its next work, or within a second when idle, until both
    /// logs are empty.
    fn empty_logs(&mut self) -> Result<()> {
        let mut emptied = true;
        for conn in [&self.index, &self.archive] {
            conn.busy_timeout(std::time::Duration::ZERO)?;
            let busy = conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| {
                r.get::<_, i64>(0)
            });
            conn.busy_timeout(std::time::Duration::from_secs(30))?;
            match busy {
                Ok(0) => {}
                Ok(_) => emptied = false,
                Err(rusqlite::Error::SqliteFailure(e, _))
                    if e.code == rusqlite::ErrorCode::DatabaseBusy =>
                {
                    emptied = false;
                }
                Err(e) => return Err(e.into()),
            }
        }
        self.logs_hold_excised = !emptied;
        Ok(())
    }

    /// One transaction per database, committed in the order
    /// [`Self::write_excision`] describes, or rolled back for a dry run.
    fn run_excision(
        &mut self,
        request: &crate::excise::ExciseRequest,
        plan: &crate::excise::Plan,
    ) -> Result<crate::excise::Excised> {
        let embeddings = if self.location.embeddings_db().exists() {
            let e = open_write(&self.location.embeddings_db(), crate::archive::cache::SETUP)?;
            e.pragma_update(None, "synchronous", "full")?;
            e.execute_batch("BEGIN IMMEDIATE")?;
            Some(e)
        } else {
            None
        };
        let rollback = |embeddings: &Option<rusqlite::Connection>| {
            if let Some(e) = embeddings {
                let _ = e.execute_batch("ROLLBACK");
            }
        };
        let atx = self.archive.transaction()?;
        let itx = self
            .index
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let seq = bump_counter(&atx)?;
        let counter = bump_counter(&itx)?;
        let txs = crate::excise::Transactions {
            archive: &atx,
            index: &itx,
            embeddings: embeddings.as_ref(),
            seq,
            counter,
        };
        let materials = match &self.sinex {
            Some(_) => crate::primary::materials_of_inputs(&atx, plan.inputs()),
            None => Ok(Vec::new()),
        };
        let applied = materials.and_then(|materials| {
            let done = crate::excise::apply(&txs, request, plan)?;
            if let (Some(sinex), false) = (&self.sinex, request.dry_run) {
                excise_on_sinex(sinex.as_ref(), &atx, request, plan, &done, materials)?;
            }
            Ok(done)
        });
        let done = match applied {
            Ok(done) => done,
            Err(error) => {
                rollback(&embeddings);
                return Err(error);
            }
        };
        if request.dry_run {
            itx.rollback()?;
            atx.rollback()?;
            rollback(&embeddings);
            return Ok(done);
        }
        // A failed commit leaves the databases apart until the archive is
        // reopened, so this writer stops.
        let committed: Result<()> = (|| {
            if let Some(e) = &embeddings {
                e.execute_batch("COMMIT")?;
                crate::vectors::wrote();
            }
            crate::fault::check("excision.after_embeddings_commit", &request.id)?;
            atx.commit()?;
            crate::fault::check("excision.after_archive_commit", &request.id)?;
            itx.commit()?;
            crate::fault::check("excision.after_index_commit", &request.id)
        })();
        committed.map_err(|error| StoreError::ExcisionInterrupted(Box::new(error)))?;
        Ok(done)
    }

    /// Writes a group in one transaction per database. If it fails, each
    /// batch is retried alone, so one bad batch costs only itself.
    fn write_group(&mut self, group: &[Box<Batch>]) {
        if self.excisions {
            let refused = match matching(group, &self.archive) {
                Ok(refused) => refused,
                Err(error) => {
                    for b in group {
                        self.report
                            .errors
                            .push(format!("input {}: {error}", b.input.id));
                    }
                    return;
                }
            };
            if !refused.is_empty() {
                let filtered: Vec<Box<Batch>> = group
                    .iter()
                    .map(|b| Box::new(without_refused(b, &refused)))
                    .collect();
                return self.write_group_inner(&filtered);
            }
        }
        self.write_group_inner(group);
    }

    fn write_group_inner(&mut self, group: &[Box<Batch>]) {
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
        if let Some(sinex) = &self.sinex {
            let crate::staging::Prepared::Part { file, .. } = file.file else {
                return Err(StoreError::Invalid(
                    "local frames reached Sinex writer".into(),
                ));
            };
            let confirmed = sinex.send_blob(new_blob.hash, new_blob.size, &mut file.reopen()?)?;
            let tx = self.archive.transaction()?;
            crate::blobs::insert_extent(&tx, confirmed.extent(new_blob.hash, new_blob.size)?)?;
            crate::write_lock::commit(tx)?;
            return Ok(new_blob.hash);
        }
        let crate::staging::Prepared::Local { prepared } = file.file else {
            return Err(StoreError::Invalid(
                "Sinex bytes reached local segment writer".into(),
            ));
        };
        let writer = self.segment.as_mut().ok_or_else(|| {
            StoreError::Invalid("segment writer requires reopen after failed commit".into())
        })?;
        let synced = writer.append_group(vec![*prepared], |_| Ok(None))?.wait()?;
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
        self.segment
            .as_mut()
            .expect("appended writer")
            .acknowledge(&synced)?;
        Ok(new_blob.hash)
    }

    fn write_user(
        &mut self,
        request: &crate::user_writes::UserRequest,
    ) -> Result<crate::user_writes::UserOutcome> {
        let tx = self.archive.transaction()?;
        let seq = bump_counter(&tx)?;
        let outcome = crate::user_writes::apply(&tx, request, seq)?;
        if outcome.repeat {
            // A resent request: nothing to keep, not even the counter bump.
            tx.rollback()?;
        } else {
            if let Some(sinex) = &self.sinex {
                let change = crate::primary::RegisterChange::Request {
                    request: Box::new(request.clone()),
                    first_note: crate::user_writes::first_note_of_request(&tx, request)?,
                };
                let material = sinex.send_change(&change)?;
                crate::primary::change_material(&tx, &change, &material)?;
            }
            crate::write_lock::commit(tx)?;
        }
        Ok(outcome)
    }

    /// Records an excision from another archive's user state. What it
    /// removed is refused from now on; when this archive already holds any
    /// of it (collected before the import), it is refused instead, because
    /// recording the excision would not remove it.
    fn import_excision(&mut self, excision: &crate::primary::ExcisionRows) -> Result<bool> {
        let tx = self.archive.transaction()?;
        let known: bool = tx
            .prepare_cached("SELECT 1 FROM excision WHERE id = ?1")?
            .exists([&excision.id])?;
        if known {
            return Ok(false);
        }
        for (kind, value) in &excision.excised {
            let held = match kind.as_str() {
                "session" => self
                    .index
                    .prepare_cached("SELECT 1 FROM session WHERE id = ?1")?
                    .exists([value])?,
                "document" => self
                    .index
                    .prepare_cached("SELECT 1 FROM document WHERE id = ?1")?
                    .exists([value])?,
                "input" => tx
                    .prepare_cached("SELECT 1 FROM revision WHERE input = ?1")?
                    .exists([value])?,
                "blob" => {
                    let hash = BlobHash::parse(value)
                        .map_err(|e| StoreError::Invalid(format!("excised blob {value}: {e}")))?;
                    tx.prepare_cached("SELECT 1 FROM extent WHERE blob = ?1")?
                        .exists([&hash.0[..]])?
                }
                _ => false,
            };
            if held {
                return Err(StoreError::ExcisionRefused {
                    target: excision.target.clone(),
                    reason: format!(
                        "this archive holds {kind} {value}, which the imported excision removed; excise it here, then import again"
                    ),
                });
            }
        }
        let seq = bump_counter(&tx)?;
        crate::primary::insert_excision(&tx, excision, seq)?;
        if let Some(sinex) = &self.sinex {
            let change = crate::primary::RegisterChange::Excision(excision.clone());
            let material = sinex.send_change(&change)?;
            crate::primary::change_material(&tx, &change, &material)?;
        }
        crate::write_lock::commit(tx)?;
        self.excisions = true;
        Ok(true)
    }

    fn write<'a>(&mut self, batches: impl Iterator<Item = &'a Batch> + Clone) -> Result<()> {
        for b in batches.clone() {
            if crate::zip_members::zip_of(b).is_some() {
                return Err(StoreError::Invalid(
                    "a ZIP member is stored with its ZIP's group".into(),
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
                let (chain, rows) = write_bytes(&atx, batch, seq)?;
                if let (Some(sinex), Some(rows)) = (&self.sinex, &rows) {
                    staging::send_to_sinex(
                        sinex.as_ref(),
                        &atx,
                        rows,
                        None,
                        batch.new_blob.as_ref(),
                    )?;
                }
                chains.push(chain);
            }
            crate::fault::check("write.archive_commit", "")?;
            crate::write_lock::commit(atx)?;
            crate::fault::check("write.between_commits", "")?;
        }

        let indexed = batches
            .map(|b| attachments::with_stored_blobs(b, &self.archive))
            .collect::<Result<Vec<_>>>()?;
        let itx = self
            .index
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let counter = bump_counter(&itx)?;
        let mut stats = (0u64, 0u64);
        for (b, chain) in indexed.iter().zip(chains) {
            if let Some(chain) = chain {
                for p in &b.parse {
                    stats.1 += write_rows(&itx, b, p, chain, counter, self.first_build.as_ref())?;
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

/// In primary mode, before an excision commits: Sinex excises the parts and
/// register materials of the removed inputs and the requests of the removed
/// notes, then
/// keeps the excision itself, so a rebuild keeps refusing what it removed.
fn excise_on_sinex(
    sinex: &dyn crate::primary::Sinex,
    archive: &Connection,
    request: &crate::excise::ExciseRequest,
    plan: &crate::excise::Plan,
    done: &crate::excise::Excised,
    mut materials: Vec<String>,
) -> Result<()> {
    materials.extend(crate::primary::materials_of_requests(archive, &done.notes)?);
    if !materials.is_empty() {
        sinex.excise(&materials)?;
    }
    crate::primary::forget_inputs(archive, plan.inputs())?;
    crate::primary::forget_requests(archive, &done.notes)?;
    let change = crate::primary::RegisterChange::Excision(crate::primary::excision_rows(
        archive,
        &request.id,
    )?);
    let material = sinex.send_change(&change)?;
    crate::primary::change_material(archive, &change, &material)
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
