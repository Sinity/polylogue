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
use crate::blobs::{Change, FileFacts, NewBlob, Stored};
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

mod batch;
mod bytes;
mod excluded;
mod rows;
mod search;
mod staging;

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
        std::sync::Arc<hashed_blobs::Store>,
        SyncSender<Vec<Result<crate::staging::StoredInput>>>,
    ),
    FinishWriting(
        std::sync::Arc<hashed_blobs::Store>,
        SyncSender<Result<crate::staging::Finished>>,
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
    Stop,
    /// A user write is waiting in its own queue.
    Wake,
    Batch(Box<Batch>),
    Flush(SyncSender<WriteReport>),
    /// An excision, with the blob store whose files it changes.
    Excise(
        Box<crate::excise::ExciseRequest>,
        std::sync::Arc<hashed_blobs::Store>,
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
        let result = self
            .bulk
            .lock()
            .map_err(|_| ())?
            .as_ref()
            .ok_or(())?
            .send(message)
            .map_err(|_| ());
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

impl Writer {
    /// Starts the writer thread on `archive`, which the caller has locked.
    /// An empty index is built without syncs: every row in it can be
    /// recomputed, and syncing a first build of tens of GB through a busy
    /// drive costs more than the build. The first derive (the end of the
    /// pass), or an excision, ends the build: it builds what the build
    /// deferred, syncs the index and clears the mark; an index still marked
    /// when the archive is next opened is set aside and rebuilt.
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
        if empty {
            // Make the rebuild marker durable before unsynchronized writes.
            // Keep WAL throughout: an open read transaction must not prevent the
            // first build from finishing or the writer from stopping.
            index.pragma_update(None, "synchronous", "full")?;
            index.execute(
                "INSERT INTO meta(key, value) VALUES ('unsynced', '1')
                 ON CONFLICT(key) DO UPDATE SET value = '1'",
                [],
            )?;
            index.pragma_update(None, "synchronous", "off")?;
            // The log is not copied into the database file until the rows
            // are written: each copy read every page back from the log and
            // wrote it again, which cost more than writing the rows.
            index.pragma_update(None, "wal_autocheckpoint", 0)?;
        }
        // A first build inserts rows without maintaining the indexes only
        // reads use, or the unique indexes on natural keys; each is built
        // once from a sorted scan when the build ends. The writer's other
        // indexes stay, and an index missing from these lists is kept, which
        // is the safe default.
        let deferred_indexes = if empty {
            let mut defs = Vec::new();
            for name in KEY_INDEXES.iter().chain(READ_ONLY_INDEXES) {
                let sql: Option<String> = index
                    .query_row(
                        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?1",
                        [name],
                        |r| r.get(0),
                    )
                    .optional()?;
                if let Some(sql) = sql {
                    index.execute(&format!("DROP INDEX \"{name}\""), [])?;
                    defs.push(sql);
                }
            }
            defs
        } else {
            Vec::new()
        };
        let archive_conn = crate::archive::open_archive_writer(archive)?;
        let excisions = crate::excise::any_excision(&archive_conn)?;
        let mut inner = Inner {
            archive: archive_conn,
            index,
            report: WriteReport::default(),
            database_states: HashMap::new(),
            first_build: empty.then(crate::first_build::FirstBuild::default),
            deferred_indexes,
            derive_due: true,
            derived_version: None,
            logs_hold_excised: false,
            search: Search::start(archive.index_db(), !empty)?,
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
        store: std::sync::Arc<hashed_blobs::Store>,
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

    /// Completes blobs a stop left `writing` ([`Inner::finish_writing`]).
    pub fn finish_writing(
        &self,
        store: std::sync::Arc<hashed_blobs::Store>,
    ) -> Result<crate::staging::Finished> {
        let (reply, wait) = sync_channel(1);
        self.tx
            .as_ref()
            .ok_or(StoreError::WriterGone)?
            .send(Msg::FinishWriting(store, reply))
            .map_err(|_| StoreError::WriterGone)?;
        wait.recv().map_err(|_| StoreError::WriterGone)?
    }

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
        let (reply, wait) = sync_channel(1);
        self.tx
            .as_ref()
            .ok_or(StoreError::WriterGone)?
            .send(Msg::Flush(reply))
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
        store: std::sync::Arc<hashed_blobs::Store>,
    ) -> Result<Vec<Result<crate::staging::StoredInput>>> {
        self.enqueue_group(group, store)?
            .recv()
            .map_err(|_| StoreError::WriterGone)
    }

    /// Queue a group; parsing can continue while the writer commits it.
    pub fn enqueue_group(
        &self,
        group: Vec<crate::staging::PreparedBatch>,
        store: std::sync::Arc<hashed_blobs::Store>,
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
    archive: Connection,
    index: Connection,
    report: WriteReport,
    database_states: HashMap<String, (i64, u64, crate::tool_databases::SavedDatabase)>,
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
        let mut finish_writing = Vec::new();
        let mut assets = Vec::new();
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
            let _busy = crate::timing::enter(crate::timing::Stage::Writer);
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
                    Msg::FinishWriting(store, reply) => {
                        finish_writing.push((store, reply));
                        false
                    }
                    Msg::Assets(staged, reply) => {
                        assets.push((staged, reply));
                        false
                    }
                    Msg::StoreBlob(blob, reply) => {
                        blobs.push((blob, reply));
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
                    Msg::Flush(reply) => {
                        flushes.push(reply);
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
            group.retain(|b| !self.report.failed_inputs.contains(&b.input.id));
            if !group.is_empty() {
                crate::timing::time(crate::timing::Stage::Rows, || self.write_group(&group));
                // Derived rows follow every commit; a first build derives
                // once, after its indexes exist.
                if self.first_build.is_none()
                    && let Err(e) = self.derive()
                {
                    self.report.errors.push(format!("derive: {e}"));
                }
            }
            // Later queue replies also observe release of the parsed batches'
            // memory charges, not only their committed database rows.
            drop(group);
            for (prepared, batch, reply) in single_files.drain(..) {
                let _ = reply.send(self.store_bytes(*prepared, *batch));
            }
            self.store_groups(std::mem::take(&mut file_groups));
            for (store, reply) in finish_writing.drain(..) {
                let _ = reply.send(self.finish_writing(&store));
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
            let derived_on_flush = !flushes.is_empty() && self.first_build.is_none();
            if derived_on_flush && let Err(e) = self.derive() {
                self.report.errors.push(format!("derive: {e}"));
            }
            for (staged, reply) in assets.drain(..) {
                let result = crate::assets::install(&mut self.index, &self.archive, *staged);
                self.derive_due |= result.is_ok();
                let _ = reply.send(result);
            }
            // Whatever a flush reports written is also searchable.
            if derived_on_flush && let Err(e) = crate::text_rows::index_pending(&self.index, || ())
            {
                self.report
                    .errors
                    .push(format!("write search entries: {e}"));
            }
            for reply in flushes.drain(..) {
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
        let derived = crate::timing::time(crate::timing::Stage::Derive, || {
            crate::derive::run(&mut self.index)
        })?;
        self.derive_due = false;
        self.derived_version = Some(version(&self.index)?);
        Ok(derived)
    }

    /// Ends an unsynced build: makes the index durable, then clears the mark
    /// and returns to normal syncing.
    fn finish_build(&mut self) -> Result<()> {
        if self.first_build.is_none() {
            return Ok(());
        }
        // The rows' log is copied into the database file once, before the
        // scans that build the deferred indexes: read through a log of tens
        // of GB, every page costs a search of the whole log's index. The
        // usual checkpoints resume for what those builds write.
        let copied = std::time::Instant::now();
        self.index
            .query_row("PRAGMA wal_checkpoint(PASSIVE)", [], |_| Ok(()))?;
        self.index.pragma_update(
            None,
            "wal_autocheckpoint",
            crate::archive::WAL_AUTOCHECKPOINT,
        )?;
        crate::timing::add(crate::timing::Stage::Sync, copied.elapsed());
        self.build_deferred()?;
        let synced = std::time::Instant::now();
        let i = &self.index;
        i.pragma_update(None, "synchronous", "full")?;
        // A FULL checkpoint syncs the database file, with every page the
        // unsynced checkpoints wrote, then the FULL commit syncs the log
        // with the marker's removal. No journal-mode switch can wait on old
        // readers.
        i.query_row("PRAGMA wal_checkpoint(PASSIVE)", [], |_| Ok(()))?;
        i.execute("UPDATE meta SET value = '0' WHERE key = 'unsynced'", [])?;
        i.pragma_update(None, "synchronous", "normal")?;
        crate::timing::add(crate::timing::Stage::Sync, synced.elapsed());
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
        for sql in std::mem::take(&mut self.deferred_indexes) {
            self.index.execute(&sql, [])?;
        }
        add(Stage::SecondaryIndexes, started.elapsed());
        Ok(())
    }

    /// An excision in one pass ([`crate::excise`]): plan, then one
    /// transaction per database, committed embeddings, archive, index; a dry
    /// run plans and rolls back. Deleting vectors first only costs a later
    /// purchase if a later commit fails. Once the archive commits, the
    /// excision is in force: an index that missed it is rebuilt when the
    /// archive is next opened, and the files of blobs marked for deletion go
    /// after the commits, or when the archive is next opened. Running a
    /// recorded request again finishes what is left.
    fn write_excision(
        &mut self,
        request: &crate::excise::ExciseRequest,
        store: &hashed_blobs::Store,
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
        self.finish_build()?;
        let previous: i64 = self
            .archive
            .pragma_query_value(None, "synchronous", |r| r.get(0))?;
        self.archive
            .pragma_update(None, "synchronous", previous.max(2))?;
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
                        self.report.failed_inputs.insert(b.input.id.clone());
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
            if self.report.failed_inputs.contains(&b.input.id) {
                continue;
            }
            if let Err(e) = self.write(std::iter::once(&**b)) {
                self.report.failed_inputs.insert(b.input.id.clone());
                self.report
                    .errors
                    .push(format!("input {}: {e}", b.input.id));
            }
        }
    }

    /// One user request in its own `archive.db` transaction.
    /// A blob row, then its file, then `present`: a stop between them
    /// leaves a `writing` row nothing refers to, which opening the archive
    /// finishes.
    fn store_blob(
        &mut self,
        new_blob: crate::NewBlob,
        file: Option<crate::staging::PreparedBytes>,
    ) -> Result<BlobHash> {
        let _write = crate::write_lock::hold(&self.archive)?;
        let Some(file) = file else {
            let tx = self.archive.transaction()?;
            bytes::insert_blob(&tx, &new_blob)?;
            tx.commit()?;
            return Ok(new_blob.hash);
        };
        if file.owner != self.location.root || file.hash() != new_blob.hash {
            return Err(StoreError::Invalid(
                "a blob's file is a private file of this archive with its bytes".into(),
            ));
        }
        self.archive.execute(
            "INSERT OR IGNORE INTO blob(hash, size, stored_size, place, encoding, state)
             VALUES (?1, ?2, 0, 'file', 'plain', 'writing')",
            params![&new_blob.hash.0[..], new_blob.size as i64],
        )?;
        let present: bool = self.archive.query_row(
            "SELECT state = 'present' FROM blob WHERE hash = ?1",
            [&new_blob.hash.0[..]],
            |r| r.get(0),
        )?;
        if !present {
            let crate::staging::Prepared::Local { store, prepared } = file.file else {
                return Err(StoreError::Invalid(
                    "bytes for Sinex reached the blob store".into(),
                ));
            };
            prepared.install_checked::<StoreError>(|| Ok(()))?;
            let size = i64::try_from(store.stored_size(&hashed_blobs::Hash(new_blob.hash.0))?)
                .map_err(|_| StoreError::Invalid("blob exceeds archive address space".into()))?;
            self.archive.execute(
                "UPDATE blob SET state = 'present', stored_size = ?2 WHERE hash = ?1",
                params![&new_blob.hash.0[..], size],
            )?;
        }
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
        let atx = self.archive.transaction()?;
        let seq = bump_counter(&atx)?;
        let mut chains = Vec::new();
        for b in batches.clone() {
            let (chain, rows) = write_bytes(&atx, b, seq)?;
            if let (Some(sinex), Some(rows)) = (&self.sinex, &rows) {
                staging::send_to_sinex(sinex.as_ref(), &atx, rows, None, b.new_blob.as_ref())?;
            }
            chains.push(chain);
        }
        crate::fault::check("write.archive_commit", "")?;
        crate::write_lock::commit(atx)?;
        crate::fault::check("write.between_commits", "")?;

        let itx = self
            .index
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let counter = bump_counter(&itx)?;
        let mut stats = (0u64, 0u64);
        for (b, chain) in batches.zip(chains) {
            if let Some(chain) = chain {
                for p in &b.parse {
                    if let Some(saved) = &p.database_state {
                        self.database_states.insert(b.input.id.clone(), (chain, p.parsed_upto, saved.clone()));
                    }
                    stats.1 += write_rows(&itx, b, p, chain, counter, self.first_build.as_ref())?;
                }
                if let Some((saved_chain, length, saved)) = self.database_states.get(&b.input.id)
                    && *saved_chain == chain
                {
                    crate::tool_databases::save_when_parsed(&itx, &b.input.id, chain, *length, saved)?;
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
