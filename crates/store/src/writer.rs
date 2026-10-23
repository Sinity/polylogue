//! The one writer (architecture 5.2-5.6). It runs on its own thread, owns
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
use crate::blobs::{Change, FileFacts, Piece, Stored};
use crate::{Result, StoreError};
use polylogue_model::{
    Accounting, Block, ErrorReason, Event, Json, Link, Message, MessageRow, Output, PromptKind,
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
pub(crate) use rows::promote_current;
mod search;

use crate::encoding::PreparedMessage;
#[cfg(test)]
use crate::encoding::pack_block;
use crate::encoding::{block_text, pack_event, unpack_block};
pub use batch::{Batch, InFlight, InputKey, Observation, ParseStep, WriteReport};
pub(crate) use bytes::observation_id;
use bytes::*;
use excluded::*;
use rows::*;
use search::*;

enum Msg {
    Parsers(Vec<(String, u32)>, SyncSender<Result<()>>),
    Stop,
    Batch(Box<Batch>),
    Flush(SyncSender<WriteReport>),
    /// A note, delivery or setting; answered once committed.
    User(
        Box<crate::user_writes::UserRequest>,
        SyncSender<Result<crate::user_writes::UserOutcome>>,
    ),
    /// An excision, with the blob store whose files it changes.
    Excise(
        Box<crate::excise::ExciseRequest>,
        std::sync::Arc<hashed_blobs::Store>,
        SyncSender<Result<crate::excise::Excised>>,
    ),
    /// Run the derive step over sessions whose derived rows are stale;
    /// answered with the number of sessions derived.
    Derive(SyncSender<Result<u64>>),
}

/// Handle to the writer thread. Dropping it stops the thread after the
/// batches already sent are written.
pub struct Writer {
    tx: Option<WriterQueue>,
    handle: Option<JoinHandle<()>>,
}

/// Serializes admission with shutdown. Once the stop message is queued,
/// every retained sender refuses new work rather than acknowledging a
/// message that the receiver will never process.
#[derive(Clone)]
struct WriterQueue(Arc<Mutex<Option<SyncSender<Msg>>>>);

impl WriterQueue {
    fn send(&self, message: Msg) -> std::result::Result<(), ()> {
        self.0
            .lock()
            .map_err(|_| ())?
            .as_ref()
            .ok_or(())?
            .send(message)
            .map_err(|_| ())
    }

    fn stop(&self) {
        if let Ok(mut admission) = self.0.lock()
            && let Some(sender) = admission.take()
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
    "session_collection",
    "session_live",
    "session_live_checkout",
    "message_prose",
    "message_unseen",
    "message_branch",
    "message_copy",
    "message_response",
    "message_native",
    "block_secret",
    "tool_call_tool",
    "tool_call_action",
    "tool_call_failed",
    "tool_call_path",
    "tool_call_repo_command",
    "message_disagreement_input",
    "link_child",
    "event_after",
    "event_kind",
    "commit_link_hash",
    "pr_link_number",
    "work_node_session",
    "marker_message",
    "removed_session_changed",
];

/// Batches waiting for the writer before senders block.
const QUEUE: usize = 4;

/// Input bytes a transaction covers before committing. A batch's bytes
/// stay counted against the parse pool's limit until its transaction
/// commits ([`InFlight`]); committing well before that limit lets the pool
/// keep parsing while the writer commits, instead of the two alternating.
const BYTES_PER_TRANSACTION: u64 = 128 << 20;

impl Writer {
    /// Starts the writer thread on `archive`, which the caller has locked.
    /// An empty index is built without syncs (architecture 2: "index.db
    /// built unsynced"): every row in it can be recomputed, and syncing a
    /// first build of tens of GB through a busy drive costs more than the
    /// build. The first flush syncs it and clears the mark; an index still
    /// marked when the archive is next opened is set aside and rebuilt.
    pub fn start(archive: &Archive) -> Result<Self> {
        let index = open_write(&archive.index_db(), crate::archive::cache::INDEX_WRITER)?;
        let _publication = crate::publication::write(&index)?;
        let empty: bool =
            index.query_row("SELECT NOT EXISTS (SELECT 1 FROM parse_state)", [], |r| {
                r.get(0)
            })?;
        if empty {
            index.execute(
                "INSERT INTO meta(key, value) VALUES ('unsynced', '1')
                 ON CONFLICT(key) DO UPDATE SET value = '1'",
                [],
            )?;
            index.pragma_update(None, "synchronous", "off")?;
            // Pages go straight to the database file, once, instead of to
            // the WAL and again at checkpoint: a first build wrote 2.2 times
            // what it kept in WAL mode. The rollback journal stays in memory,
            // so a failed transaction still rolls back; a crash leaves the
            // mark set and the index is rebuilt.
            index.pragma_update(None, "journal_mode", "memory")?;
        }
        // A first build inserts rows without maintaining the indexes only
        // reads use; each is built once from a sorted scan when the build
        // ends. Indexes the writer's own statements use stay, and an index
        // missing from this list is kept, which is the safe default.
        let deferred_indexes = if empty {
            let mut defs = Vec::new();
            for name in READ_ONLY_INDEXES {
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
        if crate::excise::recovery::current(&archive_conn)?
            != crate::excise::recovery::current(&index)?
            || crate::excise::vector_cleanup::pending(&archive_conn)?
            || crate::excise::blob_cleanup::pending(&archive_conn)?
        {
            return Err(StoreError::ExcisionPending);
        }
        let refused = crate::excise::Refused::load(&archive_conn)?;
        let mut inner = Inner {
            archive: archive_conn,
            index,
            report: WriteReport::default(),
            unsynced: empty,
            deferred_indexes,
            search: Search::start(archive.index_db(), !empty)?,
            location: archive.clone(),
            refused,
        };
        // Senders block once a few batches wait. The writer is the slowest
        // stage of a first build, and a parsed batch holds its rows and
        // packed bodies; a deep queue let finished batches pile up past
        // the parse pool's bytes-in-flight limit (8.4 GB on the real
        // corpus). With a short queue the pool keeps a job's bytes counted
        // until its batch is queued, so memory stays bounded by that limit.
        let (tx, rx) = sync_channel::<Msg>(QUEUE);
        let handle = std::thread::Builder::new()
            .name("polylogue-writer".into())
            .spawn(move || inner.run(&rx))?;
        Ok(Self {
            tx: Some(WriterQueue(Arc::new(Mutex::new(Some(tx))))),
            handle: Some(handle),
        })
    }

    /// Select format versions before their parse jobs start emitting rows.
    pub fn require_parsers(&self, versions: &[(&str, u32)]) -> Result<()> {
        let (reply, wait) = sync_channel(1);
        self.tx
            .as_ref()
            .ok_or(StoreError::WriterGone)?
            .send(Msg::Parsers(
                versions
                    .iter()
                    .map(|(format, version)| ((*format).to_owned(), *version))
                    .collect(),
                reply,
            ))
            .map_err(|_| StoreError::WriterGone)?;
        wait.recv().map_err(|_| StoreError::WriterGone)?
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
    /// it to commit. It is written after the batches already queued.
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
    /// derived (architecture 5.6), after the batches already queued.
    pub fn derive(&self) -> Result<u64> {
        derive_via(self.tx.as_ref().ok_or(StoreError::WriterGone)?)
    }

    /// Waits until earlier batches are committed and their memory reservations
    /// are released, then returns what was written since the previous flush.
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
    tx.send(Msg::User(Box::new(request), reply))
        .map_err(|_| StoreError::WriterGone)?;
    wait.recv().map_err(|_| StoreError::WriterGone)?
}

struct Inner {
    archive: Connection,
    index: Connection,
    report: WriteReport,
    /// The index is being built without syncs ([`Writer::start`]).
    unsynced: bool,
    /// Secondary indexes dropped for a first build, recreated when it ends.
    deferred_indexes: Vec<String>,
    /// The search thread ([`search_thread`]): woken after each commit, and
    /// allowed to write once the index is out of a first build.
    search: Search,
    location: Archive,
    /// Sessions and messages excised: their rows are never written again.
    refused: crate::excise::Refused,
}

impl Inner {
    fn run(&mut self, rx: &Receiver<Msg>) {
        let mut flushes = Vec::new();
        let mut users = Vec::new();
        let mut excisions = Vec::new();
        let mut derives = Vec::new();
        let mut parsers = Vec::new();
        while let Ok(first) = rx.recv() {
            let mut stopping = false;
            let mut group = Vec::new();
            let mut rows = 0;
            let mut bytes = 0;
            let mut push =
                |m: Msg, group: &mut Vec<Box<Batch>>, rows: &mut usize, bytes: &mut u64| match m {
                    Msg::Parsers(versions, reply) => {
                        parsers.push((versions, reply));
                        false
                    }
                    Msg::Stop => {
                        stopping = true;
                        false
                    }
                    Msg::Batch(b) => {
                        *rows += weight(&b);
                        *bytes += input_bytes(&b);
                        group.push(b);
                        true
                    }
                    Msg::Flush(reply) => {
                        flushes.push(reply);
                        false
                    }
                    Msg::User(request, reply) => {
                        users.push((request, reply));
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
                if !self.unsynced
                    && let Err(e) = crate::timing::time(crate::timing::Stage::Derive, || {
                        crate::derive::run(&mut self.index)
                    })
                {
                    self.report.errors.push(format!("derive: {e}"));
                }
            }
            // Later queue replies also observe release of the parsed batches'
            // memory reservations, not only their committed database rows.
            drop(group);
            for (versions, reply) in parsers.drain(..) {
                let result = (|| {
                    let tx = crate::publication::transaction(
                        &self.index,
                        TransactionBehavior::Deferred,
                    )?;
                    let before = tx.total_changes();
                    for (format, version) in versions {
                        rows::require_parser(&tx, &format, version)?;
                    }
                    if tx.total_changes() != before {
                        bump_counter(&tx)?;
                    }
                    tx.commit()?;
                    Ok(())
                })();
                let _ = reply.send(result);
            }
            for (request, reply) in users.drain(..) {
                let _ = reply.send(self.write_user(&request));
            }
            for (request, store, reply) in excisions.drain(..) {
                let result = self.write_excision(&request, &store);
                let interrupted = matches!(&result, Err(StoreError::ExcisionInterrupted(_)));
                let _ = reply.send(result);
                if interrupted {
                    return;
                }
            }
            if !derives.is_empty() {
                // A first build's indexes exist before anything is derived
                // from its rows.
                let result = self.settle().and_then(|()| {
                    crate::timing::time(crate::timing::Stage::Derive, || {
                        crate::derive::run(&mut self.index)
                    })
                });
                for reply in derives.drain(..) {
                    let _ = reply.send(match &result {
                        Ok(n) => Ok(*n),
                        Err(e) => Err(StoreError::Invalid(format!("derive: {e}"))),
                    });
                }
            }
            if !flushes.is_empty()
                && let Err(e) = self.settle()
            {
                self.report.errors.push(format!("sync the index: {e}"));
            }
            // Whatever a flush reports written is also derived.
            if !flushes.is_empty()
                && let Err(e) = crate::timing::time(crate::timing::Stage::Derive, || {
                    crate::derive::run(&mut self.index)
                })
            {
                self.report.errors.push(format!("derive: {e}"));
            }
            // Whatever a flush reports written is also searchable.
            if !flushes.is_empty()
                && let Err(e) = crate::text_rows::index_pending(&self.index)
            {
                self.report
                    .errors
                    .push(format!("write search entries: {e}"));
            }
            for reply in flushes.drain(..) {
                let _ = reply.send(std::mem::take(&mut self.report));
            }
            if stopping {
                break;
            }
        }
        let _ = self.settle();
    }

    /// Ends an unsynced build: makes the index durable, then clears the mark
    /// and returns to normal syncing.
    fn settle(&mut self) -> Result<()> {
        if !self.unsynced {
            return Ok(());
        }
        self.build_deferred()?;
        let _publication = crate::publication::write(&self.index)?;
        let synced = std::time::Instant::now();
        let i = &self.index;
        i.pragma_update(None, "synchronous", "full")?;
        // Back to WAL, so readers and the writer stop waiting on each
        // other; the switch syncs the database file.
        i.query_row("PRAGMA journal_mode = wal", [], |_| Ok(()))?;
        i.execute("UPDATE meta SET value = '0' WHERE key = 'unsynced'", [])?;
        i.pragma_update(None, "synchronous", "normal")?;
        crate::timing::add(crate::timing::Stage::Sync, synced.elapsed());
        self.unsynced = false;
        self.search.ready();
        Ok(())
    }

    /// Ends a first build's bulk load: writes the search entries the build
    /// marked, rebuilds the trigram index from `tool_call`, and creates the
    /// secondary indexes it dropped. Each from one ordered scan, so the
    /// indexes are written once instead of rewritten through the build.
    fn build_deferred(&mut self) -> Result<()> {
        use crate::timing::{Stage, add};
        // Bulk inserts are finished; release their cached programs before the search scan.
        self.index.flush_prepared_statement_cache();
        let started = std::time::Instant::now();
        crate::text_rows::index_marked(&self.index)?;
        add(Stage::SearchIndex, started.elapsed());
        let started = std::time::Instant::now();
        let publication = crate::publication::write(&self.index)?;
        self.index.execute(
            "INSERT INTO tool_call_trigram(tool_call_trigram) VALUES ('rebuild')",
            [],
        )?;
        add(Stage::TrigramIndex, started.elapsed());
        drop(publication);
        let started = std::time::Instant::now();
        for sql in std::mem::take(&mut self.deferred_indexes) {
            let _publication = crate::publication::write(&self.index)?;
            self.index.execute(&sql, [])?;
        }
        add(Stage::SecondaryIndexes, started.elapsed());
        Ok(())
    }

    /// One transaction per database; a dry run rolls all of them back.
    /// The archive commits first, followed by the derived stores. Local blob
    /// cleanup follows those commits; external backing roots stay read-only.
    fn write_excision(
        &mut self,
        request: &crate::excise::ExciseRequest,
        store: &hashed_blobs::Store,
    ) -> Result<crate::excise::Excised> {
        if crate::excise::recovery::current(&self.archive)?
            != crate::excise::recovery::current(&self.index)?
        {
            return Err(StoreError::ExcisionPending);
        }
        crate::archive::retired::validate(&self.location, &self.archive)?;
        // Persist removal receipts with WAL semantics even during a first build.
        if !request.dry_run {
            self.settle()?;
        }
        if request.dry_run {
            return self.write_excision_rows(request, store);
        }
        // Canonical replacement rows and cleanup intent must be durable before
        // removing old files. Ordinary ingestion keeps its existing sync mode.
        let previous: i64 = self
            .archive
            .pragma_query_value(None, "synchronous", |r| r.get(0))?;
        self.archive
            .pragma_update(None, "synchronous", previous.max(2))?;
        let result = self.write_excision_rows(request, store);
        self.archive
            .pragma_update(None, "synchronous", previous)
            .map_err(|e| StoreError::ExcisionInterrupted(Box::new(e.into())))?;
        result
    }

    fn write_excision_rows(
        &mut self,
        request: &crate::excise::ExciseRequest,
        store: &hashed_blobs::Store,
    ) -> Result<crate::excise::Excised> {
        let publication = crate::publication::write(&self.archive)?;
        let embeddings = if self.location.embeddings_db().exists() {
            let e = open_write(&self.location.embeddings_db(), crate::archive::cache::SETUP)?;
            e.pragma_update(None, "synchronous", "full")?;
            e.execute_batch("BEGIN IMMEDIATE")?;
            Some(e)
        } else {
            None
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
            store,
            seq,
            counter,
        };
        // A restored embeddings file may lack older receipts even though the
        // canonical/index revisions agree. Establish the prior policy prefix
        // before acknowledging this operation's newer watermark.
        if let Some(e) = &embeddings {
            crate::excise::vector_policy::reconcile(e, &atx)?;
        }
        let done = match crate::excise::apply(&txs, request) {
            Ok(r) => r,
            Err(e) => {
                if let Some(e) = &embeddings {
                    let _ = e.execute_batch("ROLLBACK");
                }
                return Err(e);
            }
        };
        if request.dry_run {
            itx.rollback()?;
            atx.rollback()?;
            if let Some(e) = &embeddings {
                e.execute_batch("ROLLBACK")?;
            }
            return Ok(done);
        }
        let refused = crate::excise::Refused::load(&atx)?;
        // Once COMMIT is attempted its outcome may require reconciliation.
        // Stop this writer on error; later requests must not acknowledge old rows.
        let committed: Result<()> = (|| {
            atx.commit()?;
            crate::fault::check("excision.between_commits", &request.id)?;
            itx.commit()?;
            crate::fault::check("excision.after_index_commit", &request.id)?;
            drop(publication);
            if let Some(e) = &embeddings {
                e.execute_batch("COMMIT")?;
            }
            Ok(())
        })()
        .and_then(|()| {
            crate::fault::check("excision.after_vector_commit", &request.id)?;
            crate::archive::retired::remove(&self.location, &self.archive)?;
            crate::excise::vector_cleanup::clear(&self.archive, &request.id)?;
            crate::excise::blob_cleanup::recover(&self.location, &self.archive)
        });
        if let Err(error) = committed {
            return Err(StoreError::ExcisionInterrupted(Box::new(error)));
        }
        self.refused = refused;
        Ok(done)
    }

    /// Writes a group in one transaction per database. If it fails, each
    /// batch is retried alone, so one bad batch costs only itself.
    fn write_group(&mut self, group: &[Box<Batch>]) {
        if !self.refused.is_empty() {
            let refused = match matching(group, &self.refused) {
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
    fn write_user(
        &mut self,
        request: &crate::user_writes::UserRequest,
    ) -> Result<crate::user_writes::UserOutcome> {
        let tx = crate::publication::transaction(&self.archive, TransactionBehavior::Deferred)?;
        let seq = bump_counter(&tx)?;
        let outcome = crate::user_writes::apply(&tx, request, seq)?;
        if outcome.repeat {
            // A resent request: nothing to keep, not even the counter bump.
            tx.rollback()?;
        } else {
            tx.commit()?;
        }
        Ok(outcome)
    }

    fn write<'a>(&mut self, batches: impl Iterator<Item = &'a Batch> + Clone) -> Result<()> {
        for b in batches.clone() {
            crate::fault::check("write.input", &b.input.id)?;
            for p in &b.parse {
                p.accounting.check()?;
            }
        }
        let _publication = crate::publication::write(&self.archive)?;
        let atx = self.archive.transaction()?;
        let seq = bump_counter(&atx)?;
        let mut chains = Vec::new();
        for b in batches.clone() {
            chains.push(write_bytes(&atx, b, seq)?);
        }
        crate::fault::check("write.archive_commit", "")?;
        atx.commit()?;
        crate::fault::check("write.between_commits", "")?;

        let itx = self
            .index
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let counter = bump_counter(&itx)?;
        let mut stats = (0u64, 0u64);
        for (b, chain) in batches.zip(chains) {
            if let Some(chain) = chain {
                for p in &b.parse {
                    stats.1 += write_rows(&itx, b, p, chain, counter, self.unsynced)?;
                }
            }
            stats.0 += 1;
        }
        crate::fault::check("write.index_commit", "")?;
        itx.commit()?;
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
        _ => b.piece.as_ref().map_or(0, |p| p.size),
    }
}

fn weight(b: &Batch) -> usize {
    1 + b
        .parse
        .iter()
        .map(|p| p.output.messages.len() + p.output.events.len() + p.output.links.len())
        .sum::<usize>()
}

// ---------------------------------------------------------------- archive.db

// ------------------------------------------------------------------ index.db

#[cfg(test)]
mod tests;
