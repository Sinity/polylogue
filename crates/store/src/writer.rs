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

use crate::archive::{Archive, open_write};
use crate::blobs::{Change, FileFacts, Piece, Stored};
use crate::{Result, StoreError};
use polylogue_model::{
    Accounting, Block, ErrorReason, Event, Json, Link, Message, MessageRow, Output, PromptKind,
    Role, Session, SessionId, Timestamp, ToolOutcome,
};
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, sync_channel};
use std::thread::JoinHandle;

mod batch;
mod bytes;
mod excluded;
mod rows;
mod search;

pub(crate) use batch::unpack_block;
pub use batch::{Batch, InFlight, InputKey, Observation, ParseStep, WriteReport};
use batch::{block_text, message_hash, pack_block, pack_event};
use bytes::*;
use excluded::*;
pub(crate) use rows::restore_text;
use rows::*;
use search::*;

enum Msg {
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
    tx: Option<SyncSender<Msg>>,
    handle: Option<JoinHandle<()>>,
}

/// zstd level for text bodies in `index.db`: fast enough for the writer
/// thread (about 500 MB/s per core) and about a quarter of the size.
const TEXT_LEVEL: i32 = 1;

/// Unpacks a `text.body`.
pub(crate) fn unpack_text(packed: &[u8]) -> Result<String> {
    let bytes = zstd::stream::decode_all(packed)?;
    String::from_utf8(bytes).map_err(|e| StoreError::Io(std::io::Error::other(e)))
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
        crate::cost::reprice(&index, polylogue_model::Catalog::builtin())?;
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
        let archive_conn =
            open_write(&archive.archive_db(), crate::archive::cache::ARCHIVE_WRITER)?;
        let refused = crate::excise::Refused::load(&archive_conn)?;
        let mut inner = Inner {
            archive: archive_conn,
            index,
            report: WriteReport::default(),
            unsynced: empty,
            deferred_indexes,
            search: Search::start(archive.index_db(), !empty)?,
            embeddings_db: archive.embeddings_db(),
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
            tx: Some(tx),
            handle: Some(handle),
        })
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

    /// Waits until every batch sent before this call is committed, and
    /// returns what was written since the previous flush.
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
        self.tx.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// A clonable sender into the writer, for worker threads.
#[derive(Clone)]
pub struct WriterSender(SyncSender<Msg>);

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

fn derive_via(tx: &SyncSender<Msg>) -> Result<u64> {
    let (reply, wait) = sync_channel(1);
    tx.send(Msg::Derive(reply))
        .map_err(|_| StoreError::WriterGone)?;
    wait.recv().map_err(|_| StoreError::WriterGone)?
}

fn user_via(
    tx: &SyncSender<Msg>,
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
    embeddings_db: std::path::PathBuf,
    /// Sessions and messages excised: their rows are never written again.
    refused: crate::excise::Refused,
}

impl Inner {
    fn run(&mut self, rx: &Receiver<Msg>) {
        let mut flushes = Vec::new();
        let mut users = Vec::new();
        let mut excisions = Vec::new();
        let mut derives = Vec::new();
        while let Ok(first) = rx.recv() {
            let mut group = Vec::new();
            let mut rows = 0;
            let mut bytes = 0;
            let mut push =
                |m: Msg, group: &mut Vec<Box<Batch>>, rows: &mut usize, bytes: &mut u64| match m {
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
            for (request, reply) in users.drain(..) {
                let _ = reply.send(self.write_user(&request));
            }
            for (request, store, reply) in excisions.drain(..) {
                let _ = reply.send(self.write_excision(&request, &store));
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
                && let Err(e) = write_pending_search(&self.index, i64::MAX)
            {
                self.report
                    .errors
                    .push(format!("write search entries: {e}"));
            }
            for reply in flushes.drain(..) {
                let _ = reply.send(std::mem::take(&mut self.report));
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
        const CHUNK: i64 = 100_000;
        let started = std::time::Instant::now();
        for (bit, table) in [(1i64, "text_fts"), (2, "output_fts")] {
            let mut after = 0i64;
            loop {
                let tx = self.index.transaction()?;
                let rows: Vec<(i64, Vec<u8>)> = tx
                    .prepare_cached(
                        "SELECT id, body FROM text WHERE id > ?1 AND indexed & ?2 ORDER BY id LIMIT ?3",
                    )?
                    .query_map(params![after, bit, CHUNK], |r| Ok((r.get(0)?, r.get(1)?)))?
                    .collect::<rusqlite::Result<_>>()?;
                let Some(&(last, _)) = rows.last() else {
                    break;
                };
                // Unpacking and folding run on every core; FTS5 inserts
                // them in id order on this thread.
                let folded: Vec<(i64, String)> = std::thread::scope(|s| {
                    let parts: Vec<_> = rows
                        .chunks(rows.len().div_ceil(8).max(1))
                        .map(|part| {
                            s.spawn(move || {
                                part.iter()
                                    .map(|(id, packed)| {
                                        unpack_text(packed)
                                            .map(|b| (*id, polylogue_model::text::fold(&b)))
                                    })
                                    .collect::<Result<Vec<_>>>()
                            })
                        })
                        .collect();
                    parts
                        .into_iter()
                        .map(|h| h.join().expect("fold thread"))
                        .collect::<Result<Vec<Vec<_>>>>()
                        .map(|v| v.into_iter().flatten().collect())
                })?;
                {
                    let mut ins = tx.prepare_cached(&format!(
                        "INSERT INTO {table}(rowid, body) VALUES (?1, ?2)"
                    ))?;
                    for (id, body) in &folded {
                        ins.execute(params![id, body])?;
                    }
                }
                tx.commit()?;
                after = last;
            }
        }
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

    /// One excision in its own transactions: `archive.db`, `index.db` and
    /// `embeddings.db` commit together or not at all (a dry run rolls all
    /// three back); the blob files it replaced are deleted after they
    /// commit.
    fn write_excision(
        &mut self,
        request: &crate::excise::ExciseRequest,
        store: &hashed_blobs::Store,
    ) -> Result<crate::excise::Excised> {
        let embeddings = if self.embeddings_db.exists() {
            let e = open_write(&self.embeddings_db, crate::archive::cache::SETUP)?;
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
        let (done, files) = match crate::excise::apply(&txs, request) {
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
        atx.commit()?;
        itx.commit()?;
        if let Some(e) = &embeddings {
            e.execute_batch("COMMIT")?;
        }
        for hash in files {
            if let Err(e) = store.delete(&hashed_blobs::Hash(hash.0)) {
                self.report
                    .errors
                    .push(format!("excision: delete blob {hash}: {e}"));
            }
        }
        self.refused.sessions.extend(done.sessions.iter().cloned());
        self.refused
            .messages
            .extend(crate::excise::Refused::load(&self.archive)?.messages);
        Ok(done)
    }

    /// Writes a group in one transaction per database. If it fails, each
    /// batch is retried alone, so one bad batch costs only itself.
    fn write_group(&mut self, group: &[Box<Batch>]) {
        if !self.refused.is_empty() && group.iter().any(|b| touches(b, &self.refused)) {
            let filtered: Vec<Box<Batch>> = group
                .iter()
                .map(|b| Box::new(without_refused(b, &self.refused)))
                .collect();
            return self.write_group_inner(&filtered);
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

    /// One user request in its own `archive.db` transaction.
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
            tx.commit()?;
        }
        Ok(outcome)
    }

    fn write<'a>(&mut self, batches: impl Iterator<Item = &'a Batch> + Clone) -> Result<()> {
        for b in batches.clone() {
            if let Some(p) = &b.parse {
                p.accounting.check()?;
            }
        }
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
            if let (Some(p), Some(chain)) = (&b.parse, chain) {
                stats.1 += write_rows(&itx, b, p, chain, counter, self.unsynced)?;
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
    match &b.parse {
        Some(p) => p.parsed_upto.saturating_sub(p.start),
        None => b.piece.as_ref().map_or(0, |p| p.size),
    }
}

fn weight(b: &Batch) -> usize {
    1 + b.parse.as_ref().map_or(0, |p| {
        p.output.messages.len() + p.output.events.len() + p.output.links.len()
    })
}

// ---------------------------------------------------------------- archive.db

// ------------------------------------------------------------------ index.db

#[cfg(test)]
mod tests;
