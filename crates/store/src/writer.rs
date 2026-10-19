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

use crate::blobs::{Change, FileFacts, Piece, Stored};
use crate::files::{Archive, open_write};
use crate::{Result, StoreError};
use polylogue_model::{
    Accounting, Block, ErrorReason, Event, Link, Message, MessageRow, Output, PromptKind, Role,
    Session, SessionId, SessionKind, Timestamp, TitleSource, ToolOutcome,
};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, sync_channel};
use std::thread::JoinHandle;

/// The input a batch belongs to, as `archive.db` names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputKey {
    /// `archive.db` input id: a location's stable id, or the BLAKE3 of an
    /// imported file's bytes.
    pub id: String,
    /// The origin that parses it (`claude-code`).
    pub format: String,
    /// How faithful the route is, for choosing between versions of a
    /// message (architecture 5.5): 0 live file, 1 spool, 2 import.
    pub route_rank: u8,
}

/// One look at a file that found something.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation {
    pub machine: String,
    pub host: String,
    pub path: String,
    pub facts: Option<FileFacts>,
    pub seen_at_ms: i64,
}

/// One parse step's rows and where they sit in the input's chain.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseStep {
    pub output: Output,
    pub accounting: Accounting,
    /// Where the step's bytes start in the chain. 0 means the chain was
    /// parsed from the start, so the step's counts replace earlier ones.
    pub start: u64,
    /// Offset after the last complete record parsed.
    pub parsed_upto: u64,
    pub parser_version: u32,
    /// BLAKE3 of each message's canonical JSON, in `output.messages` order,
    /// computed on the parse pool ([`ParseStep::hash_messages`]). Empty
    /// means the writer computes them.
    pub hashes: Vec<[u8; 32]>,
    /// Block texts compressed for the `text` table, by BLAKE3 of the text,
    /// prepared on the parse pool. A text missing here is compressed by the
    /// writer.
    pub packed: HashMap<[u8; 32], Vec<u8>>,
    /// Each message's blocks as stored (compressed JSON without their
    /// text), in `output.messages` and block order.
    pub bodies: Vec<Vec<Vec<u8>>>,
    /// Each event's id and compressed payload, in `output.events` order.
    pub events: Vec<(String, Vec<u8>)>,
}

impl ParseStep {
    /// Does the per-message work that needs no database off the writer
    /// thread: version hashes and compressed block texts.
    pub fn prepare(&mut self) {
        self.hashes = self
            .output
            .messages
            .iter()
            .map(|r| message_hash(&r.message))
            .collect();
        let mut packer = zstd::bulk::Compressor::new(TEXT_LEVEL).ok();
        for row in &self.output.messages {
            for block in &row.message.blocks {
                let Some(text) = block_text(block) else {
                    continue;
                };
                let hash = *blake3::hash(text.as_bytes()).as_bytes();
                if let (std::collections::hash_map::Entry::Vacant(slot), Some(p)) =
                    (self.packed.entry(hash), packer.as_mut())
                    && let Ok(bytes) = p.compress(text.as_bytes())
                {
                    slot.insert(bytes);
                }
            }
        }
        let prepared: Result<Vec<Vec<Vec<u8>>>> = self
            .output
            .messages
            .iter()
            .map(|r| r.message.blocks.iter().map(pack_block).collect())
            .collect();
        self.bodies = prepared.unwrap_or_default();
        let events: Result<Vec<(String, Vec<u8>)>> =
            self.output.events.iter().map(pack_event).collect();
        self.events = events.unwrap_or_default();
    }
}

/// A block as the `block.body` column stores it.
fn pack_block(block: &Block) -> Result<Vec<u8>> {
    let mut body = block.clone();
    take_text(&mut body);
    Ok(zstd::bulk::compress(
        &serde_json::to_vec(&body)?,
        TEXT_LEVEL,
    )?)
}

/// An event's id (computed from its session, kind, time and payload, so the
/// same record parsed again is the same row) and stored payload.
fn pack_event(e: &Event) -> Result<(String, Vec<u8>)> {
    let data = serde_json::to_vec(&e.data)?;
    let mut h = blake3::Hasher::new();
    h.update(e.session.to_string().as_bytes());
    h.update(b"\0");
    h.update(e.kind.as_bytes());
    h.update(b"\0");
    h.update(&ms(e.at).unwrap_or(i64::MIN).to_le_bytes());
    h.update(&data);
    Ok((
        h.finalize().to_hex().to_string(),
        zstd::bulk::compress(&data, TEXT_LEVEL)?,
    ))
}

/// Reads a block back from `block.body`.
pub(crate) fn unpack_block(packed: &[u8]) -> Result<Block> {
    Ok(serde_json::from_slice(&zstd::stream::decode_all(packed)?)?)
}

/// The text a block keeps in the `text` table (see [`take_text`]).
fn block_text(block: &Block) -> Option<&str> {
    match block {
        Block::Text { text } if !text.is_empty() => Some(text),
        Block::Thinking { text: Some(t), .. } if !t.is_empty() => Some(t),
        Block::ToolResult {
            output: Some(o), ..
        } if !o.is_empty() => Some(o),
        _ => None,
    }
}

fn message_hash(m: &Message) -> [u8; 32] {
    let json = serde_json::to_vec(m).unwrap_or_default();
    *blake3::hash(&json).as_bytes()
}

/// What the writer stores for one input in one step. `piece` holds the new
/// bytes (already durable when they are a file); `parse` holds the rows
/// parsed from them, if the origin is parsed.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    pub input: InputKey,
    pub change: Change,
    /// Present on the first batch of a change; absent on the later chunks
    /// of a large file stored in several batches (`continues`).
    pub observation: Option<Observation>,
    /// This batch extends the revision the previous batch of this input
    /// started, instead of starting a new one.
    pub continues: bool,
    pub piece: Option<Piece>,
    /// Bytes records carried inline (base64 images), prepared like pieces;
    /// blocks name them by hash.
    pub blobs: Vec<Piece>,
    /// The chain's stored length after this batch.
    pub length: u64,
    pub parse: Option<ParseStep>,
}

/// What the writer did since the last flush.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WriteReport {
    pub batches: u64,
    pub messages: u64,
    pub transactions: u64,
    /// Batches the writer could not store, with the reason. The rest of
    /// their transaction was retried one batch at a time.
    pub errors: Vec<String>,
}

enum Msg {
    Batch(Box<Batch>),
    Flush(SyncSender<WriteReport>),
    /// A note, delivery or setting; answered once committed.
    User(
        Box<crate::user::UserRequest>,
        SyncSender<Result<crate::user::UserOutcome>>,
    ),
    /// An excision, with the blob store whose files it changes.
    Excise(
        Box<crate::excise::ExciseRequest>,
        std::sync::Arc<hashed_blobs::Store>,
        SyncSender<Result<crate::excise::Excised>>,
    ),
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

impl Writer {
    /// Starts the writer thread on `archive`, which the caller has locked.
    /// An empty index is built without syncs (architecture 2: "index.db
    /// built unsynced"): every row in it can be recomputed, and syncing a
    /// first build of tens of GB through a busy drive costs more than the
    /// build. The first flush syncs it and clears the mark; an index still
    /// marked when the archive is next opened is set aside and rebuilt.
    pub fn start(archive: &Archive) -> Result<Self> {
        let index = open_write(&archive.index_db())?;
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
        let archive_conn = open_write(&archive.archive_db())?;
        let refused = crate::excise::Refused::load(&archive_conn)?;
        let mut inner = Inner {
            archive: archive_conn,
            index,
            report: WriteReport::default(),
            unsynced: empty,
            embeddings_db: archive.embeddings_db(),
            refused,
        };
        let (tx, rx) = sync_channel::<Msg>(256);
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
    pub fn user(&self, request: crate::user::UserRequest) -> Result<crate::user::UserOutcome> {
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
    pub fn user(&self, request: crate::user::UserRequest) -> Result<crate::user::UserOutcome> {
        user_via(&self.0, request)
    }
}

fn user_via(
    tx: &SyncSender<Msg>,
    request: crate::user::UserRequest,
) -> Result<crate::user::UserOutcome> {
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
    embeddings_db: std::path::PathBuf,
    /// Sessions and messages excised: their rows are never written again.
    refused: crate::excise::Refused,
}

impl Inner {
    fn run(&mut self, rx: &Receiver<Msg>) {
        let mut flushes = Vec::new();
        let mut users = Vec::new();
        let mut excisions = Vec::new();
        while let Ok(first) = rx.recv() {
            let mut group = Vec::new();
            let mut rows = 0;
            let mut push = |m: Msg, group: &mut Vec<Box<Batch>>, rows: &mut usize| match m {
                Msg::Batch(b) => {
                    *rows += weight(&b);
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
            };
            if push(first, &mut group, &mut rows) {
                while rows < ROWS_PER_TRANSACTION {
                    match rx.try_recv() {
                        Ok(m) => {
                            if !push(m, &mut group, &mut rows) {
                                break;
                            }
                        }
                        Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
                    }
                }
            }
            if !group.is_empty() {
                self.write_group(&group);
            }
            for (request, reply) in users.drain(..) {
                let _ = reply.send(self.write_user(&request));
            }
            for (request, store, reply) in excisions.drain(..) {
                let _ = reply.send(self.write_excision(&request, &store));
            }
            if !flushes.is_empty()
                && let Err(e) = self.settle()
            {
                self.report.errors.push(format!("sync the index: {e}"));
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
        let i = &self.index;
        i.pragma_update(None, "synchronous", "full")?;
        // Back to WAL, so readers and the writer stop waiting on each
        // other; the switch syncs the database file.
        i.query_row("PRAGMA journal_mode = wal", [], |_| Ok(()))?;
        i.execute("UPDATE meta SET value = '0' WHERE key = 'unsynced'", [])?;
        i.pragma_update(None, "synchronous", "normal")?;
        self.unsynced = false;
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
            let e = open_write(&self.embeddings_db)?;
            e.execute_batch("BEGIN IMMEDIATE")?;
            Some(e)
        } else {
            None
        };
        let atx = self.archive.transaction()?;
        let itx = self.index.transaction()?;
        let seq = bump_counter(&atx)?;
        let counter = bump_counter(&itx)?;
        let txs = crate::excise::Txs {
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
        request: &crate::user::UserRequest,
    ) -> Result<crate::user::UserOutcome> {
        let tx = self.archive.transaction()?;
        let seq = bump_counter(&tx)?;
        let outcome = crate::user::apply(&tx, request, seq)?;
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
        atx.commit()?;

        let itx = self.index.transaction()?;
        let counter = bump_counter(&itx)?;
        let mut stats = (0u64, 0u64);
        for (b, chain) in batches.zip(chains) {
            if let (Some(p), Some(chain)) = (&b.parse, chain) {
                stats.1 += write_rows(&itx, b, p, chain, counter)?;
            }
            stats.0 += 1;
        }
        itx.commit()?;
        self.report.batches += stats.0;
        self.report.messages += stats.1;
        self.report.transactions += 1;
        Ok(())
    }
}

/// Whether a batch's rows name an excised session or message.
fn touches(b: &Batch, r: &crate::excise::Refused) -> bool {
    let Some(p) = &b.parse else {
        return false;
    };
    let o = &p.output;
    o.sessions
        .iter()
        .any(|s| r.sessions.contains(&s.id.to_string()))
        || o.messages.iter().any(|m| refused_message(&m.message, r))
        || o.events
            .iter()
            .any(|e| r.sessions.contains(&e.session.to_string()))
        || o.links.iter().any(|l| {
            r.sessions.contains(&l.parent.to_string()) || r.sessions.contains(&l.child.to_string())
        })
}

fn refused_message(m: &Message, r: &crate::excise::Refused) -> bool {
    r.messages.contains(&m.id.to_string()) || r.sessions.contains(&m.session.to_string())
}

/// A copy of a batch without the rows of excised sessions and messages:
/// they are never written again, from any input (architecture 5.6). The
/// bytes and parse position stay, so the input is not parsed again for
/// them.
fn without_refused(b: &Batch, r: &crate::excise::Refused) -> Batch {
    let mut out = b.clone();
    let Some(p) = &mut out.parse else {
        return out;
    };
    let keep_msgs: Vec<bool> = p
        .output
        .messages
        .iter()
        .map(|m| !refused_message(&m.message, r))
        .collect();
    let keep_events: Vec<bool> = p
        .output
        .events
        .iter()
        .map(|e| !r.sessions.contains(&e.session.to_string()))
        .collect();
    let mut i = 0;
    p.output.messages.retain(|_| {
        i += 1;
        keep_msgs[i - 1]
    });
    if p.hashes.len() == keep_msgs.len() {
        let mut i = 0;
        p.hashes.retain(|_| {
            i += 1;
            keep_msgs[i - 1]
        });
    }
    if p.bodies.len() == keep_msgs.len() {
        let mut i = 0;
        p.bodies.retain(|_| {
            i += 1;
            keep_msgs[i - 1]
        });
    }
    let mut i = 0;
    p.output.events.retain(|_| {
        i += 1;
        keep_events[i - 1]
    });
    if p.events.len() == keep_events.len() {
        let mut i = 0;
        p.events.retain(|_| {
            i += 1;
            keep_events[i - 1]
        });
    }
    p.output
        .sessions
        .retain(|s| !r.sessions.contains(&s.id.to_string()));
    p.output.links.retain(|l| {
        !r.sessions.contains(&l.parent.to_string()) && !r.sessions.contains(&l.child.to_string())
    });
    p.output
        .background
        .retain(|u| !r.sessions.contains(&u.session.to_string()));
    out
}

fn weight(b: &Batch) -> usize {
    1 + b.parse.as_ref().map_or(0, |p| {
        p.output.messages.len() + p.output.events.len() + p.output.links.len()
    })
}

/// Increments the database's write counter; one value per transaction.
fn bump_counter(tx: &Transaction<'_>) -> Result<i64> {
    let next: i64 = tx
        .query_row(
            "SELECT CAST(value AS INTEGER) FROM meta WHERE key = 'write_counter'",
            [],
            |r| r.get(0),
        )
        .optional()?
        .unwrap_or(0)
        + 1;
    tx.execute(
        "INSERT INTO meta(key, value) VALUES ('write_counter', ?1)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        [next.to_string()],
    )?;
    Ok(next)
}

// ---------------------------------------------------------------- archive.db

/// Stores the bytes' rows and returns the revision chain the batch's
/// records are in, or `None` when there is none (a disappearance).
fn write_bytes(tx: &Transaction<'_>, b: &Batch, seq: i64) -> Result<Option<i64>> {
    tx.prepare_cached("INSERT OR IGNORE INTO input(id, format) VALUES (?1, ?2)")?
        .execute(params![b.input.id, b.input.format])?;
    for piece in &b.blobs {
        insert_blob(tx, piece)?;
    }
    let latest: Option<(i64, i64, i64)> = tx
        .prepare_cached(
            "SELECT number, chain, length FROM revision WHERE input = ?1
             ORDER BY number DESC LIMIT 1",
        )?
        .query_row([&b.input.id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .optional()?;
    let next = latest.map_or(1, |l| l.0 + 1);

    let (number, chain, offset) = match (&b.change, b.continues, latest) {
        (Change::Disappeared, _, l) => {
            insert_observation(tx, b, "deleted", None, seq)?;
            return Ok(l.map(|l| l.1));
        }
        (_, true, Some((number, chain, length))) => {
            tx.prepare_cached("UPDATE revision SET length = ?3 WHERE input = ?1 AND number = ?2")?
                .execute(params![b.input.id, number, b.length as i64])?;
            (number, chain, length as u64)
        }
        (Change::Unchanged, _, Some((number, chain, _))) => (number, chain, b.length),
        (Change::Appended { from }, false, Some((_, chain, _))) => {
            insert_revision(tx, b, next, chain, "appended", seq)?;
            (next, chain, *from)
        }
        (change, _, _) => {
            let produced = match (change, b.input.route_rank) {
                (_, 2) => "imported",
                (Change::First, _) => "first",
                _ => "rewritten",
            };
            insert_revision(tx, b, next, next, produced, seq)?;
            (next, next, 0)
        }
    };

    if let Some(piece) = &b.piece {
        insert_blob(tx, piece)?;
        tx.prepare_cached(
            "INSERT OR IGNORE INTO extent(input, chain, offset, length, blob, blob_offset)
             VALUES (?1, ?2, ?3, ?4, ?5, 0)",
        )?
        .execute(params![
            b.input.id,
            chain,
            offset as i64,
            piece.size as i64,
            &piece.hash.0[..]
        ])?;
    }
    if b.observation.is_some() && !b.continues {
        let class = match &b.change {
            Change::First if b.input.route_rank == 2 => "imported",
            Change::First => "first",
            Change::Appended { .. } => "appended",
            Change::Rewritten => "rewritten",
            Change::Unchanged => "back",
            Change::Disappeared => "deleted",
        };
        insert_observation(tx, b, class, Some(number), seq)?;
    }
    Ok(Some(chain))
}

fn insert_revision(
    tx: &Transaction<'_>,
    b: &Batch,
    number: i64,
    chain: i64,
    produced: &str,
    seq: i64,
) -> Result<()> {
    tx.prepare_cached(
        "INSERT INTO revision(input, number, produced, chain, length, local_seq)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
    )?
    .execute(params![
        b.input.id,
        number,
        produced,
        chain,
        b.length as i64,
        seq
    ])?;
    Ok(())
}

fn insert_blob(tx: &Transaction<'_>, piece: &Piece) -> Result<()> {
    let (place, stored_size, data) = match &piece.stored {
        Stored::Inline(z) => ("inline", z.len() as u64, Some(z.as_slice())),
        Stored::File { stored_size } => ("file", *stored_size, None),
    };
    tx.prepare_cached(
        "INSERT OR IGNORE INTO blob(hash, size, stored_size, place, encoding, state, data)
         VALUES (?1, ?2, ?3, ?4, 'plain', 'present', ?5)",
    )?
    .execute(params![
        &piece.hash.0[..],
        piece.size as i64,
        stored_size as i64,
        place,
        data
    ])?;
    Ok(())
}

fn insert_observation(
    tx: &Transaction<'_>,
    b: &Batch,
    class: &str,
    revision: Option<i64>,
    seq: i64,
) -> Result<()> {
    let Some(o) = &b.observation else {
        return Ok(());
    };
    tx.prepare_cached("INSERT OR IGNORE INTO machine(id, host, first_seen) VALUES (?1, ?2, ?3)")?
        .execute(params![o.machine, o.host, o.seen_at_ms])?;
    let f = o.facts;
    tx.prepare_cached(
        "INSERT INTO observation(id, input, machine, path, class, revision, length, device, inode,
             size, mtime_ns, ctime_ns, polylogue_version, seen_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
    )?
    .execute(params![
        format!("{:012x}-{:08x}-{}", o.seen_at_ms, seq, b.input.id),
        b.input.id,
        o.machine,
        o.path,
        class,
        revision,
        revision.map(|_| b.length as i64),
        f.map(|f| f.device as i64),
        f.map(|f| f.inode as i64),
        f.map(|f| f.size as i64),
        f.map(|f| f.mtime_ns as i64),
        f.map(|f| f.ctime_ns as i64),
        env!("CARGO_PKG_VERSION"),
        o.seen_at_ms,
    ])?;
    Ok(())
}

// ------------------------------------------------------------------ index.db

/// Writes a parse step's rows; returns the number of message rows written.
fn write_rows(
    tx: &Transaction<'_>,
    b: &Batch,
    p: &ParseStep,
    chain: i64,
    counter: i64,
) -> Result<u64> {
    tx.prepare_cached("INSERT OR IGNORE INTO input_ref(id) VALUES (?1)")?
        .execute([&b.input.id])?;
    let input_pk: i64 = tx
        .prepare_cached("SELECT pk FROM input_ref WHERE id = ?1")?
        .query_row([&b.input.id], |r| r.get(0))?;
    let ctx = Ctx {
        tx,
        input_pk,
        chain,
        rank: i64::from(b.input.route_rank),
        parser_version: i64::from(p.parser_version),
        counter,
        packed: &p.packed,
    };
    let mut sessions: HashMap<String, i64> = HashMap::new();
    for s in &p.output.sessions {
        let pk = ctx.upsert_session(s)?;
        sessions.insert(s.id.to_string(), pk);
    }
    let mut written = 0;
    let mut activity: HashMap<i64, i64> = HashMap::new();
    for (i, row) in p.output.messages.iter().enumerate() {
        let session_pk = ctx.session_pk(&mut sessions, &row.message.session)?;
        let hash = p
            .hashes
            .get(i)
            .copied()
            .unwrap_or_else(|| message_hash(&row.message));
        let bodies = match p.bodies.get(i) {
            Some(b) if b.len() == row.message.blocks.len() => b.clone(),
            _ => row
                .message
                .blocks
                .iter()
                .map(pack_block)
                .collect::<Result<_>>()?,
        };
        written += u64::from(ctx.write_message(row, session_pk, hash, &bodies)?);
        if let Some(at) = ms(row.message.at) {
            let last = activity.entry(session_pk).or_insert(at);
            *last = (*last).max(at);
        }
    }
    for (session_pk, at) in activity {
        tx.prepare_cached(
            "UPDATE session SET last_activity_at = MAX(COALESCE(last_activity_at, ?2), ?2),
                 changed = ?3
             WHERE pk = ?1",
        )?
        .execute(params![session_pk, at, counter])?;
    }
    for (i, e) in p.output.events.iter().enumerate() {
        let session_pk = ctx.session_pk(&mut sessions, &e.session)?;
        let packed = match p.events.get(i) {
            Some(prepared) if p.events.len() == p.output.events.len() => prepared.clone(),
            _ => pack_event(e)?,
        };
        ctx.write_event(e, session_pk, &packed)?;
    }
    for l in &p.output.links {
        ctx.write_link(l)?;
    }
    for u in &p.output.background {
        let (outcome, reason, exit) = outcome_columns(&u.outcome);
        tx.prepare_cached(
            "UPDATE tool_call SET outcome = ?1, outcome_reason = ?2, exit_code = ?3
             WHERE session_pk = (SELECT pk FROM session WHERE id = ?4) AND call_id = ?5",
        )?
        .execute(params![
            outcome,
            reason,
            exit,
            u.session.to_string(),
            u.call_id
        ])?;
    }
    for pk in sessions.values() {
        tx.prepare_cached(
            "INSERT OR IGNORE INTO parsed(input_pk, chain, session_pk, parser_version)
             VALUES (?1, ?2, ?3, ?4)",
        )?
        .execute(params![input_pk, chain, pk, ctx.parser_version])?;
    }
    ctx.write_parse_state(b, p)?;
    Ok(written)
}

struct Ctx<'a, 't> {
    tx: &'a Transaction<'t>,
    input_pk: i64,
    chain: i64,
    rank: i64,
    parser_version: i64,
    counter: i64,
    packed: &'a HashMap<[u8; 32], Vec<u8>>,
}

/// The preferred version of a message, as compared with a new one.
struct Version {
    input_pk: i64,
    chain: i64,
    start: i64,
    end: i64,
    /// Whether the version's session is a main session. A spawned
    /// session's file can repeat its parent's messages; the parent's own
    /// record is the original.
    main: bool,
    complete: bool,
    rank: i64,
    hash: Vec<u8>,
}

impl Version {
    /// Architecture 5.5 and 5.5a. Within one input: a newer revision chain
    /// wins (a rewrite), and within a chain the earliest wire record wins;
    /// the same record sent again (a response still being written) replaces
    /// itself. Across inputs: the originating session's own record over a
    /// copy in a session it spawned, then a complete version, then the
    /// more faithful route, then the smaller content hash, so the choice
    /// does not depend on which input arrived first.
    fn beats(&self, cur: &Version) -> bool {
        if self.input_pk == cur.input_pk {
            if self.chain != cur.chain {
                return self.chain > cur.chain;
            }
            return self.start <= cur.start;
        }
        if self.main != cur.main {
            return self.main;
        }
        if self.complete != cur.complete {
            return self.complete;
        }
        if self.rank != cur.rank {
            return self.rank < cur.rank;
        }
        self.hash < cur.hash
    }
}

fn ms(t: Option<Timestamp>) -> Option<i64> {
    t.map(Timestamp::unix_ms)
}

fn name<T: serde::Serialize>(v: &T) -> String {
    match serde_json::to_value(v) {
        Ok(serde_json::Value::String(s)) => s,
        _ => String::new(),
    }
}

fn parse_name<T: serde::de::DeserializeOwned>(s: &str) -> Option<T> {
    serde_json::from_value(serde_json::Value::String(s.to_owned())).ok()
}

impl Ctx<'_, '_> {
    fn session_pk(&self, cache: &mut HashMap<String, i64>, id: &SessionId) -> Result<i64> {
        let key = id.to_string();
        if let Some(pk) = cache.get(&key) {
            return Ok(*pk);
        }
        let pk = self.upsert_session(&Session::new(id.clone(), SessionKind::Main))?;
        cache.insert(key, pk);
        Ok(pk)
    }

    /// Merges a session's facts into its row (architecture 5.5a), creating
    /// the row on first sight. Pending links naming it as their child are
    /// resolved here.
    fn upsert_session(&self, s: &Session) -> Result<i64> {
        let id = s.id.to_string();
        let existing = self
            .tx
            .prepare_cached(
                "SELECT pk, kind, title, title_source, cwd_json, branch, harness_version, started_at
                 FROM session WHERE id = ?1",
            )?
            .query_row([&id], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, Option<String>>(2)?,
                    r.get::<_, Option<String>>(3)?,
                    r.get::<_, Option<String>>(4)?,
                    r.get::<_, Option<String>>(5)?,
                    r.get::<_, Option<String>>(6)?,
                    r.get::<_, Option<i64>>(7)?,
                ))
            })
            .optional()?;
        let (pk, merged) = match existing {
            Some((pk, kind, title, title_source, cwd, branch, version, started)) => {
                let mut cur =
                    Session::new(s.id.clone(), parse_name(&kind).unwrap_or(SessionKind::Main));
                if let (Some(text), Some(source)) = (
                    title,
                    title_source.as_deref().and_then(parse_name::<TitleSource>),
                ) {
                    cur.offer_title(&text, source);
                }
                cur.cwd = cwd
                    .and_then(|c| serde_json::from_str(&c).ok())
                    .unwrap_or_default();
                cur.git_branch = branch;
                cur.tool_version = version;
                cur.started = started.and_then(Timestamp::from_unix_ms);
                let before = cur.clone();
                cur.merge(s);
                if cur == before {
                    return Ok(pk);
                }
                (Some(pk), cur)
            }
            None => (None, s.clone()),
        };
        let cwd_json = serde_json::to_string(&merged.cwd)?;
        let title = merged.title.as_ref().map(|t| t.text.clone());
        let title_source = merged.title.as_ref().map(|t| name(&t.source));
        let cols = params![
            id,
            merged.id.origin().as_str(),
            merged.id.native().as_str(),
            name(&merged.kind),
            title,
            title_source,
            cwd_json,
            merged.cwd.last(),
            merged.git_branch,
            merged.tool_version,
            ms(merged.started),
            self.counter,
        ];
        match pk {
            Some(pk) => {
                self.tx
                    .prepare_cached(
                        "UPDATE session SET title = ?5, title_source = ?6, cwd_json = ?7, cwd = ?8,
                             branch = ?9, harness_version = ?10, started_at = ?11, changed = ?12
                         WHERE id = ?1 AND origin = ?2 AND native_id = ?3 AND kind = ?4",
                    )?
                    .execute(cols)?;
                Ok(pk)
            }
            None => {
                self.tx
                    .prepare_cached(
                        "INSERT INTO session(id, origin, native_id, kind, title, title_source,
                             cwd_json, cwd, branch, harness_version, started_at, changed, harness)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?2)",
                    )?
                    .execute(cols)?;
                let pk = self.tx.last_insert_rowid();
                self.tx
                    .prepare_cached(
                        "UPDATE link SET child_pk = ?1 WHERE child_id = ?2 AND child_pk IS NULL",
                    )?
                    .execute(params![pk, id])?;
                Ok(pk)
            }
        }
    }

    /// Stores one version of a message and makes it the shown one when it
    /// beats the current one. Returns whether the message row was written.
    fn write_message(
        &self,
        row: &MessageRow,
        session_pk: i64,
        hash: [u8; 32],
        bodies: &[Vec<u8>],
    ) -> Result<bool> {
        let m = &row.message;
        let offset = row.offset as i64;
        let new = Version {
            input_pk: self.input_pk,
            chain: self.chain,
            start: offset,
            end: row.end.max(row.offset) as i64,
            main: self.session_kind(session_pk)? == "main",
            complete: !(m.role == Role::Assistant && m.stop_reason.is_none()),
            rank: self.rank,
            hash: hash.to_vec(),
        };
        let cur = self
            .tx
            .prepare_cached(
                "SELECT m.pk, m.order_input_pk, m.seq, m.input_pk, m.source_chain, m.source_start,
                        m.in_progress, m.route_rank, s.version_hash, m.session_pk, ss.kind
                 FROM message m LEFT JOIN message_source s
                   ON s.message_pk = m.pk AND s.preferred = 1
                 JOIN session ss ON ss.pk = m.session_pk
                 WHERE m.id = ?1",
            )?
            .query_row([m.id.to_string()], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    (r.get::<_, i64>(1)?, r.get::<_, i64>(2)?),
                    Version {
                        input_pk: r.get(3)?,
                        chain: r.get::<_, Option<i64>>(4)?.unwrap_or(0),
                        start: r.get::<_, Option<i64>>(5)?.unwrap_or(0),
                        end: 0,
                        main: r.get::<_, String>(10)? == "main",
                        complete: r.get::<_, i64>(6)? == 0,
                        rank: r.get(7)?,
                        hash: r.get::<_, Option<Vec<u8>>>(8)?.unwrap_or_default(),
                    },
                    (r.get::<_, i64>(9)?, r.get::<_, String>(10)?),
                ))
            })
            .optional()?;

        let Some((pk, order, preferred, (cur_session, cur_kind))) = cur else {
            let pk = self.insert_message(m, session_pk, &new, (self.input_pk, offset))?;
            self.upsert_source(pk, &new, true)?;
            self.insert_blocks(pk, session_pk, m, bodies)?;
            return Ok(true);
        };

        // Within one chain the earliest record of a message wins: a later
        // duplicate (a repeated uuid) is not another version.
        let prior: Option<i64> = self
            .tx
            .prepare_cached(
                "SELECT source_start FROM message_source
                 WHERE message_pk = ?1 AND input_pk = ?2 AND chain = ?3",
            )?
            .query_row(params![pk, self.input_pk, self.chain], |r| r.get(0))
            .optional()?;
        if prior.is_some_and(|start| offset > start) {
            return Ok(false);
        }
        let new_order = order.min((self.input_pk, offset));
        if !new.beats(&preferred) {
            self.upsert_source(pk, &new, false)?;
            if new_order != order {
                self.tx
                    .prepare_cached(
                        "UPDATE message SET order_input_pk = ?2, seq = ?3 WHERE pk = ?1",
                    )?
                    .execute(params![pk, new_order.0, new_order.1])?;
            }
            return Ok(false);
        }
        // A message is stored once, in the session where it originated
        // (architecture 5.4). A spawned session's file can repeat its
        // parent's messages (a compaction call, a side question replaying
        // the history); such a copy may become the shown version, but it
        // never moves the message out of a main session.
        let session_pk = if cur_session != session_pk
            && cur_kind == "main"
            && self.session_kind(session_pk)? != "main"
        {
            cur_session
        } else {
            session_pk
        };
        self.tx
            .prepare_cached("UPDATE message_source SET preferred = 0 WHERE message_pk = ?1")?
            .execute([pk])?;
        self.upsert_source(pk, &new, true)?;
        self.clear_blocks(pk)?;
        self.update_message(pk, m, session_pk, &new, new_order)?;
        self.insert_blocks(pk, session_pk, m, bodies)?;
        Ok(true)
    }

    fn upsert_source(&self, pk: i64, v: &Version, preferred: bool) -> Result<()> {
        self.tx
            .prepare_cached(
                "INSERT INTO message_source(message_pk, input_pk, chain, source_start, source_end,
                     preferred, complete, version_hash)
                 VALUES (?1, ?2, ?3, ?4, ?8, ?5, ?6, ?7)
                 ON CONFLICT(message_pk, input_pk, chain) DO UPDATE SET
                     source_start = excluded.source_start, source_end = excluded.source_end,
                     preferred = excluded.preferred, complete = excluded.complete,
                     version_hash = excluded.version_hash",
            )?
            .execute(params![
                pk,
                v.input_pk,
                v.chain,
                v.start,
                i64::from(preferred),
                i64::from(v.complete),
                v.hash,
                v.end
            ])?;
        Ok(())
    }

    fn session_kind(&self, session_pk: i64) -> Result<String> {
        Ok(self
            .tx
            .prepare_cached("SELECT kind FROM session WHERE pk = ?1")?
            .query_row([session_pk], |r| r.get(0))?)
    }

    fn insert_message(
        &self,
        m: &Message,
        session_pk: i64,
        v: &Version,
        order: (i64, i64),
    ) -> Result<i64> {
        self.set_message(None, m, session_pk, v, order)
    }

    fn update_message(
        &self,
        pk: i64,
        m: &Message,
        session_pk: i64,
        v: &Version,
        order: (i64, i64),
    ) -> Result<()> {
        self.set_message(Some(pk), m, session_pk, v, order)
            .map(drop)
    }

    /// Writes every column of the preferred version in one statement: an
    /// insert for a new message, an update when another version won.
    fn set_message(
        &self,
        pk: Option<i64>,
        m: &Message,
        session_pk: i64,
        v: &Version,
        order: (i64, i64),
    ) -> Result<i64> {
        let written_by = match (m.role, m.prompt_kind) {
            (Role::Assistant, _) => "model",
            (Role::System, _) | (Role::User, Some(PromptKind::Harness)) => "harness",
            (Role::User, Some(_)) => "human",
            (Role::User, None) => "unknown",
        };
        let words: i64 = m
            .blocks
            .iter()
            .filter_map(Block::text)
            .map(|t| t.split_whitespace().count() as i64)
            .sum();
        let attrs = {
            let mut a = serde_json::Map::new();
            if let Some(c) = &m.reported_cost_usd {
                a.insert("reported_cost_usd".into(), c.clone().into());
            }
            if !m.production.is_empty() {
                a.insert(
                    "production".into(),
                    serde_json::to_value(&m.production).unwrap_or_default(),
                );
            }
            (!a.is_empty()).then(|| serde_json::Value::Object(a).to_string())
        };
        let u = &m.usage;
        let n = |x: Option<u64>| x.map(|x| x as i64);
        let sql = if pk.is_some() {
            "UPDATE message SET session_pk = ?2, order_input_pk = ?3, seq = ?4, route_rank = ?5,
                 role = ?6, written_by = ?7, prompt_kind = ?8, model = ?9, at = ?10,
                 parent_id = ?11, input_pk = ?12, source_chain = ?13, source_start = ?14,
                 source_end = ?27, in_progress = ?15, input_tokens = ?16, output_tokens = ?17,
                 cache_read_tokens = ?18, cache_write_tokens = ?19, reasoning_tokens = ?20,
                 service_tier = ?21, stop_reason = ?22, duration_ms = ?23, attrs_json = ?24,
                 words = ?25, parser_version = ?26
             WHERE pk = ?1"
        } else {
            "INSERT INTO message(id, session_pk, order_input_pk, seq, route_rank, role, written_by,
                 prompt_kind, model, at, parent_id, input_pk, source_chain, source_start,
                 source_end, in_progress, input_tokens, output_tokens, cache_read_tokens,
                 cache_write_tokens, reasoning_tokens, service_tier, stop_reason, duration_ms,
                 attrs_json, words, parser_version)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?27, ?15, ?16,
                 ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26)"
        };
        let first: rusqlite::types::Value = match pk {
            Some(pk) => pk.into(),
            None => m.id.to_string().into(),
        };
        self.tx.prepare_cached(sql)?.execute(params![
            first,
            session_pk,
            order.0,
            order.1,
            v.rank,
            name(&m.role),
            written_by,
            m.prompt_kind.map(|k| name(&k)),
            m.model,
            ms(m.at),
            m.parent.as_ref().map(ToString::to_string),
            v.input_pk,
            v.chain,
            v.start,
            i64::from(!v.complete),
            n(u.input),
            n(u.output),
            n(u.cached_input),
            n(u.cache_write),
            n(u.reasoning),
            u.service_tier,
            m.stop_reason,
            n(m.duration_ms),
            attrs,
            words,
            self.parser_version,
            v.end,
        ])?;
        Ok(pk.unwrap_or_else(|| self.tx.last_insert_rowid()))
    }

    /// Removes a message's blocks, keeping search entries and tool-call
    /// pairs in step, before another version replaces them. It runs before
    /// the message row changes: a version from another session moves the
    /// message, and its old blocks must not outlive the move.
    fn clear_blocks(&self, message_pk: i64) -> Result<()> {
        // Calls of the old blocks go, with their trigram entries; results of
        // the old blocks are unpaired from their calls.
        let old_calls: Vec<(i64, Option<String>, Option<String>)> = self
            .tx
            .prepare_cached(
                "SELECT t.pk, t.command, t.path FROM tool_call t JOIN block b ON b.pk = t.call_block_pk
                 WHERE b.message_pk = ?1",
            )?
            .query_map([message_pk], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .collect::<rusqlite::Result<_>>()?;
        for (pk, command, path) in old_calls {
            self.tx
                .prepare_cached(
                    "INSERT INTO tool_call_trigram(tool_call_trigram, rowid, command, path)
                     VALUES ('delete', ?1, ?2, ?3)",
                )?
                .execute(params![pk, command, path])?;
            self.tx
                .prepare_cached("DELETE FROM tool_call WHERE pk = ?1")?
                .execute([pk])?;
        }
        self.tx
            .prepare_cached(
                "UPDATE tool_call SET result_block_pk = NULL, outcome = 'unknown',
                     outcome_reason = 'awaiting_result', exit_code = NULL
                 WHERE result_block_pk IN (SELECT pk FROM block WHERE message_pk = ?1)",
            )?
            .execute([message_pk])?;
        self.tx
            .prepare_cached("DELETE FROM block WHERE message_pk = ?1")?
            .execute([message_pk])?;
        Ok(())
    }

    /// Inserts a message's blocks; `bodies` are their stored forms
    /// ([`pack_block`]), in the same order.
    fn insert_blocks(
        &self,
        message_pk: i64,
        session_pk: i64,
        m: &Message,
        bodies: &[Vec<u8>],
    ) -> Result<()> {
        for (idx, (block, body)) in m.blocks.iter().zip(bodies).enumerate() {
            self.insert_block(message_pk, session_pk, idx as i64, block, body, ms(m.at))?;
        }
        Ok(())
    }

    fn insert_block(
        &self,
        message_pk: i64,
        session_pk: i64,
        idx: i64,
        block: &Block,
        body: &[u8],
        at: Option<i64>,
    ) -> Result<()> {
        let bit = if matches!(block, Block::ToolResult { .. }) {
            2
        } else {
            1
        };
        let text_id = match block_text(block) {
            Some(t) => Some(self.intern(t, bit)?),
            None => None,
        };
        let (kind, call_id, attachment) = match block {
            Block::Text { .. } => ("text", None, None),
            Block::Thinking { .. } => ("thinking", None, None),
            Block::ToolCall { call_id, .. } => ("tool_call", call_id.as_deref(), None),
            Block::ToolResult { call_id, .. } => ("tool_result", call_id.as_deref(), None),
            Block::Image(a) => ("image", None, Some(a)),
            Block::File(a) => ("file", None, Some(a)),
            Block::Other { .. } => ("other", None, None),
        };
        let (signature, encrypted) = match block {
            Block::Thinking {
                signature,
                redacted,
                ..
            } => (signature.as_deref(), *redacted),
            _ => (None, false),
        };
        let other_type = match block {
            Block::Other { type_name, .. } => Some(type_name.as_str()),
            _ => None,
        };
        self.tx
            .prepare_cached(
                "INSERT INTO block(message_pk, session_pk, idx, kind, text_id, body, call_id,
                     blob_hash, media_type, file_name, size, url, direction, signature, encrypted,
                     other_type)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
            )?
            .execute(params![
                message_pk,
                session_pk,
                idx,
                kind,
                text_id,
                body,
                call_id,
                attachment.and_then(|a| a.blob).map(|h| h.0.to_vec()),
                attachment.and_then(|a| a.media_type.as_deref()),
                attachment.and_then(|a| a.name.as_deref()),
                attachment.and_then(|a| a.size).map(|s| s as i64),
                attachment.and_then(|a| a.url.as_deref()),
                attachment.map(|a| name(&a.direction)),
                signature,
                i64::from(encrypted),
                other_type,
            ])?;
        let block_pk = self.tx.last_insert_rowid();

        match block {
            Block::ToolCall {
                call_id,
                tool,
                command,
                path,
                ..
            } => self.insert_call(block_pk, session_pk, call_id, tool, command, path, at),
            Block::ToolResult {
                call_id: Some(call_id),
                outcome,
                ..
            } => {
                let (outcome, reason, exit) = outcome_columns(outcome);
                self.tx
                    .prepare_cached(
                        "UPDATE tool_call SET result_block_pk = ?1, outcome = ?2, outcome_reason = ?3,
                             exit_code = ?4
                         WHERE session_pk = ?5 AND call_id = ?6",
                    )?
                    .execute(params![block_pk, outcome, reason, exit, session_pk, call_id])?;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_call(
        &self,
        block_pk: i64,
        session_pk: i64,
        call_id: &Option<String>,
        tool: &Option<String>,
        command: &Option<String>,
        path: &Option<String>,
        at: Option<i64>,
    ) -> Result<()> {
        let tool = tool.as_deref().unwrap_or("unknown");
        self.tx
            .prepare_cached(
                "INSERT INTO tool_call(session_pk, call_block_pk, call_id, at, tool, action, command,
                     path, outcome, outcome_reason)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'unknown', 'awaiting_result')",
            )?
            .execute(params![
                session_pk,
                block_pk,
                call_id,
                at,
                tool,
                action(tool),
                command,
                path
            ])?;
        let pk = self.tx.last_insert_rowid();
        self.tx
            .prepare_cached(
                "INSERT INTO tool_call_trigram(rowid, command, path) VALUES (?1, ?2, ?3)",
            )?
            .execute(params![pk, command, path])?;
        // The result may have been stored first (a late call record, or a
        // message replaced after its result arrived).
        let Some(call_id) = call_id else {
            return Ok(());
        };
        let result: Option<(i64, Vec<u8>)> = self
            .tx
            .prepare_cached(
                "SELECT pk, body FROM block
                 WHERE session_pk = ?1 AND call_id = ?2 AND kind = 'tool_result'
                 ORDER BY pk LIMIT 1",
            )?
            .query_row(params![session_pk, call_id], |r| Ok((r.get(0)?, r.get(1)?)))
            .optional()?;
        if let Some((result_pk, body)) = result
            && let Ok(Block::ToolResult { outcome, .. }) = unpack_block(&body)
        {
            let (outcome, reason, exit) = outcome_columns(&outcome);
            self.tx
                .prepare_cached(
                    "UPDATE tool_call SET result_block_pk = ?2, outcome = ?3, outcome_reason = ?4,
                         exit_code = ?5
                     WHERE pk = ?1",
                )?
                .execute(params![pk, result_pk, outcome, reason, exit])?;
        }
        Ok(())
    }

    /// The text row for `body`, stored once per distinct content, and its
    /// entry in the search index `bit` names (1 prose, 2 tool output).
    fn intern(&self, body: &str, bit: i64) -> Result<i64> {
        let hash = blake3::hash(body.as_bytes());
        let found: Option<(i64, i64)> = self
            .tx
            .prepare_cached("SELECT id, indexed FROM text WHERE hash = ?1")?
            .query_row([&hash.as_bytes()[..]], |r| Ok((r.get(0)?, r.get(1)?)))
            .optional()?;
        let (id, indexed) = match found {
            Some(f) => f,
            None => {
                let packed = match self.packed.get(hash.as_bytes()) {
                    Some(p) => std::borrow::Cow::Borrowed(p.as_slice()),
                    None => zstd::bulk::compress(body.as_bytes(), TEXT_LEVEL)?.into(),
                };
                self.tx
                    .prepare_cached("INSERT INTO text(hash, body, indexed) VALUES (?1, ?2, 0)")?
                    .execute(params![&hash.as_bytes()[..], &*packed])?;
                (self.tx.last_insert_rowid(), 0)
            }
        };
        if indexed & bit == 0 {
            let sql = if bit == 1 {
                "INSERT INTO text_fts(rowid, body) VALUES (?1, ?2)"
            } else {
                "INSERT INTO output_fts(rowid, body) VALUES (?1, ?2)"
            };
            self.tx
                .prepare_cached(sql)?
                .execute(params![id, polylogue_model::text::fold(body)])?;
            self.tx
                .prepare_cached("UPDATE text SET indexed = indexed | ?2 WHERE id = ?1")?
                .execute(params![id, bit])?;
        }
        Ok(id)
    }

    /// `packed` is the event's id and stored payload ([`pack_event`]).
    fn write_event(&self, e: &Event, session_pk: i64, packed: &(String, Vec<u8>)) -> Result<()> {
        let (id, data) = packed;
        let after: Option<(i64, i64)> = match &e.after {
            Some(id) => self
                .tx
                .prepare_cached("SELECT pk, seq FROM message WHERE id = ?1")?
                .query_row([id.to_string()], |r| Ok((r.get(0)?, r.get(1)?)))
                .optional()?,
            None => None,
        };
        // An event with its own byte range sorts by it among the input's
        // messages (whose `seq` is their record offset); otherwise it takes
        // the position of the message it follows.
        let seq = e
            .range
            .map(|(start, _)| start as i64)
            .or(after.map(|a| a.1))
            .unwrap_or(0);
        self.tx
            .prepare_cached(
                "INSERT OR IGNORE INTO event(id, session_pk, seq, after_message_pk, kind, at,
                     input_pk, parser_version, source_chain, source_start, source_end, data)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            )?
            .execute(params![
                id,
                session_pk,
                seq,
                after.map(|a| a.0),
                e.kind,
                ms(e.at),
                self.input_pk,
                self.parser_version,
                e.range.map(|_| self.chain),
                e.range.map(|r| r.0 as i64),
                e.range.map(|r| r.1 as i64),
                data,
            ])?;
        Ok(())
    }

    /// One link per kind, parent and child; parts one file lacks are filled
    /// from another (the SQL form of [`Link::merge`]).
    fn write_link(&self, l: &Link) -> Result<()> {
        let kind = name(&l.kind);
        let child = l.child.to_string();
        let child_pk: Option<i64> = self
            .tx
            .prepare_cached("SELECT pk FROM session WHERE id = ?1")?
            .query_row([&child], |r| r.get(0))
            .optional()?;
        self.tx
            .prepare_cached(
                "INSERT INTO link(parent_id, child_id, child_pk, kind, how, branch_point,
                     spawning_call, evidence, input_pk, parser_version)
                 VALUES (?1, ?2, ?3, ?4, ?4, ?5, ?6, ?7, ?8, ?9)
                 ON CONFLICT(parent_id, child_id, kind, how) DO UPDATE SET
                     child_pk = COALESCE(link.child_pk, excluded.child_pk),
                     branch_point = COALESCE(link.branch_point, excluded.branch_point),
                     spawning_call = COALESCE(link.spawning_call, excluded.spawning_call)",
            )?
            .execute(params![
                l.parent.to_string(),
                child,
                child_pk,
                kind,
                l.at.as_ref().map(ToString::to_string),
                l.call_id,
                l.evidence,
                self.input_pk,
                self.parser_version,
            ])?;
        Ok(())
    }

    /// Parse position and accounting for the chain, committed with the rows
    /// they cover. A step from offset 0 replaces the chain's counts; a
    /// continuing step adds to them.
    fn write_parse_state(&self, b: &Batch, p: &ParseStep) -> Result<()> {
        let a = &p.accounting;
        let used: u64 = a.materialized.values().sum();
        let unknown: u64 = a.unknown.values().sum();
        let refused: u64 = a.refused.values().sum();
        let status = if p.parsed_upto >= b.length {
            "done"
        } else {
            "parsing"
        };
        let whole = p.start == 0;
        let sql = if whole {
            "INSERT INTO parse_state(input_pk, chain, parser_version, status, parsed_upto, framed,
                 used, unrecognized_n, undecodable)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(input_pk, chain) DO UPDATE SET parser_version = excluded.parser_version,
                 status = excluded.status, parsed_upto = excluded.parsed_upto,
                 framed = excluded.framed, used = excluded.used,
                 unrecognized_n = excluded.unrecognized_n, undecodable = excluded.undecodable"
        } else {
            "INSERT INTO parse_state(input_pk, chain, parser_version, status, parsed_upto, framed,
                 used, unrecognized_n, undecodable)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(input_pk, chain) DO UPDATE SET parser_version = excluded.parser_version,
                 status = excluded.status, parsed_upto = excluded.parsed_upto,
                 framed = framed + excluded.framed, used = used + excluded.used,
                 unrecognized_n = unrecognized_n + excluded.unrecognized_n,
                 undecodable = undecodable + excluded.undecodable"
        };
        self.tx.prepare_cached(sql)?.execute(params![
            self.input_pk,
            self.chain,
            self.parser_version,
            status,
            p.parsed_upto as i64,
            a.framed as i64,
            used as i64,
            unknown as i64,
            refused as i64,
        ])?;
        if whole {
            self.tx
                .prepare_cached("DELETE FROM unrecognized WHERE input_pk = ?1 AND chain = ?2")?
                .execute(params![self.input_pk, self.chain])?;
        }
        let groups = [
            ("record:", &a.unknown, 1),
            ("undecodable:", &a.refused, 0),
            ("field:", &a.unread_fields, 0),
        ];
        for (prefix, map, counted_only) in groups {
            for (what, count) in map {
                self.tx
                    .prepare_cached(
                        "INSERT INTO unrecognized(input_pk, chain, format, what, count, counted_only)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                         ON CONFLICT(input_pk, chain, what) DO UPDATE SET count = count + excluded.count",
                    )?
                    .execute(params![
                        self.input_pk,
                        self.chain,
                        b.input.format,
                        format!("{prefix}{what}"),
                        *count as i64,
                        counted_only,
                    ])?;
            }
        }
        Ok(())
    }
}

/// Moves a block's prose or output text out, for the `text` table; the
/// body keeps an empty string in its place. Returns the text and which
/// search index it belongs in (1 prose, 2 tool output).
fn take_text(block: &mut Block) -> (Option<String>, i64) {
    match block {
        Block::Text { text } if !text.is_empty() => (Some(std::mem::take(text)), 1),
        Block::Thinking { text: Some(t), .. } if !t.is_empty() => (Some(std::mem::take(t)), 1),
        Block::ToolResult {
            output: Some(o), ..
        } if !o.is_empty() => (Some(std::mem::take(o)), 2),
        _ => (None, 0),
    }
}

/// Puts a block's text back from its `text` row.
pub(crate) fn restore_text(block: &mut Block, text: String) {
    match block {
        Block::Text { text: t } => *t = text,
        Block::Thinking { text: t, .. } => *t = Some(text),
        Block::ToolResult { output, .. } => *output = Some(text),
        _ => {}
    }
}

fn outcome_columns(o: &ToolOutcome) -> (&'static str, Option<String>, Option<i64>) {
    match o {
        ToolOutcome::Ok => ("ok", None, None),
        ToolOutcome::Error { reason } => match reason {
            Some(ErrorReason::ExitCode(code)) => ("error", Some("exit_code".into()), Some(*code)),
            Some(ErrorReason::Denied(label)) => ("error", Some(format!("denied:{label}")), None),
            Some(ErrorReason::Interrupted) => ("error", Some("interrupted".into()), None),
            Some(ErrorReason::TimedOut) => ("error", Some("timed_out".into()), None),
            None => ("error", None, None),
        },
        ToolOutcome::NoResult => ("unknown", Some("no_result".into()), None),
        ToolOutcome::Unknown { reason } => ("unknown", Some(name(reason)), None),
    }
}

/// The kind of action a tool performs, from its name.
fn action(tool: &str) -> &'static str {
    match tool {
        "Bash" | "BashOutput" | "KillShell" | "shell" | "exec_command" | "local_shell" => "shell",
        "Read" | "NotebookRead" => "file_read",
        "Write" => "file_write",
        "Edit" | "MultiEdit" | "NotebookEdit" | "apply_patch" => "file_edit",
        "Grep" | "Glob" | "LS" | "ToolSearch" => "search",
        "WebFetch" | "WebSearch" | "web_search" => "web",
        "Task" | "Agent" => "subagent",
        t if t.starts_with("mcp__") => "mcp",
        _ => "other",
    }
}

#[cfg(test)]
mod tests;
