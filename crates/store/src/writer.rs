//! The index writer. It runs on its own thread, owns the only writing
//! connection to `index.db`, and writes what is parsed from stored bytes in
//! large transactions: sessions, messages and their versions, blocks, search
//! entries, tool calls, events and links, together with the parse position
//! they cover. It never writes `archive.db`, which it reads attached
//! read-only.
//!
//! The index follows the archive: it receives only parses of bytes the
//! archive has already committed. Captured bytes are put
//! ([`crate::bytes`]) and their rows committed by the archive writer
//! ([`crate::archive::writer`]) before their parsed rows reach this writer
//! ([`WriterSender::store_group`] does both in order). A crash between the
//! two leaves the index behind its bytes; the loop parses them again, and
//! every write here is an upsert, so that changes nothing. What follows from
//! archive commits alone (chain progress of stored chains, alternate text)
//! is applied here from what the archive writer hands over with each commit,
//! up to the position kept in the index's `meta` as `archive_position`.

use crate::ConnExt as _;
use crate::OptionExt as _;
pub use crate::archive::writer::HistorySlot;
#[cfg(any(test, feature = "fixture"))]
pub use crate::archive::writer::{DispatchPause, DispatchPhase};
pub use crate::archive::writer::{ParentLifetime, Queued, UserSubmission};
use crate::archive::{Archive, bump_counter, open_runtime_writer};
use crate::blobs::{Change, FileFacts, NewBlob};
use crate::{Result, StoreError};
use polylogue_model::{
    BlobHash, Block, ErrorReason, Event, Json, Link, Message, Output, RecordCounts, Session,
    SessionId, Timestamp, ToolAction, ToolOutcome,
};
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

mod assets;
mod attachments;
mod batch;
pub mod build;
mod capture;
mod chain;
mod compose;
mod derive_thread;
mod hook_fields;
pub(crate) mod identity;
pub(crate) use readings::{read_message, shown_message};
pub(crate) mod readings;
mod rows;
mod search;
mod shown;
pub use chain::{ChainParse, ChainSink};

use crate::archive::writer::captured::stored_reading_chain;
use crate::encoding::{PreparedDocument, PreparedMessage, block_text, unpack_block};
pub use batch::{
    Batch, DatabaseLog, ImportMember, InContainer, InFlight, InputKey, Observation, ParseStep,
    WriteReport,
};
use search::*;

enum Msg {
    Assets(
        Box<crate::export_items::ExportItems>,
        SyncSender<Result<()>>,
    ),
    Coverage(
        Box<polylogue_model::capture::Coverage>,
        SyncSender<Result<()>>,
    ),
    /// Sessions whose live state changed with the clock alone: each gets a
    /// new change counter, so the change feed carries the transition.
    LiveClock(Vec<String>, SyncSender<Result<()>>),
    /// Load a build's runs, compose (on this many threads), index and
    /// install the build file.
    Build(
        Vec<std::path::PathBuf>,
        usize,
        Option<crossbeam_channel::Sender<&'static str>>,
        SyncSender<Result<Vec<build::BuildPhase>>>,
    ),
    ReplaceChain(InputKey, i64, u64, Box<ChainParse>, SyncSender<Result<u64>>),
    Stop,
    /// The archive committed: what follows from it in the index is applied.
    Wake,
    /// Parsed rows of stored bytes, answered once committed when a caller
    /// waits for them.
    Batch(capture::Parsed),
    Flush(SyncSender<Result<WriteReport>>, bool),
    /// Run the derive step over sessions whose derived rows are stale;
    /// answered with the number of sessions derived.
    Derive(SyncSender<Result<u64>>),
    DeriveInBackground(SyncSender<Result<bool>>),
    /// One wave of derived values from the derive thread, answered with
    /// which sessions it applied.
    DeriveWrite(Box<crate::summary::Write>, SyncSender<Result<Vec<bool>>>),
    /// One batch of full-text entries from the search thread, answered with
    /// the number written.
    SearchWrite(
        Box<crate::text_rows::SearchEntries>,
        SyncSender<Result<u64>>,
    ),
}

/// The two writers of an archive and its byte home. Dropping it stops the
/// archive writer after the writes already accepted, then the index writer
/// after the batches already sent.
pub struct Writer {
    sender: WriterSender,
    archive: Option<crate::archive::writer::ArchiveWriter>,
    /// The archive's stop request ([`Archive::stop`]). It ends the writer's
    /// long work early: a derive pass after the transaction in progress, a
    /// first build's ending (its deferred indexes) at its next step, replay
    /// of stored bytes by rolling back its unfinished index transaction. What committed stays; the
    /// rest is done at the next start. Capture and authored writes are not
    /// affected.
    stop: crate::cancellation::Cancellation,
    handle: Option<JoinHandle<()>>,
}

/// Orders new work against shutdown. Once the stop message is queued,
/// every remaining sender refuses new work rather than acknowledging a
/// message that the receiver will never process.
#[derive(Clone)]
struct WriterQueue {
    bulk: Arc<Mutex<Option<SyncSender<Msg>>>>,
    /// A second sender into the queue that only wakes the writer to follow
    /// the archive. Its lock is held only for one `try_send`, never while a
    /// sender waits for room.
    wake: Arc<Mutex<Option<SyncSender<Msg>>>>,
    location: Archive,
    commits: crate::completion::CommitEvents,
    /// The archive writer and the byte home the parsed rows' bytes go to first.
    archive: crate::archive::writer::ArchiveSender,
    bytes: Arc<dyn crate::bytes::Bytes>,
    /// The archive position the index has taken in, for [`WriterSender::followed`].
    published: Arc<Published>,
}

/// The archive position the index writer has taken in, as its last commit
/// recorded it, and whether the writer has ended.
#[derive(Default)]
struct Published {
    state: Mutex<(u64, bool)>,
    changed: std::sync::Condvar,
}

impl Published {
    fn publish(&self, position: crate::archive::writer::Position) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.0 = state.0.max(position.0);
        self.changed.notify_all();
    }

    fn close(&self) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).1 = true;
        self.changed.notify_all();
    }
}

impl WriterQueue {
    fn send(&self, message: Msg) -> std::result::Result<(), ()> {
        let counted = matches!(&message, Msg::Batch(..));
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

    fn ask<T>(&self, message: impl FnOnce(SyncSender<Result<T>>) -> Msg) -> Result<T> {
        let (reply, wait) = sync_channel(1);
        self.send(message(reply))
            .map_err(|_| StoreError::WriterGone)?;
        wait.recv().map_err(|_| StoreError::WriterGone)?
    }

    fn stop(&self) {
        if let Ok(mut wake) = self.wake.lock() {
            wake.take();
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

/// Groups waiting for the writer before senders block. A batch's bytes stay
/// counted against the parse pool's limit until its transaction commits
/// ([`InFlight`]), so that limit bounds their memory, not this depth. The
/// writer commits every group waiting at the start of its turn together,
/// up to [`ROWS_PER_TRANSACTION`] and [`BYTES_PER_TRANSACTION`]; with four
/// waiting at most, a pass over many small files committed some twenty
/// files per transaction, each transaction rewriting a page of every index
/// it touched.
const QUEUE: usize = 256;

/// Input bytes a transaction covers before committing. A batch's bytes
/// stay counted against the parse pool's limit until its transaction
/// commits ([`InFlight`]); committing well before that limit lets the pool
/// keep parsing while the writer commits, instead of the two alternating.
const BYTES_PER_TRANSACTION: u64 = 128 << 20;

/// The writer thread's name; its busy time is read from its activity.
pub(crate) const WRITER_THREAD: &str = "polylogue-writer";

/// The writer's index connection: the private build file while a build runs,
/// otherwise `index.db`, with the archive attached read-only.
fn open_index_writer(archive: &Archive, building: bool) -> Result<Connection> {
    let index = if building {
        crate::archive::index_build::open_writer(archive)?
    } else {
        open_runtime_writer(
            &archive.index_db(),
            archive.memory_budget().writer_cache_mib(),
        )?
    };
    let uri = format!("file:{}?mode=ro", archive.archive_db().display());
    index.execute("ATTACH DATABASE ?1 AS a", [uri])?;
    crate::source::views(&index)?;
    Ok(index)
}

impl Writer {
    /// Starts the writers of `archive`, which the caller has locked, with
    /// its segment log as the byte home.
    pub fn start(archive: &Archive) -> Result<Self> {
        Self::start_with(archive, Arc::new(crate::bytes::Segments::start(archive)?))
    }

    /// Starts the archive writer and the index writer of `archive` with
    /// `bytes` as its byte home. A build writes the private index file and
    /// defers search and secondary indexes; a stop keeps its committed rows
    /// and positions for the next start, and a crash discards it
    /// ([`crate::archive::index_build`]).
    pub fn start_with(archive: &Archive, bytes: Arc<dyn crate::bytes::Bytes>) -> Result<Self> {
        // A build writes its private file ([`crate::archive::index_build`]).
        let building = archive.index_build.building();
        let index = open_index_writer(archive, building)?;
        let installed = archive.root.join("index.db");
        let commits = crate::completion::CommitEvents::default();
        let archive_writer =
            crate::archive::writer::ArchiveWriter::start(archive, commits.clone())?;
        let archive_reads = crate::archive::open_read(&archive.archive_db())?;
        let (tx, rx) = sync_channel::<Msg>(QUEUE);
        let stop = archive.stop().clone();
        let queue = WriterQueue {
            wake: Arc::new(Mutex::new(Some(tx.clone()))),
            bulk: Arc::new(Mutex::new(Some(tx))),
            location: archive.clone(),
            commits: commits.clone(),
            archive: archive_writer.sender(),
            bytes,
            published: Arc::default(),
        };
        let follow = {
            let wake = queue.wake.clone();
            commits.subscribe_with_wake(
                crate::completion::Committed {
                    content: true,
                    authored: true,
                    query_records: false,
                },
                move || {
                    if let Ok(wake) = wake.lock()
                        && let Some(wake) = wake.as_ref()
                    {
                        let _ = wake.try_send(Msg::Wake);
                    }
                },
            )
        };
        if let Some(kept) =
            crate::meta_get(&index, ARCHIVE_POSITION)?.and_then(|kept| kept.parse().ok())
        {
            queue
                .published
                .publish(crate::archive::writer::Position(kept));
        }
        let mut inner = Inner {
            follow,
            published: Arc::clone(&queue.published),
            stop: stop.clone(),
            commits: commits.clone(),
            followed: archive_writer.sender(),
            archive: archive_reads,
            index,
            report: WriteReport::default(),
            // Both read the installed index, once a build has ended.
            search: Search::start(
                installed.clone(),
                queue.clone(),
                !building,
                archive.memory_budget().search_cache_mib(),
            )?,
            derive: derive_thread::DeriveThread::start(
                installed,
                queue.clone(),
                stop.child(),
                archive.memory_budget().search_cache_mib(),
            )?,
            location: archive.clone(),
            build_job: None,
        };

        let handle = polylogue_model::spawn_named(WRITER_THREAD, move || inner.run(&rx))?;
        Ok(Self {
            sender: WriterSender(queue),
            archive: Some(archive_writer),
            stop,
            handle: Some(handle),
        })
    }

    /// A cheap handle the parse pool's threads send through.
    pub fn sender(&self) -> WriterSender {
        self.sender.clone()
    }
}

impl std::ops::Deref for Writer {
    type Target = WriterSender;

    fn deref(&self) -> &Self::Target {
        &self.sender
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        self.stop();
    }
}

impl Writer {
    /// Ends the archive writer after the writes it accepted, then the index
    /// writer after what it was sent, so the index takes in every archive
    /// commit before it stops; later calls fail with [`StoreError::WriterGone`].
    pub fn stop(&mut self) {
        let started =
            (self.handle.is_some() && self.stop.is_cancelled()).then(std::time::Instant::now);
        if started.is_some() {
            crate::log!(info, "shutdown writer drain started");
        }
        if let Some(mut archive) = self.archive.take() {
            archive.stop();
        }
        self.sender.0.stop();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.sender.0.bytes.close();
        if let Some(started) = started {
            crate::log!(
                info,
                "shutdown writer drain finished in {:?}",
                started.elapsed()
            );
        }
    }
}

/// A clonable sender into the writer, for worker threads.
#[derive(Clone)]
pub struct WriterSender(WriterQueue);

impl WriterSender {
    /// The archive writer: user writes, restorations and the groups the
    /// archive commits.
    pub fn archive(&self) -> &crate::archive::writer::ArchiveSender {
        &self.0.archive
    }

    /// The byte home captured bytes are put into.
    pub fn bytes(&self) -> &Arc<dyn crate::bytes::Bytes> {
        &self.0.bytes
    }

    /// The archive position the index has taken in: everything the index
    /// keeps of archive rows committed at or before it (alternate text and
    /// its search entries, chain progress) is in the index.
    pub fn followed_position(&self) -> crate::archive::writer::Position {
        crate::archive::writer::Position(
            self.0
                .published
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .0,
        )
    }

    /// Waits until the index has taken in the archive commit at `position`.
    /// Only the index's own products of archive rows need it; authored rows
    /// are read from the attached archive. Ends with the limit's deadline or
    /// cancellation, the archive's stop request, or the writer's end, each
    /// with the error a read gets.
    pub fn followed(
        &self,
        position: crate::archive::writer::Position,
        limit: Option<&crate::read::limit::ReadLimit>,
    ) -> Result<()> {
        let published = &self.0.published;
        let stop = self.0.location.stop().clone();
        let wake = |published: Arc<Published>| {
            move || {
                let _state = published.state.lock();
                published.changed.notify_all();
            }
        };
        let _stopped = stop.on_cancel(wake(Arc::clone(published)));
        let _cancelled =
            limit.map(|limit| limit.cancellation.on_cancel(wake(Arc::clone(published))));
        let mut state = published.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if state.0 >= position.0 {
                return Ok(());
            }
            if let Some(limit) = limit {
                limit.check()?;
            }
            stop.check()?;
            if state.1 {
                return Err(StoreError::WriterGone);
            }
            let wait = limit
                .and_then(crate::read::limit::ReadLimit::remaining)
                .unwrap_or(std::time::Duration::from_secs(60));
            state = published
                .changed
                .wait_timeout(state, wait)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    pub fn completion_commits(&self) -> crate::completion::CommitSubscription {
        self.0.commits.subscribe()
    }

    /// The logical index commits made since the writer started; see
    /// [`crate::completion::CommitEvents::made`].
    pub fn commits_made(&self) -> u64 {
        self.0.commits.made()
    }

    /// Subscribe before the caller's first index read; the callback only
    /// queues a wake in its existing inbox.
    pub fn completion_commits_with_wake(
        &self,
        interests: crate::completion::Committed,
        wake: impl Fn() + Send + Sync + 'static,
    ) -> crate::completion::CommitSubscription {
        self.0.commits.subscribe_with_wake(interests, wake)
    }

    /// Replaces one chain's parsed rows in one index transaction.
    pub fn replace_chain(
        &self,
        input: InputKey,
        chain: i64,
        length: u64,
        reading: ChainParse,
    ) -> Result<u64> {
        self.queue_chain(input, chain, length, reading)?
            .recv()
            .map_err(|_| StoreError::WriterGone)?
    }

    /// Ends a standalone build: loads `runs` ([`build::BuildRun`]) into the
    /// build file, composes every session, writes the indexes and search
    /// entries and installs the file. Returns the phases it took.
    pub fn build(&self, runs: Vec<std::path::PathBuf>) -> Result<Vec<build::BuildPhase>> {
        // Two composers, so that batches meet the writer out of order.
        self.0.ask(|reply| Msg::Build(runs, 2, None, reply))
    }

    /// [`Self::build`] without waiting, its sessions composed by
    /// `composers` threads: each phase's name is sent to `announce` as it
    /// starts, and the phases come on the returned receiver once the file
    /// is installed.
    pub fn start_build(
        &self,
        runs: Vec<std::path::PathBuf>,
        composers: usize,
        announce: crossbeam_channel::Sender<&'static str>,
    ) -> Result<Receiver<Result<Vec<build::BuildPhase>>>> {
        let (reply, wait) = sync_channel(1);
        self.0
            .send(Msg::Build(runs, composers, Some(announce), reply))
            .map_err(|_| StoreError::WriterGone)?;
        Ok(wait)
    }

    /// Queue a chain without waiting for its transaction to commit.
    pub fn queue_chain(
        &self,
        input: InputKey,
        chain: i64,
        length: u64,
        mut reading: ChainParse,
    ) -> Result<Receiver<Result<u64>>> {
        reading.finish()?;
        reading.check(&input.id, chain)?;
        // Bytes the reading decoded (inline images) are stored first: the
        // index receives parses of stored bytes only.
        if !reading.decoded.is_empty() {
            for result in self.store_group(std::mem::take(&mut reading.decoded))? {
                result?;
            }
        }
        let (reply, wait) = sync_channel(1);
        if self
            .0
            .send(Msg::ReplaceChain(
                input,
                chain,
                length,
                Box::new(reading),
                reply,
            ))
            .is_err()
        {
            return Err(StoreError::WriterGone);
        }
        Ok(wait)
    }

    /// Accepts one completed discovery walk; it is visible once its index
    /// transaction commits.
    pub fn coverage(&self, coverage: polylogue_model::capture::Coverage) -> Result<()> {
        self.0.ask(|reply| Msg::Coverage(Box::new(coverage), reply))
    }

    /// Puts sessions whose live state changed with the clock alone on the change feed.
    pub fn live_clock(&self, sessions: Vec<String>) -> Result<()> {
        self.0.ask(|reply| Msg::LiveClock(sessions, reply))
    }

    /// Derives the products of sessions changed since they were last derived,
    /// after the batches already queued.
    pub fn derive(&self) -> Result<u64> {
        self.0.ask(Msg::Derive)
    }

    /// Continue pending derivations without waiting for their computation.
    /// Returns whether derived values remain pending at this queue boundary.
    /// This does not finalize a first build; [`Self::derive`] owns that boundary.
    pub fn derive_in_background(&self) -> Result<bool> {
        self.0.ask(Msg::DeriveInBackground)
    }

    /// Installs export associations derived from committed source observations.
    pub fn import_assets(&self, staged: crate::export_items::ExportItems) -> Result<()> {
        self.0.ask(|reply| Msg::Assets(Box::new(staged), reply))
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

    /// The archive's queued groups commit first, so a report covers every
    /// write handed over before the flush.
    fn flush_with_derivation(&self, derive: bool) -> Result<WriteReport> {
        self.0.archive.flush()?;
        self.0.ask(|reply| Msg::Flush(reply, derive))
    }
}

struct Inner {
    /// Ends a derive pass or a first build's ending early.
    stop: crate::cancellation::Cancellation,
    commits: crate::completion::CommitEvents,
    /// Hands over what archive commits changed, and their reports.
    followed: crate::archive::writer::ArchiveSender,
    /// Wakes this writer once after archive commits, until it is drained.
    follow: crate::completion::CommitSubscription,
    published: Arc<Published>,
    /// `archive.db`, read only.
    archive: Connection,
    index: Connection,
    report: WriteReport,
    /// The search thread ([`search_thread`]): woken after each commit, and
    /// allowed to write once the index is out of a first build.
    search: Search,
    /// Runs derive passes, sending their values here to be applied.
    derive: derive_thread::DeriveThread,
    location: Archive,
    /// A build being loaded, composed and installed, advanced one step at a
    /// time between the messages the writer serves ([`build::BuildJob`]).
    build_job: Option<build::BuildJob>,
}

impl Inner {
    /// A build is in progress: the build file holds no rows until the batch
    /// build loads them ([`build`]).
    fn building(&self) -> bool {
        self.location.index_build.building()
    }

    fn run(&mut self, rx: &Receiver<Msg>) {
        let _worker = crate::timing::worker();
        let mut flushes = Vec::new();
        let mut derives = Vec::new();
        let mut background_derives = Vec::new();
        let mut derive_writes = Vec::new();
        let mut search_writes = Vec::new();
        let mut assets = Vec::new();
        let mut coverage = Vec::new();
        let mut live_clock = Vec::new();
        let mut replacements = Vec::new();
        let mut builds = Vec::new();
        // An archive commit woke the writer and what it handed over may not
        // be applied yet.
        let mut following = false;
        loop {
            // While a build runs, the writer takes a waiting message or else
            // goes on with the build's next step.
            let first = if self.build_job.is_some() {
                match rx.try_recv() {
                    Ok(first) => first,
                    Err(TryRecvError::Empty) => {
                        let _busy = crate::timing::start_stage(crate::timing::Stage::Writer);
                        let _doing = crate::activity::doing("building the index");
                        self.advance_build();
                        continue;
                    }
                    Err(TryRecvError::Disconnected) => break,
                }
            } else if following {
                // What archive commits hand over goes into the next batch's
                // transaction when one follows at once, as it does after a
                // capture; otherwise it is applied in a transaction of its own.
                match rx.recv_timeout(FOLLOW_AFTER) {
                    Ok(first) => first,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        following = false;
                        if let Err(error) = self.follow_archive() {
                            self.report.errors.push(polylogue_model::Failure::new(
                                "following the archive",
                                &error,
                            ));
                        }
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
            let _busy = crate::timing::start_stage(crate::timing::Stage::Writer);
            let _doing = crate::activity::doing("writing");
            let mut stopping = false;
            let mut group = Vec::new();
            let mut rows = 0;
            let mut bytes = 0;
            let mut push = |m: Msg,
                            group: &mut Vec<capture::Parsed>,
                            rows: &mut usize,
                            bytes: &mut u64| match m {
                Msg::Assets(staged, reply) => {
                    assets.push((staged, reply));
                    false
                }
                Msg::Coverage(fact, reply) => {
                    coverage.push((fact, reply));
                    false
                }
                Msg::LiveClock(sessions, reply) => {
                    live_clock.push((sessions, reply));
                    false
                }
                Msg::ReplaceChain(input, chain, length, reading, reply) => {
                    replacements.push((input, chain, length, *reading, reply));
                    false
                }
                Msg::Build(runs, composers, announce, reply) => {
                    builds.push((runs, composers, announce, reply));
                    false
                }
                Msg::Stop => {
                    stopping = true;
                    false
                }
                Msg::Wake => {
                    following = true;
                    false
                }
                Msg::Batch(parsed) => {
                    crate::timing::queue(crate::timing::Queue::Writer, -1);
                    for b in &parsed.batches {
                        *rows += weight(b);
                        *bytes += input_bytes(b);
                    }
                    group.push(parsed);
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
                Msg::DeriveInBackground(reply) => {
                    background_derives.push(reply);
                    false
                }
                Msg::DeriveWrite(write, reply) => {
                    derive_writes.push((write, reply));
                    false
                }
                Msg::SearchWrite(entries, reply) => {
                    search_writes.push((entries, reply));
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
                crate::timing::time(crate::timing::Stage::Rows, || self.write_group(group));
            }
            for (write, reply) in derive_writes.drain(..) {
                let result = crate::summary::apply(&self.index, &write);
                if result.is_ok() {
                    self.commits
                        .committed(crate::completion::Committed::CONTENT);
                }
                let _ = reply.send(result);
            }
            for (entries, reply) in search_writes.drain(..) {
                let result = crate::text_rows::apply_entries(&self.index, &entries);
                // Settled search text is a commit its readers see: status
                // stops saying search text is pending.
                if matches!(result, Ok(applied) if applied > 0) {
                    self.commits
                        .committed(crate::completion::Committed::CONTENT);
                }
                let _ = reply.send(result);
            }
            for (input, chain, length, reading, reply) in replacements.drain(..) {
                let _ = reply.send(crate::timing::time(crate::timing::Stage::Rows, || {
                    self.replace_chain(input, chain, length, reading)
                }));
            }
            for (runs, composers, announce, reply) in builds.drain(..) {
                if self.build_job.is_some() {
                    let _ = reply.send(Err(StoreError::Invalid(
                        "a build is already being loaded".into(),
                    )));
                } else {
                    self.build_job = Some(build::BuildJob::new(runs, composers, announce, reply));
                }
            }

            for reply in derives.drain(..) {
                // A build in progress holds no rows to derive from until it
                // is installed.
                if self.building() {
                    let _ = reply.send(Ok(0));
                } else {
                    self.request_derive(derive_thread::Waiting::Derive(reply));
                }
            }
            for (staged, reply) in assets.drain(..) {
                let before = self.index.total_changes();
                let result = self.install_assets(*staged);
                if result.is_ok() && self.index.total_changes() != before {
                    self.commits
                        .committed(crate::completion::Committed::CONTENT);
                }
                let _ = reply.send(result);
            }
            for reply in background_derives.drain(..) {
                let pending = crate::summary::caught_up(&self.index).map(|current| !current);
                let requested = pending.as_ref().is_ok_and(|pending| *pending);
                // A caller with no new reading must not wait for an unrelated
                // computation, including the work preparing that computation.
                let _ = reply.send(pending);
                if requested && !self.building() {
                    self.request_derive(derive_thread::Waiting::Nobody);
                }
            }
            for (fact, reply) in coverage.drain(..) {
                let _ = reply.send(self.write_coverage(*fact));
            }
            for (sessions, reply) in live_clock.drain(..) {
                let _ = reply.send(self.write_live_clock(&sessions));
            }
            if !flushes.is_empty() || self.build_job.is_some() {
                following = false;
                if let Err(error) = self.follow_archive() {
                    self.report.errors.push(polylogue_model::Failure::new(
                        "following the archive",
                        &error,
                    ));
                }
            }
            for (reply, derive) in flushes.drain(..) {
                let mut report = std::mem::take(&mut self.report);
                merge_report(&mut report, self.followed.take_report());
                // Whatever a flush reports written is also derived, once the
                // index is out of a build.
                if derive && !self.building() {
                    self.request_derive(derive_thread::Waiting::Flush(reply, Box::new(report)));
                } else {
                    let _ = reply.send(Ok(report));
                }
            }
            self.restart_full_logs();
            // A steady stream of messages still lets the build go on.
            self.advance_build();

            if stopping {
                break;
            }
        }
        // A derive pass waiting for a write this loop will not apply ends.
        for message in rx.try_iter() {
            drop(message);
        }
        self.derive.stop();
        self.search.stop();
        let closed = self.follow_archive();
        self.published.close();
        if let Err(error) = closed {
            crate::log!(error, "writer: following the archive: {error}");
        }
    }

    /// Applies, in an index transaction of its own, the text of alternates
    /// committed after the index's `archive_position`, which it advances to
    /// the archive's last commit.
    fn follow_archive(&mut self) -> Result<()> {
        // Drained before reading the position, so a commit after this point
        // wakes the writer again.
        while self.follow.try_recv().is_ok() {}
        let followed = self.followed.committed_position();
        let position = followed.0.to_string();
        let kept = crate::meta_get(&self.index, ARCHIVE_POSITION)?;
        if kept.as_deref() == Some(position.as_str()) {
            self.published.publish(followed);
            return Ok(());
        }
        let after: i64 = kept.and_then(|kept| kept.parse().ok()).unwrap_or(-1);
        let tx = self
            .index
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let before = tx.total_changes();
        let alternates: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM source_alternate WHERE commit_seq > ?1)",
            [after],
            |row| row.get(0),
        )?;
        if alternates {
            crate::alternates::project(&tx)?;
        }
        let changed = tx.total_changes() != before;
        if changed {
            bump_counter(&tx)?;
        }
        crate::meta_set(&tx, ARCHIVE_POSITION, &position)?;
        tx.commit()?;
        self.published.publish(followed);
        if changed {
            self.commits
                .committed(crate::completion::Committed::CONTENT);
            self.report.transactions += 1;
        }
        Ok(())
    }

    /// Projects every alternate's text into the index.
    fn project_alternate_text(&mut self) -> Result<()> {
        let before = self.index.total_changes();
        let tx = self.index.transaction()?;
        crate::alternates::project(&tx)?;
        let changed = tx.total_changes() != before;
        if changed {
            bump_counter(&tx)?;
        }
        tx.commit()?;
        if changed {
            self.commits
                .committed(crate::completion::Committed::AUTHORED);
        }
        Ok(())
    }

    /// Between transactions, starts over a write-ahead log grown past its
    /// checkpoint size ([`crate::archive::restart_full_log`]). A log still in
    /// use by a reader is tried again after the next transaction.
    fn restart_full_logs(&mut self) {
        if self.stop.is_cancelled() {
            return;
        }
        let wait = std::time::Duration::from_millis(100);
        let limit = crate::archive::WAL_AUTOCHECKPOINT;
        let mut results = vec![crate::archive::restart_full_log(
            &self.index,
            "main",
            &self.location.index_db(),
            limit,
            wait,
        )];
        // The full-text files the writer has attached, once a build is installed.
        if crate::search::files::attached(&self.index).unwrap_or(false)
            && let Ok(Some(files)) = crate::search::files::paths(&self.index)
        {
            for (n, path) in files.iter().enumerate() {
                if !self.stop.is_cancelled() {
                    results.push(crate::archive::restart_full_log(
                        &self.index,
                        &crate::search::files::schema(n),
                        path,
                        limit,
                        wait,
                    ));
                }
            }
        }
        // A failed checkpoint loses nothing: the log stays and is copied by
        // a later one.
        for result in results {
            if let Err(error) = result {
                crate::log!(error, "writer: checkpoint: {error}");
            }
        }
    }

    /// Gives each named session a new live change counter.
    fn write_live_clock(&mut self, sessions: &[String]) -> Result<()> {
        if sessions.is_empty() {
            return Ok(());
        }
        let tx = self.index.transaction()?;
        let counter = bump_counter(&tx)?;
        for session in sessions {
            tx.prepare_cached("UPDATE session SET live_changed = ?2 WHERE id = ?1")?
                .execute(rusqlite::params![session, counter])?;
        }
        crate::write_lock::commit(tx)?;
        self.commits
            .committed(crate::completion::Committed::CONTENT);
        self.report.transactions += 1;
        Ok(())
    }

    fn write_coverage(&mut self, coverage: polylogue_model::capture::Coverage) -> Result<()> {
        crate::fault::check("write.coverage", &coverage.machine)?;
        coverage
            .validate()
            .map_err(|error| StoreError::Invalid(error.into()))?;
        let tx = self.index.transaction()?;
        bump_counter(&tx)?;
        crate::coverage::apply(&tx, &coverage)?;
        crate::write_lock::commit(tx)?;
        self.commits
            .committed(crate::completion::Committed::CONTENT);
        self.report.transactions += 1;
        Ok(())
    }

    /// Starts a derive pass on the derive thread, answering `waiting` once
    /// it is done; the writer goes on with other work meanwhile. Here the
    /// writer only marks what the pass must see: sessions whose derivation
    /// version moved, alternates' text and the search entries waiting.
    fn request_derive(&mut self, waiting: derive_thread::Waiting) {
        let prepared = self
            .until_stopped(|inner| {
                if inner.stop.is_cancelled() {
                    return Err(StoreError::WriterGone);
                }
                inner.project_alternate_text()?;
                crate::text_rows::index_pending(&inner.index)?;
                crate::summary::mark_moved_versions(&mut inner.index)
            })
            .and_then(|prepared| prepared.ok_or(StoreError::WriterGone));
        match (prepared, waiting) {
            (Ok(()), waiting) => self.derive.request(waiting),
            (Err(StoreError::WriterGone), derive_thread::Waiting::Derive(reply)) => {
                let _ = reply.send(Err(StoreError::WriterGone));
            }
            (Err(e), derive_thread::Waiting::Derive(reply)) => {
                let _ = reply.send(Err(StoreError::Failed(polylogue_model::Failure::new(
                    "derive", &e,
                ))));
            }
            (Err(e), derive_thread::Waiting::Flush(reply, mut report)) => {
                report
                    .errors
                    .push(polylogue_model::Failure::new("derive", &e));
                let _ = reply.send(Ok(*report));
            }
            (Err(e), derive_thread::Waiting::Nobody) => {
                self.report
                    .errors
                    .push(polylogue_model::Failure::new("derive", &e));
            }
        }
    }

    /// Runs `work`, whose statements on the index a stop interrupts: one
    /// index statement or derive transaction over a large archive runs for
    /// minutes. Stopped, its statement and transaction roll back and the
    /// answer is `None`; the work is done again at the next start.
    fn until_stopped<T>(&mut self, work: impl FnOnce(&mut Self) -> Result<T>) -> Result<Option<T>> {
        if self.stop.is_cancelled() {
            return Ok(None);
        }
        let stop = self.stop.clone();
        self.index
            .progress_handler(1_000, Some(move || stop.is_cancelled()))?;
        let result = work(self);
        self.index.progress_handler(0, None::<fn() -> bool>)?;
        match result {
            Ok(value) => Ok(Some(value)),
            Err(StoreError::Stopped) if self.stop.is_cancelled() => Ok(None),
            Err(StoreError::Sqlite(error))
                if self.stop.is_cancelled()
                    && error.sqlite_error_code()
                        == Some(rusqlite::ErrorCode::OperationInterrupted) =>
            {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    /// Runs the build's next step and answers its caller once it ends.
    fn advance_build(&mut self) {
        let Some(mut job) = self.build_job.take() else {
            return;
        };
        job.show_step();
        let stepped = self.until_stopped(|inner| inner.build_step(&mut job));
        job.hide_step();
        match stepped {
            Ok(Some(false)) => self.build_job = Some(job),
            Ok(Some(true)) => job.finish(Ok(())),
            Ok(None) => job.finish(Err(StoreError::Stopped)),
            // A job that ran with no build in progress, or failed while
            // installing, leaves the files as they are.
            Err(error) if job.installing || !self.location.index_build.building() => {
                job.finish(Err(error))
            }
            Err(error) => {
                // The full-text fill's threads write the files the restart
                // removes: they end first.
                job.search = None;
                let error = match self.restart_build(&error.to_string()) {
                    Ok(()) => error,
                    Err(restart) => StoreError::Invalid(format!(
                        "{error}; the failed build could not be discarded: {restart}"
                    )),
                };
                job.finish(Err(error));
            }
        }
    }

    /// Discards a build that failed and makes its file again, empty, as
    /// opening the archive after a crash does. The next build reads every
    /// chain again and composes from its own runs; the failed build's
    /// readings and partly composed sessions would otherwise read as
    /// upkeep's, which composition takes from the index session by session.
    fn restart_build(&mut self, error: &str) -> Result<()> {
        self.index.flush_prepared_statement_cache();
        let failed = std::mem::replace(&mut self.index, Connection::open_in_memory()?);
        let restarted = failed
            .close()
            .map_err(|(_, error)| StoreError::from(error))
            .and_then(|()| crate::archive::index_build::restart(&self.location, error));
        // The writer goes on with the build file there is, made again or not.
        self.index = open_index_writer(&self.location, true)?;
        restarted
    }

    /// Closes the writer's connection to the build file, installs it and
    /// opens the installed `index.db`. Connection-level state (the attached
    /// register, its views and the chain-progress triggers) is set up again
    /// on the new connection. A failed install leaves the writer on the build
    /// file, which a later finish installs.
    fn install_build(&mut self) -> Result<()> {
        self.index.flush_prepared_statement_cache();
        let built = std::mem::replace(&mut self.index, Connection::open_in_memory()?);
        built.close().map_err(|(_, error)| error)?;
        let synced = std::time::Instant::now();
        let installed = crate::archive::index_build::finish(&self.location);
        crate::timing::add(crate::timing::Stage::Sync, synced.elapsed());
        self.index = open_index_writer(&self.location, self.location.index_build.building())?;
        installed
    }

    /// Writes a group in one transaction per database. If it fails, each
    /// batch is retried alone, so one bad batch costs only itself.
    fn write_group(&mut self, group: Vec<capture::Parsed>) {
        if group.iter().any(|parsed| parsed.batches.is_empty())
            && let Err(error) = self.follow_archive()
        {
            self.report.errors.push(polylogue_model::Failure::new(
                "following the archive",
                &error,
            ));
        }
        let batches = || group.iter().flat_map(|parsed| &parsed.batches);
        let replay = group.iter().all(|parsed| parsed.replay) && batches().any(cancellable_replay);
        let together = batches().count() > 1 && self.write(batches(), replay).is_ok();
        let mut answers: Vec<Vec<Result<()>>> = Vec::with_capacity(group.len());
        for parsed in &group {
            if parsed.batches.is_empty() {
                answers.push(Vec::new());
            } else if together {
                answers.push(parsed.batches.iter().map(|_| Ok(())).collect());
            } else if parsed.atomic || parsed.batches.len() == 1 {
                let replay = parsed.replay && parsed.batches.iter().all(cancellable_replay);
                let answer = self.write(parsed.batches.iter(), replay);
                answers.push(match answer {
                    Ok(()) => parsed.batches.iter().map(|_| Ok(())).collect(),
                    Err(error) => {
                        let mut answer: Vec<Result<()>> = (1..parsed.batches.len())
                            .map(|_| Err(error.for_group_member()))
                            .collect();
                        answer.insert(0, Err(error));
                        answer
                    }
                });
            } else {
                let answer = parsed
                    .batches
                    .iter()
                    .map(|b| self.write(std::iter::once(b), parsed.replay && cancellable_replay(b)))
                    .collect();
                answers.push(answer);
            }
        }
        for (parsed, answer) in group.into_iter().zip(answers) {
            let capture::Parsed {
                batches,
                atomic,
                reply,
                ..
            } = parsed;
            let inputs: Vec<String> = batches.iter().map(|b| b.input.id.clone()).collect();
            // A reply also observes the release of the batches' memory
            // charges, not only their committed rows.
            drop(batches);
            match reply {
                capture::Reply::Stored {
                    stored,
                    positions,
                    reply,
                } => {
                    let _ = reply.send(capture::stored_answer(stored, &positions, answer, atomic));
                }
                capture::Reply::Nobody => {
                    for (input, answer) in inputs.iter().zip(answer) {
                        match answer {
                            Err(StoreError::Stopped) | Ok(()) => {}
                            Err(error) => self.report.errors.push(polylogue_model::Failure::new(
                                format_args!("input {input}"),
                                &error,
                            )),
                        }
                    }
                }
            }
        }
    }

    fn write<'a>(
        &mut self,
        batches: impl Iterator<Item = &'a Batch> + Clone,
        replay: bool,
    ) -> Result<()> {
        if replay {
            self.until_stopped(|inner| inner.write_uninterrupted(batches))?
                .ok_or(StoreError::Stopped)
        } else {
            self.write_uninterrupted(batches)
        }
    }

    fn write_uninterrupted<'a>(
        &mut self,
        batches: impl Iterator<Item = &'a Batch> + Clone,
    ) -> Result<()> {
        let mut chains = Vec::new();
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
            if !matches!(b.change, Change::Reparse { .. })
                || b.new_blob.is_some()
                || b.observation.is_some()
            {
                return Err(StoreError::internal(
                    "the index writer takes parsed rows of bytes the archive has committed",
                ));
            }
            chains.push(stored_reading_chain(&self.archive, b)?);
        }
        let mut has_step = false;
        let mut all_replayed = true;
        'batches: for (batch, chain) in batches.clone().zip(&chains) {
            for step in &batch.parse {
                has_step = true;
                if step.delivery.is_none()
                    || crate::wire_shapes::admit_export(
                        &self.index,
                        &batch.input.id,
                        *chain,
                        batch.length,
                        step,
                    )?
                {
                    all_replayed = false;
                    break 'batches;
                }
            }
        }
        if has_step && all_replayed {
            return Ok(());
        }
        let indexed = batches.collect::<Vec<_>>();
        // While a build's load and composition run, the build file has no
        // indexes to write rows by: the bytes are stored, and upkeep reads
        // them once the build is installed.
        let reads = self.build_job.is_none();
        let itx = self
            .index
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        crate::fault::check("write.replay_rows", "")?;
        let counter = bump_counter(&itx)?;
        let changes = crate::change_summary::ChangeSummary::start(&itx)?;
        let mut touched = readings::Touched::default();
        let mut batches = 0u64;
        for (b, chain) in indexed.iter().zip(chains) {
            if reads {
                for p in &b.parse {
                    let source_commit = crate::source::Source::reading_commit(
                        &itx,
                        &b.input.id,
                        chain,
                        p.parsed_upto,
                    )?;
                    rows::write_step(
                        &itx,
                        b,
                        p,
                        rows::StepWrite {
                            chain,
                            source_commit: &source_commit,
                            replacement: false,
                            changes: &changes,
                        },
                        &mut touched,
                    )?;
                }
            }
            batches += 1;
        }
        let (messages, read) = rows::finish(&itx, touched, &changes, counter)?;
        self.report.item_units_read += read;
        crate::fault::check("write.index_commit", "")?;
        itx.commit()?;
        self.commits
            .committed(crate::completion::Committed::CONTENT);
        self.search.wake();
        self.report.batches += batches;
        self.report.messages += messages;
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
                + p.output.prefix_copies.len()
                + p.output.prefix_members.len()
        })
        .sum::<usize>()
}

#[cfg(test)]
mod tests;

/// Decoded blobs are outputs of stored input, not a new capture read.
/// Spooling them can finish before the unfinished index reading is cancelled.
fn replayed_input(batch: &Batch) -> bool {
    matches!(batch.change, Change::Reparse { .. })
        && batch.new_blob.is_none()
        && batch.observation.is_none()
}

/// Failure records still commit so a stop never erases their diagnosis.
fn cancellable_replay(batch: &Batch) -> bool {
    replayed_input(batch) && batch.parse.iter().all(|step| step.failure.is_none())
}

/// Packs each step's rows and settles its attachment references on the
/// sending thread, before the writer takes the batches: references name only
/// bytes the archive holds (`writer/attachments.rs`).
fn prepare_batches(location: &Archive, batches: &mut [&mut Batch]) -> Result<()> {
    if batches.iter().all(|batch| batch.parse.is_empty()) {
        return Ok(());
    }
    for batch in batches.iter_mut() {
        for step in &mut batch.parse {
            step.prepare();
            step.check_events(&batch.input.id)?;
        }
    }
    let archive = crate::archive::open_read(&location.archive_db())?;
    let pending: HashMap<_, _> = batches
        .iter()
        .flat_map(|batch| batch.new_blob.iter().chain(&batch.blobs))
        .map(|blob| (blob.hash, blob.size))
        .collect();
    for batch in batches.iter_mut() {
        attachments::normalize(batch, &archive, &pending)?;
    }
    Ok(())
}

/// How long what archive commits handed over waits for a batch whose
/// transaction it can join.
const FOLLOW_AFTER: std::time::Duration = std::time::Duration::from_millis(50);

/// The meta key of the archive position the index has followed to.
const ARCHIVE_POSITION: &str = "archive_position";

/// Adds `other`'s figures to `report`.
fn merge_report(report: &mut WriteReport, other: WriteReport) {
    report.batches += other.batches;
    report.messages += other.messages;
    report.transactions += other.transactions;
    report.segment_syncs += other.segment_syncs;
    report.segment_bytes_written += other.segment_bytes_written;
    report.blobs_appended += other.blobs_appended;
    report.blobs_deduplicated += other.blobs_deduplicated;
    report.item_units_read += other.item_units_read;
    report.errors.extend(other.errors);
}
