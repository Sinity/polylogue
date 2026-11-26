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

use crate::ConnExt as _;
use crate::OptionExt as _;
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

mod annotations;
mod assets;
mod dispatch;
pub use dispatch::ParentLifetime;
#[cfg(any(test, feature = "fixture"))]
pub use dispatch::{DispatchPause, DispatchPhase};
mod attachments;
mod authored;
mod batch;
pub mod build;
mod bytes;
mod chain;
mod compose;
mod derive_thread;
mod query_objects;
pub use query_objects::HistorySlot;
mod hook_fields;
pub(crate) mod identity;
mod saved;
mod scoped;
pub(crate) use readings::{read_message, shown_message};
pub(crate) mod readings;
mod rows;
mod search;
mod shown;
mod staging;
mod user_queue;
pub use chain::{ChainParse, ChainSink};

use crate::encoding::{PreparedDocument, PreparedMessage, block_text, unpack_block};
pub use batch::{
    Batch, DatabaseLog, ImportMember, InContainer, InFlight, InputKey, Observation, ParseStep,
    WriteReport,
};
use bytes::*;
use search::*;

enum StoreGroup {
    Prepared(
        Vec<crate::staging::PreparedBatch>,
        Arc<crate::blobs::Blobs>,
        SyncSender<Vec<Result<crate::staging::StoredInput>>>,
    ),
}

enum Msg {
    StoreBytes(
        Box<crate::staging::PreparedBytes>,
        Box<Batch>,
        SyncSender<Result<crate::staging::StoredInput>>,
    ),
    StoreGroup(StoreGroup),
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
    /// Bytes no input holds (a chat action's attachment), as one blob.
    StoreBlob(
        Box<(crate::NewBlob, Option<crate::staging::PreparedBytes>)>,
        SyncSender<Result<BlobHash>>,
    ),
    /// Load a build's runs, compose (on this many threads), index and
    /// install the build file.
    Build(
        Vec<std::path::PathBuf>,
        usize,
        Option<crossbeam_channel::Sender<&'static str>>,
        SyncSender<Result<Vec<build::BuildPhase>>>,
    ),
    ReplaceChain(InputKey, i64, u64, Box<ChainParse>, SyncSender<Result<u64>>),
    /// Record one older chain's reading from a newer chain of the same
    /// input holding the same bytes.
    CopyReading(crate::ReadingCopy, SyncSender<Result<Result<bool>>>),
    RestoreAuthored(
        Box<authored::Restoration>,
        SyncSender<Result<polylogue_model::user::UserOutcome>>,
    ),
    RestoreAnnotation(
        Box<annotations::Restoration>,
        SyncSender<Result<polylogue_model::user::UserOutcome>>,
    ),
    Stop,
    /// A user write is waiting in its own queue.
    Wake,
    Batch(Box<Batch>),
    Flush(SyncSender<Result<WriteReport>>, bool),
    /// Run the derive step over sessions whose derived rows are stale;
    /// answered with the number of sessions derived.
    Derive(SyncSender<Result<u64>>),
    DeriveInBackground(SyncSender<Result<bool>>),
    /// One wave of derived values from the derive thread, answered with
    /// which sessions it applied.
    DeriveWrite(Box<crate::derive::Write>, SyncSender<Result<Vec<bool>>>),
    /// One batch of full-text entries from the search thread, answered with
    /// the number written.
    SearchWrite(
        Box<crate::text_rows::SearchEntries>,
        SyncSender<Result<u64>>,
    ),
}

/// Handle to the writer thread. Dropping it stops the thread after the
/// batches already sent are written.
pub struct Writer {
    sender: WriterSender,
    /// The archive's stop request ([`Archive::stop`]). It ends the writer's
    /// long work early: a derive pass after the transaction in progress, a
    /// first build's ending (its deferred indexes) at its next step, replay
    /// of stored bytes and copies of identical-chain readings by rolling
    /// back their unfinished index transaction. What committed stays; the
    /// rest is done at the next start. Capture and authored writes are not
    /// affected.
    stop: crate::cancellation::Cancellation,
    handle: Option<JoinHandle<()>>,
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
    /// A second sender into the bulk queue that only wakes the writer for a
    /// user write. Its lock is held only for one `try_send`, never while a
    /// bulk sender waits for room, so a user write always sends its wake.
    wake: Arc<Mutex<Option<SyncSender<Msg>>>>,
    users: Arc<user_queue::UserQueue>,
    location: Archive,
    commits: crate::completion::CommitEvents,
    #[cfg(any(test, feature = "fixture"))]
    dispatch: dispatch::DispatchPauses,
}

/// A user request and what the writer checks before committing it.
pub struct UserSubmission {
    pub request: polylogue_model::user::UserRequest,
    /// The pinned read that selected the request.
    pub scope: Option<crate::ReadScopeToken>,
    /// Trusted catalog context.
    pub preparation: Option<crate::user_writes::PreparationContext>,
    /// The caller's deadline and cancellation.
    pub parent: Option<Arc<ParentLifetime>>,
}

impl From<polylogue_model::user::UserRequest> for UserSubmission {
    fn from(request: polylogue_model::user::UserRequest) -> Self {
        Self {
            request,
            scope: None,
            preparation: None,
            parent: None,
        }
    }
}

enum UserMsg {
    Request(
        Box<UserSubmission>,
        SyncSender<Result<crate::user_writes::WriteResult>>,
    ),
    QueryObject(
        Box<polylogue_model::query_objects::QueryObject>,
        Option<Arc<ParentLifetime>>,
        SyncSender<Result<()>>,
    ),
    QueryRun(
        Box<polylogue_model::query_objects::QueryRun>,
        Option<Arc<ParentLifetime>>,
        SyncSender<Result<()>>,
    ),
    QueryRunOutcome(
        Box<polylogue_model::query_objects::QueryRunOutcome>,
        SyncSender<Result<()>>,
    ),
    History(Box<query_objects::History>),
}

/// User writes waiting before their senders block.
const USER_QUEUE: usize = 64;

impl WriterQueue {
    fn new(
        bulk: SyncSender<Msg>,
        users: Arc<user_queue::UserQueue>,
        location: Archive,
        commits: crate::completion::CommitEvents,
        #[cfg(any(test, feature = "fixture"))] dispatch: dispatch::DispatchPauses,
    ) -> Self {
        Self {
            wake: Arc::new(Mutex::new(Some(bulk.clone()))),
            bulk: Arc::new(Mutex::new(Some(bulk))),
            users,
            location,
            commits,
            #[cfg(any(test, feature = "fixture"))]
            dispatch,
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

    fn ask<T>(&self, message: impl FnOnce(SyncSender<Result<T>>) -> Msg) -> Result<T> {
        let (reply, wait) = sync_channel(1);
        self.send(message(reply))
            .map_err(|_| StoreError::WriterGone)?;
        wait.recv().map_err(|_| StoreError::WriterGone)?
    }

    fn send_user(&self, message: UserMsg) -> std::result::Result<(), ()> {
        self.users.push_wait(message).map_err(|_| ())?;
        self.wake();
        Ok(())
    }

    /// Wakes a writer that may be waiting for bulk work. A full bulk queue
    /// refuses the wake, and needs none: the writer has messages to take
    /// and serves user writes before each one.
    fn wake(&self) {
        if let Ok(wake) = self.wake.lock()
            && let Some(wake) = wake.as_ref()
        {
            let _ = wake.try_send(Msg::Wake);
        }
    }

    /// As [`Self::send_user`], but a full queue hands the message back
    /// instead of waiting for room.
    fn try_send_user(&self, message: UserMsg) -> std::result::Result<(), Option<UserMsg>> {
        self.users.try_push(message)?;
        self.wake();
        Ok(())
    }

    fn stop(&self) {
        self.users.close();
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
/// otherwise `index.db`, with the archive's register attached.
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
    /// Starts the writer thread on `archive`, which the caller has locked.
    /// A build writes the private index file and defers search and secondary
    /// indexes; a stop keeps its committed rows and positions for the next
    /// start, and a crash discards it ([`crate::archive::index_build`]).
    pub fn start(archive: &Archive) -> Result<Self> {
        // A build writes its private file ([`crate::archive::index_build`]).
        let building = archive.index_build.building();
        let index = open_index_writer(archive, building)?;
        let installed = archive.root.join("index.db");
        let archive_conn = crate::archive::open_archive_writer(archive)?;
        crate::write_lock::sync_register(archive)?;
        crate::chain_progress::record_source_changes(&archive_conn)?;
        crate::chain_progress::open(&index)?;
        let segment = crate::blobs::open_writer(&archive_conn, &archive.root)?;
        let commits = crate::completion::CommitEvents::default();
        #[cfg(any(test, feature = "fixture"))]
        let dispatch = dispatch::DispatchPauses::default();
        let (tx, rx) = sync_channel::<Msg>(QUEUE);
        let users = Arc::new(user_queue::UserQueue::default());
        let stop = archive.stop().clone();
        let queue = WriterQueue::new(
            tx,
            Arc::clone(&users),
            archive.clone(),
            commits.clone(),
            #[cfg(any(test, feature = "fixture"))]
            dispatch.clone(),
        );
        let mut inner = Inner {
            #[cfg(any(test, feature = "fixture"))]
            dispatch: dispatch.clone(),
            stop: stop.clone(),
            commits: commits.clone(),
            segment: Some(segment),
            archive: archive_conn,
            index,
            report: WriteReport::default(),
            users,
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
    /// Ends the writer thread after what it was sent; later calls fail with
    /// [`StoreError::WriterGone`].
    pub fn stop(&mut self) {
        let started =
            (self.handle.is_some() && self.stop.is_cancelled()).then(std::time::Instant::now);
        if started.is_some() {
            crate::log!(info, "shutdown writer drain started");
        }
        self.sender.0.stop();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        if let Some(started) = started {
            crate::log!(
                info,
                "shutdown writer drain finished in {:?}",
                started.elapsed()
            );
        }
    }
}

/// Where [`WriterSender::try_enqueue_user`] left a request.
pub enum Queued {
    /// In the queue; the receiver gets its outcome once committed.
    Committing(Receiver<Result<crate::user_writes::WriteResult>>),
    /// The queue was full: the complete submission, handed back.
    Full(Box<UserSubmission>),
}

/// A clonable sender into the writer, for worker threads.
#[derive(Clone)]
pub struct WriterSender(WriterQueue);

impl WriterSender {
    pub fn completion_commits(&self) -> crate::completion::CommitSubscription {
        self.0.commits.subscribe()
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
        prepare_batches(&self.0.location, &mut batches)?;
        let (reply, wait) = sync_channel(1);
        self.0
            .send(Msg::StoreGroup(StoreGroup::Prepared(group, store, reply)))
            .map_err(|_| StoreError::WriterGone)?;
        Ok(wait)
    }

    pub fn send(&self, mut batch: Batch) -> Result<()> {
        prepare_batches(&self.0.location, &mut [&mut batch])?;
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

    /// Writes one user request and waits for its source commit.
    /// It goes ahead of batches still queued.
    pub fn user(
        &self,
        submission: impl Into<UserSubmission>,
    ) -> Result<polylogue_model::user::UserOutcome> {
        self.user_receipt(submission).map(|receipt| receipt.outcome)
    }

    /// Return the accepting transaction's exact resolved part outcomes too.
    pub fn user_receipt(
        &self,
        submission: impl Into<UserSubmission>,
    ) -> Result<crate::user_writes::WriteResult> {
        let (reply, wait) = sync_channel(1);
        self.0
            .send_user(UserMsg::Request(Box::new(submission.into()), reply))
            .map_err(|_| StoreError::WriterGone)?;
        wait.recv().map_err(|_| StoreError::WriterGone)?
    }

    /// Queues a user request without waiting for its commit or for room.
    /// A full queue hands the complete submission back as [`Queued::Full`].
    pub fn try_enqueue_user(&self, submission: impl Into<UserSubmission>) -> Result<Queued> {
        let (reply, wait) = sync_channel(1);
        match self
            .0
            .try_send_user(UserMsg::Request(Box::new(submission.into()), reply))
        {
            Ok(()) => Ok(Queued::Committing(wait)),
            Err(Some(UserMsg::Request(submission, _))) => Ok(Queued::Full(submission)),
            Err(_) => Err(StoreError::WriterGone),
        }
    }

    /// Stores one import's bytes. The batch carries the observation, never parsed rows.
    pub fn store_bytes(
        &self,
        prepared: crate::staging::PreparedBytes,
        batch: Batch,
    ) -> Result<crate::staging::StoredInput> {
        self.0
            .ask(|reply| Msg::StoreBytes(Box::new(prepared), Box::new(batch), reply))
    }

    /// Stores bytes no input holds as one blob; `file` is the private file
    /// [`crate::Archive::prepare_blob`] wrote for a blob too large to keep inline.
    pub fn store_blob(
        &self,
        new_blob: crate::NewBlob,
        file: Option<crate::staging::PreparedBytes>,
    ) -> Result<BlobHash> {
        self.0
            .ask(|reply| Msg::StoreBlob(Box::new((new_blob, file)), reply))
    }

    /// Accepts one completed discovery walk; it is visible once its index
    /// transaction commits.
    pub fn coverage(&self, coverage: polylogue_model::capture::Coverage) -> Result<()> {
        self.0.ask(|reply| Msg::Coverage(Box::new(coverage), reply))
    }

    /// Records one older chain's complete reading from its newer chain of
    /// the same bytes. A successful copy commits all its rows and source commits
    /// before answering. The inner answer is `false` when copying would not
    /// equal parsing, or an error from the copy;
    /// both leave the chain unchanged and let the caller parse it instead.
    /// Transaction, transport and stop failures are outer errors. A stop
    /// interrupts only this reconstructible copy as `Stopped`;
    /// copies already committed remain recorded.
    pub fn copy_reading(&self, copy: crate::ReadingCopy) -> Result<Result<bool>> {
        self.0.ask(|reply| Msg::CopyReading(copy, reply))
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

    fn flush_with_derivation(&self, derive: bool) -> Result<WriteReport> {
        self.0.ask(|reply| Msg::Flush(reply, derive))
    }
}

struct Inner {
    /// Ends a derive pass or a first build's ending early.
    stop: crate::cancellation::Cancellation,
    #[cfg(any(test, feature = "fixture"))]
    dispatch: dispatch::DispatchPauses,
    commits: crate::completion::CommitEvents,
    segment: Option<hashed_blobs::SegmentWriter>,
    archive: Connection,
    index: Connection,
    report: WriteReport,
    /// User requests, served between the writer's transactions.
    users: Arc<user_queue::UserQueue>,
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

impl Drop for Inner {
    fn drop(&mut self) {
        self.users.disconnect();
    }
}

impl Inner {
    /// A build is in progress: the build file holds no rows until the batch
    /// build loads them ([`build`]).
    fn building(&self) -> bool {
        self.location.index_build.building()
    }

    /// Writes every waiting user request, each answered once committed.
    fn serve_users(&mut self) {
        // New arrivals wake the next turn; this turn finishes its existing
        // user requests before bulk work, even under a continuous read load.
        for _ in 0..self.users.waiting() {
            let Some(message) = self.users.pop() else {
                break;
            };
            match message {
                UserMsg::Request(submission, reply) => {
                    let UserSubmission {
                        request,
                        scope,
                        preparation,
                        parent,
                    } = *submission;
                    let result = self
                        .check_user_parent(&request.id, parent.as_deref())
                        .and_then(|()| {
                            self.write_user(&request, scope.as_ref(), preparation.as_ref())
                        });
                    // A queued request nobody waits for still reports its failure.
                    if let Err(std::sync::mpsc::SendError(Err(error))) = reply.send(result) {
                        self.report.errors.push(polylogue_model::Failure::new(
                            format_args!("user request {}", request.id),
                            &error,
                        ));
                    }
                }
                UserMsg::QueryObject(object, parent, reply) => {
                    let result = self
                        .check_user_parent(&object.id.to_string(), parent.as_deref())
                        .and_then(|()| self.write_query_object(&object));
                    let _ = reply.send(result);
                }
                UserMsg::QueryRun(run, parent, reply) => {
                    if let Err(error) = self.check_user_parent(run.id.as_str(), parent.as_deref()) {
                        let _ = reply.send(Err(error));
                        continue;
                    }
                    let result = self.write_query_run(&run);
                    if let Err(error) = &result {
                        self.report.errors.push(polylogue_model::Failure::new(
                            format_args!("query run {}", run.id),
                            error,
                        ));
                    }
                    let _ = reply.send(result);
                }
                UserMsg::QueryRunOutcome(outcome, reply) => {
                    let result = self.write_query_run_outcome(&outcome);
                    if let Err(error) = &result {
                        self.report.errors.push(polylogue_model::Failure::new(
                            format_args!("query run outcome {}", outcome.run),
                            error,
                        ));
                    }
                    let _ = reply.send(result);
                }
                UserMsg::History(history) => {
                    if let Err(error) = self
                        .check_user_parent(history.run.id.as_str(), None)
                        .and_then(|()| self.write_history(&history))
                    {
                        self.report.errors.push(polylogue_model::Failure::new(
                            format_args!("query history {}", history.run.id),
                            &error,
                        ));
                    }
                }
            }
        }
    }

    fn run(&mut self, rx: &Receiver<Msg>) {
        let _worker = crate::timing::worker();
        let mut flushes = Vec::new();
        let mut derives = Vec::new();
        let mut background_derives = Vec::new();
        let mut derive_writes = Vec::new();
        let mut search_writes = Vec::new();
        let mut single_files = Vec::new();
        let mut file_groups = Vec::new();
        let mut assets = Vec::new();
        let mut coverage = Vec::new();
        let mut live_clock = Vec::new();
        let mut copies = Vec::new();
        let mut restorations = Vec::new();
        let mut annotation_restorations = Vec::new();
        let mut replacements = Vec::new();
        let mut builds = Vec::new();
        let mut blobs = Vec::new();
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
            } else {
                match rx.recv() {
                    Ok(first) => first,
                    Err(_) => break,
                }
            };
            let _busy = crate::timing::start_stage(crate::timing::Stage::Writer);
            let _doing = crate::activity::doing("writing");
            self.serve_users();
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
                    Msg::StoreGroup(StoreGroup::Prepared(items, store, reply)) => {
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
                    Msg::LiveClock(sessions, reply) => {
                        live_clock.push((sessions, reply));
                        false
                    }
                    Msg::CopyReading(copy, reply) => {
                        copies.push((copy, reply));
                        false
                    }
                    Msg::StoreBlob(blob, reply) => {
                        blobs.push((blob, reply));
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
                    Msg::RestoreAuthored(restoration, reply) => {
                        restorations.push((restoration, reply));
                        false
                    }
                    Msg::RestoreAnnotation(restoration, reply) => {
                        annotation_restorations.push((restoration, reply));
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
                crate::timing::time(crate::timing::Stage::Rows, || self.write_group(&group));
            }
            // Later queue replies also observe release of the parsed batches'
            // memory charges, not only their committed database rows.
            drop(group);
            for (write, reply) in derive_writes.drain(..) {
                let result = crate::derive::apply(&self.index, &write);
                if result.is_ok() {
                    self.commits
                        .committed(crate::completion::Committed::CONTENT);
                }
                let _ = reply.send(result);
            }
            for (entries, reply) in search_writes.drain(..) {
                let _ = reply.send(crate::text_rows::apply_entries(&self.index, &entries));
            }
            for (prepared, batch, reply) in single_files.drain(..) {
                let _ = reply.send(self.store_bytes(*prepared, *batch));
            }
            self.store_groups(std::mem::take(&mut file_groups));
            for (input, chain, length, reading, reply) in replacements.drain(..) {
                let _ = reply.send(crate::timing::time(crate::timing::Stage::Rows, || {
                    self.replace_chain(input, chain, length, reading)
                }));
                self.serve_users();
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
            for (blob, reply) in blobs.drain(..) {
                let (new_blob, file) = *blob;
                let _ = reply.send(self.store_blob(new_blob, file));
            }
            for (restoration, reply) in restorations.drain(..) {
                let _ = reply.send(self.restore_authored(*restoration));
            }
            for (restoration, reply) in annotation_restorations.drain(..) {
                let _ = reply.send(self.restore_annotation(*restoration));
            }
            self.serve_users();

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
                let pending = crate::derive::caught_up(&self.index).map(|current| !current);
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
            for (copy, reply) in copies.drain(..) {
                let _ = reply.send(self.write_copy_reading(&copy));
            }
            if let Err(error) = self.refresh_chain_progress() {
                self.report
                    .errors
                    .push(polylogue_model::Failure::new("chain progress", &error));
            }
            for (reply, derive) in flushes.drain(..) {
                let report = std::mem::take(&mut self.report);
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
        let closed = self
            .refresh_chain_progress()
            .and_then(|()| crate::chain_progress::mark_current(&self.index));
        if let Err(error) = closed {
            crate::log!(error, "writer: chain progress: {error}");
        }
    }

    /// Recomputes, in an index transaction of their own, the chain progress
    /// of inputs whose source rows an archive transaction changed
    /// with no index transaction after it (bytes stored without parsed rows,
    /// a restored older archive, a write whose index transaction failed).
    fn refresh_chain_progress(&mut self) -> Result<()> {
        let source_changed = crate::chain_progress::recorded(&self.archive)?;
        if source_changed.is_empty() {
            return Ok(());
        }
        let tx = self
            .index
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        bump_counter(&tx)?;
        crate::chain_progress::refresh(&tx, &source_changed)?;
        tx.commit()?;
        crate::chain_progress::forget(&self.archive, &source_changed)?;
        self.commits
            .committed(crate::completion::Committed::CONTENT);
        self.report.transactions += 1;
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
        if !self.stop.is_cancelled() {
            results.push(crate::archive::restart_full_log(
                &self.archive,
                "main",
                &self.location.archive_db(),
                limit,
                wait,
            ));
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

    fn write_copy_reading(&mut self, copy: &crate::ReadingCopy) -> Result<Result<bool>> {
        self.until_stopped(|inner| inner.copy_reading_uninterrupted(copy))?
            .ok_or(StoreError::Stopped)
    }

    fn copy_reading_uninterrupted(&mut self, copy: &crate::ReadingCopy) -> Result<Result<bool>> {
        let tx = self.index.transaction()?;
        let counter = bump_counter(&tx)?;
        let answer = crate::fault::check("write.copy_reading", &copy.input)
            .and_then(|()| crate::identical_chains::copy_reading(&tx, copy, counter));
        let answer = match answer {
            Err(StoreError::Sqlite(error))
                if error.sqlite_error_code() == Some(rusqlite::ErrorCode::OperationInterrupted) =>
            {
                // SQLite has rolled back the interrupted transaction. Do not
                // turn cancellation into a refusal followed by parsing.
                return Err(StoreError::Sqlite(error));
            }
            answer => answer,
        };
        if matches!(answer, Ok(true)) {
            crate::write_lock::commit(tx)?;
            self.commits
                .committed(crate::completion::Committed::CONTENT);
            self.report.transactions += 1;
        } else {
            // A refusal or a copy error leaves no rows, source commits or
            // write-counter change from this attempt.
            tx.rollback()?;
        }
        Ok(answer)
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
                crate::derive::mark_moved_versions(&mut inner.index)
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
        crate::chain_progress::open(&self.index)?;
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
        crate::chain_progress::install_triggers(&self.index)?;
        installed
    }

    /// Writes a group in one transaction per database. If it fails, each
    /// batch is retried alone, so one bad batch costs only itself.
    fn write_group(&mut self, group: &[Box<Batch>]) {
        let replay = group.iter().all(|batch| replayed_input(batch))
            && group.iter().any(|batch| cancellable_replay(batch));
        if group.len() > 1 && self.write(group.iter().map(|b| &**b), replay).is_ok() {
            return;
        }
        if group.len() == 1 && self.write(std::iter::once(&*group[0]), replay).is_ok() {
            return;
        }
        for b in group {
            if let Err(e) = self.write(std::iter::once(&**b), cancellable_replay(b)) {
                if matches!(e, StoreError::Stopped) {
                    continue;
                }
                self.report.errors.push(polylogue_model::Failure::new(
                    format_args!("input {}", b.input.id),
                    &e,
                ));
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
        let file = file.or_invalid("new blob has no prepared bytes")?;
        if file.owner != self.location.root
            || file.hash() != new_blob.hash
            || file.len() != new_blob.size
        {
            return Err(StoreError::Invalid(
                "prepared bytes do not match the new blob".into(),
            ));
        }
        let prepared = file.into_prepared()?;
        if self.segment.is_none() {
            self.segment = Some(crate::blobs::open_writer(
                &self.archive,
                &self.location.root,
            )?);
        }
        let writer = self.segment.as_mut().expect("opened segment writer");
        let pending = match writer.append_group(std::iter::once(Ok(prepared)), |_| Ok(None)) {
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
            crate::blobs::record_committed_end(&tx, synced.end)?;
            crate::write_lock::commit(tx)?;
            self.commits
                .committed(crate::completion::Committed::CONTENT);
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
        scope: Option<&crate::ReadScopeToken>,
        preparation: Option<&crate::user_writes::PreparationContext>,
    ) -> Result<crate::user_writes::WriteResult> {
        crate::fault::check("write.user", &request.id)?;
        scoped::require_current(&self.location, scope)?;
        let prepared_scope = self.preflight_saved(request, preparation, scope)?;
        let scope = scope.or(prepared_scope.as_ref());
        let tx = self.archive.transaction()?;
        scoped::require_current(&self.location, scope)?;
        let known: bool = tx.exists(
            "SELECT EXISTS(SELECT 1 FROM client_request WHERE id=?1)",
            [&request.id],
        )?;
        let seq = if known {
            crate::session_values::write_counter(&tx)?
        } else {
            bump_counter(&tx)?
        };
        let outcome = {
            let source_snapshot = rusqlite::Transaction::new_unchecked(
                &self.index,
                rusqlite::TransactionBehavior::Deferred,
            )?;
            crate::user_writes::apply_resolving(
                &tx,
                &source_snapshot,
                request,
                seq,
                crate::user_writes::Preparation::plain(preparation),
            )?
        };
        scoped::require_current(&self.location, scope)?;
        if known
            || (matches!(
                request.write,
                polylogue_model::user::UserWrite::MarkSeen { .. }
            ) && outcome.outcome.repeat)
        {
            tx.rollback()?;
        } else {
            crate::write_lock::commit(tx)?;
            self.commits
                .committed(crate::completion::Committed::AUTHORED);
        }
        if !matches!(
            request.write,
            polylogue_model::user::UserWrite::Deliver(_)
                | polylogue_model::user::UserWrite::DeliveredAdvice(_)
                | polylogue_model::user::UserWrite::MarkSeen { .. }
        ) {
            self.project_alternate_text()?;
        }
        Ok(outcome)
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
        let mut parsed_only = true;
        for batch in batches.clone() {
            if !matches!(batch.change, Change::Reparse { .. })
                || batch.new_blob.is_some()
                || batch.observation.is_some()
            {
                parsed_only = false;
                break;
            }
            for blob in &batch.blobs {
                let size = i64::try_from(blob.size).map_err(|_| {
                    StoreError::Invalid("decoded body exceeds the source catalog range".into())
                })?;
                if !self.archive.query_row(
                    "SELECT EXISTS(SELECT 1 FROM blob WHERE hash=?1 AND size=?2)",
                    params![blob.hash.0.as_slice(), size],
                    |row| row.get::<_, bool>(0),
                )? {
                    parsed_only = false;
                    break;
                }
            }
        }
        let mut chains = Vec::new();
        if parsed_only {
            for batch in batches.clone() {
                chains.push(Some(stored_reading_chain(&self.archive, batch)?));
            }
            let mut has_step = false;
            let mut all_replayed = true;
            'batches: for batch in batches.clone() {
                let Change::Reparse { chain } = batch.change else {
                    unreachable!()
                };
                for step in &batch.parse {
                    has_step = true;
                    if step.delivery.is_none()
                        || crate::wire_shapes::admit_export(
                            &self.index,
                            &batch.input.id,
                            chain,
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
        }
        if !parsed_only {
            let atx = self.archive.transaction()?;
            let seq = bump_counter(&atx)?;
            for batch in batches.clone() {
                let (chain, _) = write_bytes(&atx, batch, seq)?;
                chains.push(chain);
            }
            crate::fault::check("write.archive_commit", "")?;
            crate::write_lock::commit(atx)?;
            self.commits
                .committed(crate::completion::Committed::CONTENT);
            crate::fault::check("write.between_commits", "")?;
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
            if let Some(chain) = chain.filter(|_| reads) {
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
                            counter,
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
        let (messages, read) = rows::finish_upkeep(&itx, touched, &changes, counter)?;
        self.report.item_units_read += read;
        let source_changed = crate::chain_progress::recorded(&self.archive)?;
        crate::chain_progress::refresh(&itx, &source_changed)?;
        crate::fault::check("write.index_commit", "")?;
        itx.commit()?;
        crate::chain_progress::forget(&self.archive, &source_changed)?;
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
    replayed_input(batch)
        && batch.parse.iter().all(|step| {
            !matches!(
                step.progress,
                polylogue_model::batch::StepProgress::Refused { .. }
                    | polylogue_model::batch::StepProgress::Failed { .. }
            )
        })
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
