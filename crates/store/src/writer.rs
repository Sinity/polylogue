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
use polylogue_model::LockExt;
use polylogue_model::{
    BlobHash, Block, ErrorReason, Event, Json, Link, Message, MessageRow, Output, RecordCounts,
    Session, SessionId, Timestamp, ToolAction, ToolOutcome,
};
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

mod assertions;
mod assets;
mod dispatch;
pub use dispatch::ParentLifetime;
#[cfg(any(test, feature = "fixture"))]
pub use dispatch::{DispatchPause, DispatchPhase};
mod attachments;
mod authored;
mod batch;
mod bytes;
mod chain;
mod compose;
mod derive_thread;
mod follow;
mod primary;
mod primary_capture;
mod query_objects;
mod retractions;
pub use query_objects::HistorySlot;
mod saved;
mod scoped;
mod source_snapshot;
pub(crate) use retractions::{apply_source_retractions, prepare_source_retractions};
pub(crate) use rows::finish as finish_rows;
mod hook_fields;
mod older_archive;
pub use older_archive::{GroupOutcome, OlderGroup, Restored};
pub(crate) mod resolve;
mod rows;
mod search;
mod staging;
mod user_queue;
pub use chain::ChainReading;

#[cfg(test)]
use crate::encoding::pack_block;
use crate::encoding::{PreparedDocument, PreparedMessage};
use crate::encoding::{block_text, unpack_block};
pub use batch::{
    Batch, DatabaseLog, ImportMember, InContainer, InFlight, InputKey, Observation, ParseStep,
    WriteReport,
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
    /// Sessions whose live state changed with the clock alone: each gets a
    /// new change counter, so the change feed carries the transition.
    LiveClock(Vec<String>, SyncSender<Result<()>>),
    /// Bytes no input holds (a chat action's attachment), as one blob.
    StoreBlob(
        Box<(crate::NewBlob, Option<crate::staging::PreparedBytes>)>,
        SyncSender<Result<BlobHash>>,
    ),
    ReplaceChain(
        InputKey,
        i64,
        u64,
        Box<ChainReading>,
        SyncSender<Result<u64>>,
    ),
    /// Record one older chain's reading from a newer chain of the same
    /// input holding the same bytes.
    CopyReading(crate::ReadingCopy, SyncSender<Result<Result<bool>>>),
    StageConfirmed(
        Vec<crate::primary::Confirmed>,
        SyncSender<Result<crate::primary::confirmed::Step>>,
    ),
    ApplyConfirmed(
        Box<crate::primary::confirmed::Preparation>,
        crate::completion::Committed,
        SyncSender<Result<crate::primary::confirmed::Step>>,
    ),
    DrainPrimaryCaptures(SyncSender<Result<()>>),
    /// The stream follower handled a page, or failed to read one.
    StreamFollowed(Option<String>),
    CaptureBody(
        String,
        SyncSender<Result<Vec<crate::primary::PreparedBody>>>,
    ),
    Capture(String, SyncSender<Result<crate::primary::Capture>>),
    ByteCursor(
        Box<crate::primary::ByteCursorQuery>,
        SyncSender<Result<Option<crate::primary::ByteCursor>>>,
    ),
    PendingGroupHeaders(
        String,
        Option<u64>,
        usize,
        SyncSender<Result<crate::primary::PendingGroupPage>>,
    ),
    SourceSnapshot(
        crate::primary::SourceSelector,
        SyncSender<Result<crate::primary::SourceSnapshot>>,
    ),
    RestoreAuthored(
        Box<authored::Restoration>,
        SyncSender<Result<polylogue_model::user::UserOutcome>>,
    ),
    RestoreAssertion(
        Box<assertions::Restoration>,
        SyncSender<Result<polylogue_model::user::UserOutcome>>,
    ),
    /// Inputs of an archive in the older format, stored as that archive held them.
    RestoreOlder(
        Box<older_archive::OlderGroup>,
        SyncSender<Result<older_archive::GroupOutcome>>,
    ),
    Stop,
    /// A user write is waiting in its own queue.
    Wake,
    Batch(Box<Batch>),
    Flush(SyncSender<Result<WriteReport>>, bool),
    /// Run the derive step over sessions whose derived rows are stale;
    /// answered with the number of sessions derived.
    Derive(SyncSender<Result<u64>>),
    /// One wave of derived values from the derive thread, answered with
    /// which sessions it applied.
    DeriveWrite(Box<crate::derive::Write>, SyncSender<Result<Vec<bool>>>),
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
    /// In primary mode, the thread that reads the confirmed stream.
    follower: Option<follow::Follower>,
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
    resolver: Arc<resolve::Resolver>,
    commits: crate::completion::CommitEvents,
    sinex: Option<Arc<dyn crate::primary::Sinex>>,
    stop: crate::cancellation::Cancellation,
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
        SyncSender<Result<crate::user_writes::UserReceipt>>,
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
        resolver: Arc<resolve::Resolver>,
        commits: crate::completion::CommitEvents,
        sinex: Option<Arc<dyn crate::primary::Sinex>>,
        stop: crate::cancellation::Cancellation,
        #[cfg(any(test, feature = "fixture"))] dispatch: dispatch::DispatchPauses,
    ) -> Self {
        Self {
            wake: Arc::new(Mutex::new(Some(bulk.clone()))),
            bulk: Arc::new(Mutex::new(Some(bulk))),
            users,
            resolver,
            commits,
            sinex,
            stop,
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

/// Records how many stored chains a build has read, from the counts the
/// writer keeps with each change ([`crate::chain_progress`]).
fn record_build_progress(archive: &Archive, index: &Connection) -> Result<()> {
    if archive.is_primary() || !archive.index_build.building() {
        return Ok(());
    }
    let (mut chains, mut read) = (0, 0);
    for (_, _, state, n) in crate::chain_progress::counts(index)? {
        chains += n;
        if matches!(
            state,
            crate::chain_progress::State::Current | crate::chain_progress::State::NoParser
        ) {
            read += n;
        }
    }
    archive.index_build.record_progress(chains, read);
    Ok(())
}

/// The writer's index connection: the private build file while a build runs,
/// otherwise `index.db`, with a standalone archive's register attached.
fn open_index_writer(archive: &Archive, building: bool) -> Result<Connection> {
    let index = if building {
        crate::archive::index_build::open_writer(archive)?
    } else {
        open_runtime_writer(
            &archive.index_db(),
            archive.memory_budget().writer_cache_mib(),
        )?
    };
    if archive.is_primary() {
        // Changing temporary storage drops TEMP objects, so set it before triggers.
        index.pragma_update(None, "temp_store", "file")?;
        index.pragma_update(Some("temp"), "cache_size", -8192)?;
    } else {
        let uri = format!("file:{}?mode=ro", archive.archive_db().display());
        index.execute("ATTACH DATABASE ?1 AS a", [uri])?;
        crate::source::views(&index)?;
    }
    Ok(index)
}

impl Writer {
    /// Starts the writer thread on `archive`, which the caller has locked.
    /// A build writes the private index file and defers search and secondary
    /// indexes; a stop keeps its committed rows and positions for the next
    /// start, and a crash discards it ([`crate::archive::index_build`]).
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
        // A standalone build writes its private file ([`crate::archive::index_build`]).
        let building = !archive.is_primary() && archive.index_build.building();
        let index = open_index_writer(archive, building)?;
        let installed = archive.root.join("index.db");
        let archive_conn = if archive.is_primary() {
            crate::archive::open_read(&archive.index_db())?
        } else {
            let conn = crate::archive::open_archive_writer(archive)?;
            crate::write_lock::sync_register(archive)?;
            crate::chain_progress::record_source_changes(&conn)?;
            conn
        };
        crate::chain_progress::open(&index, !archive.is_primary())?;
        record_build_progress(archive, &index)?;
        let upkeep = if building {
            resolve::IndexUpkeep::AtBuildEnd(Arc::new(crate::key_maps::KeyMaps::load_with_budget(
                &index,
                archive.memory_budget().keys() / 3,
            )?))
        } else {
            resolve::IndexUpkeep::PerTransaction
        };
        let texts =
            crate::text_rows::TextIds::new(&index, building, archive.memory_budget().keys() / 3)?;
        let segment = if sinex.is_none() {
            Some(crate::blobs::open_writer(&archive_conn, &archive.root)?)
        } else {
            None
        };
        let resolver = resolve::Resolver::new(archive, &index, upkeep)?;
        let stream_source = sinex.clone();
        let commits = crate::completion::CommitEvents::default();
        #[cfg(any(test, feature = "fixture"))]
        let dispatch = dispatch::DispatchPauses::default();
        let (tx, rx) = sync_channel::<Msg>(QUEUE);
        let users = Arc::new(user_queue::UserQueue::default());
        let stop = archive.stop().clone();
        let queue = WriterQueue::new(
            tx,
            Arc::clone(&users),
            Arc::clone(&resolver),
            commits.clone(),
            sinex.clone(),
            stop.clone(),
            #[cfg(any(test, feature = "fixture"))]
            dispatch.clone(),
        );
        let mut inner = Inner {
            #[cfg(any(test, feature = "fixture"))]
            dispatch: dispatch.clone(),
            stop: stop.clone(),
            commits: commits.clone(),
            resolver: Arc::clone(&resolver),
            segment,
            archive: archive_conn,
            index,
            report: WriteReport::default(),
            texts,
            users,
            // Both open the installed index, once a build has ended.
            search: Search::start(
                installed.clone(),
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
            sinex,
            stream_error: None,
        };

        let handle = polylogue_model::spawn_named("polylogue-writer", move || inner.run(&rx))?;
        let follower = stream_source
            .map(|sinex| follow::Follower::start(sinex, WriterSender(queue.clone()), archive))
            .transpose()?;
        Ok(Self {
            sender: WriterSender(queue),
            stop,
            handle: Some(handle),
            follower,
        })
    }

    pub fn synchronize_primary(&self) -> Result<i64> {
        let follower = self
            .follower
            .as_ref()
            .or_invalid("this writer is standalone")?;
        follower.synchronize()?;
        // Acknowledged material can now produce input/reading events. Wait
        // outside the writer thread while the follower reads the next page.
        self.sender.0.ask(Msg::DrainPrimaryCaptures)?;
        follower.synchronize()
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
            eprintln!("polylogue: shutdown writer drain started");
        }
        if let Some(mut follower) = self.follower.take() {
            follower.stop();
        }
        self.sender.0.stop();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        if let Some(started) = started {
            eprintln!(
                "polylogue: shutdown writer drain finished in {:?}",
                started.elapsed()
            );
        }
    }
}

/// Where [`WriterSender::try_enqueue_user`] left a request.
pub enum Queued {
    /// In the queue; the receiver gets its outcome once committed.
    Committing(Receiver<Result<crate::user_writes::UserReceipt>>),
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
        if self.0.sinex.is_some() {
            for item in &mut group {
                if matches!(item.batch.change, Change::Reparse { .. })
                    && !item.batch.parse.is_empty()
                {
                    primary_capture::validate(item, self.0.resolver.location())?;
                    let index = crate::archive::open_read(&self.0.resolver.location().index_db())?;
                    primary::validate_primary_reading(&index, &item.batch)?;
                    drop(index);
                    let mut bodies = Vec::new();
                    for file in &item.files {
                        bodies.push((file.hash(), file.len()));
                        match self
                            .store_blob(NewBlob::whole(file.hash(), file.len()), Some(file.clone()))
                        {
                            Ok(_) | Err(StoreError::PrimaryPending { .. }) => {}
                            Err(error) => return Err(error),
                        }
                    }
                    self.wait_primary_blobs(&bodies)?;
                    self.spool_primary_batch(&mut item.batch)?;
                    item.files.clear();
                    item.batch.blobs.clear();
                }
            }
        }
        let primary = self.0.resolver.location().is_primary();
        let mut batches: Vec<_> = group
            .iter_mut()
            .filter(|item| !primary || matches!(item.batch.change, Change::Reparse { .. }))
            .map(|item| &mut item.batch)
            .collect();
        self.0.resolver.batches(&mut batches)?;
        let (reply, wait) = sync_channel(1);
        self.0
            .send(Msg::StoreGroup(group, store, reply))
            .map_err(|_| StoreError::WriterGone)?;
        Ok(wait)
    }

    pub fn send(&self, mut batch: Batch) -> Result<()> {
        if self.0.sinex.is_some() && matches!(batch.change, Change::Reparse { .. }) {
            self.spool_primary_batch(&mut batch)?;
        }
        if !self.0.resolver.location().is_primary()
            || matches!(batch.change, Change::Reparse { .. })
        {
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
        if self.0.sinex.is_some() {
            self.spool_primary_reading(&input, chain, length, &mut reading)?;
        }
        reading.resolve(&self.0.resolver, &input, chain, length)?;
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
    ) -> Result<crate::user_writes::UserReceipt> {
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
    /// [`crate::Archive::prepare_decoded`] wrote for a blob too large to keep inline.
    pub fn store_blob(
        &self,
        new_blob: crate::NewBlob,
        file: Option<crate::staging::PreparedBytes>,
    ) -> Result<BlobHash> {
        self.0
            .ask(|reply| Msg::StoreBlob(Box::new((new_blob, file)), reply))
    }

    /// Accepts one completed discovery walk. Primary visibility follows its
    /// confirmed product; standalone visibility follows the index transaction.
    pub fn coverage(&self, coverage: polylogue_model::capture::Coverage) -> Result<()> {
        self.0.ask(|reply| Msg::Coverage(Box::new(coverage), reply))
    }

    /// Records one older chain's complete reading from its newer chain of
    /// the same bytes. A successful copy commits all its rows and source commits
    /// before answering. The inner answer is `false` when copying would not
    /// equal parsing (always in primary mode), or an error from the copy;
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
    resolver: Arc<resolve::Resolver>,
    segment: Option<hashed_blobs::SegmentWriter>,
    archive: Connection,
    index: Connection,
    report: WriteReport,
    texts: crate::text_rows::TextIds,
    /// User requests, served between the writer's transactions.
    users: Arc<user_queue::UserQueue>,
    /// The search thread ([`search_thread`]): woken after each commit, and
    /// allowed to write once the index is out of a first build.
    search: Search,
    /// Runs derive passes, sending their values here to be applied.
    derive: derive_thread::DeriveThread,
    location: Archive,
    /// In primary mode, where every register row goes before it commits.
    sinex: Option<std::sync::Arc<dyn crate::primary::Sinex>>,
    /// The last stream error recorded, so a lasting outage is recorded once.
    stream_error: Option<String>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.users.disconnect();
    }
}

impl Inner {
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
                        self.report
                            .errors
                            .push(format!("user request {}: {error}", request.id));
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
                        self.report
                            .errors
                            .push(format!("query run {}: {error}", run.id));
                    }
                    let _ = reply.send(result);
                }
                UserMsg::QueryRunOutcome(outcome, reply) => {
                    let result = self.write_query_run_outcome(&outcome);
                    if let Err(error) = &result {
                        self.report
                            .errors
                            .push(format!("query run outcome {}: {error}", outcome.run));
                    }
                    let _ = reply.send(result);
                }
                UserMsg::History(history) => {
                    if let Err(error) = self
                        .check_user_parent(history.run.id.as_str(), None)
                        .and_then(|()| self.write_history(&history))
                    {
                        self.report
                            .errors
                            .push(format!("query history {}: {error}", history.run.id));
                    }
                }
            }
        }
    }

    /// The follower's page was handled: record a failed read once per
    /// distinct error (Sinex's health also shows it in status); after a
    /// successful one, prepare the events of captures Sinex confirmed and derive.
    fn stream_followed(&mut self, error: Option<String>) {
        let result = match error {
            Some(error) => Err(error),
            None => self
                .complete_ready_captures()
                .map(|()| {
                    self.request_derive(derive_thread::Waiting::Nobody);
                })
                .map_err(|error| error.to_string()),
        };
        match result {
            Ok(()) => self.stream_error = None,
            Err(error) => {
                let position = crate::primary::confirmed::checkpoint(&self.index)
                    .map(|position| position.unwrap_or(0).to_string())
                    .unwrap_or_else(|error| format!("unavailable ({error})"));
                let error = format!("confirmed stream: after position {position}: {error}");
                if self.stream_error.as_ref() != Some(&error) {
                    self.report.errors.push(error.clone());
                    self.stream_error = Some(error);
                }
            }
        }
    }

    fn run(&mut self, rx: &Receiver<Msg>) {
        let _worker = crate::timing::worker();
        let mut flushes = Vec::new();
        let mut derives = Vec::new();
        let mut derive_writes = Vec::new();
        let mut single_files = Vec::new();
        let mut file_groups = Vec::new();
        let mut assets = Vec::new();
        let mut coverage = Vec::new();
        let mut live_clock = Vec::new();
        let mut copies = Vec::new();
        let mut confirmations = Vec::new();
        let mut prepared_confirmations = Vec::new();
        let mut capture_drains = Vec::new();
        let mut followed = Vec::new();
        let mut snapshots = Vec::new();
        let mut pending_bodies = Vec::new();
        let mut private_captures = Vec::new();
        let mut byte_cursors = Vec::new();
        let mut group_pages = Vec::new();
        let mut restorations = Vec::new();
        let mut assertion_restorations = Vec::new();
        let mut replacements = Vec::new();
        let mut blobs = Vec::new();
        let mut older_groups = Vec::new();
        loop {
            let first = match rx.recv() {
                Ok(first) => first,
                Err(_) => break,
            };
            let _busy = crate::timing::writer_started();
            self.serve_users();
            let turn = Arc::clone(&self.search.turn);
            let _turn = turn.locked();
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
                    Msg::StageConfirmed(events, reply) => {
                        confirmations.push((events, reply));
                        false
                    }
                    Msg::ApplyConfirmed(prepared, kinds, reply) => {
                        prepared_confirmations.push((prepared, kinds, reply));
                        false
                    }
                    Msg::RestoreAuthored(restoration, reply) => {
                        restorations.push((restoration, reply));
                        false
                    }
                    Msg::RestoreAssertion(restoration, reply) => {
                        assertion_restorations.push((restoration, reply));
                        false
                    }
                    Msg::RestoreOlder(group, reply) => {
                        older_groups.push((group, reply));
                        false
                    }
                    Msg::ByteCursor(query, reply) => {
                        byte_cursors.push((query, reply));
                        false
                    }
                    Msg::PendingGroupHeaders(group, after, limit, reply) => {
                        group_pages.push((group, after, limit, reply));
                        false
                    }
                    Msg::Capture(identity, reply) => {
                        private_captures.push((identity, reply));
                        false
                    }
                    Msg::CaptureBody(identity, reply) => {
                        pending_bodies.push((identity, reply));
                        false
                    }
                    Msg::SourceSnapshot(source, reply) => {
                        snapshots.push((source, reply));
                        false
                    }
                    Msg::DrainPrimaryCaptures(reply) => {
                        capture_drains.push(reply);
                        false
                    }
                    Msg::StreamFollowed(error) => {
                        followed.push(error);
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
                    Msg::DeriveWrite(write, reply) => {
                        derive_writes.push((write, reply));
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
                let _ = reply.send(self.stage_confirmed(&events));
            }
            for (prepared, kinds, reply) in prepared_confirmations.drain(..) {
                let _ = reply.send(self.apply_confirmed(*prepared, kinds));
            }
            for reply in capture_drains.drain(..) {
                let _ = reply.send(self.drain_primary_captures());
            }
            for error in followed.drain(..) {
                self.stream_followed(error);
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
            for (blob, reply) in blobs.drain(..) {
                let (new_blob, file) = *blob;
                let _ = reply.send(self.store_blob(new_blob, file));
            }
            for (restoration, reply) in restorations.drain(..) {
                let _ = reply.send(self.restore_authored(*restoration));
            }
            for (restoration, reply) in assertion_restorations.drain(..) {
                let _ = reply.send(self.restore_assertion(*restoration));
            }
            for (group, reply) in older_groups.drain(..) {
                let _ = reply.send(self.restore_older(*group));
            }
            self.serve_users();

            for reply in derives.drain(..) {
                // A first build's indexes exist before anything is derived
                // from its rows.
                match self.finish_build() {
                    Ok(()) => self.request_derive(derive_thread::Waiting::Derive(reply)),
                    Err(e) => {
                        let _ = reply.send(Err(StoreError::Invalid(format!("derive: {e}"))));
                    }
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
            for (fact, reply) in coverage.drain(..) {
                let _ = reply.send(self.write_coverage(*fact));
            }
            for (sessions, reply) in live_clock.drain(..) {
                let _ = reply.send(self.write_live_clock(&sessions));
            }
            for (copy, reply) in copies.drain(..) {
                let _ = reply.send(self.write_copy_reading(&copy));
            }
            for (query, reply) in byte_cursors.drain(..) {
                let result = query.validate().and_then(|()| {
                    self.sinex
                        .as_ref()
                        .map_or(Ok(None), |sinex| sinex.byte_cursor(&query))
                });
                let _ = reply.send(result);
            }
            for (group, after, limit, reply) in group_pages.drain(..) {
                let result = if group.is_empty() || limit == 0 {
                    Err(StoreError::Invalid(
                        "group identity and positive page limit are required".into(),
                    ))
                } else {
                    self.sinex.as_ref().map_or_else(
                        || Ok(crate::primary::PendingGroupPage::default()),
                        |sinex| sinex.pending_group_headers(&group, after, limit),
                    )
                };
                let _ = reply.send(result);
            }
            for (identity, reply) in private_captures.drain(..) {
                let result = self
                    .sinex
                    .as_ref()
                    .ok_or_else(|| {
                        StoreError::Invalid(
                            "private source captures require a primary producer".into(),
                        )
                    })
                    .and_then(|sinex| sinex.capture(&identity));
                let _ = reply.send(result);
            }
            for (identity, reply) in pending_bodies.drain(..) {
                let result = self
                    .sinex
                    .as_ref()
                    .ok_or_else(|| {
                        StoreError::Invalid(
                            "pending source bodies require a primary producer".into(),
                        )
                    })
                    .and_then(|sinex| sinex.capture_body(&identity));
                let _ = reply.send(result);
            }
            for (source, reply) in snapshots.drain(..) {
                let _ = reply.send(self.source_snapshot(source));
            }
            if let Err(error) = self.refresh_chain_progress() {
                self.report.errors.push(format!("chain progress: {error}"));
            }
            if let Err(error) = record_build_progress(&self.location, &self.index) {
                self.report.errors.push(format!("build progress: {error}"));
            }
            if !flushes.is_empty()
                && self.location.is_primary()
                && let Err(error) = crate::archive::index_rebuild::remove_when_built(
                    &self.index,
                    &self.location.index_db(),
                )
            {
                self.report
                    .errors
                    .push(format!("primary cache rebuild: {error}"));
            }
            for (reply, derive) in flushes.drain(..) {
                let report = std::mem::take(&mut self.report);
                // Whatever a flush reports written is also derived, except in
                // a first build, which a flush in the middle of a pass (each
                // import item, each run of the parse pool) must not end.
                if derive && !self.resolver.upkeep().at_build_end() {
                    self.request_derive(derive_thread::Waiting::Flush(reply, Box::new(report)));
                } else {
                    let _ = reply.send(Ok(report));
                }
            }
            self.restart_full_logs();

            if stopping {
                break;
            }
        }
        // A derive pass waiting for a write this loop will not apply ends.
        for message in rx.try_iter() {
            drop(message);
        }
        self.derive.stop();
        let closed = self.refresh_chain_progress().and_then(|()| {
            if self.sinex.is_none() {
                crate::chain_progress::mark_current(&self.index)?;
            }
            Ok(())
        });
        if let Err(error) = closed {
            eprintln!("polylogue: writer: chain progress: {error}");
        }
        // A stop during a build keeps what it committed for the next start.
        if self.sinex.is_none()
            && self.location.index_build.building()
            && let Err(error) = crate::archive::index_build::keep_progress(&self.location)
        {
            eprintln!("polylogue: writer: keep the index build: {error}");
        }
    }

    /// Recomputes, in an index transaction of their own, the chain progress
    /// of inputs whose source rows a standalone archive transaction changed
    /// with no index transaction after it (bytes stored without parsed rows,
    /// a restored older archive, a write whose index transaction failed).
    fn refresh_chain_progress(&mut self) -> Result<()> {
        if self.sinex.is_some() {
            return Ok(());
        }
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
        let index =
            crate::archive::restart_full_log(&self.index, &self.location.index_db(), limit, wait);
        let archive = if self.sinex.is_none() && !self.stop.is_cancelled() {
            crate::archive::restart_full_log(
                &self.archive,
                &self.location.archive_db(),
                limit,
                wait,
            )
        } else {
            Ok(false)
        };
        // A failed checkpoint loses nothing: the log stays and is copied by
        // a later one.
        for result in [index, archive] {
            if let Err(error) = result {
                eprintln!("polylogue: writer: checkpoint: {error}");
            }
        }
    }

    /// Gives each named session a new live change counter. In primary mode
    /// the index follows the confirmed stream, which carries no clock
    /// transitions, so nothing is written there.
    fn write_live_clock(&mut self, sessions: &[String]) -> Result<()> {
        if self.sinex.is_some() || sessions.is_empty() {
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
        if self.sinex.is_some() {
            return Ok(Ok(false));
        }
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
        self.commits
            .committed(crate::completion::Committed::CONTENT);
        self.report.transactions += 1;
        Ok(())
    }

    /// Starts a derive pass on the derive thread, answering `waiting` once
    /// it is done; the writer goes on with other work meanwhile. Here the
    /// writer only marks what the pass must see: sessions whose derivation
    /// version moved, variants' text and the search entries waiting.
    fn request_derive(&mut self, waiting: derive_thread::Waiting) {
        let prepared = self
            .until_stopped(|inner| {
                if inner.stop.is_cancelled() {
                    return Err(StoreError::WriterGone);
                }
                inner.project_variant_text()?;
                if !inner.resolver.upkeep().at_build_end() {
                    crate::text_rows::index_pending(&inner.index, || ())?;
                }
                crate::derive::mark_moved_versions(&mut inner.index)
            })
            .and_then(|prepared| prepared.ok_or(StoreError::WriterGone));
        match (prepared, waiting) {
            (Ok(()), waiting) => self.derive.request(waiting),
            (Err(StoreError::WriterGone), derive_thread::Waiting::Derive(reply)) => {
                let _ = reply.send(Err(StoreError::WriterGone));
            }
            (Err(e), derive_thread::Waiting::Derive(reply)) => {
                let _ = reply.send(Err(StoreError::Invalid(format!("derive: {e}"))));
            }
            (Err(e), derive_thread::Waiting::Flush(reply, mut report)) => {
                report.errors.push(format!("derive: {e}"));
                let _ = reply.send(Ok(*report));
            }
            (Err(e), derive_thread::Waiting::Nobody) => {
                self.report.errors.push(format!("derive: {e}"));
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

    /// Ends a build once every deferred index has committed: the build file
    /// is synced and installed as `index.db` ([`crate::archive::index_build`]),
    /// and the writer continues on it. A stop or failure before the install
    /// leaves the build to resume at the next start.
    fn finish_build(&mut self) -> Result<()> {
        if !self.resolver.upkeep().at_build_end() || self.stop.is_cancelled() {
            return Ok(());
        }
        let finished = self.until_stopped(Inner::build_deferred);
        if !matches!(finished, Ok(Some(()))) {
            return finished.map(drop);
        }
        self.install_build()?;
        self.resolver
            .set_upkeep(resolve::IndexUpkeep::PerTransaction);
        self.search.ready();
        self.commits
            .committed(crate::completion::Committed::CONTENT);
        Ok(())
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
        crate::chain_progress::install_triggers(&self.index, false)?;
        installed
    }

    /// Builds search beside the file while the writer creates its secondary
    /// indexes. Unique keys are checked before the completed file is installed.
    fn build_deferred(&mut self) -> Result<()> {
        use crate::timing::{Stage, add};
        self.index.flush_prepared_statement_cache();
        let stop = self.stop.clone();
        let root = self.location.root.clone();
        crate::text_rows::build_search_indexes(&self.index, |index| {
            let started = std::time::Instant::now();
            index.execute(
                "INSERT INTO tool_call_trigram(tool_call_trigram) VALUES ('rebuild')",
                [],
            )?;
            add(Stage::TrigramIndex, started.elapsed());
            let started = std::time::Instant::now();
            for sql in crate::schema::built_at_end_sql()? {
                if stop.is_cancelled() {
                    return Err(StoreError::Stopped);
                }
                index.execute(&sql, [])?;
                crate::fault::check("index.deferred", &root.to_string_lossy())?;
            }
            // An empty first build is followed by live writes. ANALYZE would
            // record zero-cardinality partial indexes and make later pages
            // prefer a whole-table scan and sort over their ordering index.
            if index.exists(
                "SELECT EXISTS(SELECT 1 FROM session UNION ALL SELECT 1 FROM document)",
                [],
            )? {
                index.execute_batch("PRAGMA analysis_limit = 1000; ANALYZE main;")?;
            }
            add(Stage::SecondaryIndexes, started.elapsed());
            Ok(())
        })
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
            tx.execute("INSERT INTO segment(name,committed_end) VALUES (?1,?2) ON CONFLICT(name) DO UPDATE SET committed_end=excluded.committed_end",
                params!["segments/current.hb",synced.end as i64])?;
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
    ) -> Result<crate::user_writes::UserReceipt> {
        crate::fault::check("write.user", &request.id)?;
        scoped::require_current(&self.location, scope)?;
        let prepared_scope = self.preflight_saved(request, preparation, scope)?;
        let scope = scope.or(prepared_scope.as_ref());
        if self.sinex.is_some() {
            return self.primary_request(request, scope, preparation);
        }
        let tx = self.archive.transaction()?;
        scoped::require_current(&self.location, scope)?;
        let known: bool = tx.exists(
            "SELECT EXISTS(SELECT 1 FROM client_request WHERE id=?1)",
            [&request.id],
        )?;
        let seq = if known {
            tx.scalar_optional(
                "SELECT CAST(value AS INTEGER) FROM meta WHERE key='write_counter'",
                [],
            )?
            .unwrap_or(0)
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
                crate::user_writes::Preparation::plain(preparation, None),
            )?
        };
        scoped::require_current(&self.location, scope)?;
        if known {
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
        ) {
            self.project_variant_text()?;
        }
        Ok(outcome)
    }

    fn write<'a>(
        &mut self,
        batches: impl Iterator<Item = &'a Batch> + Clone,
        replay: bool,
    ) -> Result<()> {
        if replay && self.sinex.is_none() {
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
        let upkeep = self.resolver.upkeep();
        let primary = self.sinex.is_some();
        if primary {
            for batch in batches.clone() {
                self.validate_primary_reading(batch)?;
            }
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
        let mut parsed_only = true;
        for batch in batches.clone() {
            if !matches!(batch.change, Change::Reparse { .. })
                || batch.new_blob.is_some()
                || batch.observation.is_some()
            {
                parsed_only = false;
                break;
            }
            for blob in batch.blobs.iter().filter(|_| !primary) {
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
                let Change::Reparse { chain } = batch.change else {
                    unreachable!()
                };
                chains.push(Some(if primary {
                    chain
                } else {
                    stored_reading_chain(&self.archive, batch)?
                }));
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
        let text_transaction = self.texts.transaction();
        let itx = self
            .index
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        crate::fault::check("write.replay_rows", "")?;
        let counter = bump_counter(&itx)?;
        let changes = crate::change_summary::ChangeSummary::start(&itx)?;
        let mut stats = (0u64, 0u64);
        for (b, chain) in indexed.iter().zip(chains) {
            if let Some(chain) = chain {
                for p in &b.parse {
                    let resolution = p
                        .resolution
                        .as_ref()
                        .filter(|resolution| resolution.chain == chain)
                        .ok_or_else(|| {
                            StoreError::Invalid(
                                "a parsed step was not resolved for its input chain".into(),
                            )
                        })?;
                    let source_commit = crate::source::Source::reading_commit(
                        &itx,
                        &b.input.id,
                        chain,
                        p.parsed_upto,
                    )?;
                    stats.1 += write_rows(
                        &itx,
                        b,
                        p,
                        rows::ReadingWrite {
                            texts: &self.texts,
                            resolution,
                            counter,
                            upkeep: &upkeep,
                            source_commit: Some(&source_commit),
                            changes: &changes,
                            retraction: false,
                            replacement: false,
                        },
                    )?;
                }
                if b.parse.iter().any(|p| p.progress.reached_end()) {
                    rows::prune_sessions(&itx, &b.input.id, chain, counter)?;
                }
            }
            stats.0 += 1;
        }
        self.report.item_units_read += rows::finish(&itx, &changes, &upkeep, &self.texts, counter)?;
        let source_changed = if !primary {
            let changed = crate::chain_progress::recorded(&self.archive)?;
            crate::chain_progress::refresh(&itx, &changed)?;
            Some(changed)
        } else {
            None
        };
        crate::fault::check("write.index_commit", "")?;
        itx.commit()?;
        text_transaction.commit();
        if let Some(changed) = source_changed {
            crate::chain_progress::forget(&self.archive, &changed)?;
        }
        self.commits
            .committed(crate::completion::Committed::CONTENT);
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
                + p.output.prefix_copies.len()
                + p.output.prefix_members.len()
        })
        .sum::<usize>()
}

// ---------------------------------------------------------------- archive.db

// ------------------------------------------------------------------ index.db

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
