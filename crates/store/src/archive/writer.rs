//! The archive writer: one thread with the one connection that writes
//! `archive.db`'s record, apart from the blob rows each byte home records
//! when it puts bytes ([`crate::bytes`]). It commits groups (captured inputs
//! whose bytes are put, restored records) and user writes (authored records,
//! the request journal, deliveries and the other acts, query objects and
//! runs), each commit numbered by the archive's write counter, its
//! [`Position`].
//!
//! User writes have a queue of their own, served before the next bulk group:
//! a user write waits for at most the archive commit being written, never for
//! an index transaction or a rebuild step. A user write is applied once per
//! request ID; it is never withdrawn once accepted, and waiting for room in
//! the queue is the only wait a caller's deadline or cancellation ends.
//! Delivery records are accepted without waiting for room, and the ones
//! accepted and not yet committed can be read ([`ArchiveSender::pending_deliveries`]).
//!
//! The writer prepares authored writes against the index (link hashes,
//! quotes, the private copy a saved query's preflight resolves in) through a
//! read connection it opens for each turn of user writes; it never writes the
//! index. What follows from a commit in the index (chain progress, alternate
//! text) is the index writer's, which follows the archive by position
//! ([`crate::writer`]); the chains a commit changed are handed to it with the
//! commit.
use crate::archive::{Archive, bump_counter};
use crate::staging::StoredInput;
use crate::{Batch, OptionExt as _, Result, ResultExt as _, StoreError, WriteReport};
use polylogue_model::LockExt as _;
use polylogue_model::user::{UserOutcome, UserRequest, UserWrite};
use rusqlite::Connection;
use std::collections::BTreeMap;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

mod after;
#[cfg(test)]
pub(crate) use after::declared_rows as declared_after_rows;
pub mod agreement;
pub(crate) mod captured;
mod dispatch;
mod query_objects;
mod restore;
mod saved;
mod scoped;
mod user_queue;
pub use after::{ArchiveRow, Followed};
pub use dispatch::ParentLifetime;
#[cfg(any(test, feature = "fixture"))]
pub use dispatch::{DispatchPause, DispatchPhase};
pub use query_objects::HistorySlot;

/// `archive.db`'s write counter after a commit: every committed row carries
/// the position of its commit, so rows committed after a position are those
/// whose `commit_seq` is larger.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Default,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
#[serde(transparent)]
pub struct Position(pub u64);

/// A group of rows the writer commits in one transaction.
pub enum Group {
    /// The rows of captured inputs: input, revision, extents and
    /// observation. Their blobs are put already; `added` names the blobs those
    /// puts added, so each batch is told the bytes it stored. An atomic group
    /// (a package, one consistent database read) commits alone; other groups
    /// waiting together commit together.
    Captured {
        batches: Vec<Batch>,
        added: BTreeMap<polylogue_model::BlobHash, u64>,
        /// For each batch that starts a chain, the earlier chain of its input
        /// its bytes begin like, found by the caller before the commit
        /// ([`agreement`]); `None` records none.
        agreements: Vec<Option<agreement::Agreement>>,
        atomic: bool,
    },
    /// A request recorded by another archive, as its bundle states it.
    RestoredRequest(polylogue_model::authored::PreparedRequest),
    /// An annotation recorded by another archive, as its bundle states it.
    RestoredAnnotation {
        fact: polylogue_model::annotation::AnnotationFact,
        import: Option<polylogue_model::authored::ImportAttribution>,
    },
}

/// What a commit wrote, at its position.
#[derive(Debug)]
pub struct Committed<T> {
    pub position: Position,
    pub outcome: T,
}

/// What a [`Group`] wrote.
#[derive(Debug)]
pub enum Outcome {
    /// Where each captured batch's bytes are, in the order given.
    Stored(Vec<StoredInput>),
    Restored(UserOutcome),
}

/// A user request and what the writer checks before committing it.
pub struct UserSubmission {
    pub request: UserRequest,
    /// The pinned read that selected the request.
    pub scope: Option<crate::ReadScopeToken>,
    /// Trusted catalog context.
    pub preparation: Option<crate::user_writes::PreparationContext>,
    /// The caller's deadline and cancellation.
    pub parent: Option<Arc<ParentLifetime>>,
}

impl From<UserRequest> for UserSubmission {
    fn from(request: UserRequest) -> Self {
        Self {
            request,
            scope: None,
            preparation: None,
            parent: None,
        }
    }
}

/// A delivery accepted by the writer and not yet committed.
#[derive(Clone, Debug)]
pub struct PendingDelivery {
    pub request: UserRequest,
    /// The archive read counter of the snapshot used to prepare the brief.
    pub prepared_at: Option<i64>,
}

type Written = Result<Committed<crate::user_writes::WriteResult>>;

enum UserMsg {
    Request(Box<UserSubmission>, SyncSender<Written>),
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
pub(crate) const USER_QUEUE: usize = 64;

/// Bulk groups waiting before their senders block.
const BULK_QUEUE: usize = 256;

enum Bulk {
    /// A group, and where its outcome is answered; with nobody waiting, a
    /// failure is reported at the next flush.
    Commit(Box<Group>, Option<SyncSender<Result<Committed<Outcome>>>>),
    /// Answered once every group sent before it is committed.
    Flush(SyncSender<()>),
    /// A user write is waiting in its own queue.
    Wake,
    Stop,
}

/// Where [`ArchiveSender::try_submit`] left a request.
pub enum Queued {
    /// In the queue; the receiver gets its outcome once committed.
    Committing(Receiver<Written>),
    /// The queue was full: the complete submission, handed back.
    Full(Box<UserSubmission>),
}

struct Shared {
    bulk: Mutex<Option<SyncSender<Bulk>>>,
    /// A second sender into the bulk queue that only wakes the writer for a
    /// user write. Its lock is held only for one `try_send`, never while a
    /// bulk sender waits for room, so a user write always sends its wake.
    wake: Mutex<Option<SyncSender<Bulk>>>,
    users: Arc<user_queue::UserQueue>,
    location: Archive,
    /// Delivery records accepted and not yet committed, by request ID.
    pending: Mutex<Vec<PendingDelivery>>,
    report: Arc<Mutex<WriteReport>>,
    /// The position of the last commit, which the index writer follows.
    followed: Arc<Mutex<Position>>,
    commits: crate::completion::CommitEvents,
    #[cfg(any(test, feature = "fixture"))]
    dispatch: dispatch::DispatchPauses,
}

/// A clonable handle to the archive writer.
#[derive(Clone)]
pub struct ArchiveSender(Arc<Shared>);

/// The archive writer's thread. Dropping it stops the thread after the
/// writes already accepted are committed.
pub struct ArchiveWriter {
    sender: ArchiveSender,
    handle: Option<JoinHandle<()>>,
}

impl std::ops::Deref for ArchiveWriter {
    type Target = ArchiveSender;

    fn deref(&self) -> &ArchiveSender {
        &self.sender
    }
}

/// The archive writer thread's name.
pub(crate) const ARCHIVE_WRITER_THREAD: &str = "polylogue-archive";

impl ArchiveWriter {
    /// Starts the writer on `archive`, which the caller has locked. The
    /// archive and its log are synced first, so what an earlier process
    /// committed is on disk before anything is answered. `commits` is woken
    /// after each commit.
    pub fn start(archive: &Archive, commits: crate::completion::CommitEvents) -> Result<Self> {
        let conn = crate::archive::open_archive_writer(archive)?;
        crate::write_lock::sync_register(archive)?;
        let position = Position(
            u64::try_from(crate::write_counter(&conn)?).or_stored("negative write counter")?,
        );
        let (tx, rx) = sync_channel(BULK_QUEUE);
        let shared = Arc::new(Shared {
            wake: Mutex::new(Some(tx.clone())),
            bulk: Mutex::new(Some(tx)),
            users: Arc::default(),
            location: archive.clone(),
            pending: Mutex::default(),
            report: Arc::default(),
            followed: Arc::new(Mutex::new(position)),
            commits,
            #[cfg(any(test, feature = "fixture"))]
            dispatch: Default::default(),
        });
        let mut inner = Inner {
            conn,
            shared: Arc::clone(&shared),
            stop: archive.stop().clone(),
        };
        let handle = polylogue_model::spawn_named(ARCHIVE_WRITER_THREAD, move || inner.run(&rx))?;
        Ok(Self {
            sender: ArchiveSender(shared),
            handle: Some(handle),
        })
    }

    pub fn sender(&self) -> ArchiveSender {
        self.sender.clone()
    }

    /// Ends the thread after what it was handed; later calls fail with
    /// [`StoreError::WriterGone`].
    pub fn stop(&mut self) {
        self.sender.close();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ArchiveWriter {
    fn drop(&mut self) {
        self.stop();
    }
}

impl ArchiveSender {
    fn send(&self, message: Bulk) -> Result<()> {
        self.0
            .bulk
            .locked()
            .as_ref()
            .ok_or(StoreError::WriterGone)?
            .send(message)
            .map_err(|_| StoreError::WriterGone)
    }

    /// Wakes a writer that may be waiting for bulk work. A full bulk queue
    /// refuses the wake, and needs none: the writer has messages to take and
    /// serves user writes before each one.
    fn wake(&self) {
        if let Some(wake) = self.0.wake.locked().as_ref() {
            let _ = wake.try_send(Bulk::Wake);
        }
    }

    fn close(&self) {
        self.0.users.close();
        self.0.wake.locked().take();
        if let Some(sender) = self.0.bulk.locked().take() {
            let _ = sender.send(Bulk::Stop);
        }
    }

    fn send_user(&self, message: UserMsg) -> Result<()> {
        self.0
            .users
            .push_wait(message)
            .map_err(|_| StoreError::WriterGone)?;
        self.wake();
        Ok(())
    }

    /// Commits `group` and waits for its commit and sync. Groups waiting
    /// together commit together.
    pub fn commit(&self, group: Group) -> Result<Committed<Outcome>> {
        self.enqueue(group)?
            .recv()
            .map_err(|_| StoreError::WriterGone)?
    }

    /// Hands `group` over without waiting for its commit.
    pub fn enqueue(&self, group: Group) -> Result<Receiver<Result<Committed<Outcome>>>> {
        let (reply, wait) = sync_channel(1);
        self.send(Bulk::Commit(Box::new(group), Some(reply)))?;
        Ok(wait)
    }

    /// Hands `group` over with nobody waiting for it; a failure is reported
    /// at the next flush.
    pub fn hand_over(&self, group: Group) -> Result<()> {
        self.send(Bulk::Commit(Box::new(group), None))
    }

    /// Waits until every group handed over before is committed.
    pub fn flush(&self) -> Result<()> {
        let (reply, wait) = sync_channel(1);
        self.send(Bulk::Flush(reply))?;
        wait.recv().map_err(|_| StoreError::WriterGone)
    }

    /// Writes one user request once its request ID, and waits for its
    /// commit. Waiting for room in the queue ends at `limit`'s deadline or
    /// cancellation; once the request is in the queue it is committed, and
    /// this waits for that whatever the limit says. A request ID committed
    /// before answers with the first commit's outcome and writes nothing.
    pub fn submit(
        &self,
        submission: impl Into<UserSubmission>,
        limit: Option<&crate::read::limit::ReadLimit>,
    ) -> Written {
        let slot = self.0.users.reserve(limit)?;
        let (reply, wait) = sync_channel(1);
        slot.push(UserMsg::Request(Box::new(submission.into()), reply))?;
        self.wake();
        wait.recv().map_err(|_| StoreError::WriterGone)?
    }

    /// [`Self::submit`] with no limit, answering only the outcome.
    pub fn user(&self, submission: impl Into<UserSubmission>) -> Result<UserOutcome> {
        self.submit(submission, None)
            .map(|committed| committed.outcome.outcome)
    }

    /// Queues a user request without waiting for its commit or for room. A
    /// full queue hands the complete submission back as [`Queued::Full`].
    pub fn try_submit(&self, submission: impl Into<UserSubmission>) -> Result<Queued> {
        let (reply, wait) = sync_channel(1);
        match self
            .0
            .users
            .try_push(UserMsg::Request(Box::new(submission.into()), reply))
        {
            Ok(()) => {
                self.wake();
                Ok(Queued::Committing(wait))
            }
            Err(Some(UserMsg::Request(submission, _))) => Ok(Queued::Full(submission)),
            Err(_) => Err(StoreError::WriterGone),
        }
    }

    /// Accepts a delivery record (a delivery, advice already delivered, or a
    /// delivery's outcome) without waiting for room, behind the user writes
    /// already accepted. It is among [`Self::pending_deliveries`] until its
    /// commit, whose outcome the receiver gets.
    pub fn accept_delivery(
        &self,
        submission: impl Into<UserSubmission>,
        prepared_at: Option<i64>,
    ) -> Result<Receiver<Written>> {
        let submission = submission.into();
        if !is_delivery(&submission.request.write) {
            return Err(StoreError::Invalid(
                "only delivery records are accepted without a reservation".into(),
            ));
        }
        let (reply, wait) = sync_channel(1);
        let request = submission.request.clone();
        let mut pending = self.0.pending.locked();
        self.0
            .users
            .push_unbounded(UserMsg::Request(Box::new(submission), reply))
            .map_err(|_| StoreError::WriterGone)?;
        pending.push(PendingDelivery {
            request,
            prepared_at,
        });
        drop(pending);
        self.wake();
        Ok(wait)
    }

    /// The delivery records accepted and not yet committed, in the order
    /// they were accepted.
    pub fn pending_deliveries(&self) -> Vec<PendingDelivery> {
        self.0.pending.locked().clone()
    }

    /// The committed request with this ID and its recorded outcome, if any.
    pub fn committed_for(
        &self,
        request: &str,
    ) -> Result<Option<polylogue_model::authored::PreparedRequest>> {
        let conn = crate::archive::open_read(&self.0.location.archive_db())?;
        crate::authored_records::stored_request(&conn, request)
    }

    /// Takes what the writer and the puts reported since the last take.
    pub(crate) fn take_report(&self) -> WriteReport {
        std::mem::take(&mut *self.0.report.locked())
    }

    pub(crate) fn report(&self) -> &Arc<Mutex<WriteReport>> {
        &self.0.report
    }

    /// The position of the last commit, for the index writer to follow.
    pub(crate) fn committed_position(&self) -> Position {
        *self.0.followed.locked()
    }

    /// The pauses tests set on both writers' work.
    #[cfg(any(test, feature = "fixture"))]
    pub(crate) fn dispatch(&self) -> &dispatch::DispatchPauses {
        &self.0.dispatch
    }
}

fn is_delivery(write: &UserWrite) -> bool {
    matches!(
        write,
        UserWrite::Deliver(_) | UserWrite::DeliveredAdvice(_) | UserWrite::DeliveryEvent { .. }
    )
}

struct Inner {
    conn: Connection,
    shared: Arc<Shared>,
    stop: crate::cancellation::Cancellation,
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.shared.users.disconnect();
    }
}

type Waiting = (Box<Group>, Option<SyncSender<Result<Committed<Outcome>>>>);

impl Inner {
    fn run(&mut self, rx: &Receiver<Bulk>) {
        let _worker = crate::timing::worker();
        while let Ok(first) = rx.recv() {
            let _busy = crate::timing::start_stage(crate::timing::Stage::Writer);
            let _doing = crate::activity::doing("writing the archive");
            self.serve_users();
            let mut stopping = false;
            let mut groups = Vec::new();
            let mut flushes = Vec::new();
            for message in std::iter::once(first).chain(rx.try_iter()) {
                match message {
                    Bulk::Commit(group, reply) => groups.push((group, reply)),
                    Bulk::Flush(reply) => flushes.push(reply),
                    Bulk::Wake => {}
                    Bulk::Stop => stopping = true,
                }
            }
            self.commit_groups(groups);
            self.serve_users();
            for reply in flushes {
                let _ = reply.send(());
            }
            self.restart_full_log();
            if stopping {
                break;
            }
        }
        // Writes accepted before the stop commit.
        self.serve_users();
    }

    /// Commits the groups that were waiting together: independent captured
    /// groups share one transaction, retried one group at a time if it fails.
    fn commit_groups(&mut self, groups: Vec<Waiting>) {
        let (together, alone): (Vec<Waiting>, Vec<Waiting>) = groups
            .into_iter()
            .partition(|(group, _)| matches!(**group, Group::Captured { atomic: false, .. }));
        if together.len() > 1 {
            let mut batches = Vec::new();
            let mut added = BTreeMap::new();
            let mut agreements = Vec::new();
            let mut counts = Vec::new();
            for (group, _) in &together {
                let Group::Captured {
                    batches: these,
                    added: their,
                    agreements: theirs,
                    ..
                } = &**group
                else {
                    unreachable!()
                };
                counts.push(these.len());
                batches.extend(these.iter().cloned());
                agreements.extend(
                    theirs
                        .iter()
                        .copied()
                        .chain(std::iter::repeat(None))
                        .take(these.len()),
                );
                added.extend(their.iter().map(|(hash, size)| (*hash, *size)));
            }
            if let Ok(committed) = self.captured(batches, added, agreements) {
                let mut stored = committed.outcome.into_iter();
                for ((_, reply), count) in together.into_iter().zip(counts) {
                    let outcome = Outcome::Stored(stored.by_ref().take(count).collect());
                    if let Some(reply) = reply {
                        let _ = reply.send(Ok(Committed {
                            position: committed.position,
                            outcome,
                        }));
                    }
                }
                self.serve_users();
                return self.commit_alone(alone);
            }
        }
        self.commit_alone(together);
        self.commit_alone(alone);
    }

    fn commit_alone(&mut self, groups: Vec<Waiting>) {
        for (group, reply) in groups {
            let inputs = match &*group {
                Group::Captured { batches, .. } => batches
                    .iter()
                    .map(|b| b.input.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                Group::RestoredRequest(prepared) => prepared.request.id.clone(),
                Group::RestoredAnnotation { fact, .. } => fact.id.clone(),
            };
            let result = match *group {
                Group::Captured {
                    batches,
                    added,
                    agreements,
                    ..
                } => self
                    .captured(batches, added, agreements)
                    .map(|committed| Committed {
                        position: committed.position,
                        outcome: Outcome::Stored(committed.outcome),
                    }),
                Group::RestoredRequest(prepared) => {
                    self.restore_request(prepared).map(|committed| Committed {
                        position: committed.position,
                        outcome: Outcome::Restored(committed.outcome),
                    })
                }
                Group::RestoredAnnotation { fact, import } => self
                    .restore_annotation(fact, import)
                    .map(|committed| Committed {
                        position: committed.position,
                        outcome: Outcome::Restored(committed.outcome),
                    }),
            };
            match (reply, result) {
                (Some(reply), result) => {
                    let _ = reply.send(result);
                }
                (None, Err(error)) => self.failed(format_args!("input {inputs}"), &error),
                (None, Ok(_)) => {}
            }
            self.serve_users();
        }
    }

    /// The position as it stands, for a request that writes nothing.
    fn position(&self) -> Result<Position> {
        Ok(Position(
            u64::try_from(crate::write_counter(&self.conn)?).or_stored("negative write counter")?,
        ))
    }

    /// An immediate transaction on the writer's connection: the byte homes
    /// write blob rows on connections of their own, so a transaction that
    /// reads before it writes takes the write lock first.
    fn begin(&self) -> Result<rusqlite::Transaction<'_>> {
        Ok(rusqlite::Transaction::new_unchecked(
            &self.conn,
            rusqlite::TransactionBehavior::Immediate,
        )?)
    }

    /// Commits `tx`, whose write counter is `seq`, and publishes its
    /// position to the index writer before anything is answered.
    fn commit(
        &self,
        tx: rusqlite::Transaction<'_>,
        seq: i64,
        what: crate::completion::Committed,
    ) -> Result<Position> {
        crate::write_lock::commit(tx)?;
        let position = Position(u64::try_from(seq).or_stored("negative write counter")?);
        *self.shared.followed.locked() = position;
        self.shared.report.locked().transactions += 1;
        self.shared.commits.committed(what);
        Ok(position)
    }

    /// Writes every waiting user request, each answered once committed.
    fn serve_users(&mut self) {
        // New arrivals wake the next turn; this turn finishes its existing
        // user requests before bulk work, even under a continuous read load.
        let waiting = self.shared.users.waiting();
        if waiting == 0 {
            return;
        }
        // One read connection to the index for this turn's preparations; a
        // later turn opens the index then installed.
        let mut index = None;
        for _ in 0..waiting {
            let Some(message) = self.shared.users.pop() else {
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
                            if index.is_none() {
                                index = Some(self.index_reads()?);
                            }
                            let index = index.as_ref().expect("opened index reads");
                            self.write_user(index, &request, scope.as_ref(), preparation.as_ref())
                        });
                    if is_delivery(&request.write) {
                        let mut pending = self.shared.pending.locked();
                        if let Some(at) = pending
                            .iter()
                            .position(|known| known.request.id == request.id)
                        {
                            pending.remove(at);
                        }
                    }
                    // A queued request nobody waits for still reports its failure.
                    if let Err(std::sync::mpsc::SendError(Err(error))) = reply.send(result) {
                        self.shared
                            .report
                            .locked()
                            .errors
                            .push(polylogue_model::Failure::new(
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
                        self.failed(format_args!("query run {}", run.id), error);
                    }
                    let _ = reply.send(result);
                }
                UserMsg::QueryRunOutcome(outcome, reply) => {
                    let result = self.write_query_run_outcome(&outcome);
                    if let Err(error) = &result {
                        self.failed(format_args!("query run outcome {}", outcome.run), error);
                    }
                    let _ = reply.send(result);
                }
                UserMsg::History(history) => {
                    if let Err(error) = self
                        .check_user_parent(history.run.id.as_str(), None)
                        .and_then(|()| self.write_history(&history))
                    {
                        self.failed(format_args!("query history {}", history.run.id), &error);
                    }
                }
            }
        }
    }

    fn failed(&self, what: impl std::fmt::Display, error: &StoreError) {
        self.shared
            .report
            .locked()
            .errors
            .push(polylogue_model::Failure::new(what, error));
    }

    /// A read connection to the installed index, with the archive attached
    /// read-only under the source names, as the preparation of authored
    /// writes reads them. While a new archive's first index is built there
    /// is no installed one, and the build's file is read.
    fn index_reads(&self) -> Result<Connection> {
        use crate::archive::index_build::Reads;
        let location = &self.shared.location;
        let index = match location.served_index() {
            Ok(path) => location.open_index_connection(&path, Reads::Serving)?,
            Err(StoreError::IndexBuilding(_)) => {
                location.open_index_connection(&location.index_db(), Reads::Pipeline)?
            }
            Err(error) => return Err(error),
        };
        let uri = format!("file:{}?mode=ro", location.archive_db().display());
        index.execute("ATTACH DATABASE ?1 AS a", [uri])?;
        crate::source::views(&index)?;
        Ok(index)
    }

    fn write_user(
        &mut self,
        index: &Connection,
        request: &UserRequest,
        scope: Option<&crate::ReadScopeToken>,
        preparation: Option<&crate::user_writes::PreparationContext>,
    ) -> Written {
        let location = self.shared.location.clone();
        crate::fault::check("write.user", &request.id)?;
        scoped::require_current(&location, scope)?;
        let prepared_scope = self.preflight_saved(index, request, preparation, scope)?;
        let scope = scope.or(prepared_scope.as_ref());
        let tx = self.begin()?;
        scoped::require_current(&location, scope)?;
        let known = crate::ConnExt::exists(
            &*tx,
            "SELECT EXISTS(SELECT 1 FROM client_request WHERE id=?1)",
            [&request.id],
        )?;
        let seq = if known {
            crate::write_counter(&tx)?
        } else {
            bump_counter(&tx)?
        };
        let outcome = {
            let source_snapshot = rusqlite::Transaction::new_unchecked(
                index,
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
        scoped::require_current(&location, scope)?;
        if known || (matches!(request.write, UserWrite::MarkSeen { .. }) && outcome.outcome.repeat)
        {
            tx.rollback()?;
            return Ok(Committed {
                position: self.position()?,
                outcome,
            });
        }
        let position = self.commit(tx, seq, crate::completion::Committed::AUTHORED)?;
        Ok(Committed { position, outcome })
    }

    /// Between turns, starts over a write-ahead log grown past its
    /// checkpoint size ([`crate::archive::restart_full_log`]). A log still in
    /// use by a reader is tried again after the next turn.
    fn restart_full_log(&mut self) {
        if self.stop.is_cancelled() {
            return;
        }
        if let Err(error) = crate::archive::restart_full_log(
            &self.conn,
            "main",
            &self.shared.location.archive_db(),
            crate::archive::WAL_AUTOCHECKPOINT,
            std::time::Duration::from_millis(100),
        ) {
            // A failed checkpoint loses nothing: the log stays and is copied
            // by a later one.
            crate::log!(error, "archive writer: checkpoint: {error}");
        }
    }
}

#[cfg(test)]
mod tests;
