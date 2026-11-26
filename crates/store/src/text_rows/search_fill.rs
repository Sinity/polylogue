//! A first build's full-text files, filled while it composes
//! (docs/search.md: Index maintenance).
//!
//! The writer's build connection reports each text a block or document
//! block it inserts holds, and for which index, through a temporary trigger.
//! A coordinator thread keeps the first report of each, reads the texts on a
//! read connection to the build file, folds them, and hands them to one
//! thread per full-text file, which alone writes that file. Membership only
//! grows while a build composes, so the entries follow it closely; what a
//! removed block or an uncommitted text left out of step is settled once
//! composition ends and the writer has set every text's membership from the
//! blocks: the coordinator compares the membership with the entries it
//! wrote and adds and removes the difference. From then on the index's
//! triggers queue every change for upkeep, which applies a change the
//! settling already made as the same entry.

use super::{folded_text, groups, write_entry};
use crate::row::{Rows as _, row};
use crate::search::files::{FILES, file_of};
use crate::{Result, StoreError};
use polylogue_model::LockExt as _;
use rusqlite::{Connection, OpenFlags};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

row! {
    /// Where a text is: its group and its place in the group's bytes.
    struct TextPlace {
        group_id: i64,
        start: i64,
        length: i64,
    }
    statements {
        const OF_ID = "SELECT group_id, start, length FROM text WHERE id = ?1";
    }
}

row! {
    /// Which full-text indexes a text belongs to.
    struct TextMembership {
        id: i64,
        indexed: i64,
    }
    statements {
        const AFTER = "SELECT id, indexed FROM text WHERE id > ?1 ORDER BY id LIMIT ?2";
    }
}

/// Reports the writer's trigger gathers before handing them over.
const REPORTS: usize = 4096;
/// Texts read and folded together: their groups are each decoded once.
const FETCH_TEXTS: usize = 8192;
/// How long one read transaction on the build file runs at most. In the
/// build file's rollback journal a reader's lock holds off the writer's
/// commit, which waits only its busy timeout; a slow disk makes any count
/// of rows take arbitrarily long, so a read ends by time.
const READ_TIME: Duration = Duration::from_millis(100);
/// Groups read and decoded together.
const READ_GROUPS: usize = 64;
/// Text keys one settling transaction reads.
const SETTLE_ROWS: i64 = 8192;
/// Batches of entries waiting for each file's thread.
const FILE_QUEUE: usize = 8;
/// A file's thread commits after this many entries or this long.
const COMMIT_ENTRIES: usize = 25_000;
const COMMIT_TIME: Duration = Duration::from_secs(2);

const FOLLOW: &str = "
CREATE TEMP TRIGGER search_fill_block AFTER INSERT ON main.block
WHEN NEW.text_id IS NOT NULL BEGIN
  SELECT pl_search_fill(NEW.text_id, iif(NEW.kind IN ('tool_result','tool_call'), 2, 1));
END;
CREATE TEMP TRIGGER search_fill_document AFTER INSERT ON main.document_block
WHEN NEW.text_id IS NOT NULL BEGIN
  SELECT pl_search_fill(NEW.text_id, 1);
END;";

const UNFOLLOW: &str = "DROP TRIGGER IF EXISTS temp.search_fill_block;
DROP TRIGGER IF EXISTS temp.search_fill_document;";

/// What the coordinator is told.
enum Work {
    /// Texts blocks were inserted holding, with the index bit of each.
    Members(Vec<(i64, u8)>),
    /// Composition has ended and every text's membership is set: settle.
    Settle,
}

/// What a file's thread writes.
enum Entry {
    /// The text's entry in each index of `bits`, holding `folded`.
    Write { id: i64, bits: u8, folded: String },
    /// Its entry in the index `bit` leaves.
    Remove { id: i64, bit: u8 },
}

/// The bytes a batch of entries holds.
fn batch_bytes(batch: &[Entry]) -> usize {
    batch
        .iter()
        .map(|entry| match entry {
            Entry::Write { folded, .. } => folded.len() + 32,
            Entry::Remove { .. } => 32,
        })
        .sum()
}

/// What the fill holds of texts at once, at most: the texts read and
/// folded ahead of their files' threads, split between those waiting for a
/// thread and the next ones being read (a text longer than that is held
/// alone).
#[derive(Clone)]
struct Held {
    queued: Arc<AtomicUsize>,
    memory: usize,
}

/// The reports of the writer's trigger, gathered.
struct Feed {
    reports: Vec<(i64, u8)>,
    work: Option<Sender<Work>>,
}

impl Feed {
    fn hand_over(&mut self) {
        if let Some(work) = &self.work
            && !self.reports.is_empty()
        {
            // A coordinator that has stopped reports its error at `wait`.
            let _ = work.send(Work::Members(std::mem::take(&mut self.reports)));
        }
    }
}

/// The fill of a build's full-text files, from the build's start until its
/// entries are complete.
pub(crate) struct SearchFill {
    feed: Arc<Mutex<Feed>>,
    work: Option<Sender<Work>>,
    done: Receiver<Result<()>>,
    stop: Arc<AtomicBool>,
    coordinator: Option<JoinHandle<()>>,
}

impl SearchFill {
    /// Starts filling the full-text files of the build file `index` (the
    /// writer's connection to it), following the blocks it inserts, holding
    /// at most about `memory` bytes of texts at once.
    pub(crate) fn start(index: &Connection, memory: usize) -> Result<Self> {
        let build = index
            .path()
            .filter(|path| !path.is_empty())
            .map(std::path::PathBuf::from)
            .ok_or_else(|| StoreError::Invalid("a build file has no path".into()))?;
        let paths = crate::search::files::paths(index)?
            .ok_or_else(|| StoreError::Invalid("a build file names no full-text files".into()))?;
        let stop = Arc::new(AtomicBool::new(false));
        let mut entered = [Keys::default(), Keys::default()];
        let mut files = Vec::with_capacity(FILES);
        for path in &paths {
            let file = Connection::open_with_flags(
                path,
                OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
            )?;
            // Nothing reads a build's files until it ends, and a build a
            // crash interrupts is discarded with them.
            file.pragma_update(None, "journal_mode", "memory")?;
            file.pragma_update(None, "synchronous", "off")?;
            file.pragma_update(None, "cache_size", -16_384)?;
            file.busy_timeout(crate::archive::BUSY_TIMEOUT)?;
            // A build started again in this process keeps what an earlier
            // start entered.
            for (bit, table) in [(0, "text_fts"), (1, "output_fts")] {
                let ids: Vec<i64> =
                    file.values(&format!("SELECT id FROM {table}_docsize WHERE id > 0"), [])?;
                for id in ids {
                    entered[bit].insert(id);
                }
            }
            files.push(file);
        }
        // Read-write only so SQLite may recover the build file's journal,
        // as the build's other readers may; it writes nothing.
        let reader = Connection::open_with_flags(
            &build,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        reader.pragma_update(None, "query_only", true)?;
        crate::activity::trace_statements(&reader, crate::activity::Connection::Reader);
        // The writer's transactions, an index creation among them, hold the
        // build file for as long as they run.
        reader.busy_timeout(Duration::from_secs(3600))?;
        let (work, work_rx) = std::sync::mpsc::channel();
        let (done_tx, done) = std::sync::mpsc::sync_channel(1);
        let coordinator = {
            let stop = Arc::clone(&stop);
            polylogue_model::spawn_named("polylogue-search-fill", move || {
                let held = Held {
                    queued: Arc::new(AtomicUsize::new(0)),
                    memory,
                };
                let _ = done_tx.send(coordinate(reader, files, entered, &work_rx, &stop, held));
            })?
        };
        let feed = Arc::new(Mutex::new(Feed {
            reports: Vec::with_capacity(REPORTS),
            work: Some(work.clone()),
        }));
        let reports = Arc::clone(&feed);
        index.create_scalar_function(
            "pl_search_fill",
            2,
            rusqlite::functions::FunctionFlags::SQLITE_UTF8,
            move |ctx| {
                let id: i64 = ctx.get(0)?;
                let bit: u8 = ctx.get(1)?;
                let mut feed = reports.locked();
                // A fill that ended keeps no reports.
                if feed.work.is_none() {
                    return Ok(None::<i64>);
                }
                feed.reports.push((id, bit));
                if feed.reports.len() >= REPORTS {
                    feed.hand_over();
                }
                Ok(None::<i64>)
            },
        )?;
        // An earlier start on this connection that ended without settling
        // left its triggers.
        index.execute_batch(UNFOLLOW)?;
        index.execute_batch(FOLLOW)?;
        Ok(Self {
            feed,
            work: Some(work),
            done,
            stop,
            coordinator: Some(coordinator),
        })
    }

    /// Stops following the writer and settles the entries against every
    /// text's membership, which the writer has committed. The fill then
    /// ends on its own ([`SearchFill::wait`]).
    pub(crate) fn settle(&mut self, index: &Connection) -> Result<()> {
        index.execute_batch(UNFOLLOW)?;
        {
            let mut feed = self.feed.locked();
            feed.hand_over();
            feed.work = None;
        }
        if let Some(work) = self.work.take() {
            let _ = work.send(Work::Settle);
        }
        Ok(())
    }

    /// Waits up to `limit` for the fill to end: true once every entry is
    /// written and the files are closed.
    pub(crate) fn wait(&mut self, limit: Duration) -> Result<bool> {
        let result = match self.done.recv_timeout(limit) {
            Ok(result) => result,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => return Ok(false),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err(StoreError::internal(
                "the full-text fill ended without an answer",
            )),
        };
        if let Some(coordinator) = self.coordinator.take() {
            let _ = coordinator.join();
        }
        result.map(|()| true)
    }
}

impl Drop for SearchFill {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.feed.locked().work = None;
        self.work = None;
        if let Some(coordinator) = self.coordinator.take() {
            let _ = coordinator.join();
        }
    }
}

/// Keys as bits, one per text key.
#[derive(Default)]
struct Keys(Vec<u64>);

impl Keys {
    /// Adds `id`; false when it was there.
    fn insert(&mut self, id: i64) -> bool {
        let Ok(id) = usize::try_from(id) else {
            return false;
        };
        let (word, bit) = (id / 64, 1u64 << (id % 64));
        if self.0.len() <= word {
            self.0.resize(word + 1, 0);
        }
        let added = self.0[word] & bit == 0;
        self.0[word] |= bit;
        added
    }

    fn contains(&self, id: i64) -> bool {
        usize::try_from(id).is_ok_and(|id| {
            self.0
                .get(id / 64)
                .is_some_and(|word| word & (1u64 << (id % 64)) != 0)
        })
    }

    fn remove(&mut self, id: i64) {
        if let Ok(id) = usize::try_from(id)
            && let Some(word) = self.0.get_mut(id / 64)
        {
            *word &= !(1u64 << (id % 64));
        }
    }

    /// The keys in `self` and not in `other`.
    fn without<'a>(&'a self, other: &'a Keys) -> impl Iterator<Item = i64> + 'a {
        self.0.iter().enumerate().flat_map(move |(at, &word)| {
            let mut left = word & !other.0.get(at).copied().unwrap_or(0);
            std::iter::from_fn(move || {
                (left != 0).then(|| {
                    let bit = left.trailing_zeros() as usize;
                    left &= left - 1;
                    (at * 64 + bit) as i64
                })
            })
        })
    }
}

/// The coordinator: the files' threads run under it, and it answers with
/// the first error among them and itself.
fn coordinate(
    reader: Connection,
    files: Vec<Connection>,
    entered: [Keys; 2],
    work: &Receiver<Work>,
    stop: &AtomicBool,
    held: Held,
) -> Result<()> {
    let mut senders = Vec::with_capacity(files.len());
    let mut threads = Vec::with_capacity(files.len());
    for (n, file) in files.into_iter().enumerate() {
        let (sender, batches) = std::sync::mpsc::sync_channel::<Vec<Entry>>(FILE_QUEUE);
        senders.push(sender);
        threads.push(polylogue_model::spawn_named(
            format!("polylogue-search-file-{n}"),
            {
                let queued = Arc::clone(&held.queued);
                move || fill_file(file, &batches, &queued)
            },
        )?);
    }
    let mut coordinator = Coordinator {
        held,
        reader,
        entered,
        files: senders,
        wanted: Vec::new(),
        settle: false,
    };
    let result = coordinator.run(work, stop);
    // The files' threads commit and end once their queues close.
    drop(coordinator);
    let mut outcome = result;
    for thread in threads {
        let ended = thread
            .join()
            .unwrap_or_else(|_| Err(StoreError::internal("a full-text file's thread panicked")));
        if outcome.is_ok() {
            outcome = ended;
        }
    }
    outcome
}

struct Coordinator {
    held: Held,
    reader: Connection,
    /// The (text, index) entries handed to the files' threads.
    entered: [Keys; 2],
    files: Vec<SyncSender<Vec<Entry>>>,
    /// Entries newly reported, still to read.
    wanted: Vec<(i64, u8)>,
    /// The writer asked to settle.
    settle: bool,
}

impl Coordinator {
    fn run(&mut self, work: &Receiver<Work>, stop: &AtomicBool) -> Result<()> {
        loop {
            if stop.load(Ordering::Acquire) {
                return Ok(());
            }
            if self.settle {
                return self.settle_entries(work);
            }
            let next = if self.wanted.is_empty() {
                match work.recv() {
                    Ok(next) => next,
                    // The build ended without settling: its files are not used.
                    Err(_) => return Ok(()),
                }
            } else {
                match work.try_recv() {
                    Ok(next) => next,
                    Err(TryRecvError::Empty) => {
                        self.fetch(work)?;
                        continue;
                    }
                    Err(TryRecvError::Disconnected) => return Ok(()),
                }
            };
            self.take(next);
            if self.wanted.len() >= FETCH_TEXTS {
                self.fetch(work)?;
            }
        }
    }

    fn take(&mut self, work: Work) {
        match work {
            Work::Members(reports) => {
                for (id, bit) in reports {
                    if (bit == 1 || bit == 2) && self.entered[usize::from(bit - 1)].insert(id) {
                        self.wanted.push((id, bit));
                    }
                }
            }
            Work::Settle => self.settle = true,
        }
    }

    /// Hands `batch` to file `n`'s thread. While it is full, the writer's
    /// reports are taken in, so they never wait on a file.
    fn send(&mut self, n: usize, mut batch: Vec<Entry>, work: &Receiver<Work>) -> Result<()> {
        // Half of what the fill may hold waits for the files' threads.
        let bytes = batch_bytes(&batch);
        loop {
            let queued = self.held.queued.load(Ordering::Acquire);
            if queued == 0 || queued + bytes <= self.held.memory / 2 {
                break;
            }
            while let Ok(next) = work.try_recv() {
                self.take(next);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        self.held.queued.fetch_add(bytes, Ordering::AcqRel);
        loop {
            match self.files[n].try_send(batch) {
                Ok(()) => return Ok(()),
                Err(TrySendError::Full(back)) => {
                    batch = back;
                    while let Ok(next) = work.try_recv() {
                        self.take(next);
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(TrySendError::Disconnected(_)) => {
                    return Err(StoreError::internal("a full-text file's thread stopped"));
                }
            }
        }
    }

    /// Reads, folds and hands over every wanted entry.
    fn fetch(&mut self, work: &Receiver<Work>) -> Result<()> {
        let _doing = crate::activity::doing("reading texts for the search index");
        while !self.wanted.is_empty() {
            crate::activity::progressed();
            let started = Instant::now();
            let take = self.wanted.len().min(FETCH_TEXTS);
            let mut wanted: Vec<(i64, u8)> = self.wanted.drain(..take).collect();
            wanted.sort_unstable();
            // Each text once, with the indexes it enters.
            let mut texts: Vec<(i64, u8)> = Vec::with_capacity(wanted.len());
            for (id, bit) in wanted {
                match texts.last_mut() {
                    Some((last, bits)) if *last == id => *bits |= bit,
                    _ => texts.push((id, bit)),
                }
            }
            // Where each text is: (group, start, length, id, bits).
            let mut places = Vec::with_capacity(texts.len());
            let mut next = 0;
            while next < texts.len() {
                let read = Instant::now();
                let tx = self.reader.unchecked_transaction()?;
                while next < texts.len() && read.elapsed() < READ_TIME {
                    let (id, bits) = texts[next];
                    next += 1;
                    match tx.maybe(&TextPlace::OF_ID, [id])? {
                        Some(TextPlace {
                            group_id: group,
                            start,
                            length,
                        }) => {
                            places.push((group, start, length, id, bits));
                        }
                        // Not committed yet, or removed: settling decides.
                        None => {
                            for bit in [1u8, 2] {
                                if bits & bit != 0 {
                                    self.entered[usize::from(bit - 1)].remove(id);
                                }
                            }
                        }
                    }
                }
                tx.finish()?;
            }
            places.sort_unstable();
            let mut batches: Vec<Vec<Entry>> = (0..FILES).map(|_| Vec::new()).collect();
            let mut folded_bytes = 0;
            let mut at = 0;
            while at < places.len() {
                // The next groups, each read once.
                let mut groups: Vec<i64> = Vec::new();
                let mut end = at;
                while end < places.len() && groups.len() < READ_GROUPS {
                    if groups.last() != Some(&places[end].0) {
                        groups.push(places[end].0);
                    }
                    end += 1;
                }
                while end < places.len() && groups.last() == Some(&places[end].0) {
                    end += 1;
                }
                let mut bodies = Vec::with_capacity(groups.len());
                while bodies.len() < groups.len() {
                    let read = Instant::now();
                    let tx = self.reader.unchecked_transaction()?;
                    while bodies.len() < groups.len() && read.elapsed() < READ_TIME {
                        let bytes: Vec<u8> = tx
                            .value(
                                "SELECT body FROM content_group WHERE id = ?1",
                                [groups[bodies.len()]],
                            )?
                            .ok_or_else(|| StoreError::stored(0, "a text row has no group"))?;
                        bodies.push(bytes);
                    }
                    tx.finish()?;
                }
                let mut decoded: Option<(i64, Vec<u8>)> = None;
                for &(group, start, length, id, bits) in &places[at..end] {
                    if decoded.as_ref().is_none_or(|(held, _)| *held != group) {
                        let index = groups
                            .iter()
                            .position(|&g| g == group)
                            .expect("the group was read");
                        decoded = Some((group, groups::decode(&bodies[index])?));
                    }
                    let (_, body) = decoded.as_ref().expect("the group is decoded");
                    let (folded, _) = folded_text(groups::slice_decoded(body, start, length)?);
                    folded_bytes += folded.len();
                    batches[file_of(id)].push(Entry::Write { id, bits, folded });
                }
                at = end;
                // The other half of what the fill may hold is texts being
                // read: past it, they go to their files' threads.
                if folded_bytes >= self.held.memory / 2 {
                    for (n, batch) in batches.iter_mut().enumerate() {
                        if !batch.is_empty() {
                            self.send(n, std::mem::take(batch), work)?;
                        }
                    }
                    folded_bytes = 0;
                }
            }
            crate::timing::add(crate::timing::Stage::SearchIndex, started.elapsed());
            for (n, batch) in batches.into_iter().enumerate() {
                if !batch.is_empty() {
                    self.send(n, batch, work)?;
                }
            }
        }
        Ok(())
    }

    /// Brings the entries to every text's committed membership.
    fn settle_entries(&mut self, work: &Receiver<Work>) -> Result<()> {
        let _doing = crate::activity::doing("settling the search index");
        let mut members = [Keys::default(), Keys::default()];
        let mut after = 0i64;
        // Read from the writer's commit of the membership on, in short
        // transactions beside the writer's next work: a change the writer
        // makes between them is queued for upkeep by the index's triggers,
        // which apply it as the same entry.
        loop {
            let tx = self.reader.unchecked_transaction()?;
            let rows = tx.all(&TextMembership::AFTER, [after, SETTLE_ROWS])?;
            tx.finish()?;
            for &TextMembership { id, indexed } in &rows {
                for (bit, keys) in members.iter_mut().enumerate() {
                    if indexed & (1 << bit) != 0 {
                        keys.insert(id);
                    }
                }
            }
            match rows.last() {
                Some(last) if rows.len() as i64 == SETTLE_ROWS => after = last.id,
                _ => break,
            }
        }
        let mut removed: Vec<Vec<Entry>> = (0..FILES).map(|_| Vec::new()).collect();
        for (bit, members) in members.iter().enumerate() {
            let gone: Vec<i64> = self.entered[bit].without(members).collect();
            for id in gone {
                self.entered[bit].remove(id);
                removed[file_of(id)].push(Entry::Remove {
                    id,
                    bit: bit as u8 + 1,
                });
            }
            let missing: Vec<i64> = members.without(&self.entered[bit]).collect();
            for id in missing {
                self.entered[bit].insert(id);
                self.wanted.push((id, bit as u8 + 1));
            }
        }
        // A removed entry still waiting to be read is not written.
        let entered = &self.entered;
        self.wanted
            .retain(|&(id, bit)| entered[usize::from(bit - 1)].contains(id));
        for (n, batch) in removed.into_iter().enumerate() {
            if !batch.is_empty() {
                self.send(n, batch, work)?;
            }
        }
        // A text removed since its membership was read is queued for
        // upkeep by its removal.
        self.fetch(work)
    }
}

/// A full-text file's thread: writes each batch it is handed, committing
/// every [`COMMIT_ENTRIES`] entries or [`COMMIT_TIME`].
fn fill_file(file: Connection, batches: &Receiver<Vec<Entry>>, queued: &AtomicUsize) -> Result<()> {
    file.execute_batch("BEGIN")?;
    let mut written = 0usize;
    let mut since = Instant::now();
    while let Ok(batch) = batches.recv() {
        let _doing = crate::activity::doing("writing search entries");
        let bytes = batch_bytes(&batch);
        let started = Instant::now();
        for entry in &batch {
            match entry {
                Entry::Write { id, bits, folded } => {
                    for (bit, table) in [(1u8, "text_fts"), (2, "output_fts")] {
                        if bits & bit != 0 {
                            write_entry(&file, "main", table, *id, Some(folded))?;
                        }
                    }
                }
                Entry::Remove { id, bit } => {
                    let table = if *bit == 1 { "text_fts" } else { "output_fts" };
                    write_entry(&file, "main", table, *id, None)?;
                }
            }
        }
        written += batch.len();
        drop(batch);
        queued.fetch_sub(bytes, Ordering::AcqRel);
        if written >= COMMIT_ENTRIES || since.elapsed() >= COMMIT_TIME {
            file.execute_batch("COMMIT; BEGIN")?;
            written = 0;
            since = Instant::now();
        }
        crate::timing::add(crate::timing::Stage::SearchIndex, started.elapsed());
    }
    file.execute_batch("COMMIT")?;
    Ok(())
}

/// The rows this module declares statements for, for the every-statement test.
#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<TextPlace>();
    check.row::<TextMembership>();
}
