//! Tool-database reads run independently of file discovery and parsing.
//! Only the writer's bounded byte queue applies backpressure to a reader.
use crate::*;
use polylogue_store::tool_databases::{DatabaseError, DatabaseReader};
use polylogue_store::{Batch, staging::PreparedBatch};
use std::collections::VecDeque;
use std::sync::mpsc::{TryRecvError, sync_channel};
use std::thread::JoinHandle;

pub(super) struct DatabaseInput {
    pub path: PathBuf,
    pub input: InputKey,
    pub observation: Observation,
}
#[derive(Default)]
pub(super) struct Completed {
    pub finished: bool,
    pub errors: Vec<String>,
    pub changed: bool,
    pub inputs_seen: u64,
    pub bytes_stored: u64,
}
struct Worker {
    thread: JoinHandle<()>,
    result: Receiver<(DatabaseReader, Completed)>,
    progress: Receiver<()>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    bytes: Arc<std::sync::atomic::AtomicU64>,
}
#[derive(Default)]
pub(super) struct Databases {
    reader: Option<DatabaseReader>,
    worker: Option<Worker>,
}
impl Databases {
    pub fn pending(&self) -> bool {
        self.worker.is_some()
    }
    pub fn start(
        &mut self,
        inputs: Vec<DatabaseInput>,
        archive: Archive,
        writer: polylogue_store::WriterSender,
        wake: Option<Sender<LoopMsg>>,
    ) -> Result<(), EngineError> {
        if self.pending() || inputs.is_empty() {
            return Ok(());
        }
        let mut reader = self.reader.take().unwrap_or_default();
        let (tx, result) = sync_channel(1);
        let (notify, progress) = sync_channel(1);
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        reader.stop_when(Arc::clone(&stop));
        let cancelled = Arc::clone(&stop);
        let bytes = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let captured = Arc::clone(&bytes);
        let thread = std::thread::Builder::new()
            .name("polylogue-databases".into())
            .spawn(move || {
                let _worker = polylogue_store::timing::worker();
                let mut completed = Completed::default();
                let mut pending = Vec::new();
                for source in &inputs {
                    if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    let Some(due) = reader.observe(&source.path) else {
                        continue;
                    };
                    completed.inputs_seen += 1;
                    let mut queue = match InputWriter::new(
                        &archive,
                        &writer,
                        source,
                        notify.clone(),
                        wake.clone(),
                        Arc::clone(&captured),
                    ) {
                        Ok(queue) => queue,
                        Err(error) => {
                            completed
                                .errors
                                .push(format!("read database {}: {error}", source.path.display()));
                            continue;
                        }
                    };
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        reader.read(
                            &source.path,
                            &archive,
                            &source.input,
                            &source.observation,
                            due,
                            |bytes| queue.append(bytes),
                        )
                    }));
                    match outcome {
                        Ok(Ok(_)) => completed.changed = true,
                        Ok(Err(error)) => {
                            reader.forget(&source.path);
                            completed.changed = true;
                            completed
                                .errors
                                .push(format!("read database {}: {error}", source.path.display()));
                        }
                        Err(_) => {
                            completed.changed = true;
                            completed.errors.push(format!(
                                "read database {}: worker panicked",
                                source.path.display()
                            ));
                            reader = DatabaseReader::default();
                            reader.stop_when(Arc::clone(&cancelled));
                        }
                    }
                    // Another source can be read while these groups commit.
                    pending.push((source, queue));
                }
                for (source, mut queue) in pending {
                    if let Err(error) = queue.finish() {
                        reader.forget(&source.path);
                        completed.changed = true;
                        completed
                            .errors
                            .push(format!("write database {}: {error}", source.path.display()));
                    }
                }
                let _ = tx.send((reader, completed));
                if let Some(wake) = wake {
                    let _ = wake.send(LoopMsg::Changed);
                }
            })?;
        self.worker = Some(Worker {
            thread,
            result,
            progress,
            stop,
            bytes,
        });
        Ok(())
    }
    pub fn completed(&mut self, wait: bool) -> Completed {
        let Some(worker) = &self.worker else {
            return Completed::default();
        };
        let changed = worker.progress.try_recv().is_ok();
        let result = if wait {
            worker.result.recv().map_err(|_| TryRecvError::Disconnected)
        } else {
            worker.result.try_recv()
        };
        let bytes_stored = worker.bytes.swap(0, std::sync::atomic::Ordering::Relaxed);
        match result {
            Ok((reader, completed)) => {
                let worker = self.worker.take().expect("active worker");
                let _ = worker.thread.join();
                self.reader = Some(reader);
                Completed {
                    finished: true,
                    changed: changed || completed.changed,
                    bytes_stored,
                    ..completed
                }
            }
            Err(TryRecvError::Empty) => Completed {
                changed,
                bytes_stored,
                ..Completed::default()
            },
            Err(TryRecvError::Disconnected) => {
                let worker = self.worker.take().expect("active worker");
                let _ = worker.thread.join();
                Completed {
                    finished: true,
                    errors: vec!["database worker stopped".into()],
                    changed: true,
                    bytes_stored,
                    ..Completed::default()
                }
            }
        }
    }
}
impl Drop for Databases {
    fn drop(&mut self) {
        if let Some(worker) = &self.worker {
            worker
                .stop
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        self.completed(true);
    }
}

type Acknowledgment = Receiver<Vec<polylogue_store::Result<polylogue_store::staging::StoredInput>>>;
struct InputWriter<'a> {
    archive: &'a Archive,
    writer: &'a polylogue_store::WriterSender,
    source: &'a DatabaseInput,
    blobs: Arc<hashed_blobs::Store>,
    length: u64,
    exists: bool,
    waiting: VecDeque<(u64, Acknowledgment)>,
    bytes: u64,
    notify: SyncSender<()>,
    wake: Option<Sender<LoopMsg>>,
    captured: Arc<std::sync::atomic::AtomicU64>,
}
impl<'a> InputWriter<'a> {
    fn new(
        archive: &'a Archive,
        writer: &'a polylogue_store::WriterSender,
        source: &'a DatabaseInput,
        notify: SyncSender<()>,
        wake: Option<Sender<LoopMsg>>,
        captured: Arc<std::sync::atomic::AtomicU64>,
    ) -> Result<Self, DatabaseError> {
        let reads = ReadPool::open(archive)?;
        let reader = reads.reader()?;
        let progress = reader.input_progress(&source.input.id)?;
        let out = Self {
            archive,
            writer,
            source,
            blobs: Arc::new(archive.blobs()?),
            length: progress.as_ref().map_or(0, |p| p.length),
            exists: progress.is_some(),
            waiting: VecDeque::new(),
            bytes: 0,
            notify,
            wake,
            captured,
        };
        Ok(out)
    }
    fn append(&mut self, bytes: Vec<u8>) -> Result<(), DatabaseError> {
        let size = bytes.len() as u64;
        self.length += size;
        let (new_blob, files) = if bytes.is_empty() {
            (None, Vec::new())
        } else {
            let (blob, file) =
                polylogue_store::timing::time(polylogue_store::timing::Stage::Store, || {
                    self.archive.prepare_blob(&bytes)
                })?;
            (Some(blob), file.into_iter().collect())
        };
        let batch = Batch {
            input: self.source.input.clone(),
            change: if !self.exists {
                Change::First
            } else if size > 0 {
                Change::Appended {
                    from: self.length - size,
                }
            } else {
                Change::Unchanged
            },
            observation: Some(self.source.observation.clone()),
            continues: false,
            new_blob,
            blobs: vec![],
            length: self.length,
            parse: vec![],
            release: Default::default(),
        };
        self.exists = true;
        let ack = self.writer.enqueue_group(
            vec![PreparedBatch { batch, files }],
            Arc::clone(&self.blobs),
        )?;
        self.waiting.push_back((size, ack));
        self.bytes += size;
        // Groups already queued may commit while the next group is read.
        while self.bytes >= 128 << 20 {
            self.acknowledge()?;
        }
        self.acknowledge_ready()?;
        Ok(())
    }
    fn accepted(
        &mut self,
        size: u64,
        results: Vec<polylogue_store::Result<polylogue_store::staging::StoredInput>>,
    ) -> Result<(), DatabaseError> {
        self.bytes -= size;
        if self.notify.try_send(()).is_ok()
            && let Some(wake) = &self.wake
        {
            let _ = wake.send(LoopMsg::Changed);
        }
        for result in results {
            result?;
        }
        self.captured
            .fetch_add(size, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
    fn acknowledge_ready(&mut self) -> Result<(), DatabaseError> {
        while let Some((_, ack)) = self.waiting.front() {
            match ack.try_recv() {
                Ok(results) => {
                    let (size, _) = self.waiting.pop_front().expect("waiting group");
                    self.accepted(size, results)?;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    return Err(polylogue_store::StoreError::WriterGone.into());
                }
            }
        }
        Ok(())
    }
    fn acknowledge(&mut self) -> Result<(), DatabaseError> {
        if let Some((size, ack)) = self.waiting.pop_front() {
            let results = ack
                .recv()
                .map_err(|_| polylogue_store::StoreError::WriterGone)?;
            self.accepted(size, results)?;
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<(), DatabaseError> {
        let mut error = None;
        while !self.waiting.is_empty() {
            if let Err(e) = self.acknowledge() {
                error.get_or_insert(e);
            }
        }
        error.map_or(Ok(()), Err)
    }
}

impl Engine {
    pub(super) fn refresh_database_progress(&mut self) -> Result<(), EngineError> {
        let reader = self.reads.reader()?;
        for (path, name) in &self.databases {
            if self.refused.contains(path) {
                continue;
            }
            let origin = polylogue_sources::origin::by_name(name).ok_or_else(|| {
                polylogue_store::StoreError::Invalid(format!("unknown database origin {name}"))
            })?;
            let input_id = self
                .inputs
                .get(path)
                .filter(|state| state.origin.name == *name)
                .map(|state| state.input_id.clone())
                .unwrap_or_else(|| self.location_id(path));
            let Some(progress) = reader.input_progress(&input_id)? else {
                continue;
            };
            self.inputs.insert(
                path.clone(),
                InputState {
                    input_id,
                    origin,
                    facts: None,
                    stored: progress.length,
                    parsed_upto: progress.parsed_upto,
                    parsed: progress.parsed,
                    waiting: progress.waiting,
                    parser_state: None,
                    tail: None,
                    chain: Some(progress.chain),
                    failed: None,
                    failed_history: HashMap::new(),
                },
            );
        }
        Ok(())
    }

    pub(super) fn capture_databases(
        &mut self,
        found: &[collect::Found],
        force: bool,
        pass: &mut Pass,
    ) -> Result<(bool, bool), EngineError> {
        let mut new_database = false;
        for f in found.iter().filter(|f| f.origin.is_row_dump(&f.path)) {
            new_database |= self
                .databases
                .insert(f.path.clone(), f.origin.name)
                .is_none();
        }
        let completed = self.captures.completed(false);
        pass.inputs_seen += completed.inputs_seen;
        pass.bytes_stored += completed.bytes_stored;
        pass.errors.extend(completed.errors);
        if completed.finished && self.capture_checked.is_some() {
            self.capture_checked = Some(Instant::now());
        }
        if completed.changed {
            self.refresh_database_progress()?;
        }
        let capture_due = force
            || new_database
            || self
                .capture_checked
                .is_none_or(|at| at.elapsed() >= self.config.live_every);
        let mut deferred_database = false;
        if capture_due && !self.captures.pending() {
            deferred_database = self.databases.keys().any(|path| {
                !self.refused.contains(path)
                    && self.inputs.get(path).is_some_and(|state| {
                        state.stored > 0 && (!state.parsed || state.parsed_upto < state.stored)
                    })
            });
            self.capture_checked = (!deferred_database).then(Instant::now);
            let databases = self
                .databases
                .iter()
                .filter(|(path, _)| !self.refused.contains(*path))
                .filter(|(path, _)| {
                    self.inputs.get(*path).is_none_or(|state| {
                        state.stored == 0 || (state.parsed && state.parsed_upto == state.stored)
                    })
                })
                .map(|(path, origin)| databases::DatabaseInput {
                    path: path.clone(),
                    input: InputKey {
                        id: self
                            .inputs
                            .get(path)
                            .filter(|state| state.origin.name == *origin)
                            .map(|state| state.input_id.clone())
                            .unwrap_or_else(|| self.location_id(path)),
                        format: (*origin).into(),
                        arrival_rank: 0,
                    },
                    observation: self.observation(path, None),
                })
                .collect();
            self.captures.start(
                databases,
                self.archive.clone(),
                self.writer.sender()?,
                self.wake.clone(),
            )?;
        }
        pass.capture_pending |= self.captures.pending() || deferred_database;
        Ok((deferred_database, completed.changed))
    }
}

#[cfg(test)]
mod tests;
