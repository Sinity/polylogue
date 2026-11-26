//! Tool-database reads run independently of file discovery and parsing.
//! Only the writer's bounded byte queue applies backpressure to a reader.
use crate::*;
use polylogue_store::tool_databases::{DatabaseError, DatabaseReader};
use polylogue_store::{Batch, staging::PreparedBatch};
use std::sync::mpsc::{TryRecvError, sync_channel};
use std::thread::JoinHandle;

pub(super) struct DatabaseInput {
    pub path: PathBuf,
    pub input: InputKey,
    pub observation: Observation,
    pub source_read: crate::capture_read::SourceReadContext,
}
#[derive(Default)]
pub(super) struct Completed {
    pub finished: bool,
    pub errors: Vec<polylogue_model::Failure>,
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
    scope: polylogue_store::ReadScopeToken,
}
#[derive(Default)]
pub(super) struct Databases {
    reader: Option<(DatabaseReader, polylogue_store::ReadScopeToken)>,
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
        wake: Option<crossbeam_channel::Sender<LoopMsg>>,
    ) -> Result<(), EngineError> {
        if self.pending() || inputs.is_empty() {
            return Ok(());
        }
        let scope = archive.scope_token()?;
        let mut reader = self
            .reader
            .take()
            .filter(|(_, scope)| scope.check().is_ok())
            .map(|(reader, _)| reader)
            .unwrap_or_default();
        let (tx, result) = sync_channel(1);
        let (notify, progress) = sync_channel(1);
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        reader.stop_when(Arc::clone(&stop));
        let cancelled = Arc::clone(&stop);
        let bytes = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let captured = Arc::clone(&bytes);
        let thread = polylogue_model::spawn_named("polylogue-databases", move || {
            let _worker = polylogue_store::timing::worker();
            let mut completed = Completed::default();
            let mut pending = Vec::new();
            for source in &inputs {
                if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                if let Err(error) =
                    reader.recall(&source.path, &archive, &source.input, &source.observation)
                {
                    completed.errors.push(
                        polylogue_model::Failure::new(source.path.display(), &error)
                            .doing("read database"),
                    );
                    continue;
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
                        completed.errors.push(
                            polylogue_model::Failure::new(source.path.display(), &error)
                                .doing("read database"),
                        );
                        continue;
                    }
                };
                let outcome = polylogue_model::caught(|| {
                    reader.read(
                        &source.path,
                        &archive,
                        &source.input,
                        &source.observation,
                        due,
                        |bytes, observation| queue.append(bytes, observation),
                    )
                });
                match outcome {
                    Ok(Ok(report)) => {
                        completed.changed = true;
                        if let Err(error) = queue.enqueue_staged(report.snapshot) {
                            reader.forget(&source.path);
                            completed.errors.push(
                                polylogue_model::Failure::new(source.path.display(), &error)
                                    .doing("stage database"),
                            );
                            continue;
                        }
                    }
                    Ok(Err(error)) => {
                        reader.forget(&source.path);
                        completed.changed = true;
                        completed.errors.push(
                            polylogue_model::Failure::new(source.path.display(), &error)
                                .doing("read database"),
                        );
                        continue;
                    }
                    Err(panic) => {
                        completed.changed = true;
                        completed.errors.push(
                            polylogue_model::Failure::new(source.path.display(), &panic)
                                .doing("read database"),
                        );
                        reader = DatabaseReader::default();
                        reader.stop_when(Arc::clone(&cancelled));
                        continue;
                    }
                }
                // Another source can be read while these groups commit.
                pending.push((source, queue));
            }
            for (source, mut queue) in pending {
                match queue.finish() {
                    Ok(()) => {}
                    Err(error) => {
                        reader.forget(&source.path);
                        completed.changed = true;
                        completed.errors.push(
                            polylogue_model::Failure::new(source.path.display(), &error)
                                .doing("write database"),
                        );
                    }
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
            scope,
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
                self.reader = Some((reader, worker.scope));
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
                    errors: vec![polylogue_model::Failure::bug(
                        "tool databases",
                        "the database worker ended before its reply",
                    )],
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
    length: u64,
    exists: bool,
    staged: Vec<PreparedBatch>,
    acknowledgment: Option<Acknowledgment>,
    bytes: u64,
    notify: SyncSender<()>,
    wake: Option<crossbeam_channel::Sender<LoopMsg>>,
    captured: Arc<std::sync::atomic::AtomicU64>,
}
impl<'a> InputWriter<'a> {
    fn new(
        archive: &'a Archive,
        writer: &'a polylogue_store::WriterSender,
        source: &'a DatabaseInput,
        notify: SyncSender<()>,
        wake: Option<crossbeam_channel::Sender<LoopMsg>>,
        captured: Arc<std::sync::atomic::AtomicU64>,
    ) -> Result<Self, DatabaseError> {
        let reads = ReadPool::open(archive)?;
        let reader = reads.reader()?;
        let progress = reader.input_progress(&source.input.id)?;
        let out = Self {
            archive,
            writer,
            source,
            length: progress.as_ref().map_or(0, |p| p.length),
            exists: progress.is_some(),
            staged: Vec::new(),
            acknowledgment: None,
            bytes: 0,
            notify,
            wake,
            captured,
        };
        Ok(out)
    }
    fn append(&mut self, bytes: Vec<u8>, observation: &Observation) -> Result<(), DatabaseError> {
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
        let source_read = new_blob.as_ref().map(|blob| {
            self.source.source_read.finish(
                polylogue_store::capture_read::RawRange {
                    start: self.length - size,
                    end: self.length,
                },
                blob.hash,
            )
        });
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
            observation: Some(observation.clone()),
            continues: !self.staged.is_empty(),
            new_blob,
            blobs: vec![],
            length: self.length,
            parse: vec![],
            capture_read: source_read,
            release: Default::default(),
        };
        self.exists = true;
        // Only closed encodings and small batch metadata survive a callback;
        // no provider payload or open body descriptor accumulates in memory.
        self.staged.push(PreparedBatch { batch, files });
        self.bytes += size;
        Ok(())
    }
    fn enqueue_staged(
        &mut self,
        snapshot: Option<polylogue_store::capture_read::DatabaseSnapshotToken>,
    ) -> Result<(), DatabaseError> {
        if self.staged.is_empty() {
            return Ok(());
        }
        let snapshot = snapshot.ok_or_else(|| {
            polylogue_store::StoreError::internal("database sample has no consistent-read token")
        })?;
        let start = self.staged.iter().find_map(|part| {
            part.batch
                .capture_read
                .as_ref()
                .map(|source_read| source_read.range.start)
        });
        if let Some(start) = start {
            let window = polylogue_store::capture_read::RawRange {
                start,
                end: self.length,
            };
            for part in &mut self.staged {
                if let Some(source_read) = &mut part.batch.capture_read {
                    source_read.generation =
                        Some(polylogue_store::capture_read::ObservedState::Database {
                            stream: polylogue_store::capture_read::RawStream {
                                source: source_read.source.clone(),
                                input: part.batch.input.clone(),
                                role: polylogue_store::capture_read::RawStreamRole::DatabaseDump,
                                namespace: None,
                            },
                            snapshot: snapshot.clone(),
                            window,
                        });
                    source_read.raw_total =
                        (source_read.range.end == window.end).then_some(window.end);
                }
            }
        }
        // The read transaction has completed before the group is written.
        self.acknowledgment = Some(
            self.writer
                .enqueue_group(std::mem::take(&mut self.staged))?,
        );
        Ok(())
    }
    fn finish(&mut self) -> Result<(), DatabaseError> {
        let Some(ack) = self.acknowledgment.take() else {
            return Ok(());
        };
        let results = ack
            .recv()
            .map_err(|_| polylogue_store::StoreError::WriterGone)?;
        for result in results {
            result?;
        }
        self.captured
            .fetch_add(self.bytes, std::sync::atomic::Ordering::Relaxed);
        if self.notify.try_send(()).is_ok()
            && let Some(wake) = &self.wake
        {
            let _ = wake.send(LoopMsg::Changed);
        }
        Ok(())
    }
}

impl Engine {
    /// The stored input ID of a database file, or its location ID before
    /// the first capture.
    fn database_input_id(&self, path: &Path, origin: &OriginSpec) -> String {
        self.inputs
            .get(path)
            .filter(|state| state.origin.name == origin.name)
            .map(|state| state.input_id.clone())
            .unwrap_or_else(|| self.location_id(path))
    }

    pub(super) fn refresh_database_progress(&mut self) -> Result<(), EngineError> {
        let reader = self.reads.reader()?;
        for (path, &origin) in &self.databases {
            let input_id = self.database_input_id(path, origin);
            let Some(progress) = reader.input_progress(&input_id)? else {
                continue;
            };
            self.inputs
                .entry(path.clone())
                .or_insert_with(|| InputState::new(input_id.clone(), origin))
                .adopt_confirmed(&input_id, origin, None, Some(&progress));
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
            new_database |= self.databases.insert(f.path.clone(), f.origin).is_none();
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
                self.inputs
                    .get(path)
                    .is_some_and(|state| state.stored > 0 && state.needs_reading())
            });
            let sources: Vec<_> = self
                .databases
                .iter()
                .filter(|(path, _)| {
                    self.inputs
                        .get(*path)
                        .is_none_or(|state| state.stored == 0 || !state.needs_reading())
                })
                .map(|(path, origin)| (path.clone(), *origin))
                .collect();
            let mut databases = Vec::with_capacity(sources.len());
            for (path, origin) in sources {
                let source_read = self.source_read_context(
                    &path,
                    origin.name,
                    polylogue_store::capture_read::ReadRoute::NativeCapture,
                );
                databases.push(databases::DatabaseInput {
                    input: InputKey {
                        id: self.database_input_id(&path, origin),
                        format: origin.name.into(),
                        route: polylogue_model::Route::Live,
                    },
                    observation: self.observation(&path, None),
                    source_read,
                    path,
                });
            }
            let starting = !databases.is_empty();
            self.captures.start(
                databases,
                self.archive.clone(),
                self.writer.sender(),
                self.wake.clone(),
            )?;
            if starting {
                self.capture_checked = Some(Instant::now());
            }
        }
        Ok((deferred_database, completed.changed))
    }
}

#[cfg(test)]
mod tests;
