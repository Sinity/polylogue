//! Tool-database reads run independently of file discovery and parsing.
//! Only the writer's bounded byte queue applies backpressure to a reader.
use crate::*;
use polylogue_store::tool_databases::{Comparison, ExportError, Exporter};
use polylogue_store::{Batch, ParseStep, staging::PreparedBatch};
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
}
struct Worker {
    thread: JoinHandle<()>,
    result: Receiver<(Exporter, Completed)>,
    progress: Receiver<()>,
    stop: Arc<std::sync::atomic::AtomicBool>,
}
#[derive(Default)]
pub(super) struct Databases {
    exporter: Option<Exporter>,
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
        let mut exporter = self
            .exporter
            .take()
            .unwrap_or_else(|| Exporter::with_cursor(polylogue_sources::origin::table_cursor));
        let (tx, result) = sync_channel(1);
        let (notify, progress) = sync_channel(1);
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        exporter.stop_when(Arc::clone(&stop));
        let cancelled = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("polylogue-databases".into())
            .spawn(move || {
                let _worker = polylogue_store::timing::worker();
                let mut completed = Completed::default();
                let mut pending = Vec::new();
                for source in &inputs {
                    if cancelled.load(std::sync::atomic::Ordering::Relaxed) { break; }
                    let Some(due) = exporter.observe(&source.path) else {
                        continue;
                    };
                    let mut queue = match InputWriter::new(
                        &archive,
                        &writer,
                        source,
                        notify.clone(),
                        wake.clone(),
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
                        exporter.read(
                            &source.path,
                            &archive,
                            &source.input,
                            &source.observation,
                            due,
                            |bytes, update| queue.append(bytes, update),
                        )
                    }));
                    match outcome {
                        Ok(Ok(_)) => completed.changed = true,
                        Ok(Err(error)) => {
                            exporter.forget(&source.path);
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
                            exporter =
                                Exporter::with_cursor(polylogue_sources::origin::table_cursor);
                        }
                    }
                    // Another source can be read while these groups commit.
                    pending.push((source, queue));
                }
                for (source, mut queue) in pending {
                    if let Err(error) = queue.finish() {
                        exporter.forget(&source.path);
                        completed.changed = true;
                        completed
                            .errors
                            .push(format!("write database {}: {error}", source.path.display()));
                    }
                }
                let _ = tx.send((exporter, completed));
                if let Some(wake) = wake {
                    let _ = wake.send(LoopMsg::Changed);
                }
            })?;
        self.worker = Some(Worker {
            thread,
            result,
            progress,
            stop,
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
        match result {
            Ok((exporter, completed)) => {
                let worker = self.worker.take().expect("active worker");
                let _ = worker.thread.join();
                self.exporter = Some(exporter);
                Completed {
                    finished: true,
                    changed: changed || completed.changed,
                    ..completed
                }
            }
            Err(TryRecvError::Empty) => Completed {
                changed,
                ..Completed::default()
            },
            Err(TryRecvError::Disconnected) => {
                let worker = self.worker.take().expect("active worker");
                let _ = worker.thread.join();
                Completed {
                    finished: true,
                    errors: vec!["database worker stopped".into()],
                    changed: true,
                }
            }
        }
    }
}
impl Drop for Databases {
    fn drop(&mut self) {
        if let Some(worker) = &self.worker {
            worker.stop.store(true,std::sync::atomic::Ordering::Relaxed);
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
}
impl<'a> InputWriter<'a> {
    fn new(
        archive: &'a Archive,
        writer: &'a polylogue_store::WriterSender,
        source: &'a DatabaseInput,
        notify: SyncSender<()>,
        wake: Option<Sender<LoopMsg>>,
    ) -> Result<Self, ExportError> {
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
        };
        if let Some(progress) = progress {
            let from = reader
                .tool_database_length(&source.input.id)?
                .unwrap_or(if progress.parsed { progress.length } else { 0 });
            if from < progress.length {
                // Only a stop between the byte commit and the comparison
                // commit leaves this suffix. An ordinary restart reads none.
                let chain =
                    reader.stored_chain(&source.input.id, progress.chain, progress.length)?;
                let mut start = from;
                let mut pending = Vec::new();
                while start < progress.length {
                    let end = progress
                        .length
                        .min(start + polylogue_store::tool_databases::APPEND_BYTES as u64);
                    pending.extend(chain.read_range(&out.blobs, start, end)?);
                    if let Some(last) = pending.iter().rposition(|b| *b == b'\n') {
                        let mut update = Comparison::records(&pending[..=last], None)?;
                        update.length = end - (pending.len() - last - 1) as u64;
                        let batch = Batch {
                            input: source.input.clone(),
                            change: Change::Reparse {
                                chain: progress.chain,
                            },
                            observation: None,
                            continues: false,
                            new_blob: None,
                            blobs: vec![],
                            length: progress.length,
                            parse: vec![ParseStep::comparison(update)],
                            release: Default::default(),
                        };
                        out.writer
                            .store_group(
                                vec![PreparedBatch {
                                    batch,
                                    files: vec![],
                                }],
                                Arc::clone(&out.blobs),
                            )?
                            .into_iter()
                            .collect::<polylogue_store::Result<Vec<_>>>()?;
                        pending.drain(..=last);
                    }
                    start = end;
                }
                if !pending.is_empty() {
                    return Err(ExportError::Invalid(
                        "stored row dump has an incomplete record".into(),
                    ));
                }
            }
        }
        Ok(out)
    }
    fn append(&mut self, bytes: Vec<u8>, mut update: Comparison) -> Result<(), ExportError> {
        let size = bytes.len() as u64;
        self.length += size;
        update.length = self.length;
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
            parse: vec![ParseStep::comparison(update)],
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
    ) -> Result<(), ExportError> {
        self.bytes -= size;
        if self.notify.try_send(()).is_ok()
            && let Some(wake) = &self.wake
        {
            let _ = wake.send(LoopMsg::Changed);
        }
        for result in results {
            result?;
        }
        Ok(())
    }
    fn acknowledge_ready(&mut self) -> Result<(), ExportError> {
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
    fn acknowledge(&mut self) -> Result<(), ExportError> {
        if let Some((size, ack)) = self.waiting.pop_front() {
            let results = ack
                .recv()
                .map_err(|_| polylogue_store::StoreError::WriterGone)?;
            self.accepted(size, results)?;
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<(), ExportError> {
        let mut error = None;
        while !self.waiting.is_empty() {
            if let Err(e) = self.acknowledge() {
                error.get_or_insert(e);
            }
        }
        error.map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod tests;
