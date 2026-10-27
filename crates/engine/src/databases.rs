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
    pub errors: Vec<String>,
    pub changed: bool,
}
struct Worker {
    thread: JoinHandle<()>,
    result: Receiver<(Exporter, Completed)>,
}
#[derive(Default)]
pub(super) struct Databases {
    exporter: Option<Exporter>,
    worker: Option<Worker>,
}
impl Databases {
    pub fn pending(&self) -> bool { self.worker.is_some() }
    pub fn start(&mut self, inputs: Vec<DatabaseInput>, archive: Archive, writer: polylogue_store::WriterSender) -> Result<(), EngineError> {
        if self.pending() || inputs.is_empty() { return Ok(()); }
        let mut exporter = self.exporter.take().unwrap_or_else(|| Exporter::with_cursor(polylogue_sources::origin::table_cursor));
        let (tx, result) = sync_channel(1);
        let thread = std::thread::Builder::new().name("polylogue-databases".into()).spawn(move || {
            let mut completed = Completed::default();
            for source in inputs {
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<bool,ExportError> {
                    let mut queue = InputWriter::new(&archive, &writer, &source)?;
                    let done = exporter.export(&source.path, &archive, &source.input, &source.observation,
                        |bytes, update| queue.append(bytes, update));
                    // A stopped scan can still have acknowledged groups. Its
                    // next attempt compares against those committed hashes.
                    let written = queue.finish();
                    let changed = done?.is_some();
                    written?;
                    Ok(changed)
                }));
                match outcome {
                    Ok(Ok(changed)) => completed.changed |= changed,
                    Ok(Err(error)) => { exporter.forget(&source.path); completed.changed = true; completed.errors.push(format!("read database {}: {error}",source.path.display())); }
                    Err(_) => {
                        completed.errors.push(format!("read database {}: worker panicked",source.path.display()));
                        exporter = Exporter::with_cursor(polylogue_sources::origin::table_cursor);
                    }
                }
            }
            let _ = tx.send((exporter, completed));
        })?;
        self.worker = Some(Worker { thread, result });
        Ok(())
    }
    pub fn completed(&mut self, wait: bool) -> Completed {
        let Some(worker) = &self.worker else { return Completed::default(); };
        let result = if wait { worker.result.recv().map_err(|_|TryRecvError::Disconnected) } else { worker.result.try_recv() };
        match result {
            Ok((exporter, completed)) => {
                let worker = self.worker.take().expect("active worker");
                let _ = worker.thread.join();
                self.exporter = Some(exporter);
                completed
            }
            Err(TryRecvError::Empty) => Completed::default(),
            Err(TryRecvError::Disconnected) => {
                let worker = self.worker.take().expect("active worker");
                let _ = worker.thread.join();
                Completed { errors: vec!["database worker stopped".into()], changed: true }
            }
        }
    }
}
impl Drop for Databases {
    fn drop(&mut self) { self.completed(true); }
}

type Acknowledgment = Receiver<Vec<polylogue_store::Result<polylogue_store::staging::StoredInput>>>;
struct InputWriter<'a> {
    archive: &'a Archive,
    writer: &'a polylogue_store::WriterSender,
    source: &'a DatabaseInput,
    blobs: Arc<hashed_blobs::Store>,
    length: u64,
    exists: bool,
    waiting: VecDeque<(u64,Acknowledgment)>,
    bytes: u64,
}
impl<'a> InputWriter<'a> {
    fn new(archive: &'a Archive, writer: &'a polylogue_store::WriterSender, source: &'a DatabaseInput) -> Result<Self,ExportError> {
        let reads = ReadPool::open(archive)?;
        let reader = reads.reader()?;
        let progress = reader.input_progress(&source.input.id)?;
        let mut out = Self { archive, writer, source, blobs:Arc::new(archive.blobs()?), length:progress.as_ref().map_or(0,|p|p.length), exists:progress.is_some(), waiting:VecDeque::new(), bytes:0 };
        if let Some(progress) = progress {
            let from = reader.tool_database_length(&source.input.id)?.unwrap_or(0);
            if from < progress.length {
                // Only a stop between the byte commit and the comparison
                // commit leaves this suffix. An ordinary restart reads none.
                let chain = reader.stored_chain(&source.input.id,progress.chain,progress.length)?;
                let mut start = from;
                let mut pending = Vec::new();
                while start < progress.length {
                    let end = progress.length.min(start + polylogue_store::tool_databases::APPEND_BYTES as u64);
                    pending.extend(chain.read_range(&out.blobs,start,end)?);
                    if let Some(last) = pending.iter().rposition(|b| *b == b'\n') {
                        let mut update = Comparison::records(&pending[..=last],None)?;
                        update.length = end - (pending.len()-last-1) as u64;
                        let batch = Batch { input:source.input.clone(), change:Change::Reparse{chain:progress.chain}, observation:None, continues:false, new_blob:None, blobs:vec![], length:progress.length, parse:vec![ParseStep::comparison(update)],release:Default::default() };
                        out.writer.store_group(vec![PreparedBatch{batch,files:vec![]}],Arc::clone(&out.blobs))?.into_iter().collect::<polylogue_store::Result<Vec<_>>>()?;
                        pending.drain(..=last);
                    }
                    start = end;
                }
                if !pending.is_empty() { return Err(ExportError::Invalid("stored row dump has an incomplete record".into())); }
            }
        }
        Ok(out)
    }
    fn append(&mut self, bytes: Vec<u8>, mut update: Comparison) -> Result<(),ExportError> {
        let size = bytes.len() as u64;
        self.length += size;
        update.length = self.length;
        let (new_blob,files) = if bytes.is_empty() { (None,Vec::new()) } else {
            let (blob,file) = polylogue_store::timing::time(polylogue_store::timing::Stage::Store, || self.archive.prepare_blob(&bytes))?;
            (Some(blob),file.into_iter().collect())
        };
        let batch = Batch { input:self.source.input.clone(), change:if !self.exists {Change::First} else if size > 0 {Change::Appended { from:self.length-size }} else {Change::Unchanged}, observation:Some(self.source.observation.clone()), continues:false,new_blob,blobs:vec![],length:self.length,parse:vec![ParseStep::comparison(update)],release:Default::default() };
        self.exists = true;
        let ack = self.writer.enqueue_group(vec![PreparedBatch{batch,files}],Arc::clone(&self.blobs))?;
        self.waiting.push_back((size,ack)); self.bytes += size;
        // Groups already queued may commit while the next group is read.
        while self.bytes >= 128 << 20 { self.acknowledge()?; }
        Ok(())
    }
    fn acknowledge(&mut self) -> Result<(),ExportError> {
        if let Some((size,ack))=self.waiting.pop_front() {
            for result in ack.recv().map_err(|_| polylogue_store::StoreError::WriterGone)? { result?; }
            self.bytes -= size;
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<(),ExportError> {
        let mut error = None;
        while !self.waiting.is_empty() { if let Err(e)=self.acknowledge() { error.get_or_insert(e); } }
        error.map_or(Ok(()),Err)
    }
}
