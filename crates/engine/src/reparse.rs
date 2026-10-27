//! Parse stored chains in parallel, then commit each complete reading once.
use crate::*;
use polylogue_store::writer::ChainReading;
use std::collections::VecDeque;
use std::sync::Condvar;

struct Memory {
    used: Mutex<u64>,
    room: Condvar,
    limit: u64,
}

struct Charge<'a>(&'a Memory, u64);

impl Memory {
    fn take(&self, bytes: u64) -> Charge<'_> {
        let mut used = self.used.lock().expect("reading memory");
        while *used != 0 && used.saturating_add(bytes) > self.limit {
            used = self.room.wait(used).expect("reading memory");
        }
        *used += bytes;
        Charge(self, bytes)
    }
}

impl Drop for Charge<'_> {
    fn drop(&mut self) {
        *self.0.used.lock().expect("reading memory") -= self.1;
        self.0.room.notify_all();
    }
}

struct Prepared {
    plan: Plan,
    reading: Result<Option<ChainReading>, EngineError>,
    bytes_read: u64,
}

impl Engine {
    pub(crate) fn replace_readings(
        &mut self,
        mut plans: Vec<Plan>,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        if plans.is_empty() {
            return Ok(());
        }
        let memory = Memory {
            used: Mutex::new(0),
            room: Condvar::new(),
            limit: self.config.bytes_in_flight.max(1),
        };
        plans.sort_by_key(|plan| plan.priority == Priority::Bulk);
        let mut busy = HashMap::<PathBuf, usize>::new();
        for plan in &plans {
            *busy.entry(plan.path.clone()).or_default() += 1;
        }
        let mut remaining = plans.len();
        let count = remaining;
        let waiting = Mutex::new(VecDeque::from(plans));
        let (finished, results) = channel();
        let workers = self.config.workers.max(1).min(count);
        let read_memory = self.config.reader_memory_per_connection()?;
        let archive = self.archive.clone();
        let blobs = Arc::clone(&self.blobs);
        let writer = self.writer.sender()?;
        polylogue_store::timing::reading_queue(count as i64, 0);
        std::thread::scope(|scope| -> Result<(), EngineError> {
            for worker in 0..workers {
                let finished = finished.clone();
                let waiting = &waiting;
                let memory = &memory;
                let archive = &archive;
                let blobs = &blobs;
                std::thread::Builder::new()
                    .name(format!("pl-reading-{worker}"))
                    .spawn_scoped(scope, move || {
                        loop {
                            let plan = { waiting.lock().expect("reading queue").pop_front() };
                            let Some(mut plan) = plan else { break };
                            polylogue_store::timing::reading_queue(-1, 0);
                            let mut bytes_read = 0;
                            let _busy = polylogue_store::timing::reading_worker(worker);
                            let reading =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    prepare(
                                        &mut plan,
                                        archive,
                                        blobs,
                                        read_memory,
                                        memory,
                                        &mut bytes_read,
                                    )
                                }))
                                .unwrap_or_else(|_| {
                                    Err(EngineError::Parse(
                                        "the replacement parser panicked".into(),
                                    ))
                                });
                            polylogue_store::timing::reading_queue(0, 1);
                            if finished
                                .send(Prepared {
                                    plan,
                                    reading,
                                    bytes_read,
                                })
                                .is_err()
                            {
                                return;
                            }
                        }
                    })
                    .expect("spawn a reading worker");
            }
            drop(finished);
            let mut commits = Vec::new();
            let mut checked = Instant::now();
            let every = self.config.live_every.max(Duration::from_millis(10));
            while remaining > 0 {
                self.reparse_live(&busy, &mut checked, pass)?;
                let Prepared {
                    mut plan,
                    reading,
                    bytes_read,
                } = match results.recv_timeout(every) {
                    Ok(prepared) => prepared,
                    Err(RecvTimeoutError::Timeout) => continue,
                    Err(RecvTimeoutError::Disconnected) => {
                        return Err(std::io::Error::other(
                            "a reading worker stopped before returning its input",
                        )
                        .into());
                    }
                };
                remaining -= 1;
                polylogue_store::timing::reading_queue(0, -1);
                pass.bytes_read += bytes_read;
                match reading {
                    Ok(Some(reading)) => {
                        let Change::Reparse { chain } = plan.change else {
                            return Err(std::io::Error::other(
                                "a replacement needs a stored chain",
                            )
                            .into());
                        };
                        match writer.queue_chain(plan.input.clone(), chain, plan.size, reading) {
                            Ok(reply) => commits.push((plan, reply)),
                            Err(error) => {
                                release_path(&mut busy, &plan.path);
                                pass.errors.push(format!("re-parse: {error}"));
                            }
                        }
                    }
                    Ok(None) => release_path(&mut busy, &plan.path),
                    Err(error) => {
                        pass.errors
                            .push(format!("re-parse {}: {error}", plan.path.display()));
                        release_path(&mut busy, &plan.path);
                        if matches!(error, EngineError::Parse(_)) {
                            plan.failed = true;
                            self.finish_plan(plan);
                        }
                    }
                }
            }
            for (plan, reply) in commits {
                let written = loop {
                    self.reparse_live(&busy, &mut checked, pass)?;
                    match reply.recv_timeout(every) {
                        Ok(written) => break written,
                        Err(RecvTimeoutError::Timeout) => continue,
                        Err(RecvTimeoutError::Disconnected) => {
                            return Err(polylogue_store::StoreError::WriterGone.into());
                        }
                    }
                };
                release_path(&mut busy, &plan.path);
                match written {
                    Ok(written) => {
                        pass.messages_written += written;
                        self.finish_plan(plan);
                    }
                    // The old reading remains committed and the next pass retries.
                    Err(error) => pass.errors.push(format!("re-parse: {error}")),
                }
            }
            Ok(())
        })
    }

    fn reparse_live(
        &mut self,
        busy: &HashMap<PathBuf, usize>,
        checked: &mut Instant,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        if checked.elapsed() >= self.config.live_every {
            *checked = Instant::now();
            let live = self.live_plans(&busy.keys().cloned().collect())?;
            self.execute(live, pass)?;
        }
        Ok(())
    }
}

fn release_path(busy: &mut HashMap<PathBuf, usize>, path: &Path) {
    let count = busy.get_mut(path).expect("reading path");
    *count -= 1;
    if *count == 0 {
        busy.remove(path);
    }
}

fn prepare(
    plan: &mut Plan,
    archive: &Archive,
    blobs: &Arc<hashed_blobs::Store>,
    read_memory: u64,
    memory: &Memory,
    bytes_read: &mut u64,
) -> Result<Option<ChainReading>, EngineError> {
    if let pool::Bytes::Unloaded {
        chain,
        length,
        decoded,
    } = plan.source
    {
        let reads = ReadPool::with_budget(archive, 1, read_memory)?;
        let stored = match reads.reader()?.stored_chain(&plan.input.id, chain, length) {
            Ok(stored) => stored,
            Err(polylogue_store::StoreError::InputPending { .. }) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        plan.source = if decoded {
            pool::Bytes::Import(Arc::new(Mutex::new(read::Imported::from_chain(
                stored,
                Arc::clone(blobs),
            ))))
        } else {
            pool::Bytes::Stored(Arc::new(stored))
        };
    }
    polylogue_store::fault::check("reparse.prepare", &plan.input.id)?;
    let mut reading = ChainReading::new()?;
    if let Some(polylogue_sources::Parse::Export(mut parser)) = (plan.origin.open)(&plan.path) {
        let _charge = memory.take(plan.size.min(plan.chunk));
        *bytes_read +=
            crate::import::exports::parse_export(plan, &mut *parser, blobs, |mut item| {
                let mut decoded = Vec::new();
                let mut files = Vec::new();
                for blob in std::mem::take(&mut item.output.blobs) {
                    let (new, file) = archive.prepare_decoded(&blob.bytes)?;
                    if new.hash != blob.hash || new.size != blob.size {
                        return Err(std::io::Error::other(
                            "decoded bytes differ from their declared hash or length",
                        )
                        .into());
                    }
                    decoded.push(new);
                    files.extend(file);
                }
                let mut step = polylogue_store::ParseStep::new(
                    item.output,
                    item.counts,
                    0..if item.done { plan.size } else { 0 },
                    item.done,
                );
                step.order_from = Some(item.order_from);
                step.repeats_whole_conversations = item.repeats;
                step.reading_version = plan.origin.reading_version();
                polylogue_store::timing::time(polylogue_store::timing::Stage::Store, || {
                    reading.push(&mut step)
                })?;
                keep_decoded(&mut reading, plan, decoded, files);
                Ok(())
            })?;
        plan.parse_pos = plan.size;
    } else {
        while !plan.finished() {
            let job = plan.next_job();
            let bytes = job.end - job.start;
            let _charge = memory.take(bytes);
            *bytes_read += bytes;
            let (batch, done) = pool::run(job, archive, blobs, false);
            if let Some(error) = &done.error {
                return Err(if error.starts_with("parse ") {
                    EngineError::Parse(error.clone())
                } else {
                    std::io::Error::other(error.clone()).into()
                });
            }
            let Some(mut batch) = batch else {
                return Err(std::io::Error::other("a replacement returned no rows").into());
            };
            for step in &mut batch.batch.parse {
                polylogue_store::timing::time(polylogue_store::timing::Stage::Store, || {
                    reading.push(step)
                })?;
            }
            batch.batch.parse.clear();
            if !batch.batch.blobs.is_empty() {
                reading.decoded(batch);
            }
            plan.apply(done, true);
        }
    }
    Ok(Some(reading))
}

fn keep_decoded(
    reading: &mut ChainReading,
    plan: &Plan,
    blobs: Vec<polylogue_store::NewBlob>,
    files: Vec<polylogue_store::staging::PreparedBytes>,
) {
    if !blobs.is_empty() {
        reading.decoded(polylogue_store::staging::PreparedBatch {
            batch: polylogue_store::Batch {
                input: plan.input.clone(),
                change: plan.change.clone(),
                observation: None,
                continues: false,
                new_blob: None,
                blobs,
                length: plan.size,
                parse: Vec::new(),
                release: Default::default(),
            },
            files,
        });
    }
}

#[cfg(test)]
mod tests;
