//! A standalone build of the index (docs/architecture.md, one path): every
//! stored chain due a reading is read from its first byte, at the length the
//! record holds when the build starts, by as many workers as the engine
//! parses with. Each worker writes whole chains' readings into sorted runs
//! of its own ([`polylogue_store::writer::build::BuildWorker`]); the writer then
//! loads the runs, composes every session once, writes the indexes and
//! search entries and installs the file. Requests are answered while the
//! build runs: the loop thread takes them between handing chains to the
//! workers and while it waits on the writer, and the workers, which touch
//! neither the writer's index nor a reader's, go on reading. Nothing a
//! worker has read is read again. What upkeep writes meanwhile goes into the
//! build file, and the build reads only chains not read there. Upkeep brings
//! the installed index forward from the build's position.
use crate::plan::Plan;
use crate::{Engine, EngineError, LoopMsg, Pass, has_reading, pool};
use polylogue_store::writer::ChainSink as _;
use polylogue_store::writer::build::{BuildPhase, BuildWorker, PhaseClock};
use polylogue_store::{Change, InputKey};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, TrySendError, sync_channel};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// Inputs whose chains are planned per page.
const PLANNED_INPUTS: usize = 256;

/// How often the loop thread looks for a request while it waits.
const REQUEST_POLL: Duration = Duration::from_millis(5);

/// The memory one input byte takes while its range is parsed: a parsed
/// range holds about ten times its bytes.
const PREPARED_PER_INPUT_BYTE: u64 = 10;

/// The memory the prepared readings in flight may take at once. A
/// worker reads a chain one range at a time, so a chain holds at most a
/// range's share; a worker waits before it starts a chain until that share
/// fits, and a chain whose share is the whole allowance runs alone.
pub(crate) struct Admission {
    held: Mutex<u64>,
    room: Condvar,
    limit: u64,
}

pub(crate) struct Admitted<'a> {
    admission: &'a Admission,
    bytes: u64,
}

impl Admission {
    pub(crate) fn new(limit: u64) -> Self {
        Self {
            held: Mutex::new(0),
            room: Condvar::new(),
            limit: limit.max(1),
        }
    }

    /// Waits until `bytes` fit beside what is held, or until `stopped`.
    pub(crate) fn admit(
        &self,
        bytes: u64,
        stopped: impl Fn() -> bool,
    ) -> Result<Admitted<'_>, EngineError> {
        let bytes = bytes.clamp(1, self.limit);
        let mut held = self.held.lock().map_err(|_| EngineError::Stopped)?;
        while *held + bytes > self.limit {
            if stopped() {
                return Err(EngineError::Stopped);
            }
            held = self
                .room
                .wait_timeout(held, std::time::Duration::from_millis(100))
                .map_err(|_| EngineError::Stopped)?
                .0;
        }
        *held += bytes;
        Ok(Admitted {
            admission: self,
            bytes,
        })
    }

    #[cfg(test)]
    pub(crate) fn held(&self) -> u64 {
        *self.held.lock().unwrap()
    }
}

impl Drop for Admitted<'_> {
    fn drop(&mut self) {
        if let Ok(mut held) = self.admission.held.lock() {
            *held -= self.bytes;
        }
        self.admission.room.notify_all();
    }
}

/// What the workers share while they prepare runs.
struct Prepare {
    archive: polylogue_store::Archive,
    blobs: Arc<polylogue_store::blobs::Blobs>,
    writer: polylogue_store::WriterSender,
    /// Its flag ends the build's reads when the build is abandoned.
    settings: pool::ChainSettings,
    admission: Admission,
    /// The most input bytes one range of a chain reads.
    chunk: u64,
    plans: Mutex<Receiver<Plan>>,
    next_run: AtomicU64,
    /// Each worker's sort buffer.
    sort_memory: usize,
    chains: AtomicU64,
    chains_read: AtomicU64,
    bytes: AtomicU64,
    bytes_read: AtomicU64,
    /// Bytes the parser read, summed over every chain.
    parsed: AtomicU64,
    runs: Mutex<Vec<(u64, PathBuf)>>,
    failed: Mutex<Vec<(Plan, String)>>,
}

impl Prepare {
    fn abandoned(&self) -> bool {
        self.settings.yielding.load(Ordering::Acquire) || self.archive.stop().is_cancelled()
    }

    fn progress(&self) {
        self.archive.record_build_progress(
            self.chains.load(Ordering::Relaxed),
            self.chains_read.load(Ordering::Relaxed),
        );
        self.archive.record_build_bytes(
            self.bytes.load(Ordering::Relaxed),
            self.bytes_read.load(Ordering::Relaxed),
        );
    }

    /// One worker. A worker that fails, or a build abandoned, takes the
    /// remaining chains off the queue unread, so the planner never waits.
    fn work(&self) -> Result<(), EngineError> {
        let worked = self.work_until_done();
        while let Ok(plans) = self.plans.lock()
            && plans.recv().is_ok()
        {}
        worked
    }

    /// Takes chains until none is left, reading each into the worker's
    /// stream.
    fn work_until_done(&self) -> Result<(), EngineError> {
        let mut run: Option<(u64, BuildWorker)> = None;
        loop {
            let plan = self.plans.lock().map_err(|_| EngineError::Stopped)?.recv();
            let Ok(mut plan) = plan else { break };
            if self.abandoned() {
                return Err(EngineError::Stopped);
            }
            let Change::Reparse { chain } = plan.change else {
                continue;
            };
            let (input, length) = (plan.input.clone(), plan.size);
            let admitted = self
                .admission
                .admit(length.min(self.chunk) * PREPARED_PER_INPUT_BYTE, || {
                    self.abandoned()
                })?;
            if run.is_none() {
                let number = self.next_run.fetch_add(1, Ordering::Relaxed);
                run = Some((
                    number,
                    BuildWorker::create(&self.archive, number, self.sort_memory)?,
                ));
            }
            let (_, worker) = run.as_mut().expect("a worker was opened");
            let mut read = 0;
            // The chain's steps go into the worker's scratch index as they
            // are parsed; an unfinished chain leaves nothing there.
            let written = worker
                .begin(&self.writer, &input, chain, length)
                .map_err(EngineError::from)
                .and_then(|mut open| {
                    let complete = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        pool::read_chain(
                            &mut plan,
                            &self.archive,
                            &self.blobs,
                            &self.settings,
                            &mut read,
                            &mut open,
                            |open, batch| open.decoded(batch),
                        )
                    }))
                    .unwrap_or_else(|_| Err(EngineError::Parse("the parser panicked".into())))?;
                    if !complete {
                        return Err(EngineError::Stopped);
                    }
                    open.finish()?;
                    Ok(())
                });
            drop(admitted);
            self.parsed.fetch_add(read, Ordering::Relaxed);
            match written {
                Ok(()) => {}
                Err(EngineError::Stopped) => return Err(EngineError::Stopped),
                // A chain whose reading failed is left out of the build; its
                // failure is recorded once the index is installed.
                Err(error) => self
                    .failed
                    .lock()
                    .map_err(|_| EngineError::Stopped)?
                    .push((plan, format!("build: {}: chain {chain}: {error}", input.id))),
            }
            self.chains_read.fetch_add(1, Ordering::Relaxed);
            self.bytes_read.fetch_add(length, Ordering::Relaxed);
            self.progress();
        }
        if let Some((number, run)) = run {
            let paths = run.finish()?;
            self.runs
                .lock()
                .map_err(|_| EngineError::Stopped)?
                .extend(paths.into_iter().map(|path| (number, path)));
        }
        Ok(())
    }
}

impl Engine {
    /// Builds the index from the stored record and installs it. Returns the
    /// pass the build made, imports answered meanwhile included, and the
    /// phases it took.
    pub(crate) fn build_index(&mut self) -> Result<(Pass, Vec<BuildPhase>), EngineError> {
        let mut pass = Pass::default();
        // Messages that name an asset find it when they are composed: the
        // asset lookup comes from stored observations and sidecars.
        self.refresh_import_assets(&mut pass)?;
        let through = self.reads.reader()?.last_input_id()?;
        self.enter_build_phase("read chains");
        let workers = self.config.workers.max(1);
        let (plans_tx, plans_rx) = sync_channel::<Plan>(workers * 2);
        let prepare = Prepare {
            archive: self.archive.clone(),
            blobs: Arc::clone(&self.blobs),
            writer: self.writer.sender(),
            settings: pool::ChainSettings {
                stopped_attempt: self.stopped_attempt,
                read_memory: self.config.reader_memory_per_connection()?,
                yielding: Arc::new(AtomicBool::new(false)),
            },
            admission: Admission::new(self.archive.memory_budget().total() / 3),
            chunk: self.config.chunk.max(1),
            plans: Mutex::new(plans_rx),
            next_run: AtomicU64::new(0),
            sort_memory: (self.archive.memory_budget().total() / 8) as usize / workers,
            chains: AtomicU64::new(0),
            chains_read: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            parsed: AtomicU64::new(0),
            runs: Mutex::new(Vec::new()),
            failed: Mutex::new(Vec::new()),
        };
        let clock = PhaseClock::start("read chains");
        let prepared = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|n| {
                    std::thread::Builder::new()
                        .name(format!("polylogue-build-{n}"))
                        .spawn_scoped(scope, || prepare.work())
                })
                .collect::<std::io::Result<_>>()?;
            let mut result = self.plan_build(&prepare, plans_tx, through, &mut pass);
            while result.is_ok() && !handles.iter().all(|handle| handle.is_finished()) {
                result = self.answer_waiting(&mut pass).map(drop);
                std::thread::sleep(REQUEST_POLL);
            }
            if result.is_err() {
                prepare.settings.yielding.store(true, Ordering::Release);
            }
            for handle in handles {
                let worked = handle.join().unwrap_or_else(|_| {
                    Err(std::io::Error::other("a build worker panicked").into())
                });
                if result.is_ok()
                    && let Err(error) = worked
                {
                    result = Err(error);
                }
            }
            result
        });
        prepared?;
        let mut runs = prepare
            .runs
            .into_inner()
            .map_err(|_| EngineError::Stopped)?;
        runs.sort();
        let failed = prepare
            .failed
            .into_inner()
            .map_err(|_| EngineError::Stopped)?;
        pass.bytes_read += prepare.parsed.load(Ordering::Relaxed);
        let mut phases = vec![clock.end(prepare.chains_read.load(Ordering::Relaxed))];
        let (announce, announced) = std::sync::mpsc::channel();
        let built = self
            .writer
            .sender()
            .start_build(runs.into_iter().map(|(_, path)| path).collect(), announce)?;
        // The writer drops its sender when the build ends; requests are
        // answered while it works.
        loop {
            match announced.recv_timeout(REQUEST_POLL) {
                Ok(phase) => self.enter_build_phase(phase),
                Err(RecvTimeoutError::Timeout) => {
                    self.answer_waiting(&mut pass)?;
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        phases.extend(
            built
                .recv()
                .map_err(|_| polylogue_store::StoreError::WriterGone)??,
        );
        // Positions read before the build named the build file's empty
        // index; the next pass reads them from the installed one.
        self.discard_confirmed_state();
        for (mut plan, error) in failed {
            self.record_failure(&mut plan, error, &mut pass);
        }
        pass.messages_written = self.reads.reader()?.status()?.messages;
        Ok((pass, phases))
    }

    /// Answers the requests waiting; true when one was answered. A stop
    /// request ends the build, and stays queued for the loop.
    pub(crate) fn answer_waiting(&mut self, pass: &mut Pass) -> Result<bool, EngineError> {
        let mut answered = false;
        while self.poll_request() {
            let Some(message) = self
                .requests
                .as_mut()
                .and_then(|requests| requests.pending.take())
            else {
                break;
            };
            if matches!(message, LoopMsg::Stop) {
                if let Some(requests) = &mut self.requests {
                    requests.pending = Some(message);
                }
                return Err(EngineError::Stopped);
            }
            let mut summary = crate::RunSummary::default();
            self.answer(message, &|_| {}, true, &mut summary);
            if let Some(requests) = &mut self.requests {
                requests
                    .yielding
                    .store(false, std::sync::atomic::Ordering::Release);
            }
            pass.add(summary.work);
            self.imports_during_build += summary.imports;
            answered = true;
        }
        Ok(answered)
    }

    /// Sends every stored chain due a reading to the workers, a page of
    /// inputs at a time, answering requests whenever the workers are busy.
    /// No read is held while a plan waits for a worker: a worker's decoded
    /// bytes are stored through the writer meanwhile.
    fn plan_build(
        &mut self,
        prepare: &Prepare,
        plans: std::sync::mpsc::SyncSender<Plan>,
        through: Option<String>,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        let versions = crate::reading_contracts()
            .into_iter()
            .filter_map(|contract| match contract {
                polylogue_store::ReadingContract::Expected { origin, version } => {
                    Some((origin, version))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        let Some(through) = through else {
            return Ok(());
        };
        let mut after = None::<String>;
        loop {
            self.archive.stop().check()?;
            let reader = self.reads.reader()?;
            let ids: Vec<String> = reader
                .input_ids_after(after.as_deref(), PLANNED_INPUTS)?
                .into_iter()
                .filter(|input| *input <= through)
                .collect();
            let Some(last) = ids.last().cloned() else {
                return Ok(());
            };
            let behind = reader.chains_behind_index(
                &versions,
                polylogue_store::StoppedChains::ForAttempt(self.stopped_attempt),
                polylogue_store::read::RevisionChains::All,
                Some(&ids),
            )?;
            drop(reader);
            let mut page = Vec::new();
            for c in behind {
                let Some(plan) = self.build_plan(c)? else {
                    continue;
                };
                // What upkeep does before it reads a stored chain: a bundle's
                // authored records and a transcript's local files an
                // interrupted capture left unstored are stored first.
                if let Err(error) = self
                    .restore_captured_authored(&plan, pass)
                    .and_then(|()| self.observe_local_references(&plan, pass))
                {
                    let error = format!("before reading {}: {error}", plan.path.display());
                    prepare
                        .failed
                        .lock()
                        .map_err(|_| EngineError::Stopped)?
                        .push((plan, error));
                    continue;
                }
                prepare.chains.fetch_add(1, Ordering::Relaxed);
                prepare.bytes.fetch_add(plan.size, Ordering::Relaxed);
                page.push(plan);
            }
            prepare.progress();
            for mut plan in page {
                loop {
                    match plans.try_send(plan) {
                        Ok(()) => break,
                        Err(TrySendError::Full(back)) => {
                            plan = back;
                            if !self.answer_waiting(pass)? {
                                std::thread::sleep(REQUEST_POLL);
                            }
                        }
                        Err(TrySendError::Disconnected(_)) => return Ok(()),
                    }
                }
            }
            after = Some(last);
        }
    }

    /// The plan reading a stored chain from its first byte, or `None` for a
    /// chain no parser reads: upkeep records why it is refused.
    fn build_plan(
        &self,
        c: polylogue_store::read::BehindChain,
    ) -> Result<Option<Plan>, EngineError> {
        let state = c.progress;
        let Some(origin) = polylogue_sources::origin::by_name(&c.source.format)
            .filter(|origin| has_reading(origin))
        else {
            return Ok(None);
        };
        let path = {
            use std::os::unix::ffi::OsStringExt;
            c.source
                .path_bytes
                .map(|bytes| PathBuf::from(std::ffi::OsString::from_vec(bytes)))
                .or_else(|| c.source.path.map(PathBuf::from))
        };
        let Some(path) = path.filter(|path| crate::pass::unread_path(origin, path).is_none())
        else {
            return Ok(None);
        };
        let input = InputKey {
            id: c.input,
            format: origin.name.into(),
            route: if state.imported {
                origin.route.imported()
            } else {
                origin.route
            },
        };
        let mut fresh = state;
        fresh.parsed = false;
        fresh.parsed_upto = 0;
        Ok(Some(
            self.reparse_plan(path, origin, input, &fresh, None, false)?,
        ))
    }
}

#[cfg(test)]
mod tests;
