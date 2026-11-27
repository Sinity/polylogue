//! An index rebuild (docs/architecture.md, one path): every stored chain is
//! read from its first byte, at the length the archive holds when it is
//! planned, by as many workers as the engine parses with. Each worker writes
//! whole chains' parses into sorted files of its own
//! ([`polylogue_store::index::rebuild::RebuildWorker`]); the rebuild then
//! runs its remaining stages on a thread of its own and installs the file
//! ([`polylogue_store::index::rebuild::Rebuild::finish`]). Requests are
//! answered while it runs: the loop thread takes them between handing
//! chains to the workers and while it waits on the rebuild. The index writer
//! never writes the rebuild's file: it writes the installed index, when there
//! is a current one, and refuses index writes otherwise. Incremental updates
//! bring the installed rebuild forward from the archive position it covers.
use crate::plan::Plan;
use crate::{Engine, EngineError, LoopMsg, Pass, has_reading, pool};
use crossbeam_channel::{Receiver, Select, Sender, TryRecvError, TrySendError};
use polylogue_model::{Cause, Caused as _, LockExt as _};
use polylogue_store::archive::build_progress::{CountUnit, StepCount};
use polylogue_store::index::rebuild::{Rebuild, RebuildStage, RebuildWorker, StageRecord};
use polylogue_store::writer::ChainSink as _;
use polylogue_store::{Change, InputKey};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Inputs whose chains are planned per page.
const PLANNED_INPUTS: usize = 256;

/// The memory one input byte takes while its range is parsed: a parsed
/// range holds about ten times its bytes.
const PREPARED_PER_INPUT_BYTE: u64 = 10;

/// A chain's rows in its worker's scratch index, held in memory until the
/// chain ends, take about a sixth of its bytes: the reading tables of an
/// index built from the reproduction archive hold 1.7 GB of rows and keys
/// for 11.5 GB of chains.
const SCRATCH_DIVISOR: u64 = 6;

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
        let mut held = self.held.locked();
        while *held + bytes > self.limit {
            if stopped() {
                return Err(EngineError::Stopped);
            }
            held = polylogue_model::unpoisoned(
                self.room
                    .wait_timeout(held, std::time::Duration::from_millis(100)),
            )
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
        *self.admission.held.locked() -= self.bytes;
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
    plans: Receiver<Plan>,
    next_run: AtomicU64,
    /// Each worker's own memory: its sort buffer and scratch cache.
    worker_memory: usize,
    chains: AtomicU64,
    chains_read: AtomicU64,
    bytes: AtomicU64,
    bytes_read: AtomicU64,
    /// Bytes the parser read, summed over every chain.
    parsed: AtomicU64,
    runs: Mutex<Vec<(u64, PathBuf)>>,
    /// Chains whose bytes cannot be read under their reading version:
    /// recorded as their failed parse state once the index is installed.
    failed: Mutex<Vec<(Plan, polylogue_model::ChainFailure)>>,
    /// Chains left out for a failure that is not their bytes' (a busy
    /// database, a bug): reported, and read again after the install.
    left_out: Mutex<Vec<polylogue_model::Failure>>,
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
        let (bytes, bytes_read) = (
            self.bytes.load(Ordering::Relaxed),
            self.bytes_read.load(Ordering::Relaxed),
        );
        self.archive.record_build_bytes(bytes, bytes_read);
        self.archive.record_build_count(
            RebuildStage::Parsing.name(),
            StepCount {
                done: bytes_read,
                total: Some(bytes),
                unit: CountUnit::Bytes,
            },
        );
    }

    /// One worker. A worker that fails, or a build abandoned, takes the
    /// remaining chains off the queue unread, so the planner never waits.
    fn work(&self) -> Result<(), EngineError> {
        let worked = polylogue_model::caught(|| self.work_until_done())
            .unwrap_or_else(|panic| Err(EngineError::internal(panic)));
        while self.plans.recv().is_ok() {}
        worked
    }

    /// Keeps a chain's failure: its bytes' own is recorded on the chain, a
    /// failure the machine or a stop causes ends the build, and any other
    /// leaves the chain out to be read again after the install.
    fn chain_failed(&self, plan: Plan, error: EngineError) -> Result<(), EngineError> {
        let subject = plan.path.display().to_string();
        match error {
            EngineError::Chain(failure) => self.failed.locked().push((plan, failure)),
            error => match error.cause() {
                Cause::Stopping | Cause::Environment => return Err(error),
                Cause::Transient
                | Cause::Bug
                | Cause::StoredData
                | Cause::Request
                | Cause::Content => self
                    .left_out
                    .locked()
                    .push(polylogue_model::Failure::new(subject, &error).doing("build")),
            },
        }
        Ok(())
    }

    /// Takes chains until none is left, reading each into the worker's
    /// stream.
    fn work_until_done(&self) -> Result<(), EngineError> {
        let mut run: Option<(u64, RebuildWorker)> = None;
        loop {
            let plan = self.plans.recv();
            let Ok(mut plan) = plan else { break };
            if self.abandoned() {
                return Err(EngineError::Stopped);
            }
            let Change::Reparse { chain } = plan.change else {
                continue;
            };
            let (input, length) = (plan.input.clone(), plan.size);
            let _doing =
                polylogue_store::activity::doing_on("reading a stored file", input.id.clone());
            let admitted = self.admission.admit(
                length.min(self.chunk) * PREPARED_PER_INPUT_BYTE + length / SCRATCH_DIVISOR,
                || self.abandoned(),
            )?;
            if run.is_none() {
                let number = self.next_run.fetch_add(1, Ordering::Relaxed);
                run = Some((
                    number,
                    RebuildWorker::create(&self.archive, number, self.worker_memory)?,
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
                    let complete = polylogue_model::caught(|| {
                        pool::read_chain(
                            &mut plan,
                            &self.archive,
                            &self.blobs,
                            &self.settings,
                            &mut read,
                            &mut open,
                            |open, batch| open.decoded(batch),
                        )
                    })
                    .unwrap_or_else(|panic| {
                        Err(polylogue_model::ChainFailure::from(panic).into())
                    })?;
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
                // A chain whose reading failed is left out of the build.
                Err(error) => self.chain_failed(plan, error)?,
            }
            self.chains_read.fetch_add(1, Ordering::Relaxed);
            self.bytes_read.fetch_add(length, Ordering::Relaxed);
            self.progress();
        }
        if let Some((number, run)) = run {
            let paths = run.finish()?;
            self.runs
                .locked()
                .extend(paths.into_iter().map(|path| (number, path)));
        }
        Ok(())
    }
}

impl Engine {
    /// Rebuilds the index from the archive and installs it. Returns the
    /// pass the rebuild made, imports answered meanwhile included, and its
    /// stages' records.
    pub(crate) fn build_index(&mut self) -> Result<(Pass, Vec<StageRecord>), EngineError> {
        let mut pass = Pass::default();
        let mut rebuild = Rebuild::start(&self.archive, self.config.clock.clone())?;
        // Messages that name an asset find it when they are assembled: the
        // asset lookup comes from stored observations and sidecars.
        self.refresh_import_assets(&mut pass, Some(&mut rebuild))?;
        let through = self.reads.reader()?.last_input_id()?;
        let workers = self.config.workers.max(1);
        self.log_build_step(
            RebuildStage::Parsing.name(),
            format_args!("{workers} parse workers"),
        );
        let (plans_tx, plans_rx) = crossbeam_channel::bounded::<Plan>(workers * 2);
        let prepare = Prepare {
            archive: self.archive.clone(),
            blobs: Arc::clone(&self.blobs),
            writer: self.writer.sender(),
            settings: pool::ChainSettings {
                read_memory: self.config.reader_memory_per_connection()?,
                yielding: Arc::new(AtomicBool::new(false)),
            },
            admission: Admission::new(self.archive.memory_budget().prepared_in_flight()),
            chunk: self.config.chunk.max(1),
            plans: plans_rx,
            next_run: AtomicU64::new(0),
            worker_memory: self.archive.memory_budget().build_worker_memory(workers),
            chains: AtomicU64::new(0),
            chains_read: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            parsed: AtomicU64::new(0),
            runs: Mutex::new(Vec::new()),
            failed: Mutex::new(Vec::new()),
            left_out: Mutex::new(Vec::new()),
        };
        let prepared = std::thread::scope(|scope| {
            // Each worker holds a sender it never sends on: the channel
            // disconnects when the last worker ends.
            let (running, finished) = crossbeam_channel::bounded::<std::convert::Infallible>(0);
            let prepare = &prepare;
            let handles: Vec<_> = (0..workers)
                .map(|n| {
                    let running = running.clone();
                    std::thread::Builder::new()
                        .name(format!("polylogue-build-{n}"))
                        .spawn_scoped(scope, move || {
                            let _running = running;
                            prepare.work()
                        })
                })
                .collect::<std::io::Result<_>>()?;
            drop(running);
            let mut result = self
                .plan_build(prepare, plans_tx, through, &mut pass)
                .and_then(|()| {
                    self.receive_while_answering(&finished, &mut pass)
                        .map(|_| ())
                });
            if result.is_err() {
                prepare.settings.yielding.store(true, Ordering::Release);
            }
            for handle in handles {
                // A worker catches its own panics; a join failure is one
                // raised while it was unwinding.
                let worked = handle.join().unwrap_or_else(|panic| {
                    Err(EngineError::internal(
                        polylogue_model::cause::panic_message(panic.as_ref()),
                    ))
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
        let mut runs = polylogue_model::unpoisoned(prepare.runs.into_inner());
        runs.sort();
        let failed = polylogue_model::unpoisoned(prepare.failed.into_inner());
        pass.errors
            .extend(polylogue_model::unpoisoned(prepare.left_out.into_inner()));
        pass.bytes_read += prepare.parsed.load(Ordering::Relaxed);
        let parsed = prepare.bytes_read.load(Ordering::Relaxed);
        // The rebuild's later stages run on a thread of their own; requests
        // are answered while they run.
        let (done, finished) = crossbeam_channel::bounded(1);
        let writer = self.writer.sender();
        let runs = runs.into_iter().map(|(_, path)| path).collect();
        let stop = self.archive.stop().clone();
        let built = std::thread::scope(|scope| {
            // The worker must end before the engine can release the archive.
            std::thread::Builder::new()
                .name("polylogue-rebuild".into())
                .spawn_scoped(scope, move || {
                    let built = polylogue_model::caught(|| {
                        polylogue_store::cancellation::under(&stop, || {
                            rebuild.finish(runs, parsed, workers, &writer)
                        })
                    })
                    .unwrap_or_else(|panic| Err(polylogue_store::StoreError::internal(panic)));
                    let _ = done.send(built);
                })?;
            let answer = self.receive_while_answering(&finished, &mut pass)?;
            let answer = answer.ok_or_else(|| {
                polylogue_store::StoreError::internal("the rebuild ended without an answer")
            })?;
            Ok::<_, EngineError>(answer?)
        });
        // Positions read before the rebuild named the build file's empty
        // index; the next pass reads them from the installed one, or, when
        // the rebuild failed, from the build file made again.
        self.discard_confirmed_state();
        let records = built?;
        for (mut plan, failure) in failed {
            self.record_failure(&mut plan, failure, &mut pass);
        }
        pass.messages_written = self.reads.reader()?.status(&[])?.messages;
        Ok((pass, records))
    }

    /// Receives a message from `work`, answering each request that arrives
    /// meanwhile. An empty answer means the sender has disconnected.
    fn receive_while_answering<T>(
        &mut self,
        work: &Receiver<T>,
        pass: &mut Pass,
    ) -> Result<Option<T>, EngineError> {
        loop {
            let requests = self.waiting_requests();
            let mut select = Select::new();
            let ready = select.recv(work);
            if let Some(requests) = &requests {
                select.recv(requests);
            }
            if select.ready() == ready {
                match work.try_recv() {
                    Ok(answer) => return Ok(Some(answer)),
                    Err(TryRecvError::Disconnected) => return Ok(None),
                    Err(TryRecvError::Empty) => continue,
                }
            }
            self.answer_waiting(pass)?;
        }
    }

    /// The channel requests arrive on while none is being answered.
    fn waiting_requests(&self) -> Option<Receiver<LoopMsg>> {
        self.requests
            .as_ref()
            .filter(|requests| !requests.serving)
            .map(|requests| requests.receiver().clone())
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
                if !self.archive.stop().is_cancelled() {
                    self.archive.stop().cancel();
                }
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
        plans: Sender<Plan>,
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
        // The whole build's chains and bytes are counted before the first
        // page is planned, so its progress has its total from the start.
        let counting = std::time::Instant::now();
        let (chains, bytes) = self
            .reads
            .reader()?
            .count_chains_behind_index(&versions, &through)?;
        prepare.chains.store(chains, Ordering::Relaxed);
        prepare.bytes.store(bytes, Ordering::Relaxed);
        prepare.progress();
        polylogue_store::log!(
            info,
            "index: {chains} chains and {:.0} MB to read, counted in {} ms",
            bytes as f64 / 1e6,
            counting.elapsed().as_millis()
        );
        // What planning finds is the exact total: the count includes chains
        // whose recorded refusal planning leaves out, and a file stored
        // meanwhile can have grown.
        let (mut planned_chains, mut planned_bytes) = (0, 0);
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
                prepare.chains.store(planned_chains, Ordering::Relaxed);
                prepare.bytes.store(planned_bytes, Ordering::Relaxed);
                prepare.progress();
                return Ok(());
            };
            let behind = reader.chains_behind_index(
                &versions,
                polylogue_store::read::RevisionChains::All,
                Some(&ids),
            )?;
            drop(reader);
            let mut page = Vec::new();
            for c in behind {
                // The count holds the chains of formats with a reading; one
                // the build plans is read whatever its format says.
                let expected = versions
                    .iter()
                    .any(|(origin, _)| *origin == c.source.format);
                let length = c.stored;
                let plan = self.build_plan(c)?;
                if expected || plan.is_some() {
                    planned_chains += 1;
                    planned_bytes += length;
                    prepare.chains.fetch_max(planned_chains, Ordering::Relaxed);
                    prepare.bytes.fetch_max(planned_bytes, Ordering::Relaxed);
                }
                // A counted chain no parser reads, or one that cannot be
                // read, is done once planning finds it.
                let done = || {
                    prepare.chains_read.fetch_add(1, Ordering::Relaxed);
                    prepare.bytes_read.fetch_add(length, Ordering::Relaxed);
                };
                let Some(plan) = plan else {
                    if expected {
                        done();
                    }
                    continue;
                };
                // What upkeep does before it reads a stored chain: a bundle's
                // authored records and a transcript's local files an
                // interrupted capture left unstored are stored first.
                if let Err(error) = self
                    .restore_captured_authored(&plan, pass)
                    .and_then(|()| self.observe_local_references(&plan, pass))
                {
                    prepare.chain_failed(plan, error)?;
                    done();
                    continue;
                }
                page.push(plan);
            }
            prepare.progress();
            for mut plan in page {
                // A plan waits for room with the workers; requests are
                // answered meanwhile.
                loop {
                    let requests = self.waiting_requests();
                    let mut select = Select::new();
                    let room = select.send(&plans);
                    if let Some(requests) = &requests {
                        select.recv(requests);
                    }
                    if select.ready() != room {
                        self.answer_waiting(pass)?;
                        continue;
                    }
                    match plans.try_send(plan) {
                        Ok(()) => break,
                        Err(TrySendError::Full(back)) => plan = back,
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
            route: if c.imported {
                origin.route.imported()
            } else {
                origin.route
            },
        };
        Ok(Some(self.reparse_plan(
            path,
            origin,
            input,
            c.chain,
            c.stored,
            &polylogue_model::chain::ChainState::Unparsed,
            0,
            None,
            false,
        )?))
    }
}

#[cfg(test)]
mod tests;
