//! A standalone build of the index (docs/architecture.md, one path): every
//! stored chain due a reading is read from its first byte, at the length the
//! record holds when the build starts, by as many workers as the engine
//! parses with. Each worker writes whole chains' readings into run files of
//! its own ([`polylogue_store::writer::build::BuildRun`]); the writer then
//! loads the runs, composes every session once, writes the indexes and
//! search entries and installs the file. Upkeep brings the index forward
//! from there: bytes stored while the build ran are read by the next pass.
use crate::plan::Plan;
use crate::{Engine, EngineError, Pass, has_reading, pool};
use polylogue_store::writer::build::{BuildPhase, BuildRun, PhaseClock};
use polylogue_store::{Change, InputKey};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, sync_channel};
use std::sync::{Arc, Condvar, Mutex};

/// Inputs whose chains are planned per page.
const PLANNED_INPUTS: usize = 256;

/// The input bytes whose prepared readings may be in memory at once. A
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
struct Prepare<'a> {
    archive: &'a polylogue_store::Archive,
    blobs: &'a Arc<polylogue_store::blobs::Blobs>,
    writer: polylogue_store::WriterSender,
    settings: pool::ChainSettings,
    admission: Admission,
    /// The most input bytes one range of a chain reads.
    chunk: u64,
    plans: Mutex<Receiver<Plan>>,
    next_run: AtomicU64,
    chains: AtomicU64,
    chains_read: AtomicU64,
    bytes: AtomicU64,
    bytes_read: AtomicU64,
    runs: Mutex<Vec<(u64, PathBuf)>>,
    errors: Mutex<Vec<String>>,
}

impl Prepare<'_> {
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

    /// One worker. A worker that fails takes the remaining chains off the
    /// queue unread, so the planner never waits on it.
    fn work(&self) -> Result<(), EngineError> {
        let worked = self.work_until_failed();
        if worked.is_err() {
            while let Ok(plans) = self.plans.lock()
                && plans.recv().is_ok()
            {}
        }
        worked
    }

    /// Takes chains until none is left, writing each into the worker's
    /// current run and starting another once that one is full.
    fn work_until_failed(&self) -> Result<(), EngineError> {
        let mut run: Option<(u64, BuildRun)> = None;
        loop {
            let plan = self.plans.lock().map_err(|_| EngineError::Stopped)?.recv();
            let Ok(mut plan) = plan else { break };
            self.archive.stop().check()?;
            let Change::Reparse { chain } = plan.change else {
                continue;
            };
            let (input, length) = (plan.input.clone(), plan.size);
            let admitted = self.admission.admit(length.min(self.chunk), || {
                self.archive.stop().is_cancelled()
            })?;
            let mut read = 0;
            let reading = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pool::run_chain(
                    &mut plan,
                    self.archive,
                    self.blobs,
                    &self.settings,
                    None,
                    &mut read,
                )
            }))
            .unwrap_or_else(|_| Err(EngineError::Parse("the parser panicked".into())));
            let written = match reading {
                Ok(Some(reading)) => {
                    if run.is_none() {
                        let number = self.next_run.fetch_add(1, Ordering::Relaxed);
                        run = Some((number, BuildRun::create(self.archive, number)?));
                    }
                    let (_, current) = run.as_mut().expect("a run was opened");
                    current
                        .write(&self.writer, &input, chain, length, reading)
                        .map_err(EngineError::from)
                }
                Ok(None) => return Err(EngineError::Stopped),
                Err(EngineError::Stopped) => return Err(EngineError::Stopped),
                Err(error) => Err(error),
            };
            drop(admitted);
            // A chain whose reading failed is left out; upkeep reads it
            // after the install and records the failure.
            if let Err(error) = written {
                self.errors
                    .lock()
                    .map_err(|_| EngineError::Stopped)?
                    .push(format!("build: {}: chain {chain}: {error}", input.id));
            }
            self.chains_read.fetch_add(1, Ordering::Relaxed);
            self.bytes_read.fetch_add(length, Ordering::Relaxed);
            self.progress();
            if let Some((_, current)) = &run
                && current.full()?
            {
                self.close(run.take())?;
            }
        }
        self.close(run)
    }

    fn close(&self, run: Option<(u64, BuildRun)>) -> Result<(), EngineError> {
        if let Some((number, run)) = run
            && let Some(path) = run.finish()?
        {
            self.runs
                .lock()
                .map_err(|_| EngineError::Stopped)?
                .push((number, path));
        }
        Ok(())
    }
}

impl Engine {
    /// Builds the index from the stored record and installs it. Returns the
    /// pass the build made and the phases it took.
    pub(crate) fn build_index(&mut self) -> Result<(Pass, Vec<BuildPhase>), EngineError> {
        let mut pass = Pass::default();
        // Messages that name an asset find it when they are composed: the
        // asset lookup comes from stored observations and sidecars.
        self.refresh_import_assets(&mut pass)?;
        let workers = self.config.workers.max(1);
        let (plans_tx, plans_rx) = sync_channel::<Plan>(workers * 2);
        let prepare = Prepare {
            archive: &self.archive,
            blobs: &self.blobs,
            writer: self.writer.sender(),
            settings: pool::ChainSettings {
                stopped_attempt: self.stopped_attempt,
                read_memory: self.config.reader_memory_per_connection()?,
                yielding: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            },
            admission: Admission::new(self.config.bytes_in_flight),
            chunk: self.config.chunk.max(1),
            plans: Mutex::new(plans_rx),
            next_run: AtomicU64::new(0),
            chains: AtomicU64::new(0),
            chains_read: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            runs: Mutex::new(Vec::new()),
            errors: Mutex::new(Vec::new()),
        };
        let clock = PhaseClock::start("prepare");
        let planned = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|n| {
                    std::thread::Builder::new()
                        .name(format!("polylogue-build-{n}"))
                        .spawn_scoped(scope, || prepare.work())
                })
                .collect::<std::io::Result<_>>()?;
            let planned = self.plan_build(&prepare, plans_tx);
            let mut result = planned;
            for handle in handles {
                let worked = handle.join().unwrap_or_else(|_| {
                    Err(std::io::Error::other("a build worker panicked").into())
                });
                if result.is_ok() {
                    result = worked;
                }
            }
            result
        });
        planned?;
        let mut runs = prepare
            .runs
            .into_inner()
            .map_err(|_| EngineError::Stopped)?;
        runs.sort();
        pass.errors.extend(
            prepare
                .errors
                .into_inner()
                .map_err(|_| EngineError::Stopped)?,
        );
        pass.bytes_read = prepare.bytes_read.load(Ordering::Relaxed);
        let mut phases = vec![clock.end(prepare.chains_read.load(Ordering::Relaxed))];
        phases.extend(
            self.writer
                .sender()
                .build(runs.into_iter().map(|(_, path)| path).collect())?,
        );
        // Positions read before the build named the build file's empty
        // index; the next pass reads them from the installed one.
        self.discard_confirmed_state();
        pass.messages_written = self.reads.reader()?.status()?.messages;
        Ok((pass, phases))
    }

    /// Sends every stored chain due a reading to the workers, a page of
    /// inputs at a time. No read is held while a plan waits for a worker:
    /// a worker's decoded bytes are stored through the writer meanwhile.
    fn plan_build(
        &self,
        prepare: &Prepare<'_>,
        plans: std::sync::mpsc::SyncSender<Plan>,
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
        let (through, mut after) = (self.reads.reader()?.last_input_id()?, None::<String>);
        let Some(through) = through else {
            return Ok(());
        };
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
            for c in behind {
                let Some(plan) = self.build_plan(c)? else {
                    continue;
                };
                prepare.chains.fetch_add(1, Ordering::Relaxed);
                prepare.bytes.fetch_add(plan.size, Ordering::Relaxed);
                prepare.progress();
                if plans.send(plan).is_err() {
                    return Ok(());
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
