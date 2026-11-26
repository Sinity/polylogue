//! One pass of the loop: collect, store, parse, derive; and the loop that
//! runs passes.

use crate::capture_read::PlannedFile;
use crate::plan::{InputState, Plan, span, whole};
use crate::*;
use crossbeam_channel::RecvTimeoutError;
use polylogue_sources::export::ExportParser;
use polylogue_store::RevisionChains;

/// The longest a burst of file events delays the next pass.
const COALESCE: Duration = Duration::from_secs(2);
/// Jobs queued in the parse pool per worker; the rest of a pass's plans
/// wait outside it. A job's memory is bounded by the pool's bytes in
/// flight; this bounds how many small inputs are between planning and
/// commit, and so how many the writer can commit in one transaction.
const IN_POOL_PER_WORKER: usize = 64;
/// A known file written this recently is live: the session being written.
const RECENT_NS: i128 = 10 * 60 * 1_000_000_000;

/// Only the next planning page, never parser progress. Committed checkpoints
/// decide which chains in a repeated page still need work.
pub(super) struct StoredPlanning {
    phase: Option<RevisionChains>,
    after: Option<String>,
    through: Option<String>,
    archive_counter: i64,
}

/// The export parser that reads `plan`'s input item by item, when its
/// origin opens the path as an export. Every plan is routed by it before
/// it joins the pool, whose ranges read every other parser.
fn export_parser(plan: &Plan) -> Option<Box<dyn ExportParser>> {
    match (plan.origin.open)(&plan.path) {
        Some(polylogue_sources::Parse::Export(parser)) => Some(parser),
        _ => None,
    }
}

/// The paths of `current` and of the plans after it, which `execute` plans
/// itself, so a pool run meanwhile leaves them to it.
fn later_paths<'a>(current: &'a Plan, rest: &'a std::collections::VecDeque<Plan>) -> Vec<&'a Path> {
    std::iter::once(current)
        .chain(rest)
        .map(|plan| plan.path.as_path())
        .collect()
}

/// Why `origin`'s parser does not read the bytes recorded at `path`, when
/// it does not.
pub(crate) fn unread_path(origin: &OriginSpec, path: &Path) -> Option<String> {
    (origin.support == Support::Parsed && (origin.open)(path).is_none()).then(|| {
        format!(
            "no {} parser reads the recorded path {}",
            origin.name,
            path.display()
        )
    })
}

impl Requests {
    fn new(receiver: crossbeam_channel::Receiver<LoopMsg>) -> Self {
        Self {
            receiver,
            pending: None,
            yielding: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            serving: false,
        }
    }

    pub(crate) fn poll(&mut self) -> bool {
        if self.serving {
            return false;
        }
        if self.pending.is_none() {
            loop {
                match self.receiver.try_recv() {
                    // The watcher owns the paths; this is only its wake-up.
                    Ok(LoopMsg::Changed) => continue,
                    Ok(message) => {
                        self.pending = Some(message);
                        self.yielding
                            .store(true, std::sync::atomic::Ordering::Release);
                        break;
                    }
                    Err(_) => break,
                }
            }
        }
        self.pending.is_some()
    }

    fn recv_timeout(&mut self, timeout: Duration) -> Result<LoopMsg, RecvTimeoutError> {
        self.queued()
            .map_or_else(|| self.receiver.recv_timeout(timeout), Ok)
    }

    fn recv(&mut self) -> Result<LoopMsg, RecvTimeoutError> {
        self.queued().map_or_else(
            || {
                self.receiver
                    .recv()
                    .map_err(|_| RecvTimeoutError::Disconnected)
            },
            Ok,
        )
    }

    fn try_recv(&mut self) -> Result<LoopMsg, crossbeam_channel::TryRecvError> {
        self.queued().map_or_else(|| self.receiver.try_recv(), Ok)
    }

    /// The channel requests arrive on, to wait on beside other work.
    pub(crate) fn receiver(&self) -> &crossbeam_channel::Receiver<LoopMsg> {
        &self.receiver
    }

    fn queued(&mut self) -> Option<LoopMsg> {
        self.pending.take()
    }
}

impl Engine {
    /// Taking a request pauses background parsing, never capture into the spool or
    /// the request currently being answered. Workers and writes drain before
    /// the loop takes the request; the next turn reads committed positions.
    pub(super) fn poll_request(&mut self) -> bool {
        let Some(requests) = &mut self.requests else {
            return false;
        };
        requests.poll()
    }

    pub(super) fn yielding(&self) -> bool {
        self.requests.as_ref().is_some_and(|requests| {
            !requests.serving && requests.yielding.load(std::sync::atomic::Ordering::Acquire)
        })
    }

    /// Fails `plan` with the parser's `failure` of its bytes and records it
    /// as its chain's state, at the position the chain was parsed to.
    /// Recording needs stored bytes; when it cannot be made, the failure
    /// stays the loop's own.
    pub(crate) fn record_failure(
        &mut self,
        plan: &mut Plan,
        failure: polylogue_model::ChainFailure,
        pass: &mut Pass,
    ) {
        pass.errors
            .push(polylogue_model::Failure::new(plan.path.display(), &failure));
        plan.failed = true;
        self.record_stop(plan, failure, pass);
    }

    /// Fails `plan` with `error`. Only a failure of the chain's own bytes is
    /// recorded as its parse state, standing until they change; any other
    /// leaves the chain due, to be read again.
    pub(crate) fn chain_failed(
        &mut self,
        plan: &mut Plan,
        doing: &str,
        error: EngineError,
        pass: &mut Pass,
    ) {
        match error {
            EngineError::Chain(failure) => self.record_failure(plan, failure, pass),
            error => {
                pass.errors
                    .push(polylogue_model::Failure::new(plan.path.display(), &error).doing(doing));
                plan.failed = true;
            }
        }
    }

    /// Records the chain failed at the plan's existing source cursor.
    pub(crate) fn record_stop(
        &mut self,
        plan: &mut Plan,
        failure: polylogue_model::ChainFailure,
        pass: &mut Pass,
    ) {
        if plan.failure_recorded || plan.stored_upto == 0 {
            return;
        }
        match self.send_failure(plan, failure) {
            Ok(()) => plan.failure_recorded = true,
            Err(error) => pass.errors.push(
                polylogue_model::Failure::new(plan.path.display(), &error)
                    .doing("record the failure"),
            ),
        }
    }

    /// Writes the chain's failure at the plan's cursor, at the length the
    /// plan has stored.
    fn send_failure(
        &mut self,
        plan: &Plan,
        failure: polylogue_model::ChainFailure,
    ) -> Result<(), EngineError> {
        let mut step = polylogue_store::ParseStep::failed(plan.parse_pos..plan.parse_pos, failure);
        step.reset = false;
        step.parse_version = plan.origin.parse_version();
        step.prepare();
        let batch = polylogue_store::Batch {
            input: plan.input.clone(),
            // The bytes are stored: the record names the chain they are in.
            change: match plan.change {
                Change::Reparse { chain } => Change::Reparse { chain },
                _ => Change::Unchanged,
            },
            observation: None,
            continues: false,
            new_blob: None,
            blobs: Vec::new(),
            length: plan.stored_upto,
            parse: vec![step],
            capture_read: None,
            release: Default::default(),
        };
        self.writer.send(batch)?;
        match self.flush_writes()?.failed() {
            None => Ok(()),
            Some(error) => Err(error.into()),
        }
    }

    /// Starts writing the repository lookups for the working directories
    /// the index holds, at most once a minute, on their own thread
    /// ([`repo_lookup::Background`]); a later pass collects what it writes.
    fn refresh_lookups(&mut self) -> Result<(), polylogue_model::Failure> {
        if self
            .lookups_at
            .is_some_and(|at| at.elapsed() < std::time::Duration::from_secs(60))
        {
            return Ok(());
        }
        if self.lookups.running() {
            return Ok(());
        }
        let rows = self
            .reads
            .reader()
            .and_then(|r| {
                r.query(
                    "SELECT DISTINCT cwd, cwd_json FROM session WHERE cwd IS NOT NULL",
                    &[],
                )
            })
            .map_err(|e| polylogue_model::Failure::new("repository lookups", &e))?;
        let dir = self
            .config
            .spool_dir(polylogue_sources::repo_lookup::ORIGIN);
        let read_at =
            polylogue_model::Timestamp::from_unix_ms(self.config.clock.now().as_millisecond())
                .map(|t| t.to_string())
                .unwrap_or_default();
        let started = self.lookups.start(rows, dir.clone(), read_at);
        if !matches!(started, Ok(false)) {
            self.lookups_at = Some(Instant::now());
        }
        started.map(|_| ()).map_err(|e| {
            polylogue_model::Failure::new(
                format_args!("repository lookups in {}", dir.display()),
                &EngineError::from(e),
            )
        })
    }

    /// Runs each origin's collector, honoring its declared setting.
    fn run_collectors(&mut self) -> Vec<polylogue_model::Failure> {
        let mut errors = Vec::new();
        for origin in ORIGINS {
            let Some(collector) = &origin.collector else {
                continue;
            };
            if let Some(setting) = collector.setting {
                let on = self
                    .reads
                    .reader()
                    .and_then(|r| r.setting_value(setting, &self.config.machine));
                match on {
                    Ok(Some(polylogue_model::settings::Value::Boolean(true))) => {}
                    Ok(Some(polylogue_model::settings::Value::Boolean(false))) => continue,
                    Ok(_) => {
                        errors.push(polylogue_model::Failure::stated(
                            format_args!("{} collector", origin.name),
                            polylogue_model::Cause::StoredData,
                            polylogue_model::Code::Store,
                            format!("{setting} has no typed boolean value"),
                        ));
                        continue;
                    }
                    Err(e) => {
                        errors.push(polylogue_model::Failure::new(
                            format_args!("{} collector", origin.name),
                            &e,
                        ));
                        continue;
                    }
                }
            }
            let spool = self.config.root(origin.root).join(
                origin
                    .spool_dir()
                    .expect("an origin with a collector has a spool"),
            );
            let paths = polylogue_sources::origin::CollectorPaths {
                home: &self.config.home,
                archive: &self.archive.root,
                spool: &spool,
            };
            if let Err(error) = self.advance_collector(origin.name, collector, &paths) {
                errors.push(polylogue_model::Failure::new(
                    format_args!("{} collector", origin.name),
                    &error,
                ));
                continue;
            }
            errors.extend(self.collectors.run(origin.name, collector, &paths));
        }
        errors
    }

    /// Waits for database reads as well as the current file pass.
    pub fn run_pass(&mut self) -> Result<Pass, EngineError> {
        let started = Instant::now();
        self.planned.clear();
        let (mut total, seen) = match self.one_pass(true, &Default::default()) {
            Ok(done) => done,
            Err(error) => {
                if !matches!(error, EngineError::Stopped) {
                    self.failure_log.pass_ended(
                        &[polylogue_model::Failure::of(&error)],
                        crate::failure_log::Looked {
                            files: &HashSet::new(),
                            walked: false,
                            completed: false,
                        },
                    );
                }
                return Err(error);
            }
        };
        let mut paths: std::collections::BTreeSet<_> = seen.into_iter().collect();
        // A database capture can still be running after all other work ends:
        // pass again with what it wrote.
        while total.capture_pending && self.captures.pending() {
            let completed = self.captures.completed(true);
            total.inputs_seen += completed.inputs_seen;
            total.bytes_stored += completed.bytes_stored;
            total.errors.extend(completed.errors);
            self.refresh_database_progress()
                .map_err(|error| self.failed_read_turn(error))?;
            if completed.finished && self.capture_checked.is_some() {
                self.capture_checked = Some(Instant::now());
            }
            let (next, seen) = self.one_pass(false, &paths)?;
            paths.extend(seen);
            total.add(next);
        }
        total.seconds = started.elapsed().as_secs_f64();
        let planned = std::mem::take(&mut self.planned);
        self.failure_log.pass_ended(
            &total.errors,
            crate::failure_log::Looked {
                files: &planned,
                walked: total.full_scans > 0,
                completed: true,
            },
        );
        Ok(total)
    }

    pub(super) fn discard_confirmed_state(&mut self) {
        self.discard_confirmed_progress();
        self.read_scope = None;
        self.stored_planning = None;
    }

    fn discard_confirmed_progress(&mut self) {
        for state in self.inputs.values_mut() {
            state.discard_confirmed();
        }
        self.watched_loaded = false;
        self.readings_pending = true;
        self.pass_open = true;
        self.capture_checked = None;
        self.collectors_at = None;
        self.lookups.forget();
        self.lookups_at = None;
    }

    /// One pass. An archive without a current index, a new one included, stores the files no earlier pass stored, builds the index
    /// from the record as it then stands, and reads what arrived since.
    fn one_pass(
        &mut self,
        discover: bool,
        paths: &std::collections::BTreeSet<PathBuf>,
    ) -> Result<(Pass, HashSet<PathBuf>), EngineError> {
        self.found_before_build = None;
        if self.archive.index_build().is_none() {
            return self.read_pass(discover, paths);
        }
        self.plan_build_steps();
        let (mut built, mut seen) = self.read_pass(discover, paths)?;
        // Requests are answered as they come and storing goes on, as the
        // build answers them while it reads: a stream of requests never
        // keeps the build from starting.
        while self.answer_waiting(&mut built)? {
            let first = self.found_before_build.take();
            let looked_at: std::collections::BTreeSet<_> = first
                .iter()
                .flat_map(|(_, found)| found.iter().map(|f| f.path.clone()))
                .chain(paths.iter().cloned())
                .collect();
            let (more, again) = self.read_pass(false, &looked_at)?;
            built.add(more);
            seen.extend(again);
            if let Some((true, found)) = first {
                // The walk stays the record of what the sources hold.
                self.found_before_build = Some((true, found));
            }
        }
        if self.archive.stop().is_cancelled() {
            self.found_before_build = None;
            return Ok((built, seen));
        }
        built.capture_pending = false;
        built.add(self.build_index()?.0);
        // The turn after the build looks again at what the storing turn
        // found ([`Engine::read_turn`]). After a walk it is a discovery turn,
        // so a known file the walk did not find is observed gone; otherwise
        // the state of known files is loaded first, so that only the paths
        // looked at count.
        let walked = matches!(self.found_before_build, Some((true, _)));
        if !walked {
            self.load()?;
        }
        let (mut pass, more) = self.read_pass(walked, paths)?;
        // The storing turn counted the files this turn looks at again.
        pass.inputs_seen = 0;
        built.add(pass);
        seen.extend(more);
        Ok((built, seen))
    }

    /// A build's storing turn: the files no earlier pass stored are stored,
    /// the newest first ([`Engine::execute`]), and not read; the build reads
    /// them with every other stored chain. Files already known (`seen`
    /// holds them too) and stored chains wait for the turn after the build.
    fn store_for_build(
        &mut self,
        mut fresh: Vec<Plan>,
        mut pass: Pass,
        seen: HashSet<PathBuf>,
        started: Instant,
    ) -> Result<(Pass, HashSet<PathBuf>), EngineError> {
        let mut bytes = 0;
        for plan in &mut fresh {
            if matches!(plan.source, pool::Bytes::File) {
                plan.storing_only = true;
                if has_reading(plan.origin) {
                    bytes += plan.size;
                }
            }
        }
        // A rollout's capture reads its first record for the session ID that
        // names its input, and compares its bytes with what is stored; its
        // reading is the build's, like every other stored chain's. Each
        // capture adds what it stored to the build's bytes as it is recorded.
        for capture in &mut self.rollout_captures {
            capture.storing_only = true;
        }
        // The build counts what it reads once it starts; until then its
        // progress names the bytes to be stored for it.
        self.archive.record_build_bytes(bytes, 0);
        pass.inputs_changed = (fresh.len() + self.rollout_captures.len()) as u64;
        if !fresh.is_empty() || !self.rollout_captures.is_empty() {
            self.log_build_step(
                crate::build_log::STORING_FILES,
                format_args!(
                    "{} files to store, {} rollouts to capture",
                    fresh.len(),
                    self.rollout_captures.len()
                ),
            );
            self.execute(fresh, &mut pass)?;
        }
        pass.capture_pending |= self.yielding() || self.poll_request();
        self.pass_open = pass.capture_pending;
        pass.seconds = started.elapsed().as_secs_f64();
        Ok((pass, seen))
    }

    fn read_pass(
        &mut self,
        discover: bool,
        paths: &std::collections::BTreeSet<PathBuf>,
    ) -> Result<(Pass, HashSet<PathBuf>), EngineError> {
        if self
            .read_scope
            .as_ref()
            .is_some_and(|scope| scope.check().is_err())
        {
            self.discard_confirmed_state();
        }
        if self.poll_request() {
            return Ok((
                Pass {
                    capture_pending: true,
                    ..Pass::default()
                },
                HashSet::new(),
            ));
        }
        let result = (|| {
            let scope = self.cached_scope()?;
            let result = self.read_turn(
                discover || (self.config.reads_sources && !self.watched_loaded),
                paths,
            )?;
            // A completed reset can race the last SQL read. Its cached progress
            // cannot establish completion in the new scope.
            if let Err(error) = scope.check() {
                if let Some(failure) = polylogue_model::Failure::joined(&result.0.errors) {
                    return Err(polylogue_store::StoreError::Failed(failure).into());
                }
                return Err(error.into());
            }
            Ok(result)
        })();
        result.map_err(|error| match error {
            EngineError::Stopped => EngineError::Stopped,
            error => self.failed_read_turn(error),
        })
    }

    fn failed_read_turn(&mut self, error: EngineError) -> EngineError {
        let scope_changed = matches!(
            &error,
            EngineError::Store(polylogue_store::StoreError::ScopeChanged)
        );
        if scope_changed
            || self
                .read_scope
                .as_ref()
                .is_some_and(|scope| scope.check().is_err())
        {
            self.discard_confirmed_state();
        }
        if scope_changed {
            // A failed reset may leave reads blocked. Its original queued
            // writer error must remain observable instead of an endless wait.
            match self.flush_writes() {
                Ok(report) if let Some(error) = report.failed() => return error.into(),
                Err(error) => return error,
                _ => {}
            }
        }
        error
    }

    fn read_turn(
        &mut self,
        discover: bool,
        paths: &std::collections::BTreeSet<PathBuf>,
    ) -> Result<(Pass, HashSet<PathBuf>), EngineError> {
        let continuing = self.pass_open;
        let started = Instant::now();
        self.load()?;
        if discover && !self.readings_pending {
            self.readings_pending = self.stored_readings_are_behind()?;
        }
        let collectors_due = self
            .collectors_at
            .is_none_or(|at| at.elapsed() >= Duration::from_secs(60));
        let mut errors = Vec::new();
        let sources = self.config.reads_sources;
        // The turn after a build looks again at the files its storing turn
        // found, those the build stored, for what they gained meanwhile; a
        // file that failed is found again by the next walk or change.
        let found_before_build = self.found_before_build.take();
        let walked = sources && discover && found_before_build.is_none();
        if sources && (walked || collectors_due) {
            self.collectors_at = Some(Instant::now());
            errors.extend(self.run_collectors());
            errors.extend(self.refresh_lookups().err());
        }
        let (mut found, unreadable) = if let Some((_, found)) = found_before_build {
            let found = found
                .into_iter()
                .filter(|f| self.inputs.contains_key(&f.path))
                .collect();
            (found, Vec::new())
        } else if !sources {
            (Vec::new(), Vec::new())
        } else if discover {
            let scan = collect::scan_with_coverage(&self.config);
            // Coverage reports the walk; a refused report is this pass's
            // error and the walk's files are still read.
            let observed_at =
                polylogue_model::Timestamp::from_unix_ms(self.config.clock.now().as_millisecond())
                    .ok_or_else(|| {
                        EngineError::internal("capture observation time is out of range")
                    })?;
            if let Err(error) = self.writer.coverage(polylogue_model::capture::Coverage {
                version: 1,
                machine: self.config.machine.clone(),
                observed_at,
                roots: scan.roots,
            }) {
                errors.push(polylogue_model::Failure::new("capture coverage", &error));
            }
            errors.extend(scan.errors);
            (scan.found, scan.unreadable)
        } else {
            collect::scan_paths(&self.config, paths)
        };
        if sources && !discover && collectors_due {
            found.extend(collect::scan_spools(&self.config));
        }
        let mut pass = Pass {
            full_scans: u64::from(walked),
            unreadable_roots: unreadable,
            errors,
            ..Pass::default()
        };
        if walked {
            self.log_build_step(
                crate::build_log::FINDING_FILES,
                format_args!(
                    "found {} files in {:.0} s",
                    found.len(),
                    started.elapsed().as_secs_f64()
                ),
            );
        }
        let (deferred_database, database_changed) = if sources {
            self.capture_databases(&found, discover, &mut pass)?
        } else {
            (false, false)
        };
        let mut found: Vec<_> = found
            .into_iter()
            .filter(|f| !f.origin.is_row_dump(&f.path))
            .collect();
        if sources && !discover && self.live_checked.elapsed() >= self.config.live_every {
            self.live_checked = Instant::now();
            let now_ns = i128::from(self.config.clock.now().as_millisecond()) * 1_000_000;
            found.extend(collect::scan_spools(&self.config));
            found.extend(
                self.inputs
                    .iter()
                    .filter(|(path, st)| {
                        st.origin.arrival == polylogue_sources::Arrival::Live
                            && !st.origin.is_row_dump(path)
                    })
                    .filter(|(_, st)| st.facts.is_some_and(|f| now_ns - f.mtime_ns < RECENT_NS))
                    .map(|(path, st)| collect::Found {
                        path: path.clone(),
                        origin: st.origin,
                    }),
            );
        }
        found.sort_by(|a, b| a.path.cmp(&b.path));
        found.dedup_by(|a, b| a.path == b.path);
        let building = self.archive.index_build().is_some();
        if building {
            self.found_before_build = Some((discover, found.clone()));
        }
        let changed = found.iter().any(|f| {
            self.inputs.get(&f.path).is_none_or(|st| {
                std::fs::metadata(&f.path).ok().map(|m| FileFacts::of(&m)) != st.facts
            })
        });
        let disappeared = !paths.is_empty()
            && self
                .inputs
                .iter()
                .any(|(p, st)| st.facts.is_some() && collect::under(paths, p) && !p.exists());
        if !discover
            && !self.readings_pending
            && !continuing
            && paths.is_empty()
            && !changed
            && !disappeared
            && !pass.capture_pending
            && !database_changed
            && !deferred_database
            && !self.captures.pending()
        {
            pass.seconds = started.elapsed().as_secs_f64();
            return Ok((pass, HashSet::new()));
        }
        // The asset lookup comes from stored observations and sidecars, so
        // after an index rebuild it is in place before any message that
        // names an asset is parsed. The import operation refreshes its new members.
        if !continuing {
            self.refresh_import_assets(&mut pass)?;
        }

        // Files no earlier pass stored are planned and stored first. Their
        // chains are new, so nothing the catch-up of stored chains reads
        // decides them, and capturing them never waits for that backlog.
        // Known files are planned after the catch-up, from the state it leaves.
        let mut seen = HashSet::new();
        let (fresh, found): (Vec<_>, Vec<_>) = found
            .into_iter()
            .partition(|f| !self.inputs.contains_key(&f.path));
        let fresh = self.plan_found(fresh, &mut pass, &mut seen)?;
        if building {
            seen.extend(found.into_iter().map(|f| f.path));
            return self.store_for_build(fresh, pass, seen, started);
        }
        let mut changed_inputs = (fresh.len() + self.rollout_captures.len()) as u64;
        if !fresh.is_empty() || !self.rollout_captures.is_empty() {
            self.log_build_step(
                "storing and reading new files",
                format_args!(
                    "{} files to store, {} rollouts to capture",
                    fresh.len(),
                    self.rollout_captures.len()
                ),
            );
            self.execute(fresh, &mut pass)?;
        }
        // Known files that changed since they were read (sessions being
        // written) are read next, before the catch-up of stored chains;
        // unchanged ones are planned after it, from the state it leaves.
        let (changed_known, found): (Vec<_>, Vec<_>) = found.into_iter().partition(|f| {
            self.inputs
                .get(&f.path)
                .is_some_and(|st| changed_since_read(&f.path, st))
        });
        let mut changed_plans = self.plan_found(changed_known, &mut pass, &mut seen)?;
        let mut planned = HashSet::new();
        changed_plans.retain(|plan| planned.insert(plan.input.id.clone()));
        if !changed_plans.is_empty() || !self.rollout_captures.is_empty() {
            changed_inputs += (changed_plans.len() + self.rollout_captures.len()) as u64;
            self.log_build_step(
                "storing and reading changed files",
                format_args!(
                    "{} files to store, {} rollouts to capture",
                    changed_plans.len(),
                    self.rollout_captures.len()
                ),
            );
            self.execute(changed_plans, &mut pass)?;
        }
        let mut live_waiting = false;
        if self.readings_pending && !self.yielding() && !self.archive.stop().is_cancelled() {
            self.log_build_step(
                "reading the newest stored files",
                format_args!(
                    "{:.0} MB stored, {:.0} MB read so far",
                    pass.bytes_stored as f64 / 1e6,
                    pass.bytes_read as f64 / 1e6
                ),
            );
            live_waiting = self.reparse_behind(RevisionChains::Newest, &mut pass)?;
        }
        if self.yielding() || live_waiting {
            // The files found and not yet planned go to the next pass.
            seen.extend(found.into_iter().map(|f| f.path));
            pass.inputs_changed = changed_inputs;
            pass.capture_pending = true;
            pass.seconds = started.elapsed().as_secs_f64();
            return Ok((pass, seen));
        }
        let mut plans = self.plan_found(found, &mut pass, &mut seen)?;
        // Keep stored parsing reachable when a provider disappears.
        let unseen: Vec<PathBuf> = self
            .inputs
            .iter()
            .filter(|(p, state)| {
                sources
                    && (discover || state.origin.is_row_dump(p) || collect::under(paths, p))
                    && !seen.contains(*p)
            })
            .map(|(path, _)| path.clone())
            .collect();
        for path in unseen {
            let state = &self.inputs[&path];
            if sources && state.facts.is_some() && !path.try_exists()? {
                let batch = polylogue_store::Batch {
                    input: InputKey {
                        id: state.input_id.clone(),
                        format: state.origin.name.into(),
                        route: state.origin.route,
                    },
                    change: Change::Disappeared,
                    observation: Some(self.observation(&path, None)),
                    continues: false,
                    new_blob: None,
                    blobs: vec![],
                    length: state.stored,
                    parse: Vec::new(),
                    capture_read: None,
                    release: Default::default(),
                };
                self.inputs.get_mut(&path).expect("known input").facts = None;
                self.writer.send(batch)?;
            }
            if let Some(plan) = self.stored_plan(&path)? {
                seen.insert(path.clone());
                plans.push(plan);
            }
        }
        // One input is read by one plan at a time. A provider-named input
        // found at one path and remembered at another keeps the plan of the
        // path found.
        let mut planned = HashSet::new();
        plans.retain(|plan| planned.insert(plan.input.id.clone()));
        changed_inputs += (plans.len() + self.rollout_captures.len()) as u64;
        pass.inputs_changed = changed_inputs;
        if plans.is_empty() && self.rollout_captures.is_empty() {
            let report = self.flush_writes()?;
            pass.errors.extend(report.errors);
        }
        if sources && self.archive.index_build().is_some() {
            let versions = crate::reading_contracts()
                .into_iter()
                .filter_map(|contract| match contract {
                    polylogue_store::ReadingContract::Expected { origin, version } => {
                        Some((origin, version))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            let (mut bytes, mut read) = self.reads.reader()?.newest_parse_bytes(&versions)?;
            for plan in &plans {
                if matches!(plan.source, pool::Bytes::File) && has_reading(plan.origin) {
                    let old = self.inputs.get(&plan.path);
                    bytes = bytes
                        .saturating_sub(old.map_or(0, |old| old.stored))
                        .saturating_add(plan.size);
                    if plan.parse_pos == 0 {
                        read = read.saturating_sub(old.map_or(0, |old| old.parsed_upto));
                    }
                }
            }
            self.archive.record_build_bytes(bytes, read);
        }
        if !plans.is_empty() || !self.rollout_captures.is_empty() {
            self.log_build_step(
                "storing and reading changed files",
                format_args!(
                    "{} files to store, {} rollouts to capture",
                    plans.len(),
                    self.rollout_captures.len()
                ),
            );
        }
        self.execute(plans, &mut pass)?;
        if self.yielding() {
            pass.capture_pending = true;
            pass.seconds = started.elapsed().as_secs_f64();
            return Ok((pass, seen));
        }
        if self.archive.stop().is_cancelled() {
            pass.seconds = started.elapsed().as_secs_f64();
            return Ok((pass, seen));
        }
        if self.readings_pending {
            // Every input's newest chain is read: a first build ends here, so
            // its read indexes exist while the older chains are read.
            if !pass.capture_pending && !self.captures.pending() {
                self.log_build_step(
                    "finishing the index",
                    format_args!(
                        "{:.0} MB stored, {:.0} MB read",
                        pass.bytes_stored as f64 / 1e6,
                        pass.bytes_read as f64 / 1e6
                    ),
                );
                pass.sessions_derived += self.derive()?;
            }
            if !self.poll_request() {
                self.log_build_step(
                    "reading older stored files",
                    format_args!("{} sessions derived", pass.sessions_derived),
                );
                live_waiting = self.reparse_behind(RevisionChains::Older, &mut pass)?;
            }
        }
        if self.yielding() || live_waiting {
            pass.capture_pending = true;
            pass.seconds = started.elapsed().as_secs_f64();
            return Ok((pass, seen));
        }
        if self.poll_request() {
            pass.capture_pending = true;
            pass.seconds = started.elapsed().as_secs_f64();
            return Ok((pass, seen));
        }
        if deferred_database && pass.errors.is_empty() && !self.captures.pending() {
            // Local parsing may have made a deferred database runnable. Start
            // that real work in this batch; a still-pending reading just yields.
            self.capture_databases(&[], false, &mut pass)?;
        }
        pass.capture_pending |= self.captures.pending();
        if !pass.capture_pending {
            self.log_build_step(
                "finishing the index",
                format_args!(
                    "{:.0} MB stored, {:.0} MB read",
                    pass.bytes_stored as f64 / 1e6,
                    pass.bytes_read as f64 / 1e6
                ),
            );
            pass.sessions_derived += self.derive()?;
            if pass.errors.is_empty() {
                // A source admitted behind the bookmark gets the next pass;
                // new keys cannot extend this pass's upper bound forever.
                let changed = if let Some(planning) = &self.stored_planning {
                    self.reads.reader()?.change_position()?.archive != planning.archive_counter
                } else {
                    false
                };
                let again = changed && self.stored_readings_are_behind()?;
                self.readings_pending = again;
                pass.capture_pending |= again;
                self.stored_planning = None;
            }
        }
        if self.archive.index_build().is_none()
            && let Some((step, began)) = self.build_step.take()
        {
            polylogue_store::log!(
                info,
                "index: installed; {step} took {:.0} s",
                began.elapsed().as_secs_f64()
            );
        }
        self.pass_open = pass.capture_pending;
        pass.seconds = started.elapsed().as_secs_f64();
        Ok((pass, seen))
    }

    /// Plans for found files; `seen` collects their paths.
    fn plan_found(
        &mut self,
        found: Vec<collect::Found>,
        pass: &mut Pass,
        seen: &mut HashSet<PathBuf>,
    ) -> Result<Vec<Plan>, EngineError> {
        let mut plans = Vec::new();
        for f in found {
            seen.insert(f.path.clone());
            let Ok(meta) = std::fs::metadata(&f.path) else {
                continue;
            };
            pass.inputs_seen += 1;
            let now = FileFacts::of(&meta);
            if compressed::is_rollout(&f) {
                match self.rollout_step(&f, now, Priority::Bulk) {
                    Ok(compressed::RolloutStep::Planned(Some(plan))) => {
                        let plan = *plan;
                        if let Some(previous) = plans
                            .iter_mut()
                            .find(|p: &&mut Plan| p.input.id == plan.input.id)
                        {
                            *previous = plan;
                        } else {
                            plans.push(plan);
                        }
                    }
                    Ok(compressed::RolloutStep::Planned(None)) => {}
                    Ok(compressed::RolloutStep::Capture(capture)) => {
                        self.rollout_captures.push_back(*capture);
                    }
                    Err(error) => pass.errors.push(
                        polylogue_model::Failure::new(f.path.display(), &error).doing("capture"),
                    ),
                }
                continue;
            }
            if !self.track(&f) {
                continue;
            }
            if let Some(plan) = self.plan_file(&f.path, now, pass)? {
                self.planned.insert(f.path.clone());
                plans.push(plan);
            }
        }
        // The newest sessions are stored, and so readable, first.
        self.rollout_captures
            .make_contiguous()
            .sort_by_key(|capture| std::cmp::Reverse(capture.facts.mtime_ns));
        Ok(plans)
    }

    /// Reads the path's committed progress again, after a write of it failed.
    pub(crate) fn refresh_confirmed_input(&mut self, path: &Path) -> Result<(), EngineError> {
        let known = self
            .reads
            .reader()?
            .known_watched_input_at(&self.config.machine, path.as_os_str().as_encoded_bytes())?;
        if let Some(known) = known {
            self.adopt_watched_input(path, known);
        } else if let Some(state) = self.inputs.get_mut(path) {
            state.adopt_confirmed(&state.input_id.clone(), state.origin, None, None);
        }
        Ok(())
    }

    /// Starts tracking a found path its origin recognizes; false when it
    /// does not.
    fn track(&mut self, f: &collect::Found) -> bool {
        if self.inputs.contains_key(&f.path) {
            return true;
        }
        if !collect::recognize(f) {
            return false;
        }
        let id = self.location_id(&f.path);
        self.inputs
            .insert(f.path.clone(), InputState::new(id, f.origin));
        true
    }

    /// Plans for live inputs that changed since the pass began: every file
    /// in the spools, and the known files of live origins written in the
    /// last minutes (a session being written). They run ahead of the
    /// pass's bulk work. Paths with work in flight are left to it.
    pub(crate) fn live_plans(
        &mut self,
        busy: &HashSet<&Path>,
        pass: &mut Pass,
    ) -> Result<Vec<Plan>, EngineError> {
        if !self.config.reads_sources {
            return Ok(Vec::new());
        }
        let mut changes = watch::Changes::default();
        if let Some(watcher) = &mut self.watcher {
            if let Err(error) = watcher.refresh(&self.config) {
                pass.errors.push(polylogue_model::Failure::new(
                    "source watch",
                    &EngineError::from(error),
                ));
                changes.full = true;
            }
            let latest = watcher.take();
            changes.full |= latest.full;
            changes.paths = latest.paths;
            pass.errors.extend(latest.errors);
        }
        let paths = changes
            .paths
            .into_iter()
            .map(|path| collect::database_path(&self.config, &path).unwrap_or(path))
            .collect();
        let (mut found, unreadable) = if changes.full {
            pass.full_scans += 1;
            collect::scan(&self.config)
        } else {
            collect::scan_paths(&self.config, &paths)
        };
        pass.unreadable_roots.extend(unreadable);
        self.capture_databases(&found, changes.full, pass)?;
        found.retain(|file| !file.origin.is_row_dump(&file.path));
        let now_ns = i128::from(self.config.clock.now().as_millisecond()) * 1_000_000;
        found.extend(collect::scan_spools(&self.config));
        found.extend(
            self.inputs
                .iter()
                .filter(|(path, st)| {
                    st.origin.arrival == polylogue_sources::Arrival::Live
                        && !st.origin.is_row_dump(path)
                })
                .filter(|(_, st)| st.facts.is_some_and(|f| now_ns - f.mtime_ns < RECENT_NS))
                .map(|(path, st)| collect::Found {
                    path: path.clone(),
                    origin: st.origin,
                }),
        );
        found.sort_by(|left, right| left.path.cmp(&right.path));
        found.dedup_by(|left, right| left.path == right.path);
        let mut plans = Vec::new();
        let mut seen = HashSet::new();
        for f in &found {
            if busy.contains(f.path.as_path())
                || self.reparsing.contains_key(&f.path)
                || !seen.insert(f.path.as_path())
            {
                continue;
            }
            let Ok(meta) = std::fs::metadata(&f.path) else {
                continue;
            };
            let now = FileFacts::of(&meta);
            if compressed::is_rollout(f) {
                match self.rollout_step(f, now, Priority::Live) {
                    // Paths already queued or being captured are busy.
                    Ok(compressed::RolloutStep::Capture(capture)) => {
                        self.rollout_captures.push_front(*capture);
                    }
                    Ok(compressed::RolloutStep::Planned(Some(plan))) => {
                        let mut plan = *plan;
                        plan.priority = Priority::Live;
                        if let Some(previous) = plans
                            .iter_mut()
                            .find(|p: &&mut Plan| p.input.id == plan.input.id)
                        {
                            *previous = plan;
                        } else {
                            plans.push(plan);
                        }
                    }
                    Ok(compressed::RolloutStep::Planned(None)) => {}
                    Err(error) => pass.errors.push(
                        polylogue_model::Failure::new(f.path.display(), &error).doing("capture"),
                    ),
                }
                continue;
            }
            if !self.track(f) {
                continue;
            }
            // Every recently written live file is checked on each tick. An
            // unchanged, fully stored one with no capture waiting can only
            // have an unfinished reading left, which needs no source state
            // from the writer; asking the writer for each one made a pass
            // over many recent files quadratic.
            let st = &self.inputs[&f.path];
            let plan = if st.facts == Some(now) && st.stored >= now.size {
                self.stored_plan(&f.path)?
            } else {
                self.plan_file(&f.path, now, pass)?
            };
            if let Some(mut plan) = plan {
                plan.priority = Priority::Live;
                plans.push(plan);
            }
        }
        let databases: Vec<_> = self
            .inputs
            .iter()
            .filter(|(path, state)| {
                state.origin.is_row_dump(path) && !busy.contains(path.as_path())
            })
            .map(|(path, _)| path.clone())
            .collect();
        for path in databases {
            if let Some(mut plan) = self.stored_plan(&path)? {
                plan.priority = Priority::Live;
                plans.push(plan);
            }
        }
        Ok(plans)
    }

    /// The chain a file holds fewer bytes of, each equal to the stored byte
    /// at its place: its input's current chain when it is longer than the
    /// file and begins with the file's bytes.
    fn shorter_prefix(&mut self, path: &Path, now: FileFacts) -> Result<Option<i64>, EngineError> {
        let st = &self.inputs[path];
        let reader = self.reads.reader()?;
        let Some(progress) = reader.latest_chain(&st.input_id)? else {
            return Ok(None);
        };
        if now.size == 0 || progress.stored <= now.size || progress.stored < st.stored {
            return Ok(None);
        }
        let intact = reader
            .stored_chain(&st.input_id, progress.chain, now.size)
            .ok()
            .is_some_and(|chain| {
                read::stored_prefix_intact(path, &chain, &self.blobs).unwrap_or(false)
            });
        Ok(intact.then_some(progress.chain))
    }

    /// Records that a file came back holding a prefix of its stored chain:
    /// an observation with the length it holds, no revision and no chain.
    fn observe_shorter(
        &mut self,
        path: &Path,
        now: FileFacts,
        chain: i64,
    ) -> Result<Option<Plan>, EngineError> {
        let state = &self.inputs[path];
        let batch = polylogue_store::Batch {
            input: InputKey {
                id: state.input_id.clone(),
                format: state.origin.name.into(),
                route: state.origin.route,
            },
            change: Change::Shorter { chain },
            observation: Some(self.observation(path, Some(now))),
            continues: false,
            new_blob: None,
            blobs: vec![],
            length: now.size,
            parse: Vec::new(),
            capture_read: None,
            release: Default::default(),
        };
        self.writer.send(batch)?;
        let st = self.inputs.get_mut(path).expect("tracked input");
        st.facts = Some(now);
        st.unread = None;
        Ok(None)
    }

    /// Records that a file came back holding exactly its stored chain (after
    /// it was gone, or under another inode): an observation, no revision.
    fn observe_back(&mut self, path: &Path, now: FileFacts) -> Result<Option<Plan>, EngineError> {
        let state = &self.inputs[path];
        let batch = polylogue_store::Batch {
            input: InputKey {
                id: state.input_id.clone(),
                format: state.origin.name.into(),
                route: state.origin.route,
            },
            change: Change::Unchanged,
            observation: Some(self.observation(path, Some(now))),
            continues: false,
            new_blob: None,
            blobs: vec![],
            length: state.stored,
            parse: Vec::new(),
            capture_read: None,
            release: Default::default(),
        };
        self.writer.send(batch)?;
        let st = self.inputs.get_mut(path).expect("tracked input");
        st.facts = Some(now);
        st.unread = None;
        self.stored_plan(path)
    }

    /// Plans one tracked path. A file that cannot be read while it is
    /// compared with what is stored (gone, or replaced meanwhile) is that
    /// file's error and is looked at again next pass; a store failure stays
    /// the pass's.
    fn plan_file(
        &mut self,
        path: &Path,
        now: FileFacts,
        pass: &mut Pass,
    ) -> Result<Option<Plan>, EngineError> {
        match self.plan(path, now) {
            Err(error @ EngineError::Io(_)) => {
                pass.errors
                    .push(polylogue_model::Failure::new(path.display(), &error).doing("plan"));
                Ok(None)
            }
            planned => planned,
        }
    }

    /// Decides what to read for one tracked path.
    fn plan(&mut self, path: &Path, now: FileFacts) -> Result<Option<Plan>, EngineError> {
        let origin = self.inputs[path].origin;
        let source = self
            .source_read_context(
                path,
                origin.name,
                polylogue_store::capture_read::ReadRoute::NativeCapture,
            )
            .source;
        let snapshot = self.source_snapshot(source)?;
        let source_uncertain =
            snapshot.status != polylogue_store::capture_read::SourceStatus::Ready;
        let obs = self.observation(path, Some(now));
        let st = self.inputs.get_mut(path).expect("tracked input");
        if st.unread == Some(now) {
            return Ok(None);
        }
        let parsed = has_reading(st.origin);
        let mut change = polylogue_store::blobs::classify(st.facts, Some(now));
        let mut back = false;
        if matches!(change, Change::First | Change::Rewritten)
            && !source_uncertain
            && let Some(chain) = self.shorter_prefix(path, now)?
        {
            return self.observe_shorter(path, now, chain);
        }
        let st = self.inputs.get_mut(path).expect("tracked input");
        if matches!(change, Change::First | Change::Rewritten) && !source_uncertain {
            // Another inode at the path, or the file back after it was gone:
            // when it still begins with every stored byte, it continues its
            // chain (observed back, or appended) instead of starting one.
            let reader = self.reads.reader()?;
            if let Some(progress) = reader.latest_chain(&st.input_id)?
                && progress.stored > 0
                && progress.stored <= now.size
                // An index behind this engine's own stored length has
                // writes still to commit; continuing from it would start
                // the append before them.
                && progress.stored >= st.stored
                // A file or stored chain that cannot be read is not shown to
                // hold the stored bytes; the file then starts a new chain.
                && reader
                    .stored_chain(&st.input_id, progress.chain, progress.stored)
                    .ok()
                    .is_some_and(|chain| {
                        read::stored_prefix_intact(path, &chain, &self.blobs).unwrap_or(false)
                    })
            {
                if st.chain != Some(progress.chain) {
                    // The chain this path last read may not be the input's
                    // latest (another path of a provider-named input read a
                    // later one): take that chain's parser position, and its
                    // state from its committed checkpoint below.
                    (st.parsed_upto, st.parsed, st.waiting) =
                        crate::plan::cursor(&progress.state, progress.stored);
                }
                st.stored = progress.stored;
                st.chain = Some(progress.chain);
                if progress.stored == now.size {
                    back = true;
                    change = Change::Unchanged;
                } else {
                    change = Change::Appended {
                        from: progress.stored,
                    };
                }
            }
        }
        if change == Change::Unchanged && st.facts != Some(now) && !back {
            st.facts = Some(now);
        }
        if back {
            return self.observe_back(path, now);
        }
        if change == Change::Unchanged && st.stored < now.size {
            // Stored only in part (a stop in the middle of a large file).
            change = Change::Appended { from: st.stored };
        }
        let mut proven_tail = None;
        if let Change::Appended { from } = &mut change {
            // The source observation may describe a larger file than the
            // committed prefix after an interrupted import.
            *from = st.stored;
            let tail = if *from > 0 {
                let reader = self.reads.reader()?;
                let progress = reader.latest_chain(&st.input_id)?.ok_or_else(|| {
                    polylogue_store::StoreError::invalid("append has no stored prefix")
                })?;
                reader.stored_range(
                    &st.input_id,
                    progress.chain,
                    from.saturating_sub(read::TAIL as u64),
                    *from,
                    &self.blobs,
                )?
            } else {
                Vec::new()
            };
            if read::tail_intact(path, *from, &tail)? {
                proven_tail = Some(tail);
            } else {
                change = Change::Rewritten;
            }
        }
        let never_parsed = parsed && !st.parsed;
        let (stored_upto, parse_pos, parser_state, priority) = match &change {
            Change::First | Change::Rewritten => (0, 0, None, Priority::Bulk),
            Change::Appended { .. } => (
                st.stored,
                if !parsed {
                    st.stored
                } else if never_parsed {
                    0
                } else {
                    st.parsed_upto
                },
                None,
                Priority::Live,
            ),
            Change::Unchanged | Change::Disappeared => return self.stored_plan(path),
            Change::Reparse { .. } | Change::Observed { .. } | Change::Shorter { .. } => {
                unreachable!()
            }
        };
        // A document is parsed whole: any change parses it from its start.
        let (parse_pos, parser_state) = if whole(st.origin, path) {
            (0, None)
        } else {
            (parse_pos, parser_state)
        };
        let parser_chain = match change {
            Change::Appended { .. } => st.chain,
            _ => None,
        };
        if parsed && let Some(build) = self.archive.index_build() {
            let bytes = build
                .bytes
                .saturating_sub(st.stored)
                .saturating_add(now.size);
            let read = if parse_pos == 0 {
                build.bytes_read.saturating_sub(st.parsed_upto)
            } else {
                build.bytes_read
            };
            self.archive.record_build_bytes(bytes, read);
        }
        Ok(Some(Plan {
            path: path.to_path_buf(),
            origin: st.origin,
            input: InputKey {
                id: st.input_id.clone(),
                format: st.origin.name.into(),
                route: st.origin.route,
            },
            change,
            observation: Some(obs),
            facts: Some(now),
            size: now.size,
            stored_upto,
            parse_pos,
            storing_started: false,
            storing_only: source_uncertain || never_parsed && st.stored > 0,
            parser_state,
            parser_chain,
            tail: proven_tail,
            span: span(st.origin, path, self.config.chunk),
            chunk: self.config.chunk,
            priority,
            tracked: true,
            failed: false,
            failure_recorded: false,
            failed_at: None,
            replace: false,
            waiting: false,
            input_finished: false,
            source: pool::Bytes::File,
            read_to: parse_pos,
        }))
    }

    /// Parses again, from what the archive stored, every revision chain of
    /// `chains` the index is behind, except the current chain of a watched
    /// location, which follows its file ([`Engine::stored_plan`]). The same
    /// loop builds the index from stored bytes as from files, so a rebuild
    /// needs no command of its own; older chains go through the same
    /// parser and writer, where a newer chain wins. A pass reads every
    /// input's newest chain before any older chain, so a first build ends
    /// once the newest chains are read.
    ///
    /// Replacements commit each chain once. Continuing readings run in
    /// rounds with one range plan per path, preserving each path's order.
    ///
    /// True when it stopped after a page because a live input changed: the
    /// pass ends so the next one reads that input first, and this catch-up
    /// resumes after the last page it finished.
    pub(crate) fn reparse_behind(
        &mut self,
        chains: RevisionChains,
        pass: &mut Pass,
    ) -> Result<bool, EngineError> {
        let versions = crate::reading_contracts()
            .into_iter()
            .filter_map(|contract| match contract {
                polylogue_store::ReadingContract::Expected { origin, version } => {
                    Some((origin, version))
                }
                polylogue_store::ReadingContract::Kept { .. }
                | polylogue_store::ReadingContract::RawOnly { .. } => None,
            })
            .collect::<Vec<_>>();
        if chains == RevisionChains::Newest
            && self
                .stored_planning
                .as_ref()
                .is_some_and(|planning| planning.phase == Some(RevisionChains::Older))
        {
            return Ok(false);
        }
        if self
            .stored_planning
            .as_ref()
            .is_none_or(|planning| planning.phase != Some(chains))
        {
            let reader = self.reads.reader()?;
            let archive_counter = reader.change_position()?.archive;
            let through = reader.last_input_id()?;
            if self.archive.index_build().is_some() && chains == RevisionChains::Newest {
                let (bytes, read) = reader.newest_parse_bytes(&versions)?;
                self.archive.record_build_bytes(bytes, read);
            }
            self.stored_planning = Some(StoredPlanning {
                phase: Some(chains),
                after: None,
                through,
                archive_counter,
            });
        }
        loop {
            if self.archive.stop().is_cancelled() || self.poll_request() {
                return Ok(false);
            }
            let planning = self.stored_planning.as_ref().expect("planning started");
            let ids = if let Some(through) = &planning.through {
                self.reads
                    .reader()?
                    .input_ids_after(
                        planning.after.as_deref(),
                        self.config.workers.max(1) * IN_POOL_PER_WORKER,
                    )?
                    .into_iter()
                    .filter(|input| input <= through)
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            let Some(last) = ids.last().cloned() else {
                let planning = self.stored_planning.as_mut().expect("planning started");
                planning.phase = if chains == RevisionChains::Newest && !pass.capture_pending {
                    Some(RevisionChains::Older)
                } else {
                    None
                };
                planning.after = None;
                return Ok(false);
            };
            let behind = self
                .reads
                .reader()?
                .chains_behind_index(&versions, chains, Some(&ids))?;
            self.reparse_chains(behind, pass)?;
            if self.yielding() || self.archive.stop().is_cancelled() {
                return Ok(false);
            }
            self.stored_planning
                .as_mut()
                .expect("planning started")
                .after = Some(last);
            if self.live_input_changed() {
                return Ok(true);
            }
        }
    }

    /// Whether a live input written in the last minutes changed since it
    /// was read. Bulk work stops for it at its next page boundary.
    fn live_input_changed(&self) -> bool {
        if !self.config.reads_sources {
            return false;
        }
        let now_ns = i128::from(self.config.clock.now().as_millisecond()) * 1_000_000;
        self.inputs.iter().any(|(path, st)| {
            st.origin.arrival == polylogue_sources::Arrival::Live
                && !st.origin.is_row_dump(path)
                && st.facts.is_some_and(|f| now_ns - f.mtime_ns < RECENT_NS)
                && changed_since_read(path, st)
        })
    }

    fn stored_readings_are_behind(&self) -> Result<bool, EngineError> {
        use polylogue_model::freshness::{CompletenessDimension, CompletenessReason};
        Ok(
            match self
                .reads
                .reader()?
                .completeness(&crate::reading_contracts())?
                .parsing
            {
                CompletenessDimension::Incomplete(reasons)
                | CompletenessDimension::Unknown(reasons) => reasons.into_iter().any(|reason| {
                    matches!(
                        reason,
                        CompletenessReason::MissingCheckpoint
                            | CompletenessReason::ParseVersionChanged
                            | CompletenessReason::ParserBehind
                    )
                }),
                CompletenessDimension::Complete | CompletenessDimension::NotApplicable => false,
            },
        )
    }

    fn reparse_chains(
        &mut self,
        behind: Vec<polylogue_store::read::BehindChain>,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        let mut replacements = Vec::new();
        let mut parse = Vec::new();
        let watched: HashMap<String, PathBuf> = self
            .inputs
            .iter()
            .map(|(path, st)| (st.input_id.clone(), path.clone()))
            .collect();
        for c in behind {
            if self.poll_request() {
                break;
            }
            let stored_source = c.imported || !self.config.reads_sources;
            let route = |origin: &'static OriginSpec| {
                if c.imported {
                    origin.route.imported()
                } else {
                    origin.route
                }
            };
            // Where the parse starts: on from the stored state, or from the
            // first byte replacing the parse whole.
            let from = match c.due {
                polylogue_model::chain::Due::Extend { from } => Some(from),
                _ => None,
            };
            let (path, origin) = if stored_source {
                let source = &c.source;
                let Some(origin) = polylogue_sources::origin::by_name(&source.format)
                    .filter(|origin| has_reading(origin))
                else {
                    continue;
                };
                // The recorded path chooses the parser. Without one, this
                // input is refused with its reason and the pass goes on.
                let recorded_path = if !c.imported {
                    use std::os::unix::ffi::OsStringExt;
                    source
                        .path_bytes
                        .clone()
                        .map(|bytes| PathBuf::from(std::ffi::OsString::from_vec(bytes)))
                        .or_else(|| source.path.clone().map(PathBuf::from))
                } else {
                    source.path.clone().map(PathBuf::from)
                };
                let path = match recorded_path {
                    None => Err((
                        PathBuf::new(),
                        "the stored input has no recorded path".to_owned(),
                    )),
                    Some(path) => match unread_path(origin, &path) {
                        Some(reason) => Err((path, reason)),
                        None => Ok(path),
                    },
                };
                match path {
                    Ok(path) => (path, origin),
                    Err((path, reason)) => {
                        let mut plan = self.reparse_plan(
                            path,
                            origin,
                            InputKey {
                                id: c.input.clone(),
                                format: origin.name.into(),
                                route: route(origin),
                            },
                            c.chain,
                            c.stored,
                            &c.state,
                            c.state.parsed_to(),
                            None,
                            false,
                        )?;
                        pass.errors.push(refused_input(&c.input, &reason));
                        self.record_stop(
                            &mut plan,
                            polylogue_model::ChainFailure::refused(reason),
                            pass,
                        );
                        continue;
                    }
                }
            } else {
                let Some(path) = watched.get(&c.input) else {
                    continue;
                };
                let st = &self.inputs[path];
                // The current chain of a watched file follows the file
                // ([`Engine::stored_plan`]) unless it is read again whole.
                if c.current && (from.is_some() || st.unread.is_some()) {
                    continue;
                }
                (path.clone(), st.origin)
            };
            if !has_reading(origin) {
                continue;
            }
            let tracked = self.config.reads_sources
                && c.current
                && !c.imported
                && watched.get(&c.input) == Some(&path);
            let facts = tracked
                .then(|| self.inputs.get(&path).and_then(|st| st.facts))
                .flatten();
            let mut plan = self.reparse_plan(
                path,
                origin,
                InputKey {
                    id: c.input.clone(),
                    format: origin.name.into(),
                    route: route(origin),
                },
                c.chain,
                c.stored,
                &c.state,
                from.unwrap_or(0),
                facts,
                tracked,
            )?;
            plan.input_finished = c.imported;
            if c.current && origin.arrival == polylogue_sources::Arrival::Live {
                plan.priority = Priority::Live;
            }
            if from.is_none() && c.state != polylogue_model::chain::ChainState::Unparsed {
                plan.replace = true;
                replacements.push(plan);
                continue;
            }
            parse.push(plan);
        }
        self.execute_pool(Vec::new(), replacements, &[], pass)?;
        if self.yielding() {
            return Ok(());
        }
        self.execute_by_path(parse, pass)
    }

    /// Runs plans of stored chains on the parse pool together, in rounds that
    /// hold one chain per path, since the pool runs one plan per path at a
    /// time; within a path, plans keep their order.
    fn execute_by_path(&mut self, plans: Vec<Plan>, pass: &mut Pass) -> Result<(), EngineError> {
        let mut rounds: Vec<Vec<Plan>> = Vec::new();
        let mut chains_per_path: HashMap<PathBuf, usize> = HashMap::new();
        for plan in plans {
            let round = chains_per_path.entry(plan.path.clone()).or_default();
            if *round == rounds.len() {
                rounds.push(Vec::new());
            }
            rounds[*round].push(plan);
            *round += 1;
        }
        for round in rounds {
            if self.poll_request() {
                break;
            }
            self.execute(round, pass)?;
            if self.yielding() {
                break;
            }
        }
        Ok(())
    }

    /// The plan for a watched location whose current chain the index is
    /// behind while its file is gone, moved or unchanged.
    pub(crate) fn stored_plan(&mut self, path: &Path) -> Result<Option<Plan>, EngineError> {
        let Some(st) = self.inputs.get(path) else {
            return Ok(None);
        };
        let behind = st.needs_reading();
        let Some(chain) = st.chain else {
            return Ok(None);
        };
        if !has_reading(st.origin) || st.stored == 0 || !behind || st.unread.is_some() {
            return Ok(None);
        }
        let (input_id, origin, facts) = (st.input_id.clone(), st.origin, st.facts);
        let stored = st.stored;
        let latest = self
            .reads
            .reader()?
            .latest_chain(&input_id)?
            .ok_or_else(|| {
                polylogue_store::StoreError::invalid("the stored revision disappeared")
            })?;
        let imported = latest.imported || origin.route.is_imported();
        // Parse on from the stored state where it can go on from there.
        let from = match polylogue_model::chain::due(
            &latest.state,
            &polylogue_model::chain::ChainFacts {
                stored,
                repeated_by: None,
                own_from: 0,
                parser: Some(origin.parse_version()),
                finished: imported,
            },
        ) {
            polylogue_model::chain::Due::Extend { from } => from,
            _ => 0,
        };
        Ok(Some(self.reparse_plan(
            path.to_path_buf(),
            origin,
            InputKey {
                id: input_id,
                format: origin.name.into(),
                route: if imported {
                    origin.route.imported()
                } else {
                    origin.route
                },
            },
            chain,
            stored,
            &latest.state,
            from,
            facts,
            true,
        )?))
    }

    /// A parse-only plan over a chain's stored bytes, from `from`: where
    /// [`polylogue_model::chain::due`] says its parse goes on, or 0 to parse
    /// it from the start, as a replacement of a parse it has. A parser that
    /// reads whole documents always starts at 0. The parser state itself is
    /// restored as the plan starts ([`Engine::load_stored`]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reparse_plan(
        &self,
        path: PathBuf,
        origin: &'static OriginSpec,
        input: InputKey,
        chain: i64,
        stored: u64,
        state: &polylogue_model::chain::ChainState,
        from: u64,
        facts: Option<FileFacts>,
        tracked: bool,
    ) -> Result<Plan, EngineError> {
        let parser = (origin.open)(&path);
        let resumes = matches!(parser, Some(polylogue_sources::Parse::Lines(_)))
            || parser.as_ref().is_some_and(|p| p.is_capture_log());
        let parse_pos = if resumes { from } else { 0 };
        let export = matches!(parser, Some(polylogue_sources::Parse::Export(_)));
        let mut plan = Plan::stored(
            path,
            origin,
            input,
            Change::Reparse { chain },
            facts,
            stored,
            parse_pos,
            pool::Bytes::Unloaded {
                chain,
                length: stored,
                decoded: parse_pos == 0 && !export,
            },
            self.config.chunk,
            tracked,
        );
        plan.replace = *state != polylogue_model::chain::ChainState::Unparsed && parse_pos == 0;
        Ok(plan)
    }

    /// Loads a plan's stored bytes as it starts ([`pool::Bytes::Unloaded`]).
    /// False while they are still being written: the plan waits for a later
    /// pass.
    pub(crate) fn load_stored(&self, plan: &mut Plan) -> Result<bool, EngineError> {
        if matches!(plan.source, pool::Bytes::File) {
            if plan.parser_state.is_none()
                && plan.parse_pos > 0
                && !plan.storing_only
                && has_reading(plan.origin)
                && let Some(chain) = plan.parser_chain
            {
                let reader = self.reads.reader()?;
                plan.parser_state = self.restored_parser_state(
                    &reader,
                    &plan.input.id,
                    chain,
                    plan.parse_pos,
                    &plan.origin.parse_version(),
                )?;
            }
            return Ok(true);
        }
        let pool::Bytes::Unloaded {
            chain,
            length,
            mut decoded,
        } = plan.source
        else {
            return Ok(true);
        };
        if plan.parse_pos > 0 && !whole(plan.origin, &plan.path) {
            // Planned to go on from the stored state: the parser's carried
            // state there is restored, or the chain is parsed from its start.
            let reader = self.reads.reader()?;
            plan.parser_state = self.restored_parser_state(
                &reader,
                &plan.input.id,
                chain,
                plan.parse_pos,
                &plan.origin.parse_version(),
            )?;
            if plan.parser_state.is_none() {
                plan.parse_pos = 0;
                plan.replace = true;
            } else {
                plan.read_to = plan.parse_pos;
                decoded = false;
            }
        }
        let stored = self
            .reads
            .reader()?
            .stored_chain(&plan.input.id, chain, length)?;
        plan.source = if decoded {
            pool::Bytes::Import(Arc::new(Mutex::new(read::Imported::from_chain(
                stored,
                Arc::clone(&self.blobs),
            ))))
        } else {
            pool::Bytes::Stored(Arc::new(stored))
        };
        Ok(true)
    }

    /// Readies a plan for reading: the captured authored records and local
    /// references are recorded first.
    /// None when nothing is left to read, or the plan is refused or failed.
    fn admit(&mut self, plan: Plan, pass: &mut Pass) -> Result<Option<Plan>, EngineError> {
        polylogue_store::fault::check("pass.admit", &plan.path.to_string_lossy())?;
        let Some(mut plan) = self.current_plan(plan, pass)? else {
            return Ok(None);
        };
        if matches!(plan.change, Change::Reparse { .. })
            && let Some(reason) = unread_path(plan.origin, &plan.path)
        {
            // Stored bytes at a path this build's parser does not read
            // (an earlier build captured it): refused with the reason,
            // never read range by range with nothing to record.
            pass.errors.push(refused_input(&plan.input.id, &reason));
            plan.failed = true;
            self.record_stop(
                &mut plan,
                polylogue_model::ChainFailure::refused(reason),
                pass,
            );
            self.finish_plan(plan);
            return Ok(None);
        }
        if let Err(error) = self.restore_captured_authored(&plan, pass) {
            self.chain_failed(&mut plan, "authored restoration", error, pass);
            self.finish_plan(plan);
            return Ok(None);
        }
        if let Err(error) = self.observe_local_references(&plan, pass) {
            self.chain_failed(&mut plan, "local references", error, pass);
            self.finish_plan(plan);
            return Ok(None);
        }
        Ok(Some(plan))
    }

    /// The plan of a file as it is now. A file that changed after it was
    /// planned (a session being written) is planned again from its current
    /// state; one that changes again before it is read waits for the next
    /// pass, and one that cannot be read fails alone. The pass goes on with
    /// every other file either way.
    fn current_plan(
        &mut self,
        mut plan: Plan,
        pass: &mut Pass,
    ) -> Result<Option<Plan>, EngineError> {
        let mut planned_again = false;
        loop {
            match self.check_planned_file(&plan) {
                PlannedFile::Ready => return Ok(Some(plan)),
                PlannedFile::Unreadable(error) => {
                    pass.errors.push(
                        polylogue_model::Failure::new(plan.path.display(), &EngineError::Io(error))
                            .doing("read"),
                    );
                    plan.failed = true;
                    self.finish_plan(plan);
                    return Ok(None);
                }
                PlannedFile::Changed => {}
            }
            let now = match std::fs::metadata(&plan.path) {
                Ok(metadata) if !planned_again && self.inputs.contains_key(&plan.path) => {
                    FileFacts::of(&metadata)
                }
                // Changing still, or gone: the next pass finds it as it is then.
                _ => {
                    pass.capture_pending = true;
                    return Ok(None);
                }
            };
            planned_again = true;
            // Planning counts the file's bytes into the build's total again.
            if has_reading(plan.origin)
                && let Some(build) = self.archive.index_build()
            {
                self.archive.record_build_bytes(
                    build
                        .bytes
                        .saturating_sub(plan.size.saturating_sub(plan.stored_upto)),
                    build.bytes_read,
                );
            }
            let Some(mut again) = self.plan_file(&plan.path, now, pass)? else {
                return Ok(None);
            };
            again.priority = plan.priority;
            again.storing_only |= plan.storing_only;
            plan = again;
        }
    }

    /// Runs plans on the parse pool, one job per input at a time, until
    /// every input is stored and parsed up to its size; then waits for the
    /// writer to commit.
    pub(crate) fn execute(&mut self, plans: Vec<Plan>, pass: &mut Pass) -> Result<(), EngineError> {
        if plans.is_empty() && self.rollout_captures.is_empty() {
            return Ok(());
        }
        // A build keeps each input's work together. Plans that store new
        // bytes go first, the newest first, so capture does not wait for
        // stored chains to be read. Arrays run serially only after preceding
        // pool work has committed.
        let mut plans = plans;
        if self.archive.index_build().is_some() {
            let key = |plan: &Plan| {
                let stores =
                    matches!(plan.source, pool::Bytes::File) && plan.stored_upto < plan.size;
                (
                    plan.priority == Priority::Bulk,
                    !stores,
                    std::cmp::Reverse(
                        plan.facts
                            .filter(|_| stores)
                            .map_or(0, |facts| facts.mtime_ns),
                    ),
                )
            };
            plans.sort_by(|a, b| {
                key(a)
                    .cmp(&key(b))
                    .then_with(|| (&a.input.id, &a.path).cmp(&(&b.input.id, &b.path)))
            });
        } else {
            plans.sort_by_key(|p| (p.priority == Priority::Bulk, std::cmp::Reverse(p.size)));
        }
        let mut pooled = Vec::new();
        let mut replacements = Vec::new();
        let mut plans: std::collections::VecDeque<Plan> = plans.into();
        while let Some(plan) = plans.pop_front() {
            if self.archive.stop().is_cancelled() || self.poll_request() {
                break;
            }
            let Some(mut plan) = self.admit(plan, pass)? else {
                continue;
            };
            let store_first = matches!(plan.source, pool::Bytes::File)
                && plan.parse_pos == 0
                && plan.stored_upto > 0
                && !matches!(plan.change, Change::First | Change::Rewritten);
            if store_first {
                let later = later_paths(&plan, &plans);
                self.execute_pool(std::mem::take(&mut pooled), Vec::new(), &later, pass)?;
                if self.yielding() {
                    return Ok(());
                }
                if let Err(error) = self.store_plan(&mut plan, pass) {
                    pass.errors.push(
                        polylogue_model::Failure::new(plan.path.display(), &error)
                            .doing("store before re-parse"),
                    );
                    plan.failed = true;
                    self.finish_plan(plan);
                    continue;
                }
            }
            if plan.replace || store_first {
                replacements.push(plan);
                continue;
            }
            if let Some(parser) = export_parser(&plan) {
                let later = later_paths(&plan, &plans);
                self.execute_pool(std::mem::take(&mut pooled), Vec::new(), &later, pass)?;
                if self.yielding() {
                    return Ok(());
                }
                self.read_export(plan, parser, pass)?;
            } else {
                pooled.push(plan);
            }
        }
        self.execute_pool(pooled, replacements, &[], pass)?;
        if self.archive.stop().is_cancelled() {
            return Err(EngineError::Stopped);
        }
        Ok(())
    }

    /// Reads an export's items on this thread and records a failure on its
    /// chain. Pool work queued before it has committed.
    fn read_export(
        &mut self,
        mut plan: Plan,
        mut parser: Box<dyn ExportParser>,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        if !self.load_stored(&mut plan)? {
            return Ok(());
        }
        let result = polylogue_model::caught(|| self.execute_export(&mut plan, &mut *parser, pass))
            .unwrap_or_else(|panic| Err(polylogue_model::ChainFailure::from(panic).into()));
        match result {
            Ok(()) => {}
            // A stop is no failure of the export: its committed
            // deliveries stay and the next parse continues after them.
            Err(EngineError::Stopped) => return Err(EngineError::Stopped),
            Err(error) => self.chain_failed(&mut plan, "parse", error, pass),
        }
        self.finish_plan(plan);
        Ok(())
    }

    /// Runs `plans` range by range and `chains` whole on the parse pool.
    /// Live inputs found meanwhile join it, except paths in `later`, which
    /// the caller plans itself afterwards; a live export is read after the
    /// pool's work has committed.
    pub(crate) fn execute_pool(
        &mut self,
        plans: Vec<Plan>,
        chains: Vec<Plan>,
        later: &[&Path],
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        if plans.is_empty() && chains.is_empty() && self.rollout_captures.is_empty() {
            return Ok(());
        }
        if chains
            .iter()
            .any(|plan| !matches!(plan.change, Change::Reparse { .. }))
        {
            return Err(EngineError::internal("a replacement needs a stored chain"));
        }
        let mut active_chains = HashMap::<u64, PathBuf>::new();
        let result = (|| {
            let (done_tx, done_rx) = channel();
            let pool = Pool::start(
                self.config.workers,
                self.config.bytes_in_flight,
                self.writer.sender(),
                Arc::clone(&self.blobs),
                self.archive.clone(),
                done_tx,
                pool::ChainSettings {
                    read_memory: self.config.reader_memory_per_connection()?,
                    yielding: self.requests.as_ref().map_or_else(
                        || Arc::new(std::sync::atomic::AtomicBool::new(false)),
                        |requests| Arc::clone(&requests.yielding),
                    ),
                },
            );
            // Plans wait here and enter the pool a few at a time, so live inputs
            // that arrive meanwhile can go ahead of what is still waiting.
            let mut waiting: std::collections::VecDeque<Plan> = plans.into();
            let mut chains = chains;
            chains.sort_by_key(|plan| plan.priority == Priority::Bulk);
            let mut chains: std::collections::VecDeque<Plan> = chains.into();
            let mut next_chain_id = 0u64;
            let mut running: HashMap<PathBuf, Plan> = HashMap::new();
            // Inputs whose first job this run queued. An input holds its parser
            // state between jobs until its last finishes, so a new input starts
            // only when its first range can be queued at once; otherwise the
            // loop would wait inside `submit` with a new input while finished
            // ranges of started ones wait in `done_rx`, and every large input of
            // a pass would end up started, each holding its state.
            let mut started: HashSet<PathBuf> = HashSet::new();
            // Plans whose range the writer refused for its content; their
            // failures are recorded once the pool's writes are flushed.
            let mut refused: Vec<(Plan, polylogue_model::ChainFailure)> = Vec::new();
            let mut outstanding = 0usize;
            let mut checked = Instant::now();
            let room = self.config.workers.max(1) * IN_POOL_PER_WORKER;
            let every = self.config.live_every;
            let mut stopping = false;
            let mut yielding = false;
            // Storing runs beside parsing: up to half the workers store
            // rollouts while the rest parse, so stored bytes do not wait for
            // the parse backlog and parsing does not wait for every input to
            // be stored.
            let capture_room = (self.config.workers / 2).max(1);
            let mut progress = build_log::Progress::start(pass, pool.worker_cpu_seconds());
            // Rollout captures on workers, by path, with the input each names.
            let mut capturing: HashMap<PathBuf, String> = HashMap::new();
            // Live exports, read once the pool's work has committed.
            let mut exports: std::collections::VecDeque<(Plan, Box<dyn ExportParser>)> =
                std::collections::VecDeque::new();
            'work: loop {
                if !yielding && self.poll_request() {
                    yielding = true;
                    waiting.clear();
                    chains.clear();
                    exports.clear();
                    self.rollout_captures.clear();
                    // No worker read these jobs. The next turn revalidates them;
                    // chunks already in hand keep their normal commit and ACK.
                    outstanding -= self.cancel_pool(&pool, &mut active_chains, &mut capturing);
                }
                if !stopping && self.archive.stop().is_cancelled() {
                    // What workers are parsing commits; nothing else starts.
                    stopping = true;
                    waiting.clear();
                    chains.clear();
                    exports.clear();
                    self.rollout_captures.clear();
                    outstanding -= self.cancel_pool(&pool, &mut active_chains, &mut capturing);
                }
                if !stopping && !yielding && checked.elapsed() >= every {
                    checked = Instant::now();
                    let live = {
                        // A continuation owns its path while queued, too. Finish
                        // its planned bytes before observing later source changes.
                        let reparsing: Vec<_> = self.reparsing.keys().cloned().collect();
                        let queued: Vec<_> = self
                            .rollout_captures
                            .iter()
                            .map(|capture| capture.path.clone())
                            .collect();
                        let busy = running
                            .keys()
                            .map(PathBuf::as_path)
                            .chain(waiting.iter().map(|plan| plan.path.as_path()))
                            .chain(chains.iter().map(|plan| plan.path.as_path()))
                            .chain(reparsing.iter().map(PathBuf::as_path))
                            .chain(capturing.keys().map(PathBuf::as_path))
                            .chain(queued.iter().map(PathBuf::as_path))
                            .chain(exports.iter().map(|(plan, _)| plan.path.as_path()))
                            .chain(later.iter().copied())
                            .collect();
                        self.live_plans(&busy, pass)
                    };
                    match live {
                        Ok(live) => {
                            for plan in live.into_iter().rev() {
                                if plan.replace {
                                    chains.push_front(plan);
                                } else if let Some(parser) = export_parser(&plan) {
                                    exports.push_front((plan, parser));
                                } else {
                                    waiting.push_front(plan);
                                }
                            }
                        }
                        Err(e) => pass
                            .errors
                            .push(polylogue_model::Failure::new("live inputs", &e)),
                    }
                }
                while !stopping && !yielding && outstanding < room && capturing.len() < capture_room
                {
                    // One input is captured or read by one worker at a time.
                    let Some(index) = self.rollout_captures.iter().position(|capture| {
                        !capturing.contains_key(&capture.path)
                            && !capturing.values().any(|input| *input == capture.input)
                            && !self.reparsing.contains_key(&capture.path)
                            && !running.contains_key(&capture.path)
                            && !running.values().any(|plan| plan.input.id == capture.input)
                    }) else {
                        break;
                    };
                    let capture = self
                        .rollout_captures
                        .remove(index)
                        .expect("a queued capture");
                    let (path, input) = (capture.path.clone(), capture.input.clone());
                    if !pool.submit_capture(capture) {
                        break;
                    }
                    capturing.insert(path, input);
                    outstanding += 1;
                }
                // Plans of an input being captured wait for its new bytes.
                while !stopping
                    && !yielding
                    && outstanding < room
                    && let Some(index) = waiting
                        .iter()
                        .position(|plan| !capturing.values().any(|input| *input == plan.input.id))
                {
                    let mut plan = waiting.remove(index).expect("a waiting plan");
                    if self.poll_request() {
                        waiting.push_front(plan);
                        continue 'work;
                    }
                    if plan.finished() {
                        self.finish_plan(plan);
                        continue;
                    }
                    let first = !started.contains(&plan.path);
                    if first {
                        let (start, end) = plan.next_range();
                        if outstanding > 0 && !pool.has_room(end - start) {
                            waiting.push_front(plan);
                            break;
                        }
                        started.insert(plan.path.clone());
                    }
                    // Stored bytes are loaded as their plan starts, so a pass
                    // over every stored input holds only the running plans'.
                    if !self.load_stored(&mut plan)? {
                        continue;
                    }
                    if plan.finished() {
                        self.finish_plan(plan);
                        continue;
                    }
                    let mut job = plan.next_job();
                    if matches!(job.source, pool::Bytes::File) {
                        let mut context = self.source_read_context(
                            &job.path,
                            &job.input.format,
                            polylogue_store::capture_read::ReadRoute::NativeCapture,
                        );
                        context.facts = plan.facts;
                        job.source_read = Some(context);
                    }
                    let bytes = job.end - job.start;
                    if !pool.submit(job) {
                        // Stopped while waiting for room; the loop's next turn
                        // takes back what is queued.
                        break;
                    }
                    pass.bytes_read += bytes;
                    outstanding += 1;
                    running.insert(plan.path.clone(), plan);
                }
                while !stopping && !yielding && outstanding < room {
                    let Some(index) = chains.iter().position(|chain| {
                        !running.contains_key(&chain.path)
                            && !waiting.iter().any(|range| range.path == chain.path)
                            && !capturing.values().any(|input| *input == chain.input.id)
                    }) else {
                        break;
                    };
                    let plan = &chains[index];
                    if outstanding > 0 && !pool.has_room(plan.size - plan.parse_pos) {
                        break;
                    }
                    let plan = chains.remove(index).expect("waiting chain");
                    let path = plan.path.clone();
                    let id = next_chain_id;
                    next_chain_id += 1;
                    if !pool.submit_chain(id, plan) {
                        break;
                    }
                    *self.reparsing.entry(path.clone()).or_default() += 1;
                    active_chains.insert(id, path);
                    outstanding += 1;
                }
                if outstanding == 0 {
                    break;
                }
                progress.record(self, pass, || pool.worker_cpu_seconds());
                let d = match done_rx.recv_timeout(every.max(Duration::from_millis(10))) {
                    Ok(d) => d,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                };
                outstanding -= 1;
                progress.file_done(self, pass);
                let d = match d {
                    pool::Finished::Range(done) => *done,
                    pool::Finished::Chain(done) => {
                        active_chains.remove(&done.id);
                        pass.bytes_read += done.bytes_read;
                        self.finish_chain(done.plan, done.written, pass);
                        continue;
                    }
                    pool::Finished::Captured(captured) => {
                        capturing.remove(&captured.path);
                        let path = captured.path.clone();
                        let plan = match self.captured_rollout(*captured, pass) {
                            Ok(Some(plan)) if !stopping && !yielding => plan,
                            Ok(_) => continue,
                            Err(error) => {
                                pass.errors.push(
                                    polylogue_model::Failure::new(path.display(), &error)
                                        .doing("capture"),
                                );
                                continue;
                            }
                        };
                        // Its bytes are durable: its reading goes next, in
                        // place of any older plan of the same input.
                        waiting.retain(|queued| queued.input.id != plan.input.id);
                        chains.retain(|queued| queued.input.id != plan.input.id);
                        if let Some(plan) = self.admit(plan, pass)? {
                            if plan.replace {
                                chains.push_front(plan);
                            } else {
                                waiting.push_front(plan);
                            }
                        }
                        continue;
                    }
                };
                let Some(mut plan) = running.remove(&d.path) else {
                    continue;
                };
                if d.cancelled {
                    pass.bytes_stored += d.stored;
                    if let Some(error) = &d.error {
                        pass.errors
                            .push(polylogue_model::Failure::new(d.path.display(), error));
                    }
                    continue;
                }
                if d.write_failed && d.content_refused {
                    // Reading this range again is refused again: the chain stops
                    // at its cursor, failed with the writer's reason, until the
                    // bytes or the reading change.
                    if let Some(chain) = d.chain {
                        plan.parser_chain = Some(chain);
                        if d.end > plan.stored_upto {
                            plan.stored_upto = d.end;
                            plan.storing_started = true;
                        }
                    }
                    let failure = polylogue_model::ChainFailure::malformed(match &d.error {
                        Some(error) => format!("the writer refused its rows: {error}"),
                        None => "the writer refused its rows".to_owned(),
                    });
                    refused.push((plan, failure));
                    continue;
                }
                if d.write_failed {
                    if let Some(error) = &d.error {
                        pass.errors
                            .push(polylogue_model::Failure::new(d.path.display(), error));
                    }
                    if plan.tracked {
                        self.refresh_confirmed_input(&plan.path)?;
                    }
                    continue;
                }
                pass.bytes_stored += d.stored;
                match &d.error {
                    // Replaced or rewritten since it was planned: the file is
                    // planned again, from what it then holds, by the next
                    // pass, which starts at once.
                    Some(e) if e.source_changed() => pass.capture_pending = true,
                    Some(e) => pass
                        .errors
                        .push(polylogue_model::Failure::new(d.path.display(), e)),
                    None => {}
                }
                let parsed = has_reading(plan.origin);
                if parsed {
                    self.archive
                        .advance_build_bytes(d.parsed_upto.saturating_sub(plan.parse_pos));
                }
                plan.apply(d, parsed);
                if plan.finished() {
                    self.finish_plan(plan);
                } else {
                    // The next part of the same input keeps its place.
                    waiting.push_front(plan);
                }
            }
            drop(pool);
            let report: WriteReport = self.flush_writes()?;
            pass.messages_written += report.messages;
            pass.segment_syncs += report.segment_syncs;
            pass.segment_bytes_written += report.segment_bytes_written;
            pass.blobs_appended += report.blobs_appended;
            pass.blobs_deduplicated += report.blobs_deduplicated;
            pass.errors.extend(report.errors);
            for (mut plan, failure) in refused {
                self.record_failure(&mut plan, failure, pass);
                self.finish_plan(plan);
            }
            for (plan, parser) in exports {
                if self.archive.stop().is_cancelled() || self.yielding() {
                    break;
                }
                self.read_export(plan, parser, pass)?;
            }
            Ok(())
        })();
        for path in active_chains.into_values() {
            self.finish_chain_path(&path);
        }
        result
    }

    fn cancel_pool(
        &mut self,
        pool: &Pool,
        active_chains: &mut HashMap<u64, PathBuf>,
        capturing: &mut HashMap<PathBuf, String>,
    ) -> usize {
        let cancelled = pool.cancel_queued();
        let count = cancelled.ranges + cancelled.chains.len() + cancelled.captures.len();
        for (id, plan) in cancelled.chains {
            active_chains.remove(&id);
            self.finish_chain_path(&plan.path);
        }
        // Not yet read: the next pass finds these files again.
        for capture in cancelled.captures {
            capturing.remove(&capture.path);
        }
        count
    }

    fn finish_chain(
        &mut self,
        plan: Plan,
        written: Result<Option<u64>, EngineError>,
        pass: &mut Pass,
    ) {
        self.finish_chain_path(&plan.path);
        match written {
            Ok(Some(written)) => {
                pass.messages_written += written;
                self.finish_plan(plan);
            }
            Ok(None) | Err(EngineError::Stopped) => {}
            Err(error) => {
                pass.errors.push(
                    polylogue_model::Failure::new(plan.path.display(), &error).doing("re-parse"),
                );
            }
        }
    }

    fn finish_chain_path(&mut self, path: &Path) {
        let count = self.reparsing.get_mut(path).expect("running chain path");
        *count -= 1;
        if *count == 0 {
            self.reparsing.remove(path);
        }
    }

    /// Finished inputs keep their cursors; an append restores the committed checkpoint.
    pub(crate) fn finish_plan(&mut self, plan: Plan) {
        // A parse that failed before the plan stored the rest of the file
        // fails at the length the plan left stored: reading those bytes
        // again would meet the same failure.
        if let Some((failure, at)) = plan.failed_at.clone()
            && plan.stored_upto > at
            && let Err(error) = self.send_failure(&plan, failure)
        {
            polylogue_store::log!(
                warning,
                "{}: recording the failure at the stored length: {error}",
                plan.path.display()
            );
        }
        if !plan.tracked {
            return;
        }
        if let Some(st) = self.inputs.get_mut(&plan.path) {
            let new_chain = matches!(plan.change, Change::First | Change::Rewritten);
            if new_chain && plan.failed && !plan.storing_started {
                // Planning a rewrite does not replace the committed checkpoint.
                // A failed first read leaves its facts, tail and parser intact.
                st.unread = plan.facts;
                return;
            }
            if new_chain {
                st.stored = plan.stored_upto;
                st.parsed_upto = 0;
                st.parsed = false;
                st.chain = plan.parser_chain;
            } else {
                st.stored = plan.stored_upto.max(st.stored);
                if let Some(chain) = plan.parser_chain {
                    st.chain = Some(chain);
                }
            }
            if plan.failure_recorded {
                // The parse state holds the failure at the stored length, so
                // the chain waits for changed bytes or a new reading like a
                // chain that waits for a record to finish.
                st.unread = None;
                st.facts = plan.facts;
                st.parsed = true;
                st.parsed_upto = plan.parse_pos;
                st.waiting = true;
                return;
            }
            if plan.failed {
                // Tried again when the bytes change; the facts stay old, so
                // a change is still seen as one.
                st.waiting = false;
                st.unread = plan.facts;
                return;
            }
            st.unread = None;
            st.facts = plan.facts;
            st.parsed_upto = if has_reading(plan.origin) {
                plan.parse_pos.min(plan.size)
            } else {
                0
            };
            st.parsed = has_reading(plan.origin) && (!plan.storing_only || st.parsed);
            st.waiting = plan.waiting;
        }
    }

    /// Runs passes after changes under watched locations, until stopped or,
    /// for [`Lifetime::UntilIdle`],
    /// until a pass ends with no work pending and nothing queued. Without
    /// sources (`reads_sources` off) nothing is watched: the first pass
    /// parses and derives what the archive stores, and later passes run only
    /// for imports.
    pub fn run(
        &mut self,
        tx: crossbeam_channel::Sender<LoopMsg>,
        rx: crossbeam_channel::Receiver<LoopMsg>,
        status: &Arc<SharedLoopStatus>,
        lifetime: Lifetime,
        ready: impl FnOnce(),
    ) -> Result<RunSummary, EngineError> {
        self.watcher = if self.config.reads_sources {
            Some(watch::Roots::new(&self.config, tx.clone())?)
        } else {
            None
        };
        self.wake = Some(tx.clone());
        self.requests = Some(Requests::new(rx));
        // The stop request wakes a loop waiting between passes.
        let stop = self.archive.stop().clone();
        let _stop_wake = stop.on_cancel(move || {
            let _ = tx.send(LoopMsg::Stop);
        });
        ready();
        polylogue_store::cancellation::under(&stop, || self.run_passes(status, lifetime))
    }

    fn run_passes(
        &mut self,
        status: &Arc<SharedLoopStatus>,
        lifetime: Lifetime,
    ) -> Result<RunSummary, EngineError> {
        let busy = |busy: bool| {
            status.update(|status| status.busy = busy);
        };
        let mut summary = RunSummary::default();
        let mut discover = true;
        // What the last failed turn failed with, by subject, cause and code
        // (not by sentence). A turn failing again with the same failures is
        // a repeat: an idle run ends on it, and the daemon waits for a
        // change unless every failure passes by itself, which is tried again
        // after `live_every`.
        let mut failed_with: Option<Vec<(String, polylogue_model::Cause, polylogue_model::Code)>> =
            None;
        let mut repeated;
        let mut waits_for_change;
        let mut changed = std::collections::BTreeSet::new();
        let mut watch_errors = Vec::new();
        loop {
            if self.archive.stop().is_cancelled() {
                return Ok(self.stopped(summary));
            }
            discover |= self.take_watched(&mut changed, &mut watch_errors);
            busy(true);
            let pending;
            self.planned.clear();
            let mut completed = false;
            let passed = {
                let _doing = polylogue_store::activity::doing("running a pass");
                self.one_pass(discover, &changed)
            };
            let mut finished = match passed {
                Err(EngineError::Stopped) => {
                    // Writes the pass queued before stopping are committed.
                    if let Err(error) = self.flush_writes() {
                        summary
                            .work
                            .errors
                            .push(polylogue_model::Failure::of(&error));
                    }
                    return Ok(self.stopped(summary));
                }
                Ok((pass, seen)) => {
                    // Imports a build answered while it ran.
                    summary.imports += std::mem::take(&mut self.imports_during_build);
                    let yielded = self.yielding();
                    pending = pass.capture_pending || yielded;
                    if yielded {
                        // Reload committed cursors after answering the request,
                        // retaining this scope's unfinished planning page.
                        self.discard_confirmed_progress();
                    }
                    completed = true;
                    if pending {
                        changed.extend(seen);
                    } else {
                        changed.clear();
                    }
                    discover = false;
                    failed_with = None;
                    repeated = false;
                    waits_for_change = false;
                    pass
                }
                Err(e) => {
                    // A failed turn cannot show that bytes previously held in the
                    // spool were confirmed. Only a completed pass updates it.
                    pending = self.pass_open
                        || self.captures.pending()
                        || discover
                        || !changed.is_empty();
                    let mut errors = if matches!(
                        &e,
                        EngineError::Store(polylogue_store::StoreError::ScopeChanged)
                    ) {
                        Vec::new()
                    } else {
                        vec![polylogue_model::Failure::of(&e)]
                    };
                    match self.flush_writes() {
                        Ok(report) => errors.extend(report.errors),
                        Err(error) => errors.push(polylogue_model::Failure::of(&error)),
                    }
                    let keys: Vec<_> = errors
                        .iter()
                        .map(|failure| {
                            let (subject, cause, code) = failure.key();
                            (subject.to_owned(), cause, code)
                        })
                        .collect();
                    repeated = failed_with.as_ref() == Some(&keys);
                    waits_for_change = repeated
                        && keys
                            .iter()
                            .any(|(_, cause, _)| *cause != polylogue_model::Cause::Transient);
                    failed_with = Some(keys);
                    Pass {
                        full_scans: u64::from(discover && self.config.reads_sources),
                        errors,
                        capture_pending: pending,
                        ..Pass::default()
                    }
                }
            };
            finished.errors.append(&mut watch_errors);
            let planned = std::mem::take(&mut self.planned);
            self.failure_log.pass_ended(
                &finished.errors,
                crate::failure_log::Looked {
                    files: &planned,
                    walked: completed && finished.full_scans > 0,
                    completed,
                },
            );
            summary.passes += 1;
            summary.work.add(finished.clone());
            finished.errors = self.failure_log.errors();
            status.update(|s| {
                s.full_scans += finished.full_scans;
                if finished.full_scans > 0
                    || finished.inputs_seen > 0
                    || finished.inputs_changed > 0
                    || s.last_pass.capture_pending
                    || finished.capture_pending
                    || !finished.errors.is_empty()
                    || !s.last_pass.errors.is_empty()
                    || s.passes == 0
                {
                    s.last_pass = finished;
                    s.last_pass_at =
                        polylogue_model::Timestamp::from_unix_ms(self.config.clock.now_ms());
                }
                s.passes += 1;
                s.busy = pending;
            });
            if lifetime == Lifetime::UntilIdle {
                // Work queued while the pass ran (imports, changes the
                // watcher saw) is taken now; with none, the loop is idle. A
                // capture that continues gets `live_every` before the next pass.
                // A turn failing as the one before it ends the run with its
                // errors reported.
                let mut more = pending && !repeated;
                if more {
                    match self
                        .requests
                        .as_mut()
                        .expect("running inbox")
                        .recv_timeout(self.config.live_every)
                    {
                        Ok(LoopMsg::Stop) | Err(RecvTimeoutError::Disconnected) => {
                            return Ok(self.stopped(summary));
                        }
                        Ok(message) => {
                            self.answer(message, &busy, pending, &mut summary);
                        }
                        Err(RecvTimeoutError::Timeout) => {}
                    }
                }
                loop {
                    match self.requests.as_mut().expect("running inbox").try_recv() {
                        Ok(LoopMsg::Stop) | Err(crossbeam_channel::TryRecvError::Disconnected) => {
                            return Ok(self.stopped(summary));
                        }
                        Ok(message) => {
                            more = true;
                            self.answer(message, &busy, pending, &mut summary);
                        }
                        Err(crossbeam_channel::TryRecvError::Empty) => break,
                    }
                }
                more |= self.take_watched(&mut changed, &mut watch_errors);
                more |= !repeated && !changed.is_empty();
                if !more {
                    return Ok(self.stopped(summary));
                }
                continue;
            }
            let requests = self.requests.as_mut().expect("running inbox");
            let mut msg = if (!self.config.reads_sources && !pending) || waits_for_change {
                requests.recv()
            } else {
                requests.recv_timeout(self.config.live_every)
            };
            let burst = Instant::now();
            loop {
                match msg {
                    Ok(LoopMsg::Stop) | Err(RecvTimeoutError::Disconnected) => {
                        return Ok(self.stopped(summary));
                    }
                    Ok(message) => {
                        self.answer(message, &busy, pending, &mut summary);
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                }
                if !self.config.reads_sources || burst.elapsed() > COALESCE {
                    break;
                }
                // Pending captures continue after queued requests, without
                // waiting for a watcher event or a coalescing delay.
                if pending {
                    msg = self
                        .requests
                        .as_mut()
                        .expect("running inbox")
                        .try_recv()
                        .map_err(|e| match e {
                            crossbeam_channel::TryRecvError::Empty => RecvTimeoutError::Timeout,
                            crossbeam_channel::TryRecvError::Disconnected => {
                                RecvTimeoutError::Disconnected
                            }
                        });
                    if matches!(msg, Err(RecvTimeoutError::Timeout)) {
                        break;
                    }
                    continue;
                }
                // Coalesce a burst of file events into one pass.
                msg = self
                    .requests
                    .as_mut()
                    .expect("running inbox")
                    .recv_timeout(Duration::from_millis(200));
                if matches!(msg, Err(RecvTimeoutError::Timeout)) {
                    break;
                }
            }
        }
    }

    /// Moves what the watcher saw into `changed`; true when it asks for a
    /// full scan. Without sources there is no watcher and nothing changes.
    fn take_watched(
        &mut self,
        changed: &mut std::collections::BTreeSet<PathBuf>,
        errors: &mut Vec<polylogue_model::Failure>,
    ) -> bool {
        let Some(watcher) = self.watcher.as_mut() else {
            return false;
        };
        let mut full = false;
        if let Err(error) = watcher.refresh(&self.config) {
            errors.push(polylogue_model::Failure::new(
                "source watch",
                &EngineError::from(error),
            ));
            full = true;
        }
        let changes = watcher.take();
        errors.extend(changes.errors);
        full |= changes.full;
        changed.extend(
            changes
                .paths
                .into_iter()
                .map(|path| collect::database_path(&self.config, &path).unwrap_or(path)),
        );
        full
    }

    /// Answers one message other than `Stop`, adding an import's work to
    /// `summary`.
    pub(crate) fn answer(
        &mut self,
        message: LoopMsg,
        busy: &impl Fn(bool),
        pending: bool,
        summary: &mut RunSummary,
    ) {
        let LoopMsg::Import {
            paths,
            format,
            dry_run,
            authorship,
            parent,
            reply,
        } = message
        else {
            return;
        };
        if let Some(requests) = &mut self.requests {
            requests.serving = true;
            requests
                .yielding
                .store(false, std::sync::atomic::Ordering::Release);
        }
        // Queue waits belong to the producing transport. Once the import
        // is taken, the engine owns completion and its ACK.
        let report = parent
            .as_ref()
            .map_or(Ok(()), |parent| parent.check())
            .map_err(ImportFailure::Parent)
            .and_then(|()| {
                busy(true);
                let options = ImportOptions {
                    authorship: Some(authorship),
                    origin: format,
                    dry_run,
                };
                let report = self
                    .import_with_options(&paths, &options)
                    .map_err(ImportFailure::Import);
                busy(pending);
                report
            });
        if let Some(requests) = &mut self.requests {
            requests.serving = false;
        }
        summary.imports += 1;
        match &report {
            Ok(report) => summary.work.add(report.pass.clone()),
            Err(ImportFailure::Import(error)) => summary
                .work
                .errors
                .push(polylogue_model::Failure::new("import", error)),
            Err(ImportFailure::Parent(_)) => {}
        }
        let _ = reply.send(report);
    }

    fn stopped(&mut self, mut summary: RunSummary) -> RunSummary {
        summary.imports += std::mem::take(&mut self.imports_during_build);
        self.captures = Databases::default();
        self.wake = None;
        self.requests = None;
        self.watcher = None;
        self.pass_open = false;
        self.stored_planning = None;
        summary
    }
}

/// Whether the file at `path` differs from the facts its input was last
/// read or refused at. A file that is gone is left to discovery.
fn changed_since_read(path: &Path, st: &InputState) -> bool {
    std::fs::metadata(path).is_ok_and(|meta| {
        let now = FileFacts::of(&meta);
        !st.unread
            .or(st.facts)
            .is_some_and(|read| read.unchanged_as(&now))
    })
}

/// A stored input at a path this build's parser does not read: refused for
/// its bytes, with the reason.
fn refused_input(input: &str, reason: &str) -> polylogue_model::Failure {
    polylogue_model::Failure::stated(
        format_args!("refuse stored input {input}"),
        polylogue_model::Cause::Content,
        polylogue_model::Code::Import,
        reason,
    )
}

#[cfg(test)]
mod tests;
