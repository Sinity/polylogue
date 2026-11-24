//! One pass of the loop: collect, store, parse, derive; and the loop that
//! runs passes.

use crate::plan::{InputState, Plan, span, whole};
use crate::*;
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
    fn new(receiver: Receiver<LoopMsg>) -> Self {
        Self {
            receiver,
            pending: None,
            yielding: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            serving: false,
            commits: None,
            commit_wake: false,
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
                    Ok(LoopMsg::Committed) => {
                        self.commit_wake = true;
                        continue;
                    }
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

    fn try_recv(&mut self) -> Result<LoopMsg, std::sync::mpsc::TryRecvError> {
        self.queued().map_or_else(|| self.receiver.try_recv(), Ok)
    }

    fn queued(&mut self) -> Option<LoopMsg> {
        self.pending
            .take()
            .or_else(|| std::mem::take(&mut self.commit_wake).then_some(LoopMsg::Committed))
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

    /// Fails `plan` with `error` and records the failure as its chain's parse
    /// state, at the position the chain was parsed to. Recording needs stored
    /// bytes; when it cannot be made, the failure stays the loop's own.
    pub(crate) fn record_failure(&mut self, plan: &mut Plan, error: String, pass: &mut Pass) {
        pass.errors.push(error.clone());
        plan.failed = true;
        self.record_stop(
            plan,
            polylogue_model::batch::StepProgress::Failed { error },
            pass,
        );
    }

    /// Record stopped progress at the plan's existing source cursor.
    pub(crate) fn record_stop(
        &mut self,
        plan: &mut Plan,
        progress: polylogue_model::batch::StepProgress,
        pass: &mut Pass,
    ) {
        if plan.failure_recorded || plan.stored_upto == 0 {
            return;
        }
        let mut step = polylogue_store::ParseStep::new(
            Default::default(),
            Default::default(),
            plan.parse_pos..plan.parse_pos,
            false,
        );
        step.reset = false;
        step.progress = progress;
        step.parse_version = plan.origin.parse_version();
        step.stopped_attempt = Some(self.stopped_attempt);
        step.prepare();
        let batch = polylogue_store::Batch {
            required_outcome: None,
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
            base: None,
            release: Default::default(),
        };
        let recorded = self
            .writer
            .send(batch)
            .map_err(|error| error.to_string())
            .and_then(|()| {
                let written = self.flush_writes().map_err(|error| error.to_string())?;
                if written.errors.is_empty() {
                    Ok(())
                } else {
                    Err(written.errors.join("; "))
                }
            });
        match recorded {
            Ok(()) => plan.failure_recorded = true,
            Err(recording) => pass.errors.push(format!(
                "record the failure of {}: {recording}",
                plan.path.display()
            )),
        }
    }

    /// Writes the repository lookups for the working directories the index
    /// holds, at most once a minute. The files it writes are collected in
    /// this same pass.
    fn refresh_lookups(&mut self) -> Result<(), String> {
        if self
            .lookups_at
            .is_some_and(|at| at.elapsed() < std::time::Duration::from_secs(60))
        {
            return Ok(());
        }
        self.lookups_at = Some(Instant::now());
        let rows = self
            .reads
            .reader()
            .and_then(|r| {
                r.query(
                    "SELECT DISTINCT cwd, cwd_json FROM session WHERE cwd IS NOT NULL",
                    &[],
                )
            })
            .map_err(|e| format!("repository lookups: {e}"))?;
        let dir = self
            .config
            .spool_dir(polylogue_sources::repo_lookup::ORIGIN);
        let read_at =
            polylogue_model::Timestamp::from_unix_ms(self.config.clock.now().as_millisecond())
                .map(|t| t.to_string())
                .unwrap_or_default();
        repo_lookup::refresh(
            repo_lookup::working_directories(&rows),
            &dir,
            &mut self.lookups,
            &read_at,
        )
        .map(|_| ())
        .map_err(|e| format!("repository lookups in {}: {e}", dir.display()))
    }

    /// Runs each origin's collector, honoring its declared setting.
    fn run_collectors(&mut self) -> Vec<String> {
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
                        errors.push(format!(
                            "{} collector: {setting} has no typed boolean value",
                            origin.name
                        ));
                        continue;
                    }
                    Err(e) => {
                        errors.push(format!("{} collector: {e}", origin.name));
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
                errors.push(format!("{} collector: {error}", origin.name));
                continue;
            }
            errors.extend(self.collectors.run(origin.name, collector, &paths));
        }
        errors
    }

    /// Waits for database reads as well as the current file pass.
    pub fn run_pass(&mut self) -> Result<Pass, EngineError> {
        let started = Instant::now();
        let (mut total, seen) = self.one_pass(true, &Default::default())?;
        let mut paths: std::collections::BTreeSet<_> = seen.into_iter().collect();
        // A confirmed reading can still be pending after all local work ends.
        // Return that wait to the caller instead of spinning inside its pass.
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
            let (mut next, seen) = self.one_pass(false, &paths)?;
            next.capture_pending |= completed.primary_pending;
            paths.extend(seen);
            total.add(next);
        }
        total.seconds = started.elapsed().as_secs_f64();
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
        self.lookups = repo_lookup::Cache::default();
        self.lookups_at = None;
    }

    /// One pass. A standalone archive without a current index builds it
    /// first, from the record as it stands; the pass then reads what
    /// arrived since.
    fn one_pass(
        &mut self,
        discover: bool,
        paths: &std::collections::BTreeSet<PathBuf>,
    ) -> Result<(Pass, HashSet<PathBuf>), EngineError> {
        let built = if !self.archive.is_primary() && self.archive.index_build().is_some() {
            let (built, _) = self.build_index()?;
            if built.capture_pending {
                // The build yielded: the request is answered before it goes on.
                return Ok((built, HashSet::new()));
            }
            Some(built)
        } else {
            None
        };
        let (mut pass, seen) = self.read_pass(discover, paths)?;
        if let Some(built) = built {
            pass.add(built);
        }
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
        // Drain before taking a scope token: a fenced turn must leave room
        // for the later successful reset's wake. Pending imports keep the cue.
        let content_committed = self
            .requests
            .as_ref()
            .and_then(|requests| requests.commits.as_ref())
            .and_then(|commits| commits.try_recv().ok())
            .is_some_and(|committed| committed.content);
        let result = (|| {
            let scope = self.cached_scope()?;
            let result = self.read_turn(
                discover || (self.config.reads_sources && !self.watched_loaded),
                paths,
                content_committed,
            )?;
            // A completed reset can race the last SQL read. Its cached progress
            // cannot establish completion in the new scope.
            if let Err(error) = scope.check() {
                if !result.0.errors.is_empty() {
                    return Err(
                        polylogue_store::StoreError::invalid(result.0.errors.join("; ")).into(),
                    );
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
                Ok(report) if !report.errors.is_empty() => {
                    return polylogue_store::StoreError::invalid(report.errors.join("; ")).into();
                }
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
        content_committed: bool,
    ) -> Result<(Pass, HashSet<PathBuf>), EngineError> {
        let continuing = self.pass_open;
        let started = Instant::now();
        self.load()?;
        if content_committed || (discover && !self.readings_pending) {
            self.readings_pending = self.stored_readings_are_behind()?;
        }
        let collectors_due = self
            .collectors_at
            .is_none_or(|at| at.elapsed() >= Duration::from_secs(60));
        let mut errors = Vec::new();
        let sources = self.config.reads_sources;
        if sources && (discover || collectors_due) {
            self.collectors_at = Some(Instant::now());
            errors.extend(self.run_collectors());
            errors.extend(self.refresh_lookups().err());
        }
        let (mut found, unreadable) = if !sources {
            (Vec::new(), Vec::new())
        } else if discover {
            let scan = collect::scan_with_coverage(&self.config);
            // Coverage reports the walk; a refused report is this pass's
            // error and the walk's files are still read.
            let observed_at =
                polylogue_model::Timestamp::from_unix_ms(self.config.clock.now().as_millisecond())
                    .ok_or_else(|| {
                        std::io::Error::other("capture observation time is out of range")
                    })?;
            if let Err(error) = self.writer.coverage(polylogue_model::capture::Coverage {
                version: 1,
                machine: self.config.machine.clone(),
                observed_at,
                roots: scan.roots,
            }) {
                errors.push(format!("capture coverage: {error}"));
            }
            (scan.found, scan.unreadable)
        } else {
            collect::scan_paths(&self.config, paths)
        };
        if sources && !discover && collectors_due {
            found.extend(collect::scan_spools(&self.config));
        }
        let mut pass = Pass {
            full_scans: u64::from(discover && sources),
            unreadable_roots: unreadable,
            errors,
            ..Pass::default()
        };
        if discover && sources {
            self.build_phase(
                "discovery",
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
            self.shared_prefixes.get_mut().clear();
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
        let mut changed_inputs = (fresh.len() + self.rollout_captures.len()) as u64;
        if !fresh.is_empty() || !self.rollout_captures.is_empty() {
            self.build_phase(
                "storing and reading new files",
                format_args!(
                    "{} plans, {} rollouts to capture",
                    fresh.len(),
                    self.rollout_captures.len()
                ),
            );
            self.execute(fresh, &mut pass)?;
        }
        if self.readings_pending && !self.yielding() && !self.archive.stop().is_cancelled() {
            self.build_phase(
                "reading stored newest chains",
                format_args!(
                    "{:.0} MB stored, {:.0} MB read so far",
                    pass.bytes_stored as f64 / 1e6,
                    pass.bytes_read as f64 / 1e6
                ),
            );
            self.reparse_behind(RevisionChains::Newest, &mut pass)?;
        }
        if self.yielding() {
            pass.capture_pending = true;
            pass.seconds = started.elapsed().as_secs_f64();
            return Ok((pass, seen));
        }
        let mut plans = self.plan_found(found, &mut pass, &mut seen)?;
        // Keep spooled bytes and stored parsing reachable when a provider disappears.
        // Databases already observe their pending SDK captures in capture_databases.
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
            let snapshot =
                if self.archive.is_primary() && !self.inputs[&path].origin.is_row_dump(&path) {
                    self.refresh_confirmed_input(&path)?;
                    let source = self
                        .source_read_context(
                            &path,
                            self.inputs[&path].origin.name,
                            polylogue_store::primary::ReadRoute::NativeCapture,
                        )
                        .source;
                    let snapshot = self.source_snapshot(source)?;
                    if snapshot.pending.head.is_some()
                        || snapshot.base.status != polylogue_store::primary::SourceStatus::Ready
                    {
                        pass.capture_pending = true;
                        seen.insert(path.clone());
                    }
                    Some(snapshot)
                } else {
                    None
                };
            let state = &self.inputs[&path];
            if sources && state.facts.is_some() && !path.try_exists()? {
                // A gone observation names the confirmed source, never an old
                // cached identity ahead of a pending capture read or an unreadable range.
                let ready = snapshot.as_ref().is_none_or(|snapshot| {
                    snapshot.base.status == polylogue_store::primary::SourceStatus::Ready
                        && snapshot.pending.head.is_none()
                        && snapshot
                            .confirmed
                            .as_ref()
                            .is_some_and(|confirmed| confirmed.input == state.input_id)
                });
                if ready {
                    let batch = polylogue_store::Batch {
                        required_outcome: None,
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
                        base: snapshot.map(|snapshot| snapshot.base),
                        release: Default::default(),
                    };
                    if self.archive.is_primary() {
                        // An observation without new bytes takes the existing route for
                        // sending a source to Sinex; stored-reading writes cannot send it.
                        let result = self
                            .writer
                            .store_group(
                                vec![polylogue_store::staging::PreparedBatch {
                                    batch,
                                    files: Vec::new(),
                                }],
                                Arc::clone(&self.blobs),
                            )?
                            .pop()
                            .expect("one disappearance observation");
                        match result {
                            Ok(_) => self.refresh_confirmed_input(&path)?,
                            Err(polylogue_store::StoreError::PrimaryPending { .. }) => {
                                pass.capture_pending = true;
                                seen.insert(path.clone());
                            }
                            Err(error) => return Err(error.into()),
                        }
                    } else {
                        self.inputs.get_mut(&path).expect("known input").facts = None;
                        self.writer.send(batch)?;
                    }
                } else {
                    pass.capture_pending = true;
                    seen.insert(path.clone());
                }
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
            self.build_phase(
                "storing and reading changed files",
                format_args!(
                    "{} plans, {} rollouts to capture",
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
                self.build_phase(
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
                self.build_phase(
                    "reading older chains",
                    format_args!("{} sessions derived", pass.sessions_derived),
                );
                self.reparse_behind(RevisionChains::Older, &mut pass)?;
            }
        }
        if self.yielding() {
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
            self.build_phase(
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
            && let Some((phase, began)) = self.build_phase.take()
        {
            eprintln!(
                "polylogue: build: the index is installed; {phase} took {:.0} s",
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
                match self.rollout_step(&f, now, Priority::Bulk, pass) {
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
                    Err(error) => pass
                        .errors
                        .push(format!("capture {}: {error}", f.path.display())),
                }
                continue;
            }
            if !self.track(&f) {
                continue;
            }
            if let Some(plan) = self.plan(&f.path, now)? {
                plans.push(plan);
            }
        }
        // The newest sessions are stored, and so readable, first.
        self.rollout_captures
            .make_contiguous()
            .sort_by_key(|capture| std::cmp::Reverse(capture.facts.mtime_ns));
        Ok(plans)
    }

    /// Reconcile the sampler with confirmed application after a pending capture.
    /// Spooled bytes never supply these logical source coordinates.
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
                pass.errors.push(format!("source watch: {error}"));
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
                match self.rollout_step(f, now, Priority::Live, pass) {
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
                    Err(error) => pass
                        .errors
                        .push(format!("capture {}: {error}", f.path.display())),
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
            let plan = if st.facts == Some(now) && st.working.is_none() && st.stored >= now.size {
                self.stored_plan(&f.path)?
            } else {
                self.plan(&f.path, now)?
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

    /// Records that a file came back holding exactly its stored chain (after
    /// it was gone, or under another inode): an observation, no revision.
    fn observe_back(
        &mut self,
        path: &Path,
        now: FileFacts,
        base: polylogue_store::primary::SourceBase,
    ) -> Result<Option<Plan>, EngineError> {
        let state = &self.inputs[path];
        let batch = polylogue_store::Batch {
            required_outcome: None,
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
            base: Some(base),
            release: Default::default(),
        };
        if self.archive.is_primary() {
            let result = self
                .writer
                .store_group(
                    vec![polylogue_store::staging::PreparedBatch {
                        batch,
                        files: Vec::new(),
                    }],
                    Arc::clone(&self.blobs),
                )?
                .pop()
                .expect("one observation");
            match result {
                Ok(_) => self.refresh_confirmed_input(path)?,
                // Observed again on the next pass, once Sinex confirms the source.
                Err(polylogue_store::StoreError::PrimaryPending { .. }) => return Ok(None),
                Err(error) => return Err(error.into()),
            }
        } else {
            self.writer.send(batch)?;
        }
        let st = self.inputs.get_mut(path).expect("tracked input");
        st.facts = Some(now);
        st.unread = None;
        self.stored_plan(path)
    }

    /// Decides what to read for one tracked path.
    fn plan(&mut self, path: &Path, now: FileFacts) -> Result<Option<Plan>, EngineError> {
        if self.archive.is_primary() {
            self.refresh_confirmed_input(path)?;
        }
        let origin = self.inputs[path].origin;
        let source = self
            .source_read_context(
                path,
                origin.name,
                polylogue_store::primary::ReadRoute::NativeCapture,
            )
            .source;
        let snapshot = self.source_snapshot(source)?;
        let pending = snapshot.pending.head.as_ref();
        if let Some(working) = &self.inputs[path].working
            && pending.is_some_and(|latest| latest.identity == working.capture)
            && working.facts == Some(now)
            && working.stored == now.size
        {
            // This waits on bytes held in the spool, not on confirmed source progress.
            let mut plan = Plan::stored(
                path.to_path_buf(),
                origin,
                InputKey {
                    id: self.inputs[path].input_id.clone(),
                    format: origin.name.into(),
                    route: origin.route,
                },
                Change::Unchanged,
                Some(now),
                now.size,
                now.size,
                pool::Bytes::File,
                self.config.chunk,
                false,
            );
            plan.pending = Some(working.group.clone());
            plan.base = Some(snapshot.base);
            return Ok(Some(plan));
        }
        let source_uncertain =
            snapshot.base.status != polylogue_store::primary::SourceStatus::Ready;
        let obs = self.observation(path, Some(now));
        let st = self.inputs.get_mut(path).expect("tracked input");
        if st.unread == Some(now) {
            return Ok(None);
        }
        let parsed = has_reading(st.origin);
        let mut change = polylogue_store::blobs::classify(st.facts, Some(now));
        let mut back = false;
        if matches!(change, Change::First | Change::Rewritten) && !source_uncertain {
            // Another inode at the path, or the file back after it was gone:
            // when it still begins with every stored byte, it continues its
            // chain (observed back, or appended) instead of starting one.
            let reader = self.reads.reader()?;
            if let Some(progress) = reader.input_progress(&st.input_id)?
                && progress.length > 0
                && progress.length <= now.size
                // An index behind this engine's own stored length has
                // writes still to commit; continuing from it would start
                // the append before them.
                && progress.length >= st.stored
                // A file or stored chain that cannot be read is not shown to
                // hold the stored bytes; the file then starts a new chain.
                && reader
                    .stored_chain(&st.input_id, progress.chain, progress.length)
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
                    st.parsed_upto = progress.parsed_upto;
                    st.parsed = progress.parsed;
                    st.waiting = progress.waiting;
                }
                st.stored = progress.length;
                st.chain = Some(progress.chain);
                if progress.length == now.size {
                    back = true;
                    change = Change::Unchanged;
                } else {
                    change = Change::Appended {
                        from: progress.length,
                    };
                }
            }
        }
        if change == Change::Unchanged && st.facts != Some(now) && !back {
            st.facts = Some(now);
        }
        if back {
            return self.observe_back(path, now, snapshot.base);
        }
        if change == Change::Unchanged && st.stored < now.size {
            // Stored only in part (a stop in the middle of a large file).
            change = Change::Appended { from: st.stored };
        }
        if let Change::Appended { from } = &mut change {
            // The source observation may describe a larger file than the
            // committed prefix after an interrupted import.
            *from = st.stored;
            let tail = if *from > 0 {
                let reader = self.reads.reader()?;
                let progress = reader.input_progress(&st.input_id)?.ok_or_else(|| {
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
            if !read::tail_intact(path, *from, &tail)? {
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
            Change::Reparse { .. } | Change::Observed { .. } => unreachable!(),
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
            byte_read: None,
            base: Some(snapshot.base),
            pending: None,
            checkpoint: None,
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
            tail: None,
            span: span(st.origin, path, self.config.chunk),
            chunk: self.config.chunk,
            priority,
            tracked: true,
            failed: false,
            failure_recorded: false,
            replace: false,
            waiting: false,
            input_finished: false,
            source: pool::Bytes::File,
            prefix_copies: None,
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
    /// once the newest chains are read. An older chain whose bytes begin
    /// like a newer chain's is not parsed over the bytes they share: its
    /// reading is copied from that chain's as the newer chain's parse passes
    /// them, or once that chain is read ([`crate::prefix_copies`]).
    ///
    /// Replacements commit each chain once. Continuing readings run in
    /// rounds with one range plan per path, preserving each path's order.
    pub(crate) fn reparse_behind(
        &mut self,
        chains: RevisionChains,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
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
            return Ok(());
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
                return Ok(());
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
                return Ok(());
            };
            let behind = self.reads.reader()?.chains_behind_index(
                &versions,
                polylogue_store::StoppedChains::ForAttempt(self.stopped_attempt),
                chains,
                Some(&ids),
            )?;
            self.reparse_chains(behind, pass)?;
            self.shared_prefixes.get_mut().clear();
            if self.yielding() || self.archive.stop().is_cancelled() {
                return Ok(());
            }
            self.stored_planning
                .as_mut()
                .expect("planning started")
                .after = Some(last);
        }
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
        // Chains set aside by an earlier pass that ended early are found again.
        self.shared_prefixes.get_mut().clear();
        let mut replacements = Vec::new();
        let mut parse = Vec::new();
        let mut copies = Vec::new();
        let watched: HashMap<String, PathBuf> = self
            .inputs
            .iter()
            .map(|(path, st)| (st.input_id.clone(), path.clone()))
            .collect();
        for c in behind {
            if self.poll_request() {
                break;
            }
            let state = c.progress;
            let stored_source =
                state.imported || self.archive.is_primary() || !self.config.reads_sources;
            let (path, origin) = if stored_source {
                let source = &c.source;
                let Some(origin) = polylogue_sources::origin::by_name(&source.format)
                    .filter(|origin| has_reading(origin))
                else {
                    continue;
                };
                // The recorded path chooses the parser. Without one, this
                // input is refused with its reason and the pass goes on.
                let recorded_path = if !state.imported && !self.archive.is_primary() {
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
                                route: if state.imported {
                                    origin.route.imported()
                                } else {
                                    origin.route
                                },
                            },
                            &state,
                            None,
                            false,
                        )?;
                        pass.errors
                            .push(format!("refuse stored input {}: {reason}", c.input));
                        self.record_stop(
                            &mut plan,
                            polylogue_model::batch::StepProgress::Refused { reason },
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
                // Current work belongs to stored_plan. The query
                // also selects a stopped attempt whose dependencies changed.
                if c.current
                    && ((!c.reading_changed && !c.stopped_at_length) || st.unread.is_some())
                {
                    continue;
                }
                (path.clone(), st.origin)
            };
            if !has_reading(origin) {
                continue;
            }
            let tracked = self.config.reads_sources
                && c.current
                && !state.imported
                && watched.get(&c.input) == Some(&path);
            let facts = tracked
                .then(|| self.inputs.get(&path).and_then(|st| st.facts))
                .flatten();
            let mut reading_state = state.clone();
            if c.reading_changed || c.stopped_at_length {
                // A stopped attempt selected again can have no checkpoint at
                // its failed cursor. Replace its reading from stored bytes.
                reading_state.parsed = false;
                reading_state.parsed_upto = 0;
            }
            let mut plan = self.reparse_plan(
                path,
                origin,
                InputKey {
                    id: c.input,
                    format: origin.name.into(),
                    route: if state.imported {
                        origin.route.imported()
                    } else {
                        origin.route
                    },
                },
                &reading_state,
                facts,
                tracked,
            )?;
            plan.input_finished = state.imported;
            if c.current && origin.arrival == polylogue_sources::Arrival::Live {
                plan.priority = Priority::Live;
            }
            if c.previous_reading || state.parsed && (c.reading_changed || c.stopped_at_length) {
                plan.replace = true;
                replacements.push(plan);
                continue;
            }
            if !c.current
                && !state.parsed
                && !self.archive.is_primary()
                && !origin.is_row_dump(&plan.path)
                && self.copied_later(
                    &plan.input.id,
                    state.chain,
                    &origin.parse_version(),
                    plan.input_finished,
                )?
            {
                // Parsing these bytes again gives a reading another chain
                // has, or will have once read; copy it after this page reads
                // the chains it parses.
                copies.push(plan);
                continue;
            }
            parse.push(plan);
        }
        self.execute_pool(Vec::new(), replacements, pass)?;
        if self.yielding() {
            return Ok(());
        }
        self.execute_by_path(parse, pass)?;
        self.copy_identical_readings(copies, pass)
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

    /// Records each older chain set aside by [`Engine::reparse_behind`] from
    /// the reading of the chain whose bytes it shares, now that the pass has
    /// read the chains it parses. A chain the copy does not read to its end
    /// is parsed on from the copy; one whose copy would not equal its parse
    /// (its source stopped, or its reading writes rows shared by the input's
    /// chains) is parsed whole.
    fn copy_identical_readings(
        &mut self,
        copies: Vec<Plan>,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        if copies.is_empty() || self.poll_request() {
            return Ok(());
        }
        let report = self.flush_writes()?;
        pass.messages_written += report.messages;
        pass.errors.extend(report.errors);
        let mut parse = Vec::new();
        for plan in copies {
            // Each answer commits one complete chain. A request between
            // copies yields now; uncopied chains are selected next pass.
            if self.poll_request() {
                break;
            }
            let Some(chain) = plan.parser_chain else {
                parse.push(plan);
                continue;
            };
            let copy = self.copy_now(
                &plan.input.id,
                chain,
                &plan.origin.parse_version(),
                plan.input_finished,
            )?;
            // Only a parser reading ranges in order continues from a copy.
            let continues = crate::prefix_copies::parses_in_order(&plan);
            let Some(copy) = copy.filter(|copy| copy.reaches_end || continues) else {
                parse.push(plan);
                continue;
            };
            let (reaches_end, upto) = (copy.reaches_end, copy.upto);
            // A copy that would differ, or failed, is parsed instead,
            // which gives the same reading. Transaction and stop errors
            // propagate rather than becoming another parse attempt.
            let answer = self.writer.copy_reading(copy)?;
            if matches!(answer, Ok(true)) {
                pass.readings_copied += 1;
                pass.bytes_copied += upto;
                if reaches_end {
                    continue;
                }
            }
            parse.push(plan);
        }
        self.execute_by_path(parse, pass)
    }

    /// The plan for a watched location whose current chain the index is
    /// behind while its file is gone, moved or unchanged.
    pub(crate) fn stored_plan(&mut self, path: &Path) -> Result<Option<Plan>, EngineError> {
        if self.archive.is_primary() {
            self.refresh_confirmed_input(path)?;
        }
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
        let state = polylogue_store::read::ChainProgress {
            chain,
            imported: false,
            waiting: st.waiting,
            length: st.stored,
            parsed_upto: st.parsed_upto,
            parsed: st.parsed,
        };
        let imported = self
            .reads
            .reader()?
            .input_progress(&input_id)?
            .ok_or_else(|| polylogue_store::StoreError::invalid("the stored revision disappeared"))?
            .imported
            || origin.route.is_imported();
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
            &state,
            facts,
            true,
        )?))
    }

    /// A parse-only plan over a chain's stored bytes. A capture log resumes
    /// at its last parsed record, and a line input whose read was
    /// interrupted (still `parsing`) at its last committed parser
    /// checkpoint; anything else parses from the start as a replacement of
    /// the chain's reading, so a line parser rebuilds its state. The parser
    /// state itself is restored as the plan starts ([`Engine::load_stored`]).
    pub(crate) fn reparse_plan(
        &self,
        path: PathBuf,
        origin: &'static OriginSpec,
        input: InputKey,
        state: &polylogue_store::read::ChainProgress,
        facts: Option<FileFacts>,
        tracked: bool,
    ) -> Result<Plan, EngineError> {
        let parser = (origin.open)(&path);
        let independent_rows =
            origin.is_row_dump(&path) && matches!(parser, Some(polylogue_sources::Parse::Lines(_)));
        let parse_pos = if state.parsed
            && (independent_rows || parser.as_ref().is_some_and(|p| p.is_capture_log()))
        {
            state.parsed_upto
        } else if state.parsed
            && state.parsed_upto > 0
            && matches!(parser, Some(polylogue_sources::Parse::Lines(_)))
            && origin.support == Support::Parsed
        {
            let version = origin.parse_version();
            self.reads
                .reader()?
                .parser_checkpoint(&input.id, state.chain, &version)?
                .filter(|stored| {
                    stored.parse_version == version
                        && stored.parsed_upto == state.parsed_upto
                        && stored.checkpoint.is_some()
                        && stored.progress == polylogue_model::batch::ParseProgress::Parsing
                })
                .map_or(0, |stored| stored.parsed_upto)
        } else {
            0
        };
        let export = matches!(parser, Some(polylogue_sources::Parse::Export(_)));
        let mut plan = Plan::stored(
            path,
            origin,
            input,
            Change::Reparse { chain: state.chain },
            facts,
            state.length,
            parse_pos,
            pool::Bytes::Unloaded {
                chain: state.chain,
                length: state.length,
                decoded: parse_pos == 0 && !export,
            },
            self.config.chunk,
            tracked,
        );
        plan.replace = state.parsed && parse_pos == 0;
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
        if !whole(plan.origin, &plan.path) && plan.origin.support == Support::Parsed {
            let reader = self.reads.reader()?;
            if let Some(stored) =
                reader.parser_checkpoint(&plan.input.id, chain, &plan.origin.parse_version())?
                && stored.parse_version == plan.origin.parse_version()
                && stored.parsed_upto > 0
            {
                plan.parse_pos = stored.parsed_upto;
                plan.read_to = stored.parsed_upto;
                plan.replace = false;
                // A reading another chain's copy took to this chain's end
                // leaves only a record still being written.
                plan.waiting = stored.progress
                    == polylogue_model::batch::ParseProgress::Waiting { length: plan.size };
                plan.parser_state = self.restored_parser_state(
                    &reader,
                    &plan.input.id,
                    chain,
                    stored.parsed_upto,
                    &stored.parse_version,
                )?;
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

    /// Readies a plan for reading: what a byte read already holds, the
    /// captured authored records and local references are recorded first.
    /// None when nothing is left to read, or the plan is refused or failed.
    fn admit(&mut self, mut plan: Plan, pass: &mut Pass) -> Result<Option<Plan>, EngineError> {
        self.prepare_byte_read(&mut plan)?;
        if plan.byte_read.is_some() && plan.finished() {
            self.record_byte_read_status(&plan, None, pass)?;
            self.finish_plan(plan);
            return Ok(None);
        }
        if plan.pending.is_some() && !plan.tracked && plan.finished() {
            pass.capture_pending = true;
            return Ok(None);
        }
        if matches!(plan.change, Change::Reparse { .. })
            && let Some(reason) = unread_path(plan.origin, &plan.path)
        {
            // Stored bytes at a path this build's parser does not read
            // (an earlier build captured it): refused with the reason,
            // never read range by range with nothing to record.
            pass.errors
                .push(format!("refuse stored input {}: {reason}", plan.input.id));
            plan.failed = true;
            self.record_stop(
                &mut plan,
                polylogue_model::batch::StepProgress::Refused { reason },
                pass,
            );
            self.finish_plan(plan);
            return Ok(None);
        }
        if let Err(error) = self.restore_captured_authored(&plan, pass) {
            let error = format!("authored restoration {}: {error}", plan.path.display());
            self.record_failure(&mut plan, error, pass);
            self.finish_plan(plan);
            return Ok(None);
        }
        if let Err(error) = self.observe_local_references(&plan, pass) {
            let error = format!("local references {}: {error}", plan.path.display());
            self.record_failure(&mut plan, error, pass);
            self.finish_plan(plan);
            return Ok(None);
        }
        Ok(Some(plan))
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
        for plan in plans {
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
                self.execute_pool(std::mem::take(&mut pooled), Vec::new(), pass)?;
                if self.yielding() {
                    return Ok(());
                }
                if let Err(error) = self.store_plan(&mut plan, pass) {
                    plan.failed = true;
                    self.finish_plan(plan);
                    pass.errors.push(format!("store before re-parse: {error}"));
                    continue;
                }
            }
            if plan.replace || store_first {
                replacements.push(plan);
                continue;
            }
            if let Some(polylogue_sources::parse::Parse::Export(mut parser)) =
                (plan.origin.open)(&plan.path)
            {
                self.execute_pool(std::mem::take(&mut pooled), Vec::new(), pass)?;
                if self.yielding() {
                    return Ok(());
                }
                if !self.load_stored(&mut plan)? {
                    continue;
                }
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.execute_export(&mut plan, &mut *parser, pass)
                }));
                let failure = match result {
                    Ok(Ok(())) => None,
                    // A stop is no failure of the export: its committed
                    // deliveries stay and the next parse continues after them.
                    Ok(Err(EngineError::Stopped)) => return Err(EngineError::Stopped),
                    Ok(Err(error)) => Some(error.to_string()),
                    Err(_) => Some("the array parser panicked".to_owned()),
                };
                if let Some(error) = failure {
                    let error = format!("parse {}: {error}", plan.path.display());
                    self.record_failure(&mut plan, error, pass);
                }
                self.finish_plan(plan);
            } else {
                pooled.push(plan);
            }
        }
        self.execute_pool(pooled, replacements, pass)?;
        if self.archive.stop().is_cancelled() {
            return Err(EngineError::Stopped);
        }
        Ok(())
    }

    pub(crate) fn execute_pool(
        &mut self,
        plans: Vec<Plan>,
        chains: Vec<Plan>,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        if plans.is_empty() && chains.is_empty() && self.rollout_captures.is_empty() {
            return Ok(());
        }
        if chains
            .iter()
            .any(|plan| !matches!(plan.change, Change::Reparse { .. }))
        {
            return Err(std::io::Error::other("a replacement needs a stored chain").into());
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
                    stopped_attempt: self.stopped_attempt,
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
            let mut refused: Vec<(Plan, String)> = Vec::new();
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
            let mut progress = build_log::Progress::start(self, pass, pool.worker_cpu_seconds());
            // Rollout captures on workers, by path, with the input each names.
            let mut capturing: HashMap<PathBuf, String> = HashMap::new();
            'work: loop {
                if !yielding && self.poll_request() {
                    yielding = true;
                    waiting.clear();
                    chains.clear();
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
                            .collect();
                        self.live_plans(&busy, pass)
                    };
                    match live {
                        Ok(live) => {
                            for plan in live.into_iter().rev() {
                                if plan.replace {
                                    chains.push_front(plan);
                                } else {
                                    waiting.push_front(plan);
                                }
                            }
                        }
                        Err(e) => pass.errors.push(format!("live inputs: {e}")),
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
                        self.make_due_copies(&mut plan, pass)?;
                        pass.capture_pending |= plan.pending.is_some();
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
                    if first {
                        self.plan_prefix_copies(&mut plan)?;
                    }
                    // Copies whose stop the parse has reached take the reading
                    // before the next range changes it.
                    self.make_due_copies(&mut plan, pass)?;
                    if plan.finished() {
                        pass.capture_pending |= plan.pending.is_some();
                        self.finish_plan(plan);
                        continue;
                    }
                    let mut job = plan.next_job(&self.stopped_attempt);
                    if matches!(job.source, pool::Bytes::File) {
                        let mut context = if plan.byte_read.is_none()
                            && !plan.storing_started
                            && plan.base.is_some()
                        {
                            let mut context = self.source_read_context(
                                &job.path,
                                &job.input.format,
                                polylogue_store::primary::ReadRoute::NativeCapture,
                            );
                            context.base = plan.base.clone();
                            context
                        } else {
                            self.sampling_context(&job.path, &job.input.format)?
                        };
                        context.facts = plan.facts;
                        if let Some(byte_read) = &plan.byte_read {
                            job.observation = Some(self.observation(&plan.path, plan.facts));
                            if let Some(cursor) = &byte_read.cursor {
                                if job.store_from == Some(cursor.range().end) {
                                    context.previous_body = Some(cursor.last_body().into());
                                }
                                if cursor.outcome()
                                    != Some(polylogue_store::primary::CaptureOutcome::Prepared)
                                {
                                    context.required_outcome = Some(polylogue_store::primary::CaptureOutcome::UnresolvedBaseConflict);
                                }
                            }
                        }
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
                progress.log(self, pass, || pool.worker_cpu_seconds());
                let d = match done_rx.recv_timeout(every.max(Duration::from_millis(10))) {
                    Ok(d) => d,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                };
                outstanding -= 1;
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
                                pass.errors
                                    .push(format!("capture {}: {error}", path.display()));
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
                    pass.errors.extend(d.error);
                    continue;
                }
                if d.spooled_group.is_some()
                    || plan.byte_read.is_some()
                    || d.pending.is_some()
                        && d.source_read.is_some()
                        && plan.tracked
                        && matches!(plan.source, pool::Bytes::File)
                {
                    if let Some(error) = &d.error {
                        pass.errors.push(error.clone());
                    }
                    self.apply_byte_read_result(&mut plan, &d)?;
                    if plan.finished() {
                        self.record_byte_read_status(&plan, d.source_read.as_ref(), pass)?;
                        self.finish_plan(plan);
                    } else {
                        waiting.push_front(plan);
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
                    refused.push((plan, d.error.unwrap_or_default()));
                    continue;
                }
                if d.write_failed {
                    if let Some(error) = d.error {
                        pass.errors.push(error);
                    }
                    if plan.tracked {
                        self.refresh_confirmed_input(&plan.path)?;
                    }
                    continue;
                }
                pass.bytes_stored += d.stored;
                if let Some(e) = &d.error {
                    pass.errors.push(e.clone());
                }
                let parsed = has_reading(plan.origin);
                if parsed {
                    self.archive
                        .advance_build_bytes(d.parsed_upto.saturating_sub(plan.parse_pos));
                }
                plan.apply(d, parsed);
                if plan.finished() {
                    // A read failure before producing any reading has no external
                    // work to await. Earlier prepared steps still need confirmation.
                    if self.archive.is_primary()
                        && matches!(plan.change, Change::Reparse { .. })
                        && (!plan.failed || plan.failure_recorded || plan.checkpoint.is_some())
                    {
                        if self.confirmed_reading(&plan)? {
                            if plan.tracked {
                                self.refresh_confirmed_input(&plan.path)?;
                            }
                        } else {
                            pass.capture_pending = true;
                        }
                        continue;
                    }
                    self.make_due_copies(&mut plan, pass)?;
                    pass.capture_pending |= plan.pending.is_some();
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
            for (mut plan, error) in refused {
                let error = format!("the writer refused {}: {error}", plan.path.display());
                self.record_failure(&mut plan, error, pass);
                self.finish_plan(plan);
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

    /// A rejected spool cursor cannot become progress. An existing committed
    /// stop at this same cursor can keep its error and bind the retry.
    fn record_retried_stop(&mut self, plan: &mut Plan, pass: &mut Pass) -> Result<(), EngineError> {
        let Change::Reparse { chain } = plan.change else {
            return Ok(());
        };
        let version = plan.origin.parse_version();
        let stored = self
            .reads
            .reader()?
            .parser_checkpoint(&plan.input.id, chain, &version)?;
        let Some(stored) = stored.filter(|stored| {
            stored.parse_version == version
                && stored.parsed_upto == plan.parse_pos
                && stored.progress.at_length() == Some(plan.size)
        }) else {
            return Ok(());
        };
        let progress = match stored.progress {
            polylogue_model::batch::ParseProgress::Failed { error, .. } => {
                polylogue_model::batch::StepProgress::Failed { error }
            }
            polylogue_model::batch::ParseProgress::Refused { reason, .. } => {
                polylogue_model::batch::StepProgress::Refused { reason }
            }
            _ => return Ok(()),
        };
        self.record_stop(plan, progress, pass);
        if self.archive.is_primary() && plan.failure_recorded {
            match self.confirmed_reading(plan) {
                Ok(confirmed) => pass.capture_pending |= !confirmed,
                Err(error) => {
                    pass.capture_pending = true;
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    /// Saving a prepared write does not advance a cached parser cursor. Both
    /// pooled stored parsing and replacement readings wait for confirmation.
    pub(crate) fn confirmed_reading(&self, plan: &Plan) -> Result<bool, EngineError> {
        let Change::Reparse { chain } = plan.change else {
            return Err(polylogue_store::StoreError::invalid(
                "a stored reading needs its confirmed source chain",
            )
            .into());
        };
        let version = plan.origin.parse_version();
        let cursor = self
            .reads
            .reader()?
            .parser_checkpoint(&plan.input.id, chain, &version)?;
        Ok(cursor.is_some_and(|cursor| {
            if cursor.parse_version != version {
                return false;
            }
            if let Some(length) = cursor.progress.at_length() {
                return length == plan.size
                    && if cursor.progress.stopped() {
                        cursor.stopped_attempt == Some(self.stopped_attempt)
                    } else {
                        cursor.parsed_upto >= plan.parse_pos
                    };
            }
            cursor.parsed_upto >= plan.parse_pos
                && (!(plan.waiting || plan.failure_recorded) || cursor.parsed_upto >= plan.size)
        }))
    }

    fn finish_chain(
        &mut self,
        mut plan: Plan,
        written: Result<Option<u64>, EngineError>,
        pass: &mut Pass,
    ) {
        self.finish_chain_path(&plan.path);
        match written {
            Ok(Some(written)) => {
                if self.archive.is_primary() {
                    match self.confirmed_reading(&plan) {
                        Ok(true) => {
                            pass.messages_written += written;
                            if plan.tracked
                                && let Err(error) = self.refresh_confirmed_input(&plan.path)
                            {
                                pass.errors.push(format!("confirmed reading: {error}"));
                            }
                        }
                        Ok(false) => pass.capture_pending = true,
                        Err(error) => pass.errors.push(format!("confirmed reading: {error}")),
                    }
                } else {
                    pass.messages_written += written;
                    self.finish_plan(plan);
                }
            }
            Ok(None) | Err(EngineError::Stopped) => {}
            Err(EngineError::Store(polylogue_store::StoreError::PrimaryPending { .. })) => {
                pass.capture_pending = true;
            }
            Err(error) => {
                pass.errors
                    .push(format!("re-parse {}: {error}", plan.path.display()));
                if matches!(error, EngineError::Parse(_))
                    && let Err(error) = self.record_retried_stop(&mut plan, pass)
                {
                    pass.errors.push(format!("record retried stop: {error}"));
                }
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
        if plan.byte_read.is_some() {
            // A verified byte cursor never updates the input's file facts,
            // parser progress, or the latest accepted working source.
            return;
        }
        if plan.pending.is_some() && !plan.tracked {
            return;
        }
        if let Some(group) = &plan.pending {
            let source = polylogue_store::primary::SourceSelector {
                origin: plan.input.format.clone(),
                machine: self.config.machine.clone(),
                path_bytes: plan.path.as_os_str().as_encoded_bytes().to_vec(),
            };
            let snapshot = self.source_snapshot(source).ok();
            let latest = snapshot.and_then(|snapshot| snapshot.pending.head);
            let capture = latest.and_then(|latest| match latest.captured {
                polylogue_store::primary::Capture::Input(input)
                    if input.rows.input == plan.input.id
                        && input.group.as_ref().is_some_and(|g| &g.identity == group)
                        && input
                            .raw_range
                            .is_some_and(|range| range.end == plan.stored_upto) =>
                {
                    Some(latest.identity)
                }
                _ => None,
            });
            // An accepted retry of the same write can return an older immutable result.
            // Only its still-current header may update the sampler's working
            // state; a newer accepted source observation remains untouched.
            if let Some(capture) = capture
                && let Some(state) = self.inputs.get_mut(&plan.path)
            {
                state.unread = (plan.failed && !plan.failure_recorded)
                    .then_some(plan.facts)
                    .flatten();
                state.working = Some(crate::plan::WorkingState {
                    capture,
                    group: group.clone(),
                    facts: plan.facts,
                    stored: plan.stored_upto,
                });
            }
            return;
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
        tx: Sender<LoopMsg>,
        rx: Receiver<LoopMsg>,
        status: &Arc<Mutex<LoopStatus>>,
        lifetime: Lifetime,
        ready: impl FnOnce(),
    ) -> Result<RunSummary, EngineError> {
        self.watcher = if self.config.reads_sources {
            Some(watch::Roots::new(&self.config, tx.clone())?)
        } else {
            None
        };
        self.wake = Some(tx.clone());
        let mut requests = Requests::new(rx);
        if self.archive.is_primary() {
            let wake = tx.clone();
            requests.commits = Some(self.writer.completion_commits_with_wake(
                polylogue_store::completion::Committed::CONTENT,
                move || {
                    let _ = wake.send(LoopMsg::Committed);
                },
            ));
        }
        self.requests = Some(requests);
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
        status: &Arc<Mutex<LoopStatus>>,
        lifetime: Lifetime,
    ) -> Result<RunSummary, EngineError> {
        let busy = |busy: bool| {
            status.locked().busy = busy;
        };
        let mut summary = RunSummary::default();
        let mut discover = true;
        // The errors of the last failed turn. A turn that fails with the same
        // errors again waits for a change instead of being retried at once.
        let mut failed_with: Option<Vec<String>> = None;
        let mut repeated_failure;
        let mut changed = std::collections::BTreeSet::new();
        let mut watch_errors = Vec::new();
        loop {
            if self.archive.stop().is_cancelled() {
                return Ok(self.stopped(summary));
            }
            discover |= self.take_watched(&mut changed, &mut watch_errors);
            busy(true);
            let pending;
            let mut finished = match self.one_pass(discover, &changed) {
                Err(EngineError::Stopped) => {
                    // Writes the pass queued before stopping are committed.
                    if let Err(error) = self.flush_writes() {
                        summary.work.errors.push(error.to_string());
                    }
                    return Ok(self.stopped(summary));
                }
                Ok((pass, seen)) => {
                    let yielded = self.yielding();
                    pending = pass.capture_pending || yielded;
                    if yielded {
                        // Reload committed cursors after answering the request,
                        // retaining this scope's unfinished planning page.
                        self.discard_confirmed_progress();
                    }
                    if pending {
                        changed.extend(seen);
                    } else {
                        changed.clear();
                    }
                    discover = false;
                    failed_with = None;
                    repeated_failure = false;
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
                        vec![e.to_string()]
                    };
                    match self.flush_writes() {
                        Ok(report) => errors.extend(report.errors),
                        Err(error) => errors.push(error.to_string()),
                    }
                    repeated_failure = failed_with.as_ref() == Some(&errors);
                    failed_with = Some(errors.clone());
                    Pass {
                        full_scans: u64::from(discover && self.config.reads_sources),
                        errors,
                        capture_pending: pending,
                        ..Pass::default()
                    }
                }
            };
            finished.errors.append(&mut watch_errors);
            summary.passes += 1;
            summary.work.add(finished.clone());
            {
                let mut s = status.locked();
                s.full_scans += finished.full_scans;
                if finished.full_scans > 0
                    || finished.inputs_seen > 0
                    || finished.inputs_changed > 0
                    || s.last_pass.capture_pending
                    || finished.capture_pending
                    || !finished.errors.is_empty()
                    || s.passes == 0
                {
                    s.last_pass = finished;
                }
                s.passes += 1;
                s.busy = pending;
            }
            if lifetime == Lifetime::UntilIdle {
                // Work queued while the pass ran (imports, changes the
                // watcher saw) is taken now; with none, the loop is idle. A
                // capture that continues gets `live_every` before the next pass.
                // A turn failing as the one before it ends the run with its
                // errors reported.
                let mut more = pending && !repeated_failure;
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
                        Ok(LoopMsg::Stop) | Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            return Ok(self.stopped(summary));
                        }
                        Ok(message) => {
                            more = true;
                            self.answer(message, &busy, pending, &mut summary);
                        }
                        Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    }
                }
                more |= self.take_watched(&mut changed, &mut watch_errors);
                more |= !repeated_failure && !changed.is_empty();
                if !more {
                    return Ok(self.stopped(summary));
                }
                continue;
            }
            let requests = self.requests.as_mut().expect("running inbox");
            let mut msg = if (!self.config.reads_sources && !pending) || repeated_failure {
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
                            std::sync::mpsc::TryRecvError::Empty => RecvTimeoutError::Timeout,
                            std::sync::mpsc::TryRecvError::Disconnected => {
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
        errors: &mut Vec<String>,
    ) -> bool {
        let Some(watcher) = self.watcher.as_mut() else {
            return false;
        };
        let mut full = false;
        if let Err(error) = watcher.refresh(&self.config) {
            errors.push(format!("source watch: {error}"));
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
    fn answer(
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
                let report = self
                    .import_with_options(
                        &paths,
                        &ImportOptions {
                            authorship: Some(authorship),
                            origin: format,
                            dry_run,
                            ..Default::default()
                        },
                    )
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
            Err(ImportFailure::Import(error)) => {
                summary.work.errors.push(format!("import: {error}"))
            }
            Err(ImportFailure::Parent(_)) => {}
        }
        let _ = reply.send(report);
    }

    fn stopped(&mut self, summary: RunSummary) -> RunSummary {
        self.captures = Databases::default();
        self.wake = None;
        self.requests = None;
        self.watcher = None;
        self.pass_open = false;
        self.stored_planning = None;
        summary
    }
}

#[cfg(test)]
mod tests;
