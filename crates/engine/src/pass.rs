//! One pass of the loop: collect, store, parse, derive; and the loop that
//! runs passes.

use crate::plan::{InputState, Plan, span, whole};
use crate::*;

/// Parser state stays in memory for inputs written within this long.
const KEEP_STATE_NS: i128 = 6 * 3600 * 1_000_000_000;

/// The longest a burst of file events delays the next pass.
const COALESCE: Duration = Duration::from_secs(2);
/// Jobs queued in the parse pool per worker; the rest of a pass's plans
/// wait outside it.
const IN_POOL_PER_WORKER: usize = 16;
/// A known file written this recently is live: the session being written.
const RECENT_NS: i128 = 10 * 60 * 1_000_000_000;

impl Engine {
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

    /// Publish stopped progress at the plan's existing source cursor.
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
        step.reading_version = plan.origin.reading_version();
        step.stopped_attempt = Some(self.stopped_attempt);
        step.prepare();
        let batch = polylogue_store::Batch {
            required_disposition: None,
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
            acquisition: None,
            basis: None,
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
        let state_dir = self.config.state_dir();
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
                state_dir: &state_dir,
                spool: &spool,
            };
            if let Err(error) = self.settle_collector(origin.name, collector, &paths) {
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

    /// Services files and starts or observes an independent database read.
    pub fn run_turn(&mut self) -> Result<Pass, EngineError> {
        self.one_pass(true, &Default::default())
            .map(|(pass, _)| pass)
    }

    pub(super) fn discard_confirmed_state(&mut self) {
        for state in self.inputs.values_mut() {
            state.discard_confirmed();
        }
        self.read_scope = None;
        self.loaded = false;
        self.pass_open = true;
        self.capture_checked = None;
        self.collectors_at = None;
        self.lookups = repo_lookup::Cache::default();
        self.lookups_at = None;
    }

    fn one_pass(
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
        let result = (|| {
            let scope = self.cached_scope()?;
            let result = self.read_turn(discover || !self.loaded, paths)?;
            // A completed reset can race the last SQL read. Its cached progress
            // cannot establish completion in the new scope.
            if let Err(error) = scope.check() {
                if !result.0.errors.is_empty() {
                    return Err(
                        polylogue_store::StoreError::Invalid(result.0.errors.join("; ")).into(),
                    );
                }
                return Err(error.into());
            }
            Ok(result)
        })();
        result.map_err(|error| self.failed_read_turn(error))
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
                    return polylogue_store::StoreError::Invalid(report.errors.join("; ")).into();
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
    ) -> Result<(Pass, HashSet<PathBuf>), EngineError> {
        let continuing = self.pass_open;
        let started = Instant::now();
        self.load()?;
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
            self.writer.coverage(polylogue_model::capture::Coverage {
                version: 1,
                machine: self.config.machine.clone(),
                observed_at: polylogue_model::Timestamp::from_unix_ms(
                    self.config.clock.now().as_millisecond(),
                )
                .ok_or_else(|| std::io::Error::other("capture observation time is out of range"))?,
                roots: scan.roots,
            })?;
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
        let disappeared = paths.iter().any(|path| {
            self.inputs
                .iter()
                .any(|(p, st)| p.starts_with(path) && st.facts.is_some() && !p.exists())
        });
        if !discover
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
        if discover {
            self.reparse_behind(None, &mut pass)?;
        }

        let mut seen = HashSet::new();
        let mut plans = self.plan_found(found, &mut pass, &mut seen)?;
        // Keep custody and stored parsing reachable when a provider disappears.
        // Databases already observe their SDK heads in capture_databases.
        let unseen: Vec<PathBuf> = self
            .inputs
            .iter()
            .filter(|(p, state)| {
                (discover
                    || state.origin.is_row_dump(p)
                    || paths.iter().any(|path| p.starts_with(path)))
                    && !seen.contains(*p)
            })
            .map(|(path, _)| path.clone())
            .collect();
        for path in unseen {
            let snapshot =
                if self.archive.is_primary() && !self.inputs[&path].origin.is_row_dump(&path) {
                    self.refresh_confirmed_input(&path)?;
                    let source = self
                        .acquisition_context(
                            &path,
                            self.inputs[&path].origin.name,
                            polylogue_store::primary::AcquisitionRoute::NativeCapture,
                        )
                        .source;
                    let snapshot = self.writer.source_snapshot(source, "")?;
                    if snapshot.pending.head.is_some()
                        || snapshot.basis.status != polylogue_store::primary::SourceStatus::Ready
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
                // cached identity ahead of a pending acquisition or unreadable cut.
                let ready = snapshot.as_ref().is_none_or(|snapshot| {
                    snapshot.basis.status == polylogue_store::primary::SourceStatus::Ready
                        && snapshot.pending.head.is_none()
                        && snapshot
                            .confirmed
                            .as_ref()
                            .is_some_and(|confirmed| confirmed.input == state.input_id)
                });
                if ready {
                    let batch = polylogue_store::Batch {
                        required_disposition: None,
                        input: InputKey {
                            id: state.input_id.clone(),
                            format: state.origin.name.into(),
                            arrival_rank: state.origin.arrival.rank(),
                        },
                        change: Change::Disappeared,
                        observation: Some(self.observation(&path, None)),
                        continues: false,
                        new_blob: None,
                        blobs: vec![],
                        length: state.stored,
                        parse: Vec::new(),
                        acquisition: None,
                        basis: snapshot.map(|snapshot| snapshot.basis),
                        release: Default::default(),
                    };
                    if self.archive.is_primary() {
                        // Observation-only acquisition uses the existing source
                        // publication route; stored-reading writes cannot publish it.
                        let result = self
                            .writer
                            .sender()?
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
        pass.inputs_changed = plans.len() as u64;
        if plans.is_empty() {
            let report = self.flush_writes()?;
            pass.errors.extend(report.errors);
        }
        self.execute(plans, &mut pass)?;
        self.copy_same_bytes(&mut pass)?;
        if deferred_database && pass.errors.is_empty() && !self.captures.pending() {
            // Local parsing may have made a deferred database runnable. Start
            // that real work in this batch; a still-pending reading just yields.
            self.capture_databases(&[], false, &mut pass)?;
        }
        pass.capture_pending |= self.captures.pending();
        if !pass.capture_pending {
            pass.sessions_derived = self.derive()?;
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
                match self.capture_rollout(&f, now, pass) {
                    Ok(Some(plan)) => {
                        if let Some(previous) = plans
                            .iter_mut()
                            .find(|p: &&mut Plan| p.input.id == plan.input.id)
                        {
                            *previous = plan;
                        } else {
                            plans.push(plan);
                        }
                    }
                    Ok(_) => {}
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
        Ok(plans)
    }

    /// Reconcile the sampler with confirmed application after a pending capture.
    /// Private raw custody never supplies these logical source coordinates.
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
                match self.capture_rollout(f, now, pass) {
                    Ok(Some(mut plan)) => {
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
                    Ok(_) => {}
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
        basis: polylogue_store::primary::SourceBasis,
    ) -> Result<Option<Plan>, EngineError> {
        let state = &self.inputs[path];
        let batch = polylogue_store::Batch {
            required_disposition: None,
            input: InputKey {
                id: state.input_id.clone(),
                format: state.origin.name.into(),
                arrival_rank: state.origin.arrival.rank(),
            },
            change: Change::Unchanged,
            observation: Some(self.observation(path, Some(now))),
            continues: false,
            new_blob: None,
            blobs: vec![],
            length: state.stored,
            parse: Vec::new(),
            acquisition: None,
            basis: Some(basis),
            release: Default::default(),
        };
        if self.archive.is_primary() {
            let result = self
                .writer
                .sender()?
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
                // Observed again on the next pass, once the source is settled.
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
            .acquisition_context(
                path,
                origin.name,
                polylogue_store::primary::AcquisitionRoute::NativeCapture,
            )
            .source;
        let snapshot = self.writer.source_snapshot(source, "")?;
        let pending = snapshot.pending.head.as_ref();
        if let Some(working) = &self.inputs[path].working
            && pending.is_some_and(|head| head.identity == working.capture)
            && working.facts == Some(now)
            && working.stored == now.size
        {
            // This is a private-custody wait, not confirmed source progress.
            let mut plan = Plan::stored(
                path.to_path_buf(),
                origin,
                InputKey {
                    id: self.inputs[path].input_id.clone(),
                    format: origin.name.into(),
                    arrival_rank: origin.arrival.rank(),
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
            plan.basis = Some(snapshot.basis);
            return Ok(Some(plan));
        }
        let source_uncertain =
            snapshot.basis.status != polylogue_store::primary::SourceStatus::Ready;
        let obs = self.observation(path, Some(now));
        let restored = if let Some(st) = self.inputs.get(path)
            && st.parser_state.is_none()
            && st.parsed_upto > 0
        {
            let reader = self.reads.reader()?;
            let known = reader.known_watched_input_at(
                &self.config.machine,
                path.as_os_str().as_encoded_bytes(),
            )?;
            if let Some(known) = known
                && known.input_id == st.input_id
                && known.facts == st.facts
                && known.stored == st.stored
                && known.parsed_upto == st.parsed_upto
            {
                known
                    .chain
                    .map(|chain| {
                        self.restored_parser_state(
                            &reader,
                            &st.input_id,
                            chain,
                            st.parsed_upto,
                            &st.origin.reading_version(),
                        )
                    })
                    .transpose()?
                    .flatten()
            } else {
                None
            }
        } else {
            None
        };
        let mut st = self.inputs.get_mut(path).expect("tracked input");
        if st.unread == Some(now) {
            return Ok(None);
        }
        if st.parser_state.is_none() {
            st.parser_state = restored;
        }
        let parsed = has_reading(st.origin);
        let mut change = polylogue_store::blobs::classify(st.facts, Some(now));
        let mut back = false;
        let mut rejoined = None;
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
                    // state from its admitted checkpoint below.
                    st.parsed_upto = progress.parsed_upto;
                    st.parsed = progress.parsed;
                    st.waiting = progress.waiting;
                    st.parser_state = None;
                    rejoined = Some((progress.chain, progress.parsed_upto));
                }
                st.stored = progress.length;
                st.chain = Some(progress.chain);
                st.tail = None;
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
        if let Some((chain, upto)) = rejoined
            && upto > 0
        {
            let (input, version) = (st.input_id.clone(), st.origin.reading_version());
            let state = {
                let reader = self.reads.reader()?;
                self.restored_parser_state(&reader, &input, chain, upto, &version)?
            };
            st = self.inputs.get_mut(path).expect("tracked input");
            st.parser_state = state;
        }
        if change == Change::Unchanged && st.facts != Some(now) && !back {
            st.facts = Some(now);
        }
        if back {
            return self.observe_back(path, now, snapshot.basis);
        }
        if change == Change::Unchanged && st.stored < now.size {
            // Stored only in part (a stop in the middle of a large file).
            change = Change::Appended { from: st.stored };
        }
        if let Change::Appended { from } = &mut change {
            // The source observation may describe a larger file than the
            // committed prefix after an interrupted import.
            *from = st.stored;
            if st.tail.is_none() && *from > 0 {
                let reader = self.reads.reader()?;
                let progress = reader.input_progress(&st.input_id)?.ok_or_else(|| {
                    polylogue_store::StoreError::Invalid("append has no stored prefix".into())
                })?;
                st.tail = Some(reader.stored_range(
                    &st.input_id,
                    progress.chain,
                    from.saturating_sub(read::TAIL as u64),
                    *from,
                    &self.blobs,
                )?);
            }
            if let Some(tail) = &st.tail
                && !read::tail_intact(path, *from, tail)?
            {
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
                st.parser_state.take(),
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
        Ok(Some(Plan {
            raw: None,
            basis: Some(snapshot.basis),
            pending: None,
            checkpoint: None,
            path: path.to_path_buf(),
            origin: st.origin,
            input: InputKey {
                id: st.input_id.clone(),
                format: st.origin.name.into(),
                arrival_rank: st.origin.arrival.rank(),
            },
            change,
            observation: Some(obs),
            facts: Some(now),
            size: now.size,
            stored_upto,
            parse_pos,
            storing_started: false,
            parser_deferred: source_uncertain || never_parsed && st.stored > 0,
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
        }))
    }

    /// Parses again, from what the archive stored, every revision chain the
    /// index is behind, except the current chain of a watched location,
    /// which follows its file ([`Engine::stored_plan`]). The same loop builds
    /// the index from stored bytes as from files, so a rebuild needs no
    /// command of its own; older chains go through the ordinary parser and
    /// writer, where a newer chain wins. An older chain that holds exactly
    /// the bytes of a newer chain of its input is not parsed: its reading is
    /// that chain's, copied once that chain is read ([`Engine::copy_same_bytes`]).
    /// `selected` limits it to some inputs.
    ///
    /// The chains run on the parse pool together, in rounds that hold one
    /// chain per path, since the pool runs one plan per path at a time.
    pub(crate) fn reparse_behind(
        &mut self,
        selected: Option<&[String]>,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        let behind = self.reads.reader()?.chains_behind_index(
            &crate::reading_contracts()
                .into_iter()
                .filter_map(|contract| match contract {
                    polylogue_store::ReadingContract::Expected { origin, version } => {
                        Some((origin, version))
                    }
                    polylogue_store::ReadingContract::Kept { .. }
                    | polylogue_store::ReadingContract::RawOnly { .. } => None,
                })
                .collect::<Vec<_>>(),
            polylogue_store::StoppedChains::ForAttempt(self.stopped_attempt),
        )?;
        // Chains set aside by an earlier pass that ended early are found again.
        self.same_bytes.clear();
        let mut replacements = Vec::new();
        let mut parse = Vec::new();
        let watched: HashMap<String, PathBuf> = self
            .inputs
            .iter()
            .map(|(path, st)| (st.input_id.clone(), path.clone()))
            .collect();
        for c in behind {
            if selected.is_some_and(|ids| !ids.contains(&c.input)) {
                continue;
            }
            let state = c.progress;
            let (path, origin) = match &c.imported {
                Some(imported) => {
                    let Some(origin) = polylogue_sources::origin::by_name(&imported.format)
                        .filter(|origin| has_reading(origin))
                    else {
                        continue;
                    };
                    // The recorded path chooses the parser. Without one, this
                    // input is refused with its reason and the pass goes on.
                    let path = match imported.path.clone().map(PathBuf::from) {
                        None => Err((
                            PathBuf::new(),
                            "the stored import has no recorded path".to_owned(),
                        )),
                        Some(path)
                            if origin.support == Support::Parsed
                                && (origin.open)(&path).is_none() =>
                        {
                            let reason = format!(
                                "no {} parser reads the recorded path {}",
                                origin.name,
                                path.display()
                            );
                            Err((path, reason))
                        }
                        Some(path) => Ok(path),
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
                                    arrival_rank: 2,
                                },
                                &state,
                                None,
                                false,
                            );
                            pass.errors
                                .push(format!("refuse stored import {}: {reason}", c.input));
                            self.record_stop(
                                &mut plan,
                                polylogue_model::batch::StepProgress::Refused { reason },
                                pass,
                            );
                            continue;
                        }
                    }
                }
                None => {
                    let Some(path) = watched.get(&c.input) else {
                        continue;
                    };
                    let st = &self.inputs[path];
                    // Current ordinary work belongs to stored_plan. The query
                    // also selects a stopped attempt whose dependencies changed.
                    if c.current
                        && ((!c.reading_changed && !c.stopped_at_length) || st.unread.is_some())
                    {
                        continue;
                    }
                    (path.clone(), st.origin)
                }
            };
            if !has_reading(origin) {
                continue;
            }
            let tracked = c.current && c.imported.is_none() && watched.get(&c.input) == Some(&path);
            let facts = tracked
                .then(|| self.inputs.get(&path).and_then(|st| st.facts))
                .flatten();
            let mut reading_state = state.clone();
            if c.reading_changed || c.stopped_at_length {
                // A newly eligible stopped attempt can have no checkpoint at
                // its failed cursor. Replace its reading from retained bytes.
                reading_state.parsed = false;
                reading_state.parsed_upto = 0;
            }
            let mut plan = self.reparse_plan(
                path,
                origin,
                InputKey {
                    id: c.input,
                    format: origin.name.into(),
                    arrival_rank: if c.imported.is_some() {
                        2
                    } else {
                        origin.arrival.rank()
                    },
                },
                &reading_state,
                facts,
                tracked,
            );
            plan.input_finished = c.imported.is_some();
            if c.current && origin.arrival == polylogue_sources::Arrival::Live {
                plan.priority = Priority::Live;
            }
            if state.parsed && (c.reading_changed || c.stopped_at_length) {
                replacements.push(plan);
                continue;
            }
            if !c.current
                && !state.parsed
                && !self.archive.is_primary()
                && !origin.is_row_dump(&plan.path)
                && let Some(newer) = self
                    .reads
                    .reader()?
                    .newest_identical_chain(&plan.input.id, state.chain)?
            {
                // Parsing the same bytes again gives the newer chain's
                // reading; it is copied once that chain is read.
                let copy = polylogue_store::ReadingCopy {
                    input: plan.input.id.clone(),
                    from: newer,
                    to: state.chain,
                    reading_version: origin.reading_version(),
                };
                self.same_bytes.push((plan, copy));
                continue;
            }
            parse.push(plan);
        }
        self.replace_readings(replacements, pass)?;
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
            self.execute(round, pass)?;
        }
        Ok(())
    }

    /// Records each older chain set aside by [`Engine::reparse_behind`] from
    /// its newer chain of the same bytes, now that the pass has read that
    /// chain. A chain whose copy would not equal its parse (the newer chain
    /// stopped, or its reading writes rows shared by the input's chains) is
    /// parsed instead.
    fn copy_same_bytes(&mut self, pass: &mut Pass) -> Result<(), EngineError> {
        if self.same_bytes.is_empty() {
            return Ok(());
        }
        let report = self.flush_writes()?;
        pass.messages_written += report.messages;
        pass.errors.extend(report.errors);
        let (plans, copies): (Vec<Plan>, Vec<_>) =
            std::mem::take(&mut self.same_bytes).into_iter().unzip();
        let answers = self.writer.sender()?.copy_readings(copies)?;
        let mut parse = Vec::new();
        for (plan, answer) in plans.into_iter().zip(answers) {
            // A copy that would not be exact, or failed, is parsed instead,
            // which gives the same reading.
            if matches!(answer, Ok(true)) {
                pass.readings_copied += 1;
            } else {
                parse.push(plan);
            }
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
            arrival_rank: 0,
            waiting: st.waiting,
            length: st.stored,
            parsed_upto: st.parsed_upto,
            parsed: st.parsed,
        };
        let arrival_rank = self
            .reads
            .reader()?
            .input_progress(&input_id)?
            .ok_or_else(|| {
                polylogue_store::StoreError::Invalid("the stored revision disappeared".into())
            })?
            .arrival_rank
            .max(origin.arrival.rank());
        Ok(Some(self.reparse_plan(
            path.to_path_buf(),
            origin,
            InputKey {
                id: input_id,
                format: origin.name.into(),
                arrival_rank,
            },
            &state,
            facts,
            true,
        )))
    }

    /// A parse-only plan over a chain's stored bytes. A capture log resumes
    /// at its last parsed record; anything else parses from the start, so a
    /// line parser rebuilds its state. `None` while the bytes are still
    /// being written.
    fn reparse_plan(
        &self,
        path: PathBuf,
        origin: &'static OriginSpec,
        input: InputKey,
        state: &polylogue_store::read::ChainProgress,
        facts: Option<FileFacts>,
        tracked: bool,
    ) -> Plan {
        let parser = (origin.open)(&path);
        let independent_rows =
            origin.is_row_dump(&path) && matches!(parser, Some(polylogue_sources::Parse::Lines(_)));
        let parse_pos = if state.parsed
            && (independent_rows || parser.as_ref().is_some_and(|p| p.is_capture_log()))
        {
            state.parsed_upto
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
        plan
    }

    /// Loads a plan's stored bytes as it starts ([`pool::Bytes::Unloaded`]).
    /// False while they are still being written: the plan waits for a later
    /// pass.
    pub(crate) fn load_stored(&self, plan: &mut Plan) -> Result<bool, EngineError> {
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
                reader.parser_checkpoint(&plan.input.id, chain, &plan.origin.reading_version())?
                && stored.reading_version == plan.origin.reading_version()
                && stored.parsed_upto > 0
            {
                plan.parse_pos = stored.parsed_upto;
                plan.replace = false;
                plan.parser_state = self.restored_parser_state(
                    &reader,
                    &plan.input.id,
                    chain,
                    stored.parsed_upto,
                    &stored.reading_version,
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

    /// Runs plans on the parse pool, one job per input at a time, until
    /// every input is stored and parsed up to its size; then waits for the
    /// writer to commit.
    pub(crate) fn execute(&mut self, plans: Vec<Plan>, pass: &mut Pass) -> Result<(), EngineError> {
        if plans.is_empty() {
            return Ok(());
        }
        // Keep the existing priority/size order across parser kinds. An array
        // runs serially only after preceding pool work has committed.
        let mut plans = plans;
        plans.sort_by_key(|p| (p.priority == Priority::Bulk, std::cmp::Reverse(p.size)));
        let mut ordinary = Vec::new();
        let mut replacements = Vec::new();
        for mut plan in plans {
            self.prepare_raw_plan(&mut plan)?;
            if plan.raw.is_some() && plan.finished() {
                self.record_raw_status(&plan, None, pass)?;
                self.finish_plan(plan);
                continue;
            }
            if plan.pending.is_some() && !plan.tracked && plan.finished() {
                pass.capture_pending = true;
                continue;
            }

            if let Err(error) = self.restore_captured_authored(&plan, pass) {
                let error = format!("authored restoration {}: {error}", plan.path.display());
                self.record_failure(&mut plan, error, pass);

                self.finish_plan(plan);
                continue;
            }
            if let Err(error) = self.observe_local_references(&plan, pass) {
                let error = format!("local references {}: {error}", plan.path.display());
                self.record_failure(&mut plan, error, pass);
                self.finish_plan(plan);
                continue;
            }
            let store_first = matches!(plan.source, pool::Bytes::File)
                && plan.parse_pos == 0
                && plan.stored_upto > 0
                && !matches!(plan.change, Change::First | Change::Rewritten);
            if plan.replace || store_first {
                self.execute_pool(std::mem::take(&mut ordinary), pass)?;
                if store_first && let Err(error) = self.store_plan(&mut plan, pass) {
                    plan.failed = true;
                    self.finish_plan(plan);
                    pass.errors.push(format!("store before re-parse: {error}"));
                    continue;
                }
                replacements.push(plan);
                continue;
            }
            if let Some(polylogue_sources::parse::Parse::Export(mut parser)) =
                (plan.origin.open)(&plan.path)
            {
                self.execute_pool(std::mem::take(&mut ordinary), pass)?;
                if !self.load_stored(&mut plan)? {
                    continue;
                }
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.execute_export(&mut plan, &mut *parser, pass)
                }));
                let failure = match result {
                    Ok(Ok(())) => None,
                    Ok(Err(error)) => Some(error.to_string()),
                    Err(_) => Some("the array parser panicked".to_owned()),
                };
                if let Some(error) = failure {
                    let error = format!("parse {}: {error}", plan.path.display());
                    self.record_failure(&mut plan, error, pass);
                }
                self.finish_plan(plan);
            } else {
                ordinary.push(plan);
            }
        }
        self.execute_pool(ordinary, pass)?;
        self.replace_readings(replacements, pass)?;
        Ok(())
    }

    fn execute_pool(&mut self, plans: Vec<Plan>, pass: &mut Pass) -> Result<(), EngineError> {
        if plans.is_empty() {
            return Ok(());
        }
        let (done_tx, done_rx) = channel();
        let pool = Pool::start(
            self.config.workers,
            self.config.bytes_in_flight,
            self.writer.sender()?,
            Arc::clone(&self.blobs),
            self.archive.clone(),
            done_tx,
        );
        // Plans wait here and enter the pool a few at a time, so live inputs
        // that arrive meanwhile can go ahead of what is still waiting.
        let mut waiting: std::collections::VecDeque<Plan> = plans.into();
        let mut running: HashMap<PathBuf, Plan> = HashMap::new();
        // Inputs whose first job this run queued. An input holds its parser
        // state between jobs until its last finishes, so a new input starts
        // only when its first range can be queued at once; otherwise the
        // loop would wait inside `submit` with a new input while finished
        // ranges of started ones wait in `done_rx`, and every large input of
        // a pass would end up started, each holding its state.
        let mut started: HashSet<PathBuf> = HashSet::new();
        let mut outstanding = 0usize;
        let mut checked = Instant::now();
        let room = self.config.workers.max(1) * IN_POOL_PER_WORKER;
        let every = self.config.live_every;
        loop {
            if checked.elapsed() >= every {
                checked = Instant::now();
                let live = {
                    // A continuation owns its path while queued, too. Finish
                    // its planned bytes before observing later source changes.
                    let busy = running
                        .keys()
                        .map(PathBuf::as_path)
                        .chain(waiting.iter().map(|plan| plan.path.as_path()))
                        .collect();
                    self.live_plans(&busy, pass)
                };
                match live {
                    Ok(live) => {
                        for plan in live.into_iter().rev() {
                            waiting.push_front(plan);
                        }
                    }
                    Err(e) => pass.errors.push(format!("live inputs: {e}")),
                }
            }
            while outstanding < room
                && let Some(mut plan) = waiting.pop_front()
            {
                if plan.finished() {
                    pass.capture_pending |= plan.pending.is_some();
                    self.finish_plan(plan);
                    continue;
                }
                if !started.contains(&plan.path) {
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
                let mut job = plan.next_job(&self.stopped_attempt);
                if matches!(job.source, pool::Bytes::File) {
                    let mut context =
                        if plan.raw.is_none() && !plan.storing_started && plan.basis.is_some() {
                            let mut context = self.acquisition_context(
                                &job.path,
                                &job.input.format,
                                polylogue_store::primary::AcquisitionRoute::NativeCapture,
                            );
                            context.basis = plan.basis.clone();
                            context
                        } else {
                            self.sampling_context(
                                &job.path,
                                &job.input.format,
                                &job.origin.reading_version(),
                            )?
                        };
                    context.facts = plan.facts;
                    if let Some(raw) = &plan.raw {
                        job.observation = Some(self.observation(&plan.path, plan.facts));
                        if let Some(cursor) = &raw.cursor {
                            if job.store_from == Some(cursor.range().end) {
                                context.raw_predecessor = Some(cursor.last_body().into());
                            }
                            if cursor.disposition()
                                != Some(polylogue_store::primary::CaptureDisposition::Prepared)
                            {
                                context.required_disposition = Some(polylogue_store::primary::CaptureDisposition::UnresolvedBasisConflict);
                            }
                        }
                    }
                    job.acquisition = Some(context);
                }
                pass.bytes_read += job.end - job.start;
                pool.submit(job);
                outstanding += 1;
                running.insert(plan.path.clone(), plan);
            }
            if outstanding == 0 {
                break;
            }
            let d = match done_rx.recv_timeout(every.max(Duration::from_millis(10))) {
                Ok(d) => d,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            };
            outstanding -= 1;
            let Some(mut plan) = running.remove(&d.path) else {
                continue;
            };
            if d.raw_custody.is_some()
                || plan.raw.is_some()
                || d.pending.is_some()
                    && d.raw_acquisition.is_some()
                    && plan.tracked
                    && matches!(plan.source, pool::Bytes::File)
            {
                if let Some(error) = &d.error {
                    pass.errors.push(error.clone());
                }
                self.apply_raw_result(&mut plan, &d)?;
                if plan.finished() {
                    self.record_raw_status(&plan, d.raw_acquisition.as_ref(), pass)?;
                    self.finish_plan(plan);
                } else {
                    waiting.push_front(plan);
                }
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
            if d.continuation_missing && !plan.parser_deferred {
                pass.errors.push(format!(
                    "parser continuation unavailable at {}:{}; captured bytes retained",
                    plan.input.id, plan.parse_pos
                ));
            }
            if let Some(e) = &d.error {
                pass.errors.push(e.clone());
            }
            let parsed = has_reading(plan.origin);
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
        Ok(())
    }

    /// Records what a finished plan leaves for the next pass. Parser state
    /// is kept in memory only for files written recently; an older file's
    /// state is restored from its admitted checkpoint if it grows.
    pub(crate) fn finish_plan(&mut self, plan: Plan) {
        if plan.raw.is_some() {
            // A verified physical cursor never updates logical input facts,
            // parser progress, or the latest accepted working source head.
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
            let snapshot = self
                .writer
                .source_snapshot(source, &plan.origin.reading_version())
                .ok();
            let head = snapshot.and_then(|snapshot| snapshot.pending.head);
            let capture = head.and_then(|head| match head.captured {
                polylogue_store::primary::Capture::Input(input)
                    if input.rows.input == plan.input.id
                        && input.group.as_ref().is_some_and(|g| &g.identity == group)
                        && input
                            .raw_range
                            .is_some_and(|range| range.end == plan.stored_upto) =>
                {
                    Some(head.identity)
                }
                _ => None,
            });
            // An exact accepted retry can return an older immutable receipt.
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
                st.parser_state = plan
                    .checkpoint
                    .and_then(|checkpoint| checkpoint.parser_state);
                st.waiting = true;
                return;
            }
            if plan.failed {
                // Tried again when the bytes change; the facts stay old, so
                // a change is still seen as one.
                st.parser_state = None;
                st.waiting = false;
                st.unread = plan.facts;
                if plan.tail.is_some() {
                    st.tail = plan.tail;
                }
                return;
            }
            st.unread = None;
            let recent = plan.facts.is_some_and(|f| {
                i128::from(self.config.clock.now().as_millisecond()) * 1_000_000 - f.mtime_ns
                    < KEEP_STATE_NS
            });
            st.facts = plan.facts;
            st.parsed_upto = if has_reading(plan.origin) {
                plan.parse_pos.min(plan.size)
            } else {
                0
            };
            st.parsed = has_reading(plan.origin) && (!plan.parser_deferred || st.parsed);
            st.waiting = plan.waiting;
            st.parser_state = if recent { plan.parser_state } else { None };
            st.tail = if recent { plan.tail } else { None };
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
        rx: &Receiver<LoopMsg>,
        status: &Arc<Mutex<LoopStatus>>,
        lifetime: Lifetime,
        ready: impl FnOnce(),
    ) -> Result<RunSummary, EngineError> {
        let busy = |busy: bool| {
            if let Ok(mut s) = status.lock() {
                s.busy = busy;
            }
        };
        self.watcher = if self.config.reads_sources {
            Some(watch::Roots::new(&self.config, tx.clone())?)
        } else {
            None
        };
        self.wake = Some(tx.clone());
        drop(tx);
        ready();
        let mut summary = RunSummary::default();
        let mut discover = true;
        let mut changed = std::collections::BTreeSet::new();
        let mut watch_errors = Vec::new();
        loop {
            discover |= self.take_watched(&mut changed, &mut watch_errors);
            busy(true);
            let pending;
            let mut finished = match self.one_pass(discover, &changed) {
                Ok((pass, seen)) => {
                    pending = pass.capture_pending;
                    if pending {
                        changed.extend(seen);
                    } else {
                        changed.clear();
                    }
                    discover = false;
                    pass
                }
                Err(e) => {
                    // A failed turn cannot establish that previously pending
                    // custody has settled. Only a completed pass updates it.
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
            if let Ok(mut s) = status.lock() {
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
                let mut more = pending;
                if pending {
                    match rx.recv_timeout(self.config.live_every) {
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
                    match rx.try_recv() {
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
                more |= !changed.is_empty();
                if !more {
                    return Ok(self.stopped(summary));
                }
                continue;
            }
            let mut msg = rx.recv_timeout(self.config.live_every);
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
                if burst.elapsed() > COALESCE {
                    break;
                }
                // Pending captures continue after queued requests, without
                // waiting for a watcher event or a coalescing delay.
                if pending {
                    msg = rx.try_recv().map_err(|e| match e {
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
                msg = rx.recv_timeout(Duration::from_millis(200));
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
            from_archive,
            dry_run,
            authorship,
            parent,
            reply,
        } = message
        else {
            return;
        };
        // Queue waits belong to the producing transport. After
        // admission the engine owns completion and its ACK.
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
                            format,
                            from_archive,
                            dry_run,
                            ..Default::default()
                        },
                    )
                    .map_err(|e| ImportFailure::Import(e.to_string()));
                busy(pending);
                report
            });
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
        self.watcher = None;
        self.pass_open = false;
        summary
    }
}

#[cfg(test)]
mod tests;
