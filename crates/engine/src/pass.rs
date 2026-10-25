//! One pass of the loop: collect, store, parse, derive; and the loop that
//! runs passes.

use crate::plan::{FailedChain, InputState, Plan, span, whole};
use crate::*;

/// Parser state stays in memory for inputs written within this long.
const KEEP_STATE_NS: i128 = 6 * 3600 * 1_000_000_000;

/// The longest a burst of file events delays the next pass.
const COALESCE: Duration = Duration::from_secs(2);

impl Engine {
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
        let read_at = polylogue_model::Timestamp::from_unix_ms(now_ms())
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

    /// Runs the collector of every origin that has one and whose setting
    /// the operator turned on. Returns the errors to report on the pass.
    fn run_collectors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        for origin in ORIGINS {
            let Some(collector) = &origin.collector else {
                continue;
            };
            let on = self
                .reads
                .reader()
                .and_then(|r| r.setting(collector.setting, &self.config.machine));
            match on {
                Ok(Some(value)) if value == "on" => {}
                Ok(_) => continue,
                Err(e) => {
                    errors.push(format!("{} collector: {e}", origin.name));
                    continue;
                }
            }
            let spool = self.config.root(origin.root).join(
                origin
                    .spool_dir()
                    .expect("an origin with a collector has a spool"),
            );
            if let Err(e) = (collector.sync)(&self.config.state_dir(), &spool) {
                errors.push(format!("{} collector: {e}", origin.name));
            }
        }
        errors
    }

    /// Completes a finite capture batch. The daemon also discovers source
    /// changes and handles queued requests between its turns.
    pub fn converge(&mut self) -> Result<Pass, EngineError> {
        let mut total = self.converge_turn()?;
        while total.capture_pending {
            let next = self.converge_work(false)?;
            total.inputs_seen += next.inputs_seen;
            total.inputs_changed += next.inputs_changed;
            total.bytes_read += next.bytes_read;
            total.bytes_stored += next.bytes_stored;
            total.messages_written += next.messages_written;
            total.sessions_derived += next.sessions_derived;
            total.seconds += next.seconds;
            total.unreadable_roots = next.unreadable_roots;
            total.pending_inputs = next.pending_inputs;
            total.errors.extend(next.errors);
            total.capture_pending = next.capture_pending;
        }
        Ok(total)
    }

    /// Services ordinary inputs and one cooperative SQLite capture turn.
    pub fn converge_turn(&mut self) -> Result<Pass, EngineError> {
        self.converge_work(true)
    }

    fn converge_work(&mut self, discover: bool) -> Result<Pass, EngineError> {
        let started = Instant::now();
        let pending_inputs = self.writer.finish_writing(Arc::clone(&self.blobs))?.pending;
        self.load()?;
        self.restore_failed_writes()?;
        let (found, unreadable, errors) = if discover {
            let mut errors = self.run_collectors();
            errors.extend(self.refresh_lookups().err());
            let (found, unreadable) = collect::scan(&self.config);
            (found, unreadable, errors)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        let mut pass = Pass {
            pending_inputs,
            unreadable_roots: unreadable,
            errors,
            ..Pass::default()
        };
        // The asset lookup comes from stored observations and sidecars, so
        // after an index rebuild it is in place before any message that
        // names an asset is parsed. Imports stored in this pass add theirs
        // at its end.
        self.refresh_import_assets(&mut pass)?;
        if discover {
            self.reparse_behind(None, &mut pass)?;
        }
        let databases = found
            .iter()
            .filter(|f| f.origin.capture == Capture::LogicalExport)
            .filter_map(|f| {
                let name = polylogue_sources::origin::export_name(&f.path)?;
                Some((
                    f.path.clone(),
                    self.archive
                        .exports_dir()
                        .join(f.origin.name)
                        .join(self.location_id(&f.path))
                        .join(name),
                    Some(self.observation(&f.path, None)),
                ))
            })
            .collect();
        match self.exporter.turn(databases, 1024) {
            Ok(turn) => {
                pass.capture_pending = turn.pending;
                pass.errors.extend(turn.errors);
            }
            Err(e) => pass.errors.push(format!("SQLite capture: {e}")),
        }
        let mut found: Vec<_> = found
            .into_iter()
            .filter(|f| f.origin.capture != Capture::LogicalExport)
            .collect();
        // Accepted captures remain inputs even if their provider disappears
        // while a pinned source view is being read.
        for out in self.exporter.paths() {
            if let Some(origin) = ORIGINS.iter().find(|o| {
                o.capture == Capture::LogicalExport
                    && out.starts_with(self.archive.exports_dir().join(o.name))
            }) {
                found.push(collect::Found {
                    path: out.into(),
                    origin,
                });
            }
        }
        let mut plans = Vec::new();
        let mut seen = HashSet::new();
        for f in found {
            if self.refused.contains(&f.path) {
                continue;
            }
            seen.insert(f.path.clone());
            let Ok(meta) = std::fs::metadata(&f.path) else {
                continue;
            };
            pass.inputs_seen += 1;
            let now = FileFacts::of(&meta);
            if !self.inputs.contains_key(&f.path) {
                if !collect::recognize(&f) {
                    continue;
                }
                let id = self.location_id(&f.path);
                self.inputs.insert(
                    f.path.clone(),
                    InputState {
                        input_id: id,
                        origin: f.origin,
                        facts: None,
                        stored: 0,
                        parsed_upto: 0,
                        parsed: false,
                        waiting: false,
                        carry: None,
                        tail: None,
                        chain: None,
                        failed: None,
                        failed_history: HashMap::new(),
                    },
                );
            }
            let pending = pass
                .pending_inputs
                .iter()
                .any(|u| u.input == self.inputs[&f.path].input_id);
            if let Some(plan) = self.plan(&f.path, now, pending)? {
                plans.push(plan);
            }
        }
        // Paths seen before and gone now: an observation, never a delete.
        let gone: Vec<PathBuf> = self
            .inputs
            .iter()
            .filter(|(p, s)| {
                (discover || s.origin.capture == Capture::LogicalExport)
                    && s.facts.is_some()
                    && !seen.contains(*p)
                    && !p.exists()
            })
            .map(|(p, _)| p.clone())
            .collect();
        for path in gone {
            let st = self.inputs.get_mut(&path).expect("known input");
            st.facts = None;
            let batch = polylogue_store::Batch {
                input: InputKey {
                    id: st.input_id.clone(),
                    format: st.origin.name.into(),
                    route_rank: 0,
                },
                change: Change::Disappeared,
                observation: Some(Observation {
                    machine: self.config.machine.clone(),
                    host: self.config.host.clone(),
                    path: path.display().to_string(),
                    facts: None,
                    seen_at_ms: now_ms(),
                    import: None,
                }),
                continues: false,
                piece: None,
                blobs: vec![],
                length: st.stored,
                parse: Vec::new(),
                release: Default::default(),
            };
            self.writer.send(batch)?;
        }
        // Inputs whose files are gone or moved are parsed from what the
        // archive stored, when the index is behind it.
        let unseen: Vec<PathBuf> = self
            .inputs
            .iter()
            .filter(|(p, s)| {
                (discover || s.origin.capture == Capture::LogicalExport) && !seen.contains(*p)
            })
            .map(|(p, _)| p.clone())
            .collect();
        for path in unseen {
            if let Some(plan) = self.stored_plan(&path)? {
                plans.push(plan);
            }
        }
        pass.inputs_changed = plans.len() as u64;
        if plans.is_empty() {
            let report = self.flush_writes()?;
            pass.errors.extend(report.errors);
        }
        self.execute(plans, &mut pass)?;
        self.restore_failed_writes()?;
        self.refresh_import_assets(&mut pass)?;
        pass.sessions_derived = self.derive()?;
        pass.seconds = started.elapsed().as_secs_f64();
        Ok(pass)
    }

    /// Decides what to read for one tracked path.
    fn plan(
        &mut self,
        path: &Path,
        now: FileFacts,
        pending: bool,
    ) -> Result<Option<Plan>, EngineError> {
        let obs = self.observation(path, Some(now));
        let st = self.inputs.get_mut(path).expect("tracked input");
        if st.failed == Some(now) {
            return Ok(None);
        }
        let parsed = st.origin.support == Support::Parsed;
        let mut change = polylogue_store::blobs::classify(st.facts, Some(now));
        if change == Change::Unchanged && st.stored < now.size {
            // Stored only in part (a stop in the middle of a large file).
            change = Change::Appended { from: st.stored };
        }
        if pending && let Some(chain) = st.chain {
            match self
                .reads
                .reader()?
                .check_bytes_present(&st.input_id, chain, 0..st.stored)
            {
                Ok(()) => {}
                Err(polylogue_store::StoreError::InputPending { .. }) => {
                    // Bytes still being written are never replaced by what the
                    // file holds now: equal bytes are stored again under the
                    // same hash, edited bytes start a new chain.
                    change = Change::Rewritten;
                }
                Err(error) => return Err(error.into()),
            }
        }
        if let Change::Appended { from } = &mut change {
            // The source observation may describe a larger file than the
            // committed prefix after an interrupted acquisition.
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
        let (piece_pos, parse_pos, carry, priority) = match &change {
            Change::First | Change::Rewritten => {
                st.stored = 0;
                st.parsed_upto = 0;
                st.chain = None;
                (0, 0, None, Priority::Bulk)
            }
            Change::Appended { .. } => {
                let resumable = st.carry.is_some()
                    || st.parsed_upto == 0
                    || (st.origin.open)(path).is_some_and(|p| p.is_capture_log());
                if resumable && !never_parsed {
                    (st.stored, st.parsed_upto, st.carry.take(), Priority::Live)
                } else {
                    (st.stored, 0, None, Priority::Live)
                }
            }
            Change::Unchanged | Change::Disappeared => return self.stored_plan(path),
            Change::Reparse { .. } => unreachable!(),
        };
        // A document is parsed whole: any change parses it from its start.
        let (parse_pos, carry) = if whole(st.origin, path) {
            (0, None)
        } else {
            (parse_pos, carry)
        };
        Ok(Some(Plan {
            path: path.to_path_buf(),
            origin: st.origin,
            input: InputKey {
                id: st.input_id.clone(),
                format: st.origin.name.into(),
                route_rank: 0,
            },
            change,
            observation: Some(obs),
            facts: Some(now),
            size: now.size,
            piece_pos,
            parse_pos,
            piece_started: false,
            carry,
            tail: None,
            span: span(st.origin, path, self.config.chunk),
            chunk: self.config.chunk,
            priority,
            tracked: true,
            failed: false,
            waiting: false,
            source: pool::Bytes::File,
        }))
    }

    /// Parses again, from what the archive stored, every revision chain the
    /// index is behind, except the current chain of a watched location,
    /// which follows its file ([`Engine::stored_plan`]). The same loop builds
    /// the index from stored bytes as from files, so a rebuild needs no
    /// command of its own; older chains go through the ordinary parser and
    /// writer, where a newer chain wins. `selected` limits it to some inputs.
    pub(crate) fn reparse_behind(
        &mut self,
        selected: Option<&[String]>,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        let behind = self.reads.reader()?.chains_behind_index()?;
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
                        .filter(|origin| origin.support == Support::Parsed)
                    else {
                        continue;
                    };
                    let path = PathBuf::from(imported.path.clone().ok_or_else(|| {
                        std::io::Error::other(format!(
                            "stored import {} has no recorded parser path",
                            c.input
                        ))
                    })?);
                    if (origin.open)(&path).is_none() {
                        return Err(std::io::Error::other(format!(
                            "stored import {} has no parser for its recorded path",
                            c.input
                        ))
                        .into());
                    }
                    (path, origin)
                }
                None => {
                    let Some(path) = watched.get(&c.input) else {
                        continue;
                    };
                    let st = &self.inputs[path];
                    // A whole-document input's older chain is replaced by its
                    // current one; a chain that failed waits for new bytes.
                    let lines = matches!(
                        (st.origin.open)(path),
                        Some(
                            polylogue_sources::Parse::Lines(_)
                                | polylogue_sources::Parse::CaptureLog(_)
                        )
                    );
                    let failed = st.failed_history.get(&state.chain)
                        == Some(&FailedChain {
                            length: state.length,
                        });
                    if c.current || !lines || failed {
                        continue;
                    }
                    (path.clone(), st.origin)
                }
            };
            let Some(plan) = self.reparse_plan(
                path,
                origin,
                InputKey {
                    id: c.input,
                    format: origin.name.into(),
                    route_rank: state.route_rank,
                },
                &state,
                None,
                false,
            )?
            else {
                continue;
            };
            self.execute(vec![plan], pass)?;
        }
        Ok(())
    }

    /// The plan for a watched location whose current chain the index is
    /// behind while its file is gone, moved or unchanged.
    pub(crate) fn stored_plan(&mut self, path: &Path) -> Result<Option<Plan>, EngineError> {
        let Some(st) = self.inputs.get(path) else {
            return Ok(None);
        };
        let behind = (!st.waiting && st.parsed_upto < st.stored) || !st.parsed;
        let Some(chain) = st.chain else {
            return Ok(None);
        };
        if st.origin.support != Support::Parsed || st.stored == 0 || !behind {
            return Ok(None);
        }
        let (input_id, origin, facts) = (st.input_id.clone(), st.origin, st.facts);
        let state = polylogue_store::read::ChainProgress {
            chain,
            route_rank: 0,
            waiting: st.waiting,
            length: st.stored,
            parsed_upto: st.parsed_upto,
            parsed: st.parsed,
        };
        let route_rank = self
            .reads
            .reader()?
            .input_progress(&input_id)?
            .ok_or_else(|| {
                polylogue_store::StoreError::Invalid("the stored revision disappeared".into())
            })?
            .route_rank;
        self.reparse_plan(
            path.to_path_buf(),
            origin,
            InputKey {
                id: input_id,
                format: origin.name.into(),
                route_rank,
            },
            &state,
            facts,
            true,
        )
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
    ) -> Result<Option<Plan>, EngineError> {
        let stored = match self
            .reads
            .reader()?
            .stored_chain(&input.id, state.chain, state.length)
        {
            Ok(stored) => stored,
            Err(polylogue_store::StoreError::InputPending { .. }) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let parser = (origin.open)(&path);
        let parse_pos = if state.parsed && parser.as_ref().is_some_and(|p| p.is_capture_log()) {
            state.parsed_upto
        } else {
            0
        };
        let export = matches!(parser, Some(polylogue_sources::Parse::Export(_)));
        let source = if parse_pos == 0 && !export {
            pool::Bytes::Import(Arc::new(Mutex::new(read::Imported::from_chain(
                stored,
                Arc::clone(&self.blobs),
            )?)))
        } else {
            pool::Bytes::Stored(Arc::new(stored))
        };
        Ok(Some(Plan::stored(
            path,
            origin,
            input,
            Change::Reparse { chain: state.chain },
            facts,
            state.length,
            parse_pos,
            source,
            self.config.chunk,
            tracked,
        )))
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
        for mut plan in plans {
            if let Some(polylogue_sources::parse::Parse::Export(mut parser)) =
                (plan.origin.open)(&plan.path)
            {
                self.execute_pool(std::mem::take(&mut ordinary), pass)?;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.execute_export(&mut plan, &mut *parser, pass)
                }));
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        pass.errors
                            .push(format!("parse {}: {error}", plan.path.display()));
                        plan.failed = true;
                    }
                    Err(_) => {
                        pass.errors.push(format!(
                            "parse {}: array parser panicked",
                            plan.path.display()
                        ));
                        plan.failed = true;
                    }
                }
                self.settle(plan);
            } else {
                ordinary.push(plan);
            }
        }
        self.execute_pool(ordinary, pass)?;
        self.restore_failed_writes()
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
        let mut running: HashMap<PathBuf, Plan> = HashMap::new();
        let mut outstanding = 0usize;
        for mut plan in plans {
            if plan.finished() {
                self.settle(plan);
                continue;
            }
            let job = plan.next_job();
            pass.bytes_read += job.end - job.start;
            pool.submit(job);
            outstanding += 1;
            running.insert(plan.path.clone(), plan);
        }
        while outstanding > 0 {
            let Ok(d) = done_rx.recv() else {
                break;
            };
            outstanding -= 1;
            let Some(mut plan) = running.remove(&d.path) else {
                continue;
            };
            pass.bytes_stored += d.stored;
            if let Some(e) = &d.error {
                pass.errors.push(e.clone());
            }
            let parsed = plan.origin.support == Support::Parsed;
            plan.apply(d, parsed);
            if plan.finished() {
                self.settle(plan);
            } else {
                let job = plan.next_job();
                pass.bytes_read += job.end - job.start;
                pool.submit(job);
                outstanding += 1;
                running.insert(plan.path.clone(), plan);
            }
        }
        drop(pool);
        let report: WriteReport = self.flush_writes()?;
        pass.messages_written += report.messages;
        pass.errors.extend(report.errors);
        Ok(())
    }

    /// Records what a finished plan leaves for the next pass. Parser state
    /// is kept in memory only for files written recently; an older file's
    /// state is rebuilt by parsing it again if it ever grows.
    fn settle(&mut self, plan: Plan) {
        if !plan.tracked {
            if let Change::Reparse { chain } = plan.change
                && let Some(st) = self.inputs.get_mut(&plan.path)
                && st.input_id == plan.input.id
            {
                if plan.failed {
                    st.failed_history
                        .insert(chain, FailedChain { length: plan.size });
                } else {
                    st.failed_history.remove(&chain);
                }
            }
            return;
        }
        if let Some(st) = self.inputs.get_mut(&plan.path) {
            st.stored = plan.piece_pos.max(st.stored);
            if plan.failed {
                // Tried again when the bytes change; the facts stay old, so
                // a change is still seen as one.
                st.carry = None;
                st.waiting = false;
                st.failed = plan.facts;
                return;
            }
            st.failed = None;
            let recent = plan
                .facts
                .is_some_and(|f| i128::from(now_ms()) * 1_000_000 - f.mtime_ns < KEEP_STATE_NS);
            st.facts = plan.facts;
            st.parsed_upto = plan.parse_pos.min(plan.size);
            st.parsed = true;
            st.waiting = plan.waiting;
            st.carry = if recent { plan.carry } else { None };
            st.tail = plan.tail;
        }
    }

    /// Runs passes until stopped: after any change under a watched
    /// location, and at least every `rescan_every`.
    pub fn run(
        &mut self,
        tx: Sender<LoopMsg>,
        rx: &Receiver<LoopMsg>,
        status: &Arc<Mutex<LoopStatus>>,
        ready: impl FnOnce(),
    ) -> Result<(), EngineError> {
        let busy = |busy: bool| {
            if let Ok(mut s) = status.lock() {
                s.busy = busy;
            }
        };
        let mut watcher = watch::Roots::new(&self.config, tx.clone())?;
        drop(tx);
        ready();
        let mut discover = true;
        let mut discovered_at = Instant::now();
        loop {
            watcher.refresh(&self.config)?;
            discover |= discovered_at.elapsed() >= self.config.rescan_every;
            if discover {
                discovered_at = Instant::now();
            }
            busy(true);
            let pending;
            let finished = match self.converge_work(discover) {
                Ok(pass) => {
                    pending = pass.capture_pending;
                    pass
                }
                Err(e) => {
                    pending = self.exporter.pending();
                    let mut pass = status
                        .lock()
                        .map(|s| s.last_pass.clone())
                        .unwrap_or_default();
                    pass.errors = vec![e.to_string()];
                    pass.capture_pending = pending;
                    pass
                }
            };
            if let Ok(mut s) = status.lock() {
                s.last_pass = finished;
                s.passes += 1;
                s.busy = pending;
            }
            discover = false;
            let delay = if pending {
                Duration::ZERO
            } else {
                self.config.rescan_every
            };
            let mut msg = rx.recv_timeout(delay);
            let burst = Instant::now();
            loop {
                match msg {
                    Ok(LoopMsg::Stop) | Err(RecvTimeoutError::Disconnected) => {
                        self.exporter = Captures::default();
                        return Ok(());
                    }
                    Ok(LoopMsg::Import { paths, reply }) => {
                        busy(true);
                        let report = self.import(&paths).map_err(|e| e.to_string());
                        busy(pending);
                        let _ = reply.send(report);
                        discover = true;
                    }
                    Ok(LoopMsg::Excise { request, reply }) => {
                        busy(true);
                        let done = self.excise(request);
                        busy(pending);
                        let _ = reply.send(done);
                        discover = true;
                    }
                    Ok(LoopMsg::Changed) => discover = true,
                    Err(RecvTimeoutError::Timeout) => discover |= !pending,
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
}
