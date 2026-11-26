//! One typed status constructor over current archive and runtime observations.
use super::Daemon;
use polylogue_model::LockExt as _;

/// The daemon's mode as status reports it.
fn mode(d: &Daemon) -> crate::protocol::status::Mode {
    match &d.mode {
        super::ModeState::Standalone => crate::protocol::status::Mode::Standalone,
        super::ModeState::Blocked {
            configured,
            path,
            error,
        } => crate::protocol::status::Mode::Blocked {
            configured: configured.map(str::to_owned),
            path: path.clone(),
            error: error.clone(),
        },
    }
}

/// What a status answer reads from the index: read after each commit by the
/// worker that keeps the scope index ([`crate::daemon::scope`]), so that a
/// status request reads only memory and never waits for a reader, the
/// writer's lock or a query whose cost grows with the archive.
struct Snapshot {
    read_at: polylogue_model::Timestamp,
    /// The writer's commit count loaded before its reader opened: the
    /// snapshot shows at least those commits.
    commits_read: u64,
    store: Result<polylogue_store::Status, String>,
    notices: Result<crate::protocol::status::Notices, String>,
    running: Result<polylogue_store::RunningCandidates, String>,
    /// Each subscription limit's current window as of `read_at`.
    quota: Result<Vec<polylogue_store::measure::quota::CurrentWindow>, String>,
    /// The index's completeness.
    reading: polylogue_store::Result<polylogue_model::freshness::ReadCompleteness>,
}

impl Snapshot {
    fn read(d: &Daemon, reader: &polylogue_store::Reader<'_>, commits_read: u64) -> Self {
        let read_at = d.clock.now();
        let now = read_at.as_millisecond();
        let reading = reader.completeness(&polylogue_engine::reading_contracts());
        Self {
            read_at: read_at.into(),
            commits_read,
            store: reader
                .status(&polylogue_engine::reading_contracts())
                .map_err(|error| error.to_string()),
            notices: serde_json::from_value(crate::signals::recent(reader, now))
                .map_err(|problem| problem.to_string()),
            running: reader
                .running_candidates(now)
                .map_err(|error| error.to_string()),
            quota: polylogue_store::measure::quota::current(
                reader,
                now,
                &crate::daemon::measure::quota_limits(),
            )
            .map_err(|error| error.to_string()),
            reading,
        }
    }
}

/// The latest [`Snapshot`], or since when there is none.
#[derive(Clone)]
enum Latest {
    Read(std::sync::Arc<Snapshot>),
    NotRead(crate::protocol::status::NotReadSince),
}

/// The latest snapshot, replaced whole after each commit.
pub(crate) struct StatusSnapshot(std::sync::Mutex<Latest>);

impl Default for StatusSnapshot {
    fn default() -> Self {
        Self(std::sync::Mutex::new(Latest::NotRead(
            crate::protocol::status::NotReadSince::Start,
        )))
    }
}

impl StatusSnapshot {
    /// Reads a new snapshot from `reader`, opened after the writer had made
    /// `commits_read` commits, and publishes it. The lock is held only for
    /// the swap.
    pub(crate) fn refresh(
        &self,
        d: &Daemon,
        reader: &polylogue_store::Reader<'_>,
        commits_read: u64,
    ) {
        let snapshot = std::sync::Arc::new(Snapshot::read(d, reader, commits_read));
        *self.0.locked() = Latest::Read(snapshot);
        d.status.changed();
    }

    /// No index can be read while one is built: status answers from the
    /// build's progress, and once it is installed says its figures are not
    /// read yet until the next snapshot.
    pub(crate) fn clear(&self, d: &Daemon) {
        *self.0.locked() = Latest::NotRead(crate::protocol::status::NotReadSince::Install);
        d.status.changed();
    }

    /// Whether a snapshot has been read since the daemon started or the
    /// last index build began.
    #[cfg(test)]
    pub(crate) fn is_read(&self) -> bool {
        matches!(*self.0.locked(), Latest::Read(_))
    }

    fn latest(&self) -> Latest {
        self.0.locked().clone()
    }
}

type Build = (
    polylogue_store::archive::index_build::IndexBuild,
    polylogue_store::archive::build_progress::BuildProgress,
);

/// Status from the latest snapshot and the daemon's own state; degraded
/// when the index is not current. `vectors` are the counts a deep request
/// read, or why they could not be read.
pub(super) fn answer(
    d: &Daemon,
    vectors: Option<Result<polylogue_store::vectors::VectorStatus, String>>,
    whole_log: bool,
    details: bool,
) -> command_surface::Answer<crate::protocol::status::Answer> {
    use crate::protocol::status::AnswersFrom;
    // Counted first, so a change while the answer is composed counts after it.
    let changes = d.status.changes();
    let build = d.archive.build_progress();
    // A snapshot from before a build that serves no index is of an index
    // that can no longer be read.
    if build.is_some() && !d.archive.index_files().is_ok_and(|files| files.serves()) {
        return unread(d, AnswersFrom::Build, build, vectors, whole_log, changes);
    }
    let snapshot = match d.status_snapshot.latest() {
        Latest::Read(snapshot) => snapshot,
        Latest::NotRead(since) => {
            return unread(
                d,
                AnswersFrom::NotReadYet { since },
                build,
                vectors,
                whole_log,
                changes,
            );
        }
    };
    let now = d.clock.now_ms();
    let loop_status = d.status.get();
    let (mut store, totals_error) = match &snapshot.store {
        Ok(store) => (
            Some(crate::protocol::status::Store::from_snapshot(
                store.clone(),
                details,
                &loop_status.last_pass.errors,
            )),
            None,
        ),
        Err(error) => (None, Some(error.clone())),
    };
    if let Some(store) = &mut store {
        for coverage in &mut store.capture_coverage {
            if coverage.machine == d.config.identity.machine {
                coverage.machine.clone_from(&d.config.identity.host);
            }
        }
    }
    let (quota, quota_error) = match &snapshot.quota {
        Ok(windows) => (
            windows
                .iter()
                .cloned()
                .map(|window| window.at(now))
                .collect(),
            None,
        ),
        Err(error) => (Vec::new(), Some(error.clone())),
    };
    let (vectors, vectors_error) = split(vectors);
    let signals = d.signals.locked().clone();
    let (signals, signals_error) = match serde_json::from_value(signals) {
        Ok(signals) => (signals, None),
        Err(problem) => (None, Some(format!("signals: {problem}"))),
    };
    let notices = match &snapshot.notices {
        Ok(notices) => notices.clone(),
        Err(problem) => crate::protocol::status::Notices::Unavailable {
            error: problem.clone(),
        },
    };
    let running = match &snapshot.running {
        Ok(candidates) => {
            let sessions = candidates.at(now);
            crate::protocol::status::Running::Measured {
                waiting: sessions
                    .iter()
                    .filter(|s| s.state == polylogue_model::live::LiveState::Waiting)
                    .count(),
                sessions,
            }
        }
        Err(error) => crate::protocol::status::Running::Unavailable {
            error: error.clone(),
        },
    };
    let freshness = match &snapshot.reading {
        Ok(local) => super::state::freshness_of(local.clone()),
        Err(_) => {
            let mut local = polylogue_model::freshness::ReadCompleteness::empty();
            local.parsing = polylogue_model::freshness::CompletenessDimension::Unknown(Vec::new());
            local.derivation =
                polylogue_model::freshness::CompletenessDimension::Unknown(Vec::new());
            local.search = polylogue_model::freshness::CompletenessDimension::Unknown(Vec::new());
            super::state::freshness_of(local)
        }
    };
    let caught_up = build.is_none()
        && !loop_status.busy
        && !freshness.local.background_work_pending()
        && snapshot.commits_read >= d.writes.as_ref().map_or(0, |writes| writes.commits_made());
    let mut missing: Vec<String> = freshness.notes().into_iter().map(str::to_owned).collect();
    missing.extend(loop_status.last_pass.errors.iter().map(ToString::to_string));
    missing.extend(
        loop_status
            .last_pass
            .unreadable_roots
            .iter()
            .map(|path| format!("{}: unreadable", path.display())),
    );
    if let Err(problem) = &snapshot.reading {
        missing.push(format!("index completeness: {problem}"));
    }
    missing.extend(signals_error);
    let blocked = if let super::ModeState::Blocked { path, error, .. } = &d.mode {
        missing.push(format!("{}: {error}", path.display()));
        true
    } else {
        false
    };
    if let Some(error) = &totals_error {
        missing.push(format!("archive totals: {error}"));
    }
    let index = d.archive.index_status().ok();
    if let Some(failure) = index
        .as_ref()
        .and_then(|index| index.install_failure.as_ref())
    {
        missing.push(format!("{}: {}", failure.subject, failure.message));
    }
    if let Some(error) = &quota_error {
        missing.push(format!("subscription quota: {error}"));
    }
    if let Some(error) = &vectors_error {
        missing.push(format!("vector counts: {error}"));
    }
    if let crate::protocol::status::Running::Unavailable { error } = &running {
        missing.push(format!("running sessions: {error}"));
    }
    if let crate::protocol::status::Notices::Unavailable { error } = &notices {
        missing.push(format!("notices: {error}"));
    }
    let threads = polylogue_store::activity::threads(crate::protocol::status::STATEMENTS_AFTER);
    let (index_build, build_progress) = build.unzip();
    let status = crate::protocol::status::Answer {
        archive: Some(d.archive.root.clone()),
        pid: std::process::id(),
        uptime_seconds: d.started.elapsed().as_secs(),
        answers_from: AnswersFrom::Snapshot {
            read_at: snapshot.read_at,
        },
        changes,
        caught_up,
        store,
        index_build,
        index,
        build_progress,
        totals_error,
        quota,
        quota_error,
        vectors,
        vectors_error,
        verification: None,
        pending_writes: pending_writes(d),
        loop_status,
        signals,
        notices: Some(notices),
        running: Some(running),
        warm: d.warm.summary(d.scope.is_loaded()),
        mode: mode(d),
        freshness: freshness.clone(),
        recent_log: recent_log(whole_log),
        threads,
        slow_statements: if whole_log {
            polylogue_store::activity::slow_statements()
        } else {
            Vec::new()
        },
    };
    missing.extend(
        status
            .threads
            .iter()
            .filter_map(|thread| thread.worker.as_ref()?.problem(&thread.name)),
    );
    capture_note(
        d,
        if !blocked && freshness.up_to_date() && missing.is_empty() {
            command_surface::Answer::ok(status)
        } else {
            command_surface::Answer::degraded(status, missing)
        },
    )
}

/// Answers with `answer` once the status it composes has caught up after
/// `after` changes (any, without it), or when `wait` has passed, within
/// the request's own limit. Each look follows a change to what status
/// reports; nothing is polled.
pub(super) fn wait(
    call: &crate::daemon::handler::Call<'_>,
    wait: std::time::Duration,
    after: Option<u64>,
    answer: impl Fn() -> command_surface::Answer<crate::protocol::status::Answer>,
) -> Result<command_surface::Answer<crate::protocol::status::Answer>, super::Failure> {
    use polylogue_store::read::limit::ReadLimit;
    let mut wait = wait;
    if let Some(left) = call.limit.remaining() {
        wait = wait.min(left.saturating_sub(super::follow::REPLY_MARGIN));
    }
    let until = std::time::Instant::now() + wait;
    loop {
        call.limit.check()?;
        let current = answer();
        let Some(status) = current.value() else {
            return Ok(current);
        };
        let seen = status.changes;
        if (status.caught_up && after.is_none_or(|after| seen > after))
            || std::time::Instant::now() >= until
        {
            return Ok(current);
        }
        ReadLimit::set_stage("status wait");
        call.daemon.status.wait_past(seen, until);
    }
}

/// The vector counts a deep request read, or why they could not be read.
fn split<T>(read: Option<Result<T, String>>) -> (Option<T>, Option<String>) {
    match read {
        Some(Ok(value)) => (Some(value), None),
        Some(Err(error)) => (None, Some(error)),
        None => (None, None),
    }
}

/// The daemon's log records an answer carries.
fn recent_log(whole: bool) -> Vec<polylogue_store::log::Record> {
    polylogue_store::log::recent(if whole {
        polylogue_store::log::KEPT
    } else {
        crate::protocol::status::RECENT_LOG
    })
}

fn capture_note(
    d: &Daemon,
    mut answer: command_surface::Answer<crate::protocol::status::Answer>,
) -> command_surface::Answer<crate::protocol::status::Answer> {
    if polylogue_sources::ai_studio::drive::needs_authorization(&d.archive.root) {
        answer.envelope.notes.push("AI Studio through Drive is not authorized; run `polylogue install drive` to collect it".into());
    }
    answer
}

/// Status without a snapshot: while `index.db` is being built and no index
/// can be read, or before a snapshot is read since the daemon started or an
/// index was installed. It holds the daemon's own state and the build's
/// progress, has no archive figures, and is degraded.
fn unread(
    d: &Daemon,
    answers_from: crate::protocol::status::AnswersFrom,
    build: Option<Build>,
    vectors: Option<Result<polylogue_store::vectors::VectorStatus, String>>,
    whole_log: bool,
    changes: u64,
) -> command_surface::Answer<crate::protocol::status::Answer> {
    use polylogue_model::freshness::{
        CompletenessDimension, CompletenessReason, IndexFreshness, ReadCompleteness, ReadFreshness,
    };
    let building = answers_from == crate::protocol::status::AnswersFrom::Build;
    let loop_status = d.status.get();
    let signals = d.signals.locked().clone();
    let (signals, signals_error) = match serde_json::from_value(signals) {
        Ok(signals) => (signals, None),
        Err(problem) => (None, Some(format!("signals: {problem}"))),
    };
    let mode = mode(d);
    // Before a snapshot nothing is known; a build names itself, and its
    // session figures are one of its steps.
    let dimension = || {
        if building {
            CompletenessDimension::Incomplete(vec![CompletenessReason::IndexBuilding])
        } else {
            CompletenessDimension::Unknown(Vec::new())
        }
    };
    let freshness = ReadFreshness {
        local: ReadCompleteness {
            parsing: dimension(),
            derivation: dimension(),
            search: dimension(),
            confirmed_stream: CompletenessDimension::NotApplicable,
            kept_chains: 0,
            raw_only_chains: 0,
        },
        sinex: IndexFreshness::Standalone,
    };
    let (vectors, vectors_error) = split(vectors);
    let words = answers_from.words();
    let (index_build, build_progress) = build.unzip();
    let threads = polylogue_store::activity::threads(crate::protocol::status::STATEMENTS_AFTER);
    let status = crate::protocol::status::Answer {
        archive: Some(d.archive.root.clone()),
        pid: std::process::id(),
        uptime_seconds: d.started.elapsed().as_secs(),
        answers_from,
        changes,
        caught_up: false,
        store: None,
        index_build,
        index: d.archive.index_status().ok(),
        build_progress,
        totals_error: None,
        quota: Vec::new(),
        quota_error: None,
        vectors,
        vectors_error,
        verification: None,
        pending_writes: pending_writes(d),
        loop_status,
        signals,
        notices: None,
        running: None,
        warm: d.warm.summary(d.scope.is_loaded()),
        mode: mode.clone(),
        freshness: freshness.clone(),
        recent_log: recent_log(whole_log),
        threads,
        slow_statements: if whole_log {
            polylogue_store::activity::slow_statements()
        } else {
            Vec::new()
        },
    };
    let problems = status
        .threads
        .iter()
        .filter_map(|thread| thread.worker.as_ref()?.problem(&thread.name))
        .collect::<Vec<_>>();
    let mut answer = crate::protocol::read(status, &freshness);
    if let command_surface::Outcome::Degraded { missing, .. } = &mut answer.outcome {
        missing.extend(problems);
        missing.extend(signals_error);
        missing.extend(words.map(str::to_owned));
        if let crate::protocol::status::Mode::Blocked { path, error, .. } = &mode {
            missing.push(format!("{}: {error}", path.display()));
        }
    }
    capture_note(d, answer)
}

fn pending_writes(d: &Daemon) -> crate::protocol::status::PendingWrites {
    let delivery_outcomes = d.writes.as_ref().map_or(0, |writes| {
        writes
            .archive()
            .pending_deliveries()
            .iter()
            .filter(|request| {
                matches!(
                    request.request.write,
                    polylogue_model::user::UserWrite::DeliveryEvent { .. }
                )
            })
            .count() as u64
    });
    let spool = d.live_spool.locked().summary();
    let (live_event_lines, spool_error) = (spool.lines, spool.error);
    crate::protocol::status::PendingWrites {
        delivery_outcomes,
        live_event_lines,
        last_error: spool_error,
    }
}

/// Keep runtime incompleteness and verification problems in one status answer.
pub(super) fn with_verification(
    mut status: command_surface::Answer<crate::protocol::status::Answer>,
    checked: command_surface::Answer<polylogue_store::verify::Verification>,
) -> command_surface::Answer<crate::protocol::status::Answer> {
    use command_surface::{Answer, Outcome};
    status.envelope.notes.extend(checked.envelope.notes);
    let (verification, problems) = match checked.outcome {
        Outcome::Ok(found) | Outcome::Empty(found) => (found, Vec::new()),
        Outcome::Degraded { value, missing } => (value, missing),
        Outcome::Error(error) => {
            return Answer {
                envelope: status.envelope,
                outcome: Outcome::Error(error),
            };
        }
    };
    let (mut value, mut missing) = match status.outcome {
        Outcome::Ok(value) | Outcome::Empty(value) => (value, Vec::new()),
        Outcome::Degraded { value, missing } => (value, missing),
        Outcome::Error(error) => {
            return Answer {
                envelope: status.envelope,
                outcome: Outcome::Error(error),
            };
        }
    };
    value.verification = Some(verification);
    missing.extend(problems);
    status.outcome = if missing.is_empty() {
        Outcome::Ok(value)
    } else {
        Outcome::Degraded { value, missing }
    };
    status
}

#[cfg(test)]
#[path = "status/tests.rs"]
mod tests;
