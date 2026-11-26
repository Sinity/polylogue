//! One typed status constructor over current archive and runtime observations.
use super::{Daemon, error, store_error};
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
    store: Result<polylogue_store::Status, String>,
    notices: Result<crate::protocol::status::Notices, String>,
    running: Result<polylogue_store::RunningCandidates, String>,
    /// The index's completeness and the applied stream position, which
    /// freshness compares with the Sinex observations held when answering.
    reading: polylogue_store::Result<(
        polylogue_model::freshness::ReadCompleteness,
        super::state::Applied,
    )>,
}

impl Snapshot {
    fn read(d: &Daemon, reader: &polylogue_store::Reader<'_>) -> Self {
        let now = d.clock.now_ms();
        let reading = reader
            .completeness(&polylogue_engine::reading_contracts())
            .and_then(|local| Ok((local, d.applied(reader)?)));
        Self {
            store: reader.status().map_err(|error| error.to_string()),
            notices: serde_json::from_value(crate::signals::recent(reader, now))
                .map_err(|problem| problem.to_string()),
            running: reader
                .running_candidates(now)
                .map_err(|error| error.to_string()),
            reading,
        }
    }
}

/// The latest [`Snapshot`], replaced whole after each commit.
#[derive(Default)]
pub(crate) struct StatusSnapshot(std::sync::Mutex<Option<std::sync::Arc<Snapshot>>>);

impl StatusSnapshot {
    /// Reads a new snapshot from `reader` and publishes it. The lock is
    /// held only for the swap.
    pub(crate) fn refresh(&self, d: &Daemon, reader: &polylogue_store::Reader<'_>) {
        let snapshot = std::sync::Arc::new(Snapshot::read(d, reader));
        *self.0.locked() = Some(snapshot);
    }

    /// No index can be read (one is being built): status answers from the
    /// build's progress until the next snapshot.
    pub(crate) fn clear(&self) {
        *self.0.locked() = None;
    }

    /// Whether a snapshot has been read since the daemon started or the
    /// last index build began.
    #[cfg(test)]
    pub(crate) fn is_read(&self) -> bool {
        self.0.locked().is_some()
    }

    fn latest(&self) -> Option<std::sync::Arc<Snapshot>> {
        self.0.locked().clone()
    }
}

/// Why status has no snapshot to answer from while no build is running.
const WARMING: &str = "warming: the archive's status has not been read since the daemon started";

/// Status from the latest snapshot and the daemon's own state; degraded
/// when the index is not current. `vectors` are the counts a deep request
/// read, or why they could not be read.
pub(super) fn answer(
    d: &Daemon,
    vectors: Option<Result<polylogue_store::vectors::VectorStatus, String>>,
    whole_log: bool,
) -> command_surface::Answer<crate::protocol::status::Answer> {
    let Some(snapshot) = d.status_snapshot.latest() else {
        return unread(d, d.archive.index_build(), vectors, whole_log);
    };
    let now = d.clock.now_ms();
    let (store, totals_error) = match &snapshot.store {
        Ok(store) => (Some(store.clone()), None),
        Err(error) => (None, Some(error.clone())),
    };
    let (vectors, vectors_error) = match vectors {
        Some(Ok(vectors)) => (Some(vectors), None),
        Some(Err(error)) => (None, Some(error)),
        None => (None, None),
    };
    let loop_status = d.status.locked().clone();
    let signals = d.signals.locked().clone();
    let signals = match serde_json::from_value(signals) {
        Ok(signals) => signals,
        Err(problem) => return error(polylogue_model::Code::Internal, problem),
    };
    let notices = match &snapshot.notices {
        Ok(notices) => notices.clone(),
        Err(problem) => return error(polylogue_model::Code::Internal, problem),
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
        Ok((local, applied)) => d.freshness_of(local.clone(), applied),
        Err(error) => return store_error(error),
    };
    let freshness = match freshness {
        Ok(freshness) => freshness,
        Err(error) => return store_error(&error),
    };
    let mut missing: Vec<String> = freshness.notes().into_iter().map(str::to_owned).collect();
    let blocked = if let super::ModeState::Blocked { path, error, .. } = &d.mode {
        missing.push(format!("{}: {error}", path.display()));
        true
    } else {
        false
    };
    if let Some(error) = &totals_error {
        missing.push(format!("archive totals: {error}"));
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
    let status = crate::protocol::status::Answer {
        archive: Some(d.archive.root.clone()),
        pid: std::process::id(),
        uptime_seconds: d.started.elapsed().as_secs(),
        warming: !d.scope.is_loaded(),
        store,
        index_build: d.archive.index_build(),
        build_progress: d.archive.build_progress(),
        totals_error,
        vectors,
        vectors_error,
        verification: None,
        pending_writes: pending_writes(d),
        loop_status,
        signals,
        notices,
        running,
        warm: d.warm.summary(),
        mode: mode(d),
        freshness: freshness.clone(),
        recent_log: recent_log(whole_log),
        threads: polylogue_store::activity::threads(crate::protocol::status::STATEMENTS_AFTER),
    };
    capture_note(
        d,
        if !blocked && freshness.up_to_date() && missing.is_empty() {
            command_surface::Answer::ok(status)
        } else {
            command_surface::Answer::degraded(status, missing)
        },
    )
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

/// Status before any snapshot: while `index.db` is being built and no index
/// can be read, or before the first snapshot is read. It holds the daemon's
/// own state and the build's progress and is degraded until a snapshot is
/// read.
fn unread(
    d: &Daemon,
    build: Option<polylogue_store::archive::index_build::IndexBuild>,
    vectors: Option<Result<polylogue_store::vectors::VectorStatus, String>>,
    whole_log: bool,
) -> command_surface::Answer<crate::protocol::status::Answer> {
    use polylogue_model::freshness::{
        CompletenessDimension, CompletenessReason, IndexFreshness, ReadCompleteness, ReadFreshness,
    };
    let unread = build
        .as_ref()
        .map_or_else(|| WARMING.to_owned(), ToString::to_string);
    let loop_status = d.status.locked().clone();
    let signals = d.signals.locked().clone();
    let signals = match serde_json::from_value(signals) {
        Ok(signals) => signals,
        Err(problem) => return error(polylogue_model::Code::Internal, problem),
    };
    let mode = mode(d);
    // While warming nothing is known yet; a build names itself, and its
    // session figures are one of its steps.
    let (reasons, derivation) = if build.is_some() {
        (
            vec![CompletenessReason::IndexBuilding],
            CompletenessDimension::Incomplete(vec![CompletenessReason::IndexBuilding]),
        )
    } else {
        (Vec::new(), CompletenessDimension::Unknown(Vec::new()))
    };
    let dimension = |reasons: Vec<CompletenessReason>| {
        if build.is_some() {
            CompletenessDimension::Incomplete(reasons)
        } else {
            CompletenessDimension::Unknown(reasons)
        }
    };
    let freshness = ReadFreshness {
        local: ReadCompleteness {
            parsing: dimension(reasons.clone()),
            derivation,
            search: dimension(reasons.clone()),
            // Primary mode cannot say what the stream has applied before
            // the index is read.
            confirmed_stream: if d.archive.is_primary() {
                CompletenessDimension::Unknown(reasons)
            } else {
                CompletenessDimension::NotApplicable
            },
            kept_chains: 0,
            raw_only_chains: 0,
        },
        sinex: IndexFreshness::Standalone,
    };
    let (vectors, vectors_error) = match vectors {
        Some(Ok(vectors)) => (Some(vectors), None),
        Some(Err(error)) => (None, Some(error)),
        None => (None, None),
    };
    let status = crate::protocol::status::Answer {
        archive: Some(d.archive.root.clone()),
        pid: std::process::id(),
        uptime_seconds: d.started.elapsed().as_secs(),
        warming: build.is_none(),
        store: None,
        index_build: build.clone(),
        build_progress: d.archive.build_progress(),
        totals_error: Some(unread.clone()),
        vectors,
        vectors_error,
        verification: None,
        pending_writes: pending_writes(d),
        loop_status,
        signals,
        notices: crate::protocol::status::Notices::Unavailable {
            error: unread.clone(),
        },
        running: crate::protocol::status::Running::Unavailable {
            error: unread.clone(),
        },
        warm: d.warm.summary(),
        mode: mode.clone(),
        freshness: freshness.clone(),
        recent_log: recent_log(whole_log),
        threads: polylogue_store::activity::threads(crate::protocol::status::STATEMENTS_AFTER),
    };
    let mut answer = crate::protocol::read(status, &freshness);
    if let command_surface::Outcome::Degraded { missing, .. } = &mut answer.outcome {
        if build.is_none() {
            missing.push(unread);
        }
        if let crate::protocol::status::Mode::Blocked { path, error, .. } = &mode {
            missing.push(format!("{}: {error}", path.display()));
        }
    }
    capture_note(d, answer)
}

fn pending_writes(d: &Daemon) -> crate::protocol::status::PendingWrites {
    let (delivery_outcomes, outcome_error) = d.queued_deliveries.pending_outcomes();
    let spool = d.live_spool.locked().summary();
    let (live_event_lines, spool_error) = (spool.lines, spool.error);
    crate::protocol::status::PendingWrites {
        delivery_outcomes,
        live_event_lines,
        last_error: outcome_error
            .into_iter()
            .chain(spool_error)
            .reduce(|left, right| format!("{left}; {right}")),
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
