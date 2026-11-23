//! One typed status constructor over current archive and runtime observations.
use super::{Daemon, error, store_error};
use polylogue_model::LockExt as _;

/// The daemon's mode as status reports it.
fn mode(d: &Daemon) -> crate::protocol::status::Mode {
    match &d.mode {
        super::ModeState::Standalone => crate::protocol::status::Mode::Standalone,
        #[cfg(feature = "sinex")]
        super::ModeState::Primary { .. } => crate::protocol::status::Mode::Primary,
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

/// Status over an index that is not current is a degraded answer.
pub(super) fn answer(
    d: &Daemon,
    reader: &polylogue_store::Reader<'_>,
    include_vectors: bool,
) -> command_surface::Answer<crate::protocol::status::Answer> {
    // Totals are counters the writer keeps with the rows they count: reading
    // them costs the same at any archive size.
    let (store, totals_error) = match reader.status() {
        Ok(store) => (Some(store), None),
        Err(error) => (None, Some(error.to_string())),
    };
    let (vectors, vectors_error) = if include_vectors {
        match reader.vectors().and_then(|vectors| vectors.status()) {
            Ok(vectors) => (Some(vectors), None),
            Err(error) => (None, Some(error.to_string())),
        }
    } else {
        (None, None)
    };
    let loop_status = d.status.locked().clone();
    let signals = d.signals.locked().clone();
    let signals = match serde_json::from_value(signals) {
        Ok(signals) => signals,
        Err(problem) => return error(polylogue_model::Code::Internal, problem),
    };
    let notices = match serde_json::from_value(crate::signals::recent(reader, d.clock.now_ms())) {
        Ok(notices) => notices,
        Err(problem) => return error(polylogue_model::Code::Internal, problem),
    };
    let running = match reader.running(d.clock.now_ms()) {
        Ok(sessions) => crate::protocol::status::Running::Measured {
            waiting: sessions
                .iter()
                .filter(|s| s.state == polylogue_model::live::LiveState::Waiting)
                .count(),
            sessions,
        },
        Err(error) => crate::protocol::status::Running::Unavailable {
            error: error.to_string(),
        },
    };
    let freshness = match d.freshness(reader) {
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
    #[cfg(feature = "sinex")]
    let primary = primary(d, reader);
    #[cfg(feature = "sinex")]
    if let Some(crate::protocol::status::Primary::Unavailable { error }) = &primary {
        missing.push(format!("primary status: {error}"));
    }
    let status = crate::protocol::status::Answer {
        archive: Some(d.archive.root.clone()),
        pid: std::process::id(),
        uptime_seconds: d.started.elapsed().as_secs(),
        store,
        index_build: d.archive.index_build(),
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
        #[cfg(feature = "sinex")]
        primary,
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

fn capture_note(
    d: &Daemon,
    mut answer: command_surface::Answer<crate::protocol::status::Answer>,
) -> command_surface::Answer<crate::protocol::status::Answer> {
    if polylogue_sources::ai_studio::drive::needs_authorization(&d.archive.root) {
        answer.envelope.notes.push("AI Studio through Drive is not authorized; run `polylogue install drive` to collect it".into());
    }
    answer
}

/// Status while `index.db` is being built and no index can be read: the
/// daemon's own state and the build's progress, degraded until it ends.
pub(super) fn building(
    d: &Daemon,
    build: polylogue_store::archive::index_build::IndexBuild,
    include_vectors: bool,
) -> command_surface::Answer<crate::protocol::status::Answer> {
    use polylogue_model::freshness::{
        CompletenessDimension, CompletenessReason, IndexFreshness, ReadCompleteness, ReadFreshness,
    };
    let unread = build.to_string();
    let loop_status = d.status.locked().clone();
    let signals = d.signals.locked().clone();
    let signals = match serde_json::from_value(signals) {
        Ok(signals) => signals,
        Err(problem) => return error(polylogue_model::Code::Internal, problem),
    };
    let mode = mode(d);
    let freshness = ReadFreshness {
        local: ReadCompleteness {
            parsing: CompletenessDimension::Incomplete(vec![CompletenessReason::FirstBuild]),
            derivation: CompletenessDimension::Incomplete(vec![
                CompletenessReason::DerivationBehind,
            ]),
            search: CompletenessDimension::Incomplete(vec![CompletenessReason::FirstBuild]),
            confirmed_stream: CompletenessDimension::NotApplicable,
            kept_chains: 0,
            raw_only_chains: 0,
        },
        sinex: match &mode {
            crate::protocol::status::Mode::Primary => {
                IndexFreshness::primary(None, None, None, None, None)
            }
            _ => IndexFreshness::Standalone,
        },
    };
    let status = crate::protocol::status::Answer {
        archive: Some(d.archive.root.clone()),
        pid: std::process::id(),
        uptime_seconds: d.started.elapsed().as_secs(),
        store: None,
        index_build: Some(build),
        totals_error: Some(unread.clone()),
        vectors: None,
        vectors_error: include_vectors.then(|| unread.clone()),
        verification: None,
        pending_writes: pending_writes(d),
        loop_status,
        signals,
        notices: crate::protocol::status::Notices::Unavailable {
            error: unread.clone(),
        },
        running: crate::protocol::status::Running::Unavailable { error: unread },
        warm: d.warm.summary(),
        mode: mode.clone(),
        freshness: freshness.clone(),
        #[cfg(feature = "sinex")]
        primary: None,
    };
    let mut answer = crate::protocol::read(status, &freshness);
    if let crate::protocol::status::Mode::Blocked { path, error, .. } = &mode
        && let command_surface::Outcome::Degraded { missing, .. } = &mut answer.outcome
    {
        missing.push(format!("{}: {error}", path.display()));
    }
    capture_note(d, answer)
}

#[cfg(feature = "sinex")]
fn primary(
    d: &Daemon,
    reader: &polylogue_store::Reader<'_>,
) -> Option<crate::protocol::status::Primary> {
    use crate::protocol::status::Primary;
    let super::ModeState::Primary { connection } = &d.mode else {
        return None;
    };
    Some(match connection.status() {
        Ok(producer) => {
            let (applied_position, mut position_error) = match reader.confirmed_checkpoint() {
                Ok(position) => (position, None),
                Err(error) => (None, Some(error.to_string())),
            };
            let (refused_events, last_refused) = match reader.confirmed_refusals() {
                Ok((count, last)) => (Some(count), last),
                Err(error) => {
                    position_error.get_or_insert(error.to_string());
                    (None, None)
                }
            };
            let lag_positions = producer
                .sinex_position
                .zip(applied_position)
                .and_then(|(sinex, applied)| sinex.checked_sub(applied).filter(|lag| *lag >= 0));
            Primary::Observed {
                producer: Box::new(producer),
                applied_position,
                lag_positions,
                position_error,
                refused_events,
                last_refused,
            }
        }
        Err(error) => Primary::Unavailable {
            error: error.to_string(),
        },
    })
}

fn pending_writes(d: &Daemon) -> crate::protocol::status::PendingWrites {
    let (delivery_outcomes, outcome_error) = d.queued_deliveries.pending_outcomes();
    let (live_event_lines, spool_error) = d.live_spool.locked().summary();
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
