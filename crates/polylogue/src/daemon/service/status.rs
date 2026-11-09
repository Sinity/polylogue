//! One typed status constructor over current archive and runtime observations.
use super::{Daemon, error, now_ms, store_error};
use crate::protocol::status::{Counted, Counting};
use serde_json::Value;

/// Status over an index that is not current is a degraded answer.
pub(super) fn answer(
    d: &Daemon,
    reader: &polylogue_store::Reader<'_>,
) -> command_surface::Answer<Value> {
    let position = match reader.change_position() {
        Ok(position) => position.to_string(),
        Err(error) => return store_error(&error),
    };
    let counted = d.warm.counts(&position);
    let loop_status = d
        .status
        .lock()
        .map(|status| status.clone())
        .unwrap_or_default();
    let signals = d
        .signals
        .lock()
        .map(|signals| signals.clone())
        .unwrap_or_default();
    let signals = match serde_json::from_value(signals) {
        Ok(signals) => signals,
        Err(problem) => return error("internal", problem),
    };
    let notices = match serde_json::from_value(crate::signals::recent(reader, now_ms())) {
        Ok(notices) => notices,
        Err(problem) => return error("internal", problem),
    };
    let running = match reader.running(now_ms()) {
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
    let mode = match serde_json::to_value(&d.mode).and_then(serde_json::from_value) {
        Ok(mode) => mode,
        Err(problem) => return error("internal", problem),
    };
    let currency = match d.currency(reader) {
        Ok(currency) => currency,
        Err(error) => return store_error(&error),
    };
    let status = crate::protocol::status::Answer {
        archive: Some(d.archive.root.clone()),
        pid: std::process::id(),
        uptime_seconds: d.started.elapsed().as_secs(),
        store: counted.as_ref().map(|counted| counted.status.clone()),
        count_error: d.warm.count_error(),
        counted: counted.as_ref().map_or(
            Counted::Counting {
                state: Counting::Counting,
            },
            |counted| Counted::Measured {
                position: counted.position.clone(),
                current: counted.position == position,
                seconds_ago: counted.at.elapsed().as_secs_f64(),
                seconds_to_count: counted.took.as_secs_f64(),
            },
        ),
        loop_status,
        signals,
        notices,
        running,
        warm: d.warm.summary(),
        mode,
        currency: currency.clone(),
        #[cfg(feature = "sinex")]
        primary: primary(d, reader),
    };
    match serde_json::to_value(status) {
        Ok(value) => crate::protocol::read(value, &currency),
        Err(problem) => error("internal", problem),
    }
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
                Ok(refusals) => refusals,
                Err(error) => {
                    position_error.get_or_insert(error.to_string());
                    (0, None)
                }
            };
            let lag_positions = producer
                .owner_position
                .zip(applied_position)
                .and_then(|(owner, applied)| owner.checked_sub(applied).filter(|lag| *lag >= 0));
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
