//! The applied position, scope and staged delivery share the transaction that
//! applies the events.
use super::*;
use crate::primary::{
    AppliedSnapshot, CompletedDelivery, ReplayControl, ReplayState, SinexEventContext,
};

/// Both forms mark a complete position and share its pending row; their
/// required fields are disjoint.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
enum CompletionMarker {
    Snapshot(AppliedSnapshot),
    Delivery(CompletedDelivery),
}

pub(crate) fn replay_state(conn: &Connection) -> Result<ReplayState> {
    let value: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key='confirmed_replay'",
            [],
            |row| row.get(0),
        )
        .optional()?;
    Ok(match value {
        Some(value) => serde_json::from_str(&value)?,
        None => ReplayState {
            applied_position: checkpoint(conn)?,
            scope: super::super::scope::current(conn)?,
            latest_position: None,
            staged_delivery: None,
            completed_delivery: None,
            snapshot: None,
            snapshot_progress: None,
        },
    })
}

pub(crate) fn save(conn: &Connection, state: &ReplayState) -> Result<()> {
    conn.execute("INSERT INTO meta(key,value) VALUES ('confirmed_replay',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [serde_json::to_string(state)?])?;
    Ok(())
}

/// Whether `index.db` holds a row for this Sinex event. Every confirmed
/// change, including a refusal or an erasure seen before the live event,
/// leaves one, and rows are never deleted.
pub(crate) fn has_confirmed_event(conn: &Connection, event: &str) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM confirmed_event WHERE event=?1)",
        [event],
        |row| row.get(0),
    )?)
}

pub(crate) fn sinex_context(conn: &Connection, event: &str) -> Result<Option<SinexEventContext>> {
    let bytes: Option<Vec<u8>> = conn
        .query_row(
            "SELECT sinex_context FROM confirmed_event WHERE event=?1",
            [event],
            |row| row.get(0),
        )
        .optional()?
        .flatten();
    bytes.as_deref().map(unpacked).transpose()
}

fn invalid(message: &str) -> StoreError {
    StoreError::Invalid(message.into())
}
fn hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub(super) fn validate_context(context: &SinexEventContext) -> Result<()> {
    if context.kind.name.is_empty()
        || context.kind.version == 0
        || !hash(&context.kind_definition_hash)
        || !hash(&context.payload_identity.payload_hash)
        || context
            .record_hash
            .as_deref()
            .is_some_and(|value| !hash(value))
        || context
            .location
            .as_ref()
            .is_some_and(|location| location.material.is_empty())
        || context
            .origin
            .source
            .as_ref()
            .is_some_and(|source| source.name.is_empty() || source.version == 0)
        || [
            &context.origin.source_fingerprint,
            &context.origin.producer_family,
            &context.origin.producer_identity,
        ]
        .into_iter()
        .flatten()
        .any(|value| value.is_empty())
    {
        return Err(invalid("confirmed Sinex context has inconsistent metadata"));
    }
    Ok(())
}

fn validate_snapshot(state: &ReplayState, snapshot: &AppliedSnapshot) -> Result<()> {
    let progress = state
        .snapshot_progress
        .as_ref()
        .ok_or_else(|| invalid("a snapshot completion has no start"))?;
    if snapshot.position < progress.origin
        || progress
            .position
            .is_some_and(|position| position != snapshot.position)
        || state.scope.as_deref() != Some(snapshot.scope.as_str())
        || snapshot.stream_instance != progress.stream_instance
        || snapshot
            .last
            .as_ref()
            .is_some_and(|last| last.sequence == 0 || !hash(&last.fingerprint))
    {
        return Err(invalid(
            "a snapshot completion differs from its staged pages",
        ));
    }
    Ok(())
}

fn validate_delivery(delivery: &CompletedDelivery) -> Result<()> {
    if delivery.stream_instance.is_empty()
        || delivery.stream_sequence == 0
        || !hash(&delivery.fingerprint)
    {
        return Err(invalid(
            "a completed delivery needs its stream message identity",
        ));
    }
    Ok(())
}

pub(super) fn stage_control(
    conn: &Connection,
    state: &mut ReplayState,
    control: &ReplayControl,
) -> Result<()> {
    match control {
        ReplayControl::Reset { .. } | ReplayControl::SnapshotStart { .. } => {
            return Err(invalid("a scope reset must be a separate control page"));
        }
        ReplayControl::SnapshotPage { position, cursor } => {
            let progress = state
                .snapshot_progress
                .as_mut()
                .ok_or_else(|| invalid("a snapshot page has no start"))?;
            if *position < progress.origin
                || cursor.is_empty()
                || progress.position.is_some_and(|old| old != *position)
            {
                return Err(invalid(
                    "a snapshot page changed its position or did not advance",
                ));
            }
            progress.position = Some(*position);
            progress.cursor = Some(cursor.clone());
        }
        ReplayControl::SnapshotComplete { snapshot } => {
            validate_snapshot(state, snapshot)?;
            conn.execute("INSERT INTO primary_complete(position,marker) VALUES (?1,?2) ON CONFLICT(position) DO UPDATE SET marker=excluded.marker", params![snapshot.position, packed(&CompletionMarker::Snapshot(snapshot.clone()))?])?;
        }
        ReplayControl::LatestPosition { position } => {
            if *position < 0 {
                return Err(invalid("Sinex's commit position cannot be negative"));
            }
            state.latest_position = Some(*position);
        }
        ReplayControl::Delivered { delivery } => {
            if state.snapshot_progress.is_some() {
                return Err(invalid(
                    "a partial snapshot cannot acknowledge stream deliveries",
                ));
            }
            if delivery.stream_instance.is_empty() || delivery.stream_sequence == 0 {
                return Err(invalid("a staged delivery needs its stream position"));
            }
            // A partially staged page may be newer than the last completion;
            // it allows acknowledgments, but cannot choose the restart boundary.
            state.staged_delivery = Some(delivery.clone());
        }
        ReplayControl::Completion { position, delivery } => {
            if state.snapshot_progress.is_some() {
                return Err(invalid(
                    "a snapshot needs its last page before a stream completion",
                ));
            }
            if *position < 0 {
                return Err(invalid("a completed position cannot be negative"));
            }
            if state
                .applied_position
                .is_some_and(|applied| *position < applied)
            {
                return Err(invalid("a completion moved behind the applied position"));
            }
            if let Some(completed) = delivery {
                validate_delivery(completed)?;
            }
            // An old cumulative completion marker can be appended after newer
            // events. Its stream sequence cannot be a new restart boundary past
            // those events, even while they are staged but unapplied.
            let newer: bool = conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM primary_pending WHERE position>?1)",
                [position],
                |row| row.get(0),
            )?;
            let completed = delivery.as_ref().filter(|_| !newer);
            conn.execute(
                "INSERT OR IGNORE INTO primary_complete(position,marker) VALUES (?1,?2)",
                params![
                    position,
                    completed
                        .cloned()
                        .map(CompletionMarker::Delivery)
                        .as_ref()
                        .map(packed)
                        .transpose()?
                ],
            )?;
        }
    }
    Ok(())
}

pub(super) fn complete(conn: &Connection, position: i64) -> Result<()> {
    let mut state = replay_state(conn)?;
    let marker: Option<Vec<u8>> = conn.query_row(
        "SELECT marker FROM primary_complete WHERE position=?1",
        [position],
        |row| row.get(0),
    )?;
    let install_marker = state.applied_position.is_none_or(|old| position > old)
        || (state.applied_position == Some(position) && state.completed_delivery.is_none());
    match marker.as_deref().map(unpacked).transpose()? {
        Some(CompletionMarker::Snapshot(snapshot)) => {
            validate_snapshot(&state, &snapshot)?;
            if snapshot.position != position {
                return Err(invalid("snapshot completion changed position"));
            }
            state.snapshot = Some(snapshot);
            state.snapshot_progress = None;
            state.staged_delivery = None;
            state.completed_delivery = None;
        }
        delivery => {
            if state.snapshot_progress.is_some() {
                return Err(invalid("a snapshot completion needs its last page"));
            }
            let completed = match delivery {
                Some(CompletionMarker::Delivery(completed)) => {
                    validate_delivery(&completed)?;
                    Some(completed)
                }
                _ => None,
            };
            if install_marker {
                state.completed_delivery = completed;
            }
        }
    }
    state.applied_position = Some(position);
    save(conn, &state)
}
