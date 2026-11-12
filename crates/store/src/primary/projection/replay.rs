//! The applied position, scope and staged delivery share the fact transaction.
use super::*;
use crate::primary::{NativeFactContext, ReplayControl, ReplayState};

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
            head: None,
            staged_delivery: None,
            completed_delivery: None,
            basis: None,
            projection: None,
        },
    })
}

pub(crate) fn save(conn: &Connection, state: &ReplayState) -> Result<()> {
    conn.execute("INSERT INTO meta(key,value) VALUES ('confirmed_replay',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [serde_json::to_string(state)?])?;
    Ok(())
}

/// Whether `index.db` holds a fact for this Sinex event. Every confirmed
/// change, including a refusal or an erasure seen before the live event,
/// leaves one, and facts are never deleted.
pub(crate) fn has_fact(conn: &Connection, event: &str) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM primary_fact WHERE event=?1)",
        [event],
        |row| row.get(0),
    )?)
}

pub(crate) fn native_context(conn: &Connection, event: &str) -> Result<Option<NativeFactContext>> {
    let bytes: Option<Vec<u8>> = conn
        .query_row(
            "SELECT native_context FROM primary_fact WHERE event=?1",
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

pub(super) fn validate_context(context: &NativeFactContext) -> Result<()> {
    if context.kind.name.is_empty()
        || context.kind.version == 0
        || !hash(&context.kind_definition_hash)
        || !hash(&context.projection.payload_hash)
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
        return Err(invalid(
            "confirmed native context has inconsistent owner metadata",
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
        ReplayControl::Reset { .. } | ReplayControl::ProjectionStart { .. } => {
            return Err(invalid("a scope reset must be a separate control page"));
        }
        ReplayControl::ProjectionPage { position, cursor } => {
            let projection = state
                .projection
                .as_mut()
                .ok_or_else(|| invalid("a projection page has no start"))?;
            if *position < projection.origin
                || cursor.is_empty()
                || projection.position.is_some_and(|old| old != *position)
            {
                return Err(invalid(
                    "a projection page changed its cut or did not advance",
                ));
            }
            projection.position = Some(*position);
            projection.cursor = Some(cursor.clone());
        }
        ReplayControl::ProjectionComplete { basis } => {
            let projection = state
                .projection
                .as_ref()
                .ok_or_else(|| invalid("a projection completion has no start"))?;
            if basis.position < projection.origin
                || projection
                    .position
                    .is_some_and(|position| position != basis.position)
                || state.scope.as_deref() != Some(basis.scope.as_str())
                || basis.stamp.is_empty()
                || basis.catalog.is_empty()
            {
                return Err(invalid(
                    "a projection certificate differs from the staged snapshot",
                ));
            }
            conn.execute("INSERT INTO primary_complete(position,evidence) VALUES (?1,?2) ON CONFLICT(position) DO UPDATE SET evidence=excluded.evidence", params![basis.position, packed(basis)?])?;
        }
        ReplayControl::Head { position } => {
            if *position < 0 {
                return Err(invalid("Sinex's commit position cannot be negative"));
            }
            state.head = Some(*position);
        }
        ReplayControl::Delivered { delivery } => {
            if state.projection.is_some() {
                return Err(invalid(
                    "a partial projection cannot acknowledge stream deliveries",
                ));
            }
            if delivery.stream_instance.is_empty() || delivery.stream_sequence == 0 {
                return Err(invalid("a staged delivery needs its stream receipt"));
            }
            // A partially staged page may be newer than the last completion;
            // it permits ACKs, but cannot choose the restart boundary.
            state.staged_delivery = Some(delivery.clone());
        }
        ReplayControl::Completion { position, delivery } => {
            if state.projection.is_some() {
                return Err(invalid(
                    "a projection needs its owner certificate before completion",
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
            if delivery.as_ref().is_some_and(|receipt| {
                receipt.stream_instance.is_empty()
                    || receipt.stream_sequence == 0
                    || !hash(&receipt.fingerprint)
            }) {
                return Err(invalid(
                    "a completed delivery needs its native record identity",
                ));
            }
            // An old cumulative barrier can be appended after newer facts.
            // Its physical sequence cannot be a new restart boundary past
            // those facts, even while they are durably staged but unapplied.
            let newer: bool = conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM primary_pending WHERE position>?1)",
                [position],
                |row| row.get(0),
            )?;
            let receipt = delivery.as_ref().filter(|_| !newer);
            conn.execute(
                "INSERT OR IGNORE INTO primary_complete(position,delivery) VALUES (?1,?2)",
                params![position, receipt.map(packed).transpose()?],
            )?;
        }
    }
    Ok(())
}

pub(super) fn complete(conn: &Connection, position: i64) -> Result<()> {
    let mut state = replay_state(conn)?;
    let (evidence, delivery): (Option<Vec<u8>>, Option<Vec<u8>>) = conn.query_row(
        "SELECT evidence,delivery FROM primary_complete WHERE position=?1",
        [position],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    if let Some(evidence) = evidence {
        state.basis = Some(unpacked(&evidence)?);
        state.projection = None;
        state.staged_delivery = None;
        state.completed_delivery = None;
    } else if state.applied_position.is_none_or(|old| position > old)
        || (state.applied_position == Some(position) && state.completed_delivery.is_none())
    {
        state.completed_delivery = delivery.as_deref().map(unpacked).transpose()?;
    }
    state.applied_position = Some(position);
    save(conn, &state)
}
