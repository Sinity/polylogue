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
        },
    })
}

pub(crate) fn save(conn: &Connection, state: &ReplayState) -> Result<()> {
    conn.execute("INSERT INTO meta(key,value) VALUES ('confirmed_replay',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [serde_json::to_string(state)?])?;
    Ok(())
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
        ReplayControl::Reset { .. } => {
            return Err(invalid("a scope reset must be a separate control page"));
        }
        ReplayControl::Head { position } => {
            if *position < 0 {
                return Err(invalid("Sinex's commit position cannot be negative"));
            }
            state.head = Some(*position);
        }
        ReplayControl::Delivered { delivery } => {
            if delivery.stream_instance.is_empty() || delivery.stream_sequence == 0 {
                return Err(invalid("a staged delivery needs its stream receipt"));
            }
            // A consumer Sinex provisions again starts over from the stream's
            // beginning; its redeliveries at applied positions are skipped.
            state.staged_delivery = Some(delivery.clone());
        }
        ReplayControl::Completion { position } => {
            if *position < 0 {
                return Err(invalid("a completed position cannot be negative"));
            }
            if state
                .applied_position
                .is_some_and(|applied| *position < applied)
            {
                return Err(invalid("a completion moved behind the applied position"));
            }
            conn.execute(
                "INSERT OR IGNORE INTO primary_complete(position) VALUES (?1)",
                [position],
            )?;
        }
    }
    Ok(())
}

pub(super) fn complete(conn: &Connection, position: i64) -> Result<()> {
    let mut state = replay_state(conn)?;
    state.applied_position = Some(position);
    save(conn, &state)
}
