//! Signal checkpoints and delivery receipts share the authored source
//! transaction. The evaluator and delivery worker make the decisions; this
//! module stores them unchanged.
use crate::ConnExt as _;
use crate::OptionExt as _;
use crate::{Result, StoreError, source::Source};
use polylogue_model::{
    DeliverySender,
    authored::{AuthoredEffect, DeliveryRecord, RunsHere},
    signal::{Checkpoint, Commit, EventRecord, Metadata},
    user::{UserOutcome, UserRequest},
};
use rusqlite::{Connection, params};
use shared_signals::notifications::Outlets;
use std::collections::BTreeSet;

pub(crate) fn checkpoint(
    conn: &Connection,
    source: Source,
    evaluator: &str,
) -> Result<Option<Checkpoint>> {
    let body: Option<Vec<u8>> = conn.scalar_optional(
        &source.sql("SELECT body FROM @source.signal_state WHERE owner=?1"),
        [evaluator],
    )?;
    body.map(|body| crate::encoding::unpack_json(&body))
        .transpose()
}
pub(crate) fn metadata(conn: &Connection, source: Source, id: &str) -> Result<Option<Metadata>> {
    let body: Option<Vec<u8>> = conn
        .scalar_optional(
            &source.sql("SELECT signal FROM @source.delivery WHERE id=?1"),
            [id],
        )?
        .flatten();
    body.map(|body| crate::encoding::unpack_json(&body))
        .transpose()
}
pub(crate) fn events(conn: &Connection, source: Source, id: &str) -> Result<Vec<EventRecord>> {
    let mut query = conn.prepare(
        &source.sql("SELECT body FROM @source.signal_event WHERE delivery=?1 ORDER BY position"),
    )?;
    let rows = query.query_map([id], |r| r.get::<_, Vec<u8>>(0))?;
    rows.map(|body| crate::encoding::unpack_json(&body?))
        .collect()
}
fn check_pending_identity(value: &Metadata) -> Result<()> {
    let fired = &value.delivery.fired;
    if fired.id != fired.trigger.delivery_id()
        || value.notice.delivery_id != fired.id
        || value.notice.product != fired.trigger.rule.product
        || value.notice.current != fired.current
        || value.notice.outlets != Outlets::of(&value.delivery.decision)
    {
        return Err(StoreError::internal(
            "pending delivery identity disagrees with its trigger",
        ));
    }
    Ok(())
}
pub(crate) fn write(
    conn: &Connection,
    request: &UserRequest,
    commit: &Commit,
    counter: i64,
) -> Result<UserOutcome> {
    let source = Source::of(conn)?;
    if commit.checkpoint.owner != request.machine {
        return Err(StoreError::internal(
            "signal checkpoint evaluator differs from the accepting machine",
        ));
    }
    let prior = checkpoint(conn, source, &commit.checkpoint.owner)?
        .map(|s| s.identity())
        .transpose()?;
    if prior != commit.prior {
        return Err(StoreError::internal(
            "a signal checkpoint changed under its writer",
        ));
    }
    let mut ids = BTreeSet::new();
    for pending in &commit.pending_delivery {
        check_pending_identity(&pending.metadata)?;
        if pending.record.sender != DeliverySender::Signal {
            return Err(StoreError::internal(
                "signal delivery must keep its typed sender",
            ));
        }
        let id = pending.metadata.delivery.id().to_owned();
        if !ids.insert(id.clone()) || pending.metadata.execution != RunsHere::Live {
            return Err(StoreError::internal(
                "each new pending delivery must have one live identity",
            ));
        }
        let at = polylogue_model::Timestamp::from_unix_ms(pending.metadata.delivery.at)
            .or_stored("signal time is out of range")?;
        crate::authored_records::store_delivery(
            conn,
            &DeliveryRecord {
                id,
                signal: Some(pending.metadata.clone()),
                prepared: pending.record.clone(),
                at,
                events: Vec::new(),
            },
            counter,
            false,
        )?;
    }
    for (ordinal, event) in commit.events.iter().enumerate() {
        let id = event.delivery_id();
        let Some(base) = metadata(conn, source, id)? else {
            let stored: Option<i64> = conn.scalar_optional(
                &source.sql("SELECT 1 FROM @source.delivery WHERE id=?1"),
                [id],
            )?;
            return Err(if stored.is_some() {
                StoreError::internal(format!("delivery {id} is not a signal delivery"))
            } else {
                StoreError::NotFound(format!("signal delivery {id}"))
            });
        };
        if let polylogue_model::signal::Event::HostDecision { delivery } = event
            && (delivery.fired.trigger != base.delivery.fired.trigger
                || delivery.id() != base.delivery.id())
        {
            return Err(StoreError::internal(
                "host signal decision changed occurrence identity",
            ));
        }
        let last: i64 = conn.scalar(
            &source.sql(
                "SELECT COALESCE(MAX(position),0) FROM @source.signal_event WHERE delivery=?1",
            ),
            [id],
        )?;
        let position = crate::to_u64(last, "stored signal position is negative")?
            .checked_add(1)
            .or_internal("signal receipt position overflow")?;
        insert_event(
            conn,
            &EventRecord {
                request: request.id.clone(),
                ordinal: crate::to_u64(ordinal, "signal event ordinal overflow")?,
                position,
                at: request.at,
                event: event.clone(),
            },
            counter,
        )?;
    }
    store_checkpoint(conn, &commit.checkpoint, counter)?;
    Ok(UserOutcome {
        id: request.id.clone(),
        repeat: false,
        pending: false,
    })
}
fn store_checkpoint(conn: &Connection, value: &Checkpoint, counter: i64) -> Result<()> {
    let source = Source::of(conn)?;
    conn.execute(&source.sql("INSERT INTO @source.signal_state(owner,body,commit_seq) VALUES(?1,?2,?3) ON CONFLICT(owner) DO UPDATE SET body=excluded.body,commit_seq=excluded.commit_seq"),params![value.owner,crate::authored_records::pack(value)?,counter])?;
    Ok(())
}
fn ordinal(value: &EventRecord) -> Result<i64> {
    crate::to_i64(value.ordinal, "signal event ordinal exceeds SQLite range")
}
fn position(value: &EventRecord) -> Result<i64> {
    crate::to_i64(value.position, "signal event position exceeds SQLite range")
}
fn insert_event(conn: &Connection, value: &EventRecord, counter: i64) -> Result<()> {
    if value.position == 0 || value.request.is_empty() {
        return Err(StoreError::internal("signal event identity is empty"));
    }
    let source = Source::of(conn)?;
    let id = value.event.delivery_id();
    let old:Option<Vec<u8>>=conn.scalar_optional(&source.sql("SELECT body FROM @source.signal_event WHERE (request=?1 AND request_position=?2) OR (delivery=?3 AND position=?4)"),params![value.request,ordinal(value)?,id,position(value)?])?;
    if let Some(old) = old {
        let old: EventRecord = crate::encoding::unpack_json(&old)?;
        return if old == *value {
            Ok(())
        } else {
            Err(StoreError::internal("a signal event identity was reused"))
        };
    }
    conn.execute(&source.sql("INSERT INTO @source.signal_event(request,request_position,delivery,position,at,body,commit_seq) VALUES(?1,?2,?3,?4,?5,?6,?7)"),params![value.request,ordinal(value)?,id,position(value)?,value.at.unix_ms(),crate::authored_records::pack(value)?,counter])?;
    Ok(())
}
pub(crate) fn capture(
    conn: &Connection,
    request: &UserRequest,
    counter: i64,
    effect: &mut AuthoredEffect,
) -> Result<Vec<String>> {
    let source = Source::of(conn)?;
    if let polylogue_model::user::UserWrite::SignalCommit(commit) = &request.write {
        effect.signal_checkpoints.push(
            checkpoint(conn, source, &commit.checkpoint.owner)?
                .ok_or_else(|| StoreError::internal("signal checkpoint missing after commit"))?,
        );
    }
    let ids:Vec<String>=conn.prepare(&source.sql("SELECT DISTINCT delivery FROM @source.signal_event WHERE commit_seq=?1 ORDER BY delivery"))?.query_map([counter],|r|r.get(0))?.collect::<rusqlite::Result<_>>()?;
    for id in &ids {
        effect.signal_events.extend(events(conn, source, id)?);
    }
    Ok(ids)
}
pub(crate) fn apply(
    conn: &Connection,
    effect: &AuthoredEffect,
    counter: i64,
    historical: bool,
) -> Result<()> {
    if !historical {
        for checkpoint in &effect.signal_checkpoints {
            store_checkpoint(conn, checkpoint, counter)?;
        }
    }
    for event in &effect.signal_events {
        insert_event(conn, event, counter)?;
    }
    Ok(())
}
pub(crate) fn copy(from: &Connection, to: &Connection) -> Result<()> {
    let source = Source::of(from)?;
    let mut query =
        from.prepare(&source.sql("SELECT body FROM @source.signal_state ORDER BY owner"))?;
    for body in query.query_map([], |r| r.get::<_, Vec<u8>>(0))? {
        let value: Checkpoint = crate::encoding::unpack_json(&body?)?;
        store_checkpoint(to, &value, 0)?;
    }
    let mut query = from
        .prepare(&source.sql("SELECT body FROM @source.signal_event ORDER BY delivery,position"))?;
    for body in query.query_map([], |r| r.get::<_, Vec<u8>>(0))? {
        let value: EventRecord = crate::encoding::unpack_json(&body?)?;
        insert_event(to, &value, 0)?;
    }
    Ok(())
}

/// Validate imported recorded histories before any row is applied.
pub(crate) fn validate(prepared: &polylogue_model::authored::PreparedRequest) -> Result<()> {
    let effect = &prepared.effect;
    let mut identities = BTreeSet::new();
    let mut positions = std::collections::BTreeMap::<&str, Vec<u64>>::new();
    for change in &effect.deliveries {
        if let Some(metadata) = &change.delivery.signal {
            check_pending_identity(metadata)?;
            if change.delivery.id != metadata.delivery.id() {
                return Err(StoreError::internal("signal delivery row identity differs"));
            }
        }
    }
    for event in &effect.signal_events {
        let id = event.event.delivery_id();
        if event.position == 0
            || event.request.is_empty()
            || !identities.insert((&event.request, event.ordinal))
        {
            return Err(StoreError::internal(
                "signal event identity is empty or duplicated",
            ));
        }
        let metadata = effect
            .deliveries
            .iter()
            .find(|d| d.delivery.id == id)
            .and_then(|d| d.delivery.signal.as_ref())
            .or_stored("signal event has no recorded delivery")?;
        if let polylogue_model::signal::Event::HostDecision { delivery } = &event.event
            && delivery.fired.trigger != metadata.delivery.fired.trigger
        {
            return Err(StoreError::internal("signal decision changes occurrence"));
        }
        positions.entry(id).or_default().push(event.position);
    }
    for positions in positions.values_mut() {
        positions.sort_unstable();
        if positions.iter().copied().ne(1..=positions.len() as u64) {
            return Err(StoreError::internal(
                "signal event history has a gap or duplicate",
            ));
        }
    }
    match &prepared.request.write {
        polylogue_model::user::UserWrite::SignalCommit(commit) => {
            if effect.signal_checkpoints.as_slice() != [commit.checkpoint.clone()] {
                return Err(StoreError::internal(
                    "signal checkpoint differs from recorded request",
                ));
            }
            for (ordinal, event) in commit.events.iter().enumerate() {
                if !effect.signal_events.iter().any(|e| {
                    e.request == prepared.request.id
                        && e.ordinal == ordinal as u64
                        && &e.event == event
                }) {
                    return Err(StoreError::internal(
                        "signal receipt differs from recorded request",
                    ));
                }
            }
            if effect
                .signal_events
                .iter()
                .filter(|e| e.request == prepared.request.id)
                .count()
                != commit.events.len()
            {
                return Err(StoreError::internal(
                    "signal receipts beyond the recorded request",
                ));
            }
            for pending in &commit.pending_delivery {
                if !effect.deliveries.iter().any(|d| {
                    d.delivery.id == pending.metadata.delivery.id()
                        && d.delivery.signal.as_ref() == Some(&pending.metadata)
                        && d.delivery.prepared == pending.record
                }) {
                    return Err(StoreError::internal(
                        "pending delivery differs from recorded request",
                    ));
                }
            }
        }
        _ => {
            if !effect.signal_checkpoints.is_empty()
                || effect
                    .signal_events
                    .iter()
                    .any(|e| e.request == prepared.request.id)
            {
                return Err(StoreError::internal("signal changes lack a signal request"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
