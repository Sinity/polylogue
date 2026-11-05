//! Signal checkpoints and receipts share the authored source transaction. The
//! evaluator and delivery worker own decisions; this module freezes their facts.
use crate::{Result, StoreError, source::Source};
use polylogue_model::{
    authored::{AuthoredEffect, DeliveryRecord, ExecutionEligibility},
    signal::{Checkpoint, Commit, EventRecord, Metadata},
    user::{UserOutcome, UserRequest},
};
use rusqlite::{Connection, OptionalExtension, params};
use shared_signals::notifications::Outlets;
use std::collections::BTreeSet;

pub(crate) fn checkpoint(
    conn: &Connection,
    source: Source,
    owner: &str,
) -> Result<Option<Checkpoint>> {
    let body: Option<Vec<u8>> = conn
        .query_row(
            &source.sql("SELECT body FROM @source.signal_state WHERE owner=?1"),
            [owner],
            |r| r.get(0),
        )
        .optional()?;
    body.map(|body| crate::encoding::unpack_json(&body))
        .transpose()
}
pub(crate) fn metadata(conn: &Connection, source: Source, id: &str) -> Result<Option<Metadata>> {
    let body: Option<Vec<u8>> = conn
        .query_row(
            &source.sql("SELECT signal FROM @source.delivery WHERE id=?1"),
            [id],
            |r| r.get(0),
        )
        .optional()?
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
fn identity(value: &Metadata) -> Result<()> {
    let fired = &value.delivery.fired;
    if fired.id != fired.trigger.delivery_id()
        || value.notice.delivery_id != fired.id
        || value.notice.product != fired.trigger.rule.product
        || value.notice.current != fired.current
        || value.notice.outlets != Outlets::of(&value.delivery.decision)
    {
        return Err(StoreError::Invalid(
            "signal outbox identity disagrees with its trigger".into(),
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
        return Err(StoreError::Invalid(
            "signal checkpoint owner differs from accepting machine".into(),
        ));
    }
    let prior = checkpoint(conn, source, &commit.checkpoint.owner)?
        .map(|s| s.identity())
        .transpose()?;
    if prior != commit.prior {
        return Err(StoreError::Conflict("signal checkpoint changed".into()));
    }
    let mut ids = BTreeSet::new();
    for outbox in &commit.outbox {
        identity(&outbox.metadata)?;
        if outbox.record.sender != "signal" {
            return Err(StoreError::Invalid(
                "signal delivery must retain its typed sender".into(),
            ));
        }
        let id = outbox.metadata.delivery.id().to_owned();
        if !ids.insert(id.clone()) || outbox.metadata.execution != ExecutionEligibility::Live {
            return Err(StoreError::Invalid(
                "new signal outbox must have one live identity".into(),
            ));
        }
        let at = polylogue_model::Timestamp::from_unix_ms(outbox.metadata.delivery.at)
            .ok_or_else(|| StoreError::Invalid("signal time is out of range".into()))?;
        crate::primary::authored::store_delivery(
            conn,
            &DeliveryRecord {
                id,
                signal: Some(outbox.metadata.clone()),
                prepared: outbox.record.clone(),
                at,
                events: Vec::new(),
            },
            counter,
            false,
        )?;
    }
    for (ordinal, event) in commit.events.iter().enumerate() {
        let id = event.delivery_id();
        let base = metadata(conn, source, id)?
            .ok_or_else(|| StoreError::NotFound(format!("signal delivery {id}")))?;
        if let polylogue_model::signal::Event::HostDecision { delivery } = event
            && (delivery.fired.trigger != base.delivery.fired.trigger
                || delivery.id() != base.delivery.id())
        {
            return Err(StoreError::Invalid(
                "host signal decision changed occurrence identity".into(),
            ));
        }
        let position = events(conn, source, id)?.last().map_or(Ok(1), |e| {
            e.position
                .checked_add(1)
                .ok_or_else(|| StoreError::Invalid("signal receipt position overflow".into()))
        })?;
        insert_event(
            conn,
            &EventRecord {
                request: request.id.clone(),
                ordinal: u64::try_from(ordinal)
                    .map_err(|_| StoreError::Invalid("signal event ordinal overflow".into()))?,
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
    conn.execute(&source.sql("INSERT INTO @source.signal_state(owner,body,commit_seq) VALUES(?1,?2,?3) ON CONFLICT(owner) DO UPDATE SET body=excluded.body,commit_seq=excluded.commit_seq"),params![value.owner,crate::primary::authored::pack(value)?,counter])?;
    Ok(())
}
fn number(value: u64) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| StoreError::Invalid("signal receipt order exceeds SQLite range".into()))
}
fn insert_event(conn: &Connection, value: &EventRecord, counter: i64) -> Result<()> {
    if value.position == 0 || value.request.is_empty() {
        return Err(StoreError::Invalid("signal event identity is empty".into()));
    }
    let source = Source::of(conn)?;
    let id = value.event.delivery_id();
    let old:Option<Vec<u8>>=conn.query_row(&source.sql("SELECT body FROM @source.signal_event WHERE (request=?1 AND ordinal=?2) OR (delivery=?3 AND position=?4)"),params![value.request,number(value.ordinal)?,id,number(value.position)?],|r|r.get(0)).optional()?;
    if let Some(old) = old {
        let old: EventRecord = crate::encoding::unpack_json(&old)?;
        return if old == *value {
            Ok(())
        } else {
            Err(StoreError::Conflict("signal event identity reused".into()))
        };
    }
    conn.execute(&source.sql("INSERT INTO @source.signal_event(request,ordinal,delivery,position,at,body,commit_seq) VALUES(?1,?2,?3,?4,?5,?6,?7)"),params![value.request,number(value.ordinal)?,id,number(value.position)?,value.at.unix_ms(),crate::primary::authored::pack(value)?,counter])?;
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
            checkpoint(conn, source, &commit.checkpoint.owner)?.ok_or_else(|| {
                StoreError::Invalid("signal checkpoint missing after commit".into())
            })?,
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

/// Validate imported frozen histories before any row is applied.
pub(crate) fn validate(prepared: &polylogue_model::authored::PreparedRequest) -> Result<()> {
    let effect = &prepared.effect;
    let mut identities = BTreeSet::new();
    let mut positions = std::collections::BTreeMap::<&str, Vec<u64>>::new();
    for change in &effect.deliveries {
        if let Some(metadata) = &change.delivery.signal {
            identity(metadata)?;
            if change.delivery.id != metadata.delivery.id() {
                return Err(StoreError::Invalid(
                    "signal delivery row identity differs".into(),
                ));
            }
        }
    }
    for event in &effect.signal_events {
        let id = event.event.delivery_id();
        if event.position == 0
            || event.request.is_empty()
            || !identities.insert((&event.request, event.ordinal))
        {
            return Err(StoreError::Invalid(
                "signal event identity is empty or duplicated".into(),
            ));
        }
        let metadata = effect
            .deliveries
            .iter()
            .find(|d| d.delivery.id == id)
            .and_then(|d| d.delivery.signal.as_ref())
            .ok_or_else(|| StoreError::Invalid("signal event has no frozen delivery".into()))?;
        if let polylogue_model::signal::Event::HostDecision { delivery } = &event.event
            && delivery.fired.trigger != metadata.delivery.fired.trigger
        {
            return Err(StoreError::Invalid(
                "signal decision changes occurrence".into(),
            ));
        }
        positions.entry(id).or_default().push(event.position);
    }
    for positions in positions.values_mut() {
        positions.sort_unstable();
        if positions.iter().copied().ne(1..=positions.len() as u64) {
            return Err(StoreError::Invalid(
                "signal event history has a gap or duplicate".into(),
            ));
        }
    }
    match &prepared.request.write {
        polylogue_model::user::UserWrite::SignalCommit(commit) => {
            if effect.signal_checkpoints.as_slice() != [commit.checkpoint.clone()] {
                return Err(StoreError::Invalid(
                    "signal checkpoint differs from frozen request".into(),
                ));
            }
            for (ordinal, event) in commit.events.iter().enumerate() {
                if !effect.signal_events.iter().any(|e| {
                    e.request == prepared.request.id
                        && e.ordinal == ordinal as u64
                        && &e.event == event
                }) {
                    return Err(StoreError::Invalid(
                        "signal receipt differs from frozen request".into(),
                    ));
                }
            }
            for outbox in &commit.outbox {
                if !effect.deliveries.iter().any(|d| {
                    d.delivery.id == outbox.metadata.delivery.id()
                        && d.delivery.signal.as_ref() == Some(&outbox.metadata)
                        && d.delivery.prepared == outbox.record
                }) {
                    return Err(StoreError::Invalid(
                        "signal outbox differs from frozen request".into(),
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
                return Err(StoreError::Invalid(
                    "signal changes lack a signal request".into(),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
