//! Typed authored row codecs shared by preparation, confirmation and restore.
use super::*;
use polylogue_model::Timestamp;
use polylogue_model::authored::*;
use polylogue_model::user::{BackfillJob, ChatAction, UserWrite};
use std::collections::BTreeSet;

fn event_index(n: i64) -> Result<usize> {
    usize::try_from(
        n.checked_sub(1)
            .ok_or_else(|| StoreError::Invalid("event order is out of range".into()))?,
    )
    .map_err(|_| StoreError::Invalid("event order starts at one".into()))
}
fn time(at: i64) -> Result<Timestamp> {
    Timestamp::from_unix_ms(at)
        .ok_or_else(|| StoreError::Invalid("authored timestamp is out of range".into()))
}
fn ids(conn: &Connection, table: &str, filter: &str, seq: Option<i64>) -> Result<Vec<String>> {
    let source = Source::reading(conn)?;
    let sql = source.sql(&format!(
        "SELECT id FROM @source.{table} {filter} ORDER BY id"
    ));
    let mut statement = conn.prepare(&sql)?;
    let mut rows = if let Some(seq) = seq {
        statement.query([seq])?
    } else {
        statement.query([])?
    };
    let mut values = Vec::new();
    while let Some(row) = rows.next()? {
        values.push(row.get(0)?);
    }
    Ok(values)
}
fn kept(conn: &Connection, id: &str) -> Result<KeptResult> {
    let source = Source::reading(conn)?;
    let (name, query, at, read_at, rows_json): (Option<String>, String, i64, String, String) = conn
        .query_row(
            &source.sql(
                "SELECT name,query,ran_at,read_at,rows_json FROM @source.kept_result WHERE id=?1",
            ),
            [id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )?;
    Ok(KeptResult {
        id: id.into(),
        name,
        query,
        ran_at: time(at)?,
        read_at,
        rows_json,
    })
}
fn delivery_events(
    conn: &Connection,
    id: &str,
    counter: Option<i64>,
) -> Result<Vec<DeliveryEvent>> {
    delivery_events_from(conn, Source::reading(conn)?, id, counter)
}
fn delivery_events_from(
    conn: &Connection,
    source: Source,
    id: &str,
    counter: Option<i64>,
) -> Result<Vec<DeliveryEvent>> {
    let mut query =
        source.sql("SELECT what,at,detail,request FROM @source.delivery_event WHERE delivery=?1");
    if counter.is_some() {
        query.push_str(" AND commit_seq=?2");
    }
    query.push_str(" ORDER BY at,what");
    let mut statement = conn.prepare(&query)?;
    let mut rows = if let Some(counter) = counter {
        statement.query(params![id, counter])?
    } else {
        statement.query([id])?
    };
    let mut events = Vec::new();
    while let Some(row) = rows.next()? {
        let kind: String = row.get(0)?;
        events.push(DeliveryEvent {
            request: row.get(3)?,
            what: serde_json::from_value(serde_json::Value::String(kind))?,
            at: time(row.get(1)?)?,
            detail: row.get(2)?,
        });
    }
    Ok(events)
}
fn delivery(conn: &Connection, id: &str) -> Result<DeliveryRecord> {
    delivery_from(conn, Source::reading(conn)?, id)
}
pub(super) fn delivery_from(conn: &Connection, source: Source, id: &str) -> Result<DeliveryRecord> {
    struct DeliveryRow {
        recipient_kind: String,
        recipient_id: Option<String>,
        sender: String,
        event: String,
        cwd: Option<String>,
        at: i64,
        body: Vec<u8>,
        refs: String,
        left_out: String,
    }
    let DeliveryRow { recipient_kind, recipient_id, sender, event, cwd, at, body, refs, left_out } = conn.query_row(
        &source.sql("SELECT recipient_kind,recipient_id,sender,event,cwd,at,body,refs_json,left_out_json FROM @source.delivery WHERE id=?1"),
        [id],
        |r| Ok(DeliveryRow { recipient_kind: r.get(0)?, recipient_id: r.get(1)?, sender: r.get(2)?, event: r.get(3)?, cwd: r.get(4)?, at: r.get(5)?, body: r.get(6)?, refs: r.get(7)?, left_out: r.get(8)? }),
    )?;
    Ok(DeliveryRecord {
        signal: crate::signals::metadata(conn, source, id)?,
        id: id.into(),
        prepared: polylogue_model::user::Delivery {
            recipient_kind,
            recipient_id,
            sender,
            event,
            cwd,
            text: crate::encoding::unpack_text(&body)?,
            refs: serde_json::from_str(&refs)?,
            left_out: serde_json::from_str(&left_out)?,
        },
        at: time(at)?,
        events: delivery_events_from(conn, source, id, None)?,
    })
}

pub(super) fn capture(
    conn: &Connection,
    request: &UserRequest,
    counter: i64,
) -> Result<AuthoredEffect> {
    let source = Source::of(conn)?;
    let mut effect = AuthoredEffect::default();
    for id in ids(conn, "assertion", "WHERE commit_seq=?1", Some(counter))? {
        let row = crate::assertions::get(conn, &id)?.ok_or_else(|| StoreError::NotFound(id))?;
        effect.assertions.push(row.fact);
    }
    if matches!(request.write, UserWrite::KeepResult { .. }) {
        effect.kept_results.push(kept(conn, &request.id)?);
    }
    let touched = |table: &str, event_table: &str, column: &str| -> Result<Vec<String>> {
        Ok(conn.prepare(&source.sql(&format!("SELECT id FROM @source.{table} WHERE commit_seq=?1 UNION SELECT {column} FROM @source.{event_table} WHERE commit_seq=?1 ORDER BY 1")))?.query_map([counter],|r|r.get(0))?.collect::<rusqlite::Result<_>>()?)
    };
    let mut delivery_ids: BTreeSet<String> = touched("delivery", "delivery_event", "delivery")?
        .into_iter()
        .collect();
    delivery_ids.extend(crate::signals::capture(
        conn,
        request,
        counter,
        &mut effect,
    )?);
    for id in delivery_ids {
        effect.deliveries.push(DeliveryChange {
            delivery: delivery(conn, &id)?,
            added_events: delivery_events(conn, &id, Some(counter))?,
        });
    }
    for id in touched("chat_action", "chat_action_event", "action")? {
        let action = crate::chat_actions::read(conn, source.prefix(), &id)?
            .ok_or_else(|| StoreError::NotFound(id.clone()))?;
        let ns:Vec<i64>=conn.prepare(&source.sql("SELECT n FROM @source.chat_action_event WHERE action=?1 AND commit_seq=?2 ORDER BY n"))?.query_map(params![id,counter],|r|r.get(0))?.collect::<rusqlite::Result<_>>()?;
        let added_events = ns
            .into_iter()
            .map(|n| {
                action
                    .events
                    .get(event_index(n)?)
                    .cloned()
                    .ok_or_else(|| StoreError::Invalid("chat event order has a gap".into()))
            })
            .collect::<Result<_>>()?;
        effect.chats.push(ChatChange {
            action,
            added_events,
        });
    }
    for id in touched("backfill_job", "backfill_job_event", "job")? {
        let job = crate::backfill_jobs::read(conn, source.prefix(), &id)?
            .ok_or_else(|| StoreError::NotFound(id.clone()))?;
        let ns:Vec<i64>=conn.prepare(&source.sql("SELECT n FROM @source.backfill_job_event WHERE job=?1 AND commit_seq=?2 ORDER BY n"))?.query_map(params![id,counter],|r|r.get(0))?.collect::<rusqlite::Result<_>>()?;
        let added_events = ns
            .into_iter()
            .map(|n| {
                job.events
                    .get(event_index(n)?)
                    .cloned()
                    .ok_or_else(|| StoreError::Invalid("backfill event order has a gap".into()))
            })
            .collect::<Result<_>>()?;
        effect.backfills.push(BackfillChange { job, added_events });
    }
    effect.variants = crate::variants::capture(conn, request, counter)?;
    Ok(effect)
}

/// Every stored kept result passes here, so each one can be decoded by every
/// later preparation.
pub(crate) fn insert_kept(conn: &Connection, value: &KeptResult) -> Result<()> {
    let rows_json = crate::kept::evolve_body(&value.id, &value.rows_json)?;
    crate::selection::check_kept_body(conn, &value.id, &rows_json)?;
    let source = Source::of(conn)?;
    conn.execute(&source.sql("INSERT INTO @source.kept_result(id,name,query,ran_at,read_at,rows_json) VALUES(?1,?2,?3,?4,?5,?6)"),params![value.id,value.name,value.query,value.ran_at.unix_ms(),value.read_at,rows_json])?;
    Ok(())
}
pub(super) fn insert_delivery(
    conn: &Connection,
    value: &DeliveryRecord,
    counter: i64,
    historical: bool,
) -> Result<()> {
    let source = Source::of(conn)?;
    let d = &value.prepared;
    let existing: bool = conn
        .prepare(&source.sql("SELECT 1 FROM @source.delivery WHERE id=?1"))?
        .exists([&value.id])?;
    if existing {
        let old = delivery(conn, &value.id)?;
        let mut old_signal = old.signal;
        let mut new_signal = value.signal.clone();
        for signal in [&mut old_signal, &mut new_signal].into_iter().flatten() {
            signal.execution = ExecutionEligibility::Historical;
        }
        if old.prepared != value.prepared || old.at != value.at || old_signal != new_signal {
            return Err(StoreError::Conflict(value.id.clone()));
        }
        return Ok(());
    }
    conn.execute(&source.sql("INSERT INTO @source.delivery(id,recipient_kind,recipient_id,sender,event,cwd,at,content_hash,bytes,body,refs_json,left_out_json,commit_seq) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)"),params![value.id,d.recipient_kind,d.recipient_id,d.sender,d.event,d.cwd,value.at.unix_ms(),&blake3::hash(d.text.as_bytes()).as_bytes()[..],i64::try_from(d.text.len()).map_err(|_|StoreError::Invalid("delivery size is out of range".into()))?,crate::encoding::pack_text(&d.text)?,serde_json::to_string(&d.refs)?,serde_json::to_string(&d.left_out)?,counter])?;
    if let Some(mut signal) = value.signal.clone() {
        if historical {
            signal.execution = ExecutionEligibility::Historical;
        }
        conn.execute(
            &source.sql("UPDATE @source.delivery SET signal=?2 WHERE id=?1"),
            params![value.id, pack(&signal)?],
        )?;
    }
    Ok(())
}
fn append_delivery(
    conn: &Connection,
    id: &str,
    events: &[DeliveryEvent],
    counter: i64,
) -> Result<()> {
    let source = Source::of(conn)?;
    let known = delivery_events(conn, id, None)?;
    for event in events {
        if let Some(old) = known.iter().find(|old| old.request == event.request) {
            if old != event {
                return Err(StoreError::Conflict(event.request.clone()));
            }
            continue;
        }
        conn.execute(&source.sql("INSERT INTO @source.delivery_event(delivery,what,at,detail,request,commit_seq) VALUES(?1,?2,?3,?4,?5,?6)"),params![id,event.what.as_str(),event.at.unix_ms(),event.detail,event.request,counter])?;
    }
    Ok(())
}
fn insert_chat(
    conn: &Connection,
    value: &ChatAction,
    counter: i64,
    historical: bool,
) -> Result<()> {
    let source = Source::of(conn)?;
    if let Some(old) = crate::chat_actions::read(conn, source.prefix(), &value.id)? {
        if old.request_key != value.request_key
            || old.request_hash != value.request_hash
            || old.request != value.request
            || old.stage_only != value.stage_only
            || old.needs_approval != value.needs_approval
            || old.created_at != value.created_at
            || old.attachments != value.attachments
        {
            return Err(StoreError::Conflict(value.id.clone()));
        }
        return Ok(());
    }
    let execution = if historical {
        ExecutionEligibility::Historical
    } else {
        value.execution
    };
    conn.execute(&source.sql("INSERT INTO @source.chat_action(id,request_key,request_hash,request_json,stage_only,needs_approval,created_at,execution,commit_seq) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9)"),params![value.id,value.request_key,&value.request_hash.0[..],serde_json::to_string(&value.request)?,value.stage_only,value.needs_approval,value.created_at,execution.as_str(),counter])?;
    for a in &value.attachments {
        conn.execute(&source.sql("INSERT INTO @source.chat_action_attachment(action,id,blob,name,media_type,size) VALUES(?1,?2,?3,?4,?5,?6)"),params![value.id,a.id,&a.blob.0[..],a.name,a.media_type,i64::try_from(a.size).map_err(|_|StoreError::Invalid("attachment size is out of range".into()))?])?;
    }
    Ok(())
}
fn append_chat(
    conn: &Connection,
    id: &str,
    events: &[polylogue_model::user::ChatEvent],
    counter: i64,
) -> Result<()> {
    let source = Source::of(conn)?;
    let mut n: i64 = conn.query_row(
        &source.sql("SELECT COALESCE(max(n),0) FROM @source.chat_action_event WHERE action=?1"),
        [id],
        |r| r.get(0),
    )?;
    let known = crate::chat_actions::read(conn, source.prefix(), id)?
        .ok_or_else(|| StoreError::NotFound(id.into()))?
        .events;
    for event in events {
        if let Some(old) = known.iter().find(|old| old.request == event.request) {
            if old != event {
                return Err(StoreError::Conflict(event.request.clone()));
            }
            continue;
        }
        n = n
            .checked_add(1)
            .ok_or_else(|| StoreError::Invalid("chat event order is out of range".into()))?;
        conn.execute(&source.sql("INSERT INTO @source.chat_action_event(action,n,at,kind,phase,detail,by,receipt_json,request,commit_seq) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)"),params![id,n,event.at,event.kind,event.phase,event.detail,event.by,event.receipt.as_ref().map(serde_json::to_string).transpose()?,event.request,counter])?;
    }
    Ok(())
}
fn insert_backfill(
    conn: &Connection,
    value: &BackfillJob,
    counter: i64,
    historical: bool,
) -> Result<()> {
    let source = Source::of(conn)?;
    if let Some(old) = crate::backfill_jobs::read(conn, source.prefix(), &value.id)? {
        if old.provider != value.provider
            || old.since != value.since
            || old.created_at != value.created_at
        {
            return Err(StoreError::Conflict(value.id.clone()));
        }
        return Ok(());
    }
    let execution = if historical {
        ExecutionEligibility::Historical
    } else {
        value.execution
    };
    conn.execute(&source.sql("INSERT INTO @source.backfill_job(id,provider,since,created_at,execution,commit_seq) VALUES(?1,?2,?3,?4,?5,?6)"),params![value.id,value.provider,value.since,value.created_at,execution.as_str(),counter])?;
    Ok(())
}
fn append_backfill(
    conn: &Connection,
    id: &str,
    events: &[polylogue_model::user::BackfillEvent],
    counter: i64,
) -> Result<()> {
    let source = Source::of(conn)?;
    let mut n: i64 = conn.query_row(
        &source.sql("SELECT COALESCE(max(n),0) FROM @source.backfill_job_event WHERE job=?1"),
        [id],
        |r| r.get(0),
    )?;
    let known = crate::backfill_jobs::read(conn, source.prefix(), id)?
        .ok_or_else(|| StoreError::NotFound(id.into()))?
        .events;
    for event in events {
        if let Some(old) = known.iter().find(|old| old.request == event.request) {
            if old != event {
                return Err(StoreError::Conflict(event.request.clone()));
            }
            continue;
        }
        n = n
            .checked_add(1)
            .ok_or_else(|| StoreError::Invalid("backfill event order is out of range".into()))?;
        let count = |value: u64| {
            i64::try_from(value)
                .map_err(|_| StoreError::Invalid("backfill count is out of range".into()))
        };
        conn.execute(&source.sql("INSERT INTO @source.backfill_job_event(job,n,at,kind,by,position,listed,captured,unchanged,failed,detail,request,commit_seq) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)"),params![id,n,event.at,event.kind,event.by,event.position,count(event.counts.listed)?,count(event.counts.captured)?,count(event.counts.unchanged)?,count(event.counts.failed)?,event.detail,event.request,counter])?;
    }
    Ok(())
}

pub(super) fn apply(
    conn: &Connection,
    effect: &AuthoredEffect,
    counter: i64,
    historical: bool,
) -> Result<()> {
    crate::variants::apply(conn, &effect.variants, counter)?;
    crate::assertions::apply(conn, &effect.assertions, counter)?;
    for value in &effect.kept_results {
        insert_kept(conn, value)?;
    }
    for value in &effect.deliveries {
        insert_delivery(conn, &value.delivery, counter, historical)?;
        append_delivery(conn, &value.delivery.id, &value.delivery.events, counter)?;
    }
    for value in &effect.chats {
        insert_chat(conn, &value.action, counter, historical)?;
        append_chat(conn, &value.action.id, &value.action.events, counter)?;
    }
    crate::signals::apply(conn, effect, counter, historical)?;
    for value in &effect.backfills {
        insert_backfill(conn, &value.job, counter, historical)?;
        append_backfill(conn, &value.job.id, &value.job.events, counter)?;
    }
    Ok(())
}

/// Copy semantic records into private scratch. Bodies are decoded and written
/// through these same typed codecs; parser rows, captured bytes and SQL text
/// never enter the overlay.
pub(super) fn copy(from: &Connection, to: &Connection) -> Result<()> {
    let source = Source::reading(from)?;
    crate::variants::copy(from, to)?;
    crate::assertions::copy(from, to)?;
    for (table, copy) in [
        (
            "kept_result",
            copy_kept as fn(&Connection, &Connection, &str) -> Result<()>,
        ),
        ("delivery", copy_delivery),
        ("chat_action", copy_chat),
        ("backfill_job", copy_backfill),
    ] {
        let mut statement =
            from.prepare(&source.sql(&format!("SELECT id FROM @source.{table} ORDER BY id")))?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            copy(from, to, &id)?;
        }
    }
    crate::signals::copy(from, to)?;
    let sql = source.sql(
        "SELECT id,content_hash,created_at,request,outcome,effect FROM @source.client_request ORDER BY id",
    );
    let mut statement = from.prepare(&sql)?;
    let mut rows = statement.query([])?;
    let target = Source::of(to)?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let hash: Vec<u8> = row.get(1)?;
        let at: i64 = row.get(2)?;
        let request: Vec<u8> = row.get(3)?;
        let outcome: Option<Vec<u8>> = row.get(4)?;
        let effect: Option<Vec<u8>> = row.get(5)?;
        // Decode complete records before admitting them to the private view.
        let missing = || StoreError::Invalid(format!("request {id} has no frozen record"));
        let _: UserRequest = crate::encoding::unpack_json(&request)?;
        let _: UserOutcome = crate::encoding::unpack_json(outcome.as_ref().ok_or_else(missing)?)?;
        let _: AuthoredEffect = crate::encoding::unpack_json(effect.as_ref().ok_or_else(missing)?)?;
        to.execute(&target.sql("INSERT INTO @source.client_request(id,content_hash,created_at,request,outcome,effect,commit_seq) VALUES(?1,?2,?3,?4,?5,?6,0)"),params![id,hash,at,request,outcome,effect])?;
    }
    Ok(())
}
fn copy_kept(from: &Connection, to: &Connection, id: &str) -> Result<()> {
    insert_kept(to, &kept(from, id)?)
}
fn copy_delivery(from: &Connection, to: &Connection, id: &str) -> Result<()> {
    let value = delivery(from, id)?;
    insert_delivery(to, &value, 0, false)?;
    append_delivery(to, id, &value.events, 0)
}
fn copy_chat(from: &Connection, to: &Connection, id: &str) -> Result<()> {
    let source = Source::reading(from)?;
    let value = crate::chat_actions::read(from, source.prefix(), id)?
        .ok_or_else(|| StoreError::NotFound(id.into()))?;
    insert_chat(to, &value, 0, false)?;
    append_chat(to, id, &value.events, 0)
}
fn copy_backfill(from: &Connection, to: &Connection, id: &str) -> Result<()> {
    let source = Source::reading(from)?;
    let value = crate::backfill_jobs::read(from, source.prefix(), id)?
        .ok_or_else(|| StoreError::NotFound(id.into()))?;
    insert_backfill(to, &value, 0, false)?;
    append_backfill(to, id, &value.events, 0)
}
