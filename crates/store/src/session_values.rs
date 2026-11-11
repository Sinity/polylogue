//! What each session contributes to completion (`session_value`,
//! `session_first_prompt`): the models its messages name and the tools its
//! calls use, with counts, and its first typed prompt. Every write
//! transaction that changes a session's rows brings these up to date before
//! it commits, from its [change summary](crate::change_summary): each changed
//! message takes away what it contributed before the change and adds what it
//! contributes now, so a transaction's cost follows what it changed. A row
//! whose count changes takes the transaction's write counter, and the
//! completion catalog copies the rows changed since it was published.
use crate::Result;
use crate::change_summary::{ChangeSummary, OrderKey, SessionChange};
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::BTreeMap;

/// A completable field and one of its values.
pub(crate) type Value = (String, String);

/// What one message of `session_pk` contributes: its model and the tool of
/// each of its calls. Nothing when the message is not in that session.
pub(crate) fn of_message(conn: &Connection, pk: i64, session_pk: i64) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    if let Some(model) = conn
        .prepare_cached(
            "SELECT model FROM message WHERE pk = ?1 AND session_pk = ?2
             AND model IS NOT NULL AND model <> ''",
        )?
        .query_row([pk, session_pk], |row| row.get::<_, String>(0))
        .optional()?
    {
        out.push(("model".into(), model));
    }
    let mut tools = conn.prepare_cached(
        "SELECT t.tool FROM block b JOIN tool_call t ON t.call_block_pk = b.pk
         WHERE b.message_pk = ?1 AND t.session_pk = ?2",
    )?;
    for tool in tools.query_map([pk, session_pk], |row| row.get::<_, String>(0))? {
        out.push(("tool".into(), tool?));
    }
    Ok(out)
}

pub(crate) fn write_counter(conn: &Connection) -> Result<i64> {
    Ok(conn
        .query_row(
            "SELECT CAST(value AS INTEGER) FROM meta WHERE key = 'write_counter'",
            [],
            |row| row.get(0),
        )
        .optional()?
        .unwrap_or(0))
}

/// Brings every session a transaction changed up to date from its change
/// summary.
pub(crate) fn refresh(conn: &Connection, changes: &ChangeSummary) -> Result<()> {
    let counter = write_counter(conn)?;
    let mut exists = conn.prepare_cached("SELECT EXISTS(SELECT 1 FROM session WHERE pk = ?1)")?;
    for (session_pk, change) in changes.sessions(conn)? {
        if exists.query_row([session_pk], |row| row.get::<_, bool>(0))? {
            refresh_session(conn, session_pk, &change, changes, counter)?;
        }
    }
    Ok(())
}

fn refresh_session(
    conn: &Connection,
    session_pk: i64,
    change: &SessionChange,
    changes: &ChangeSummary,
    counter: i64,
) -> Result<()> {
    // The first-prompt row exists once the session's values were computed.
    let stored: Option<Option<i64>> = conn
        .prepare_cached("SELECT message_pk FROM session_first_prompt WHERE session_pk = ?1")?
        .query_row([session_pk], |row| row.get(0))
        .optional()?;
    let Some(first) = stored.filter(|_| !change.whole) else {
        return rebuild(conn, session_pk, counter);
    };
    let mut delta: BTreeMap<Value, i64> = BTreeMap::new();
    let mut candidates = Vec::new();
    for &message in &change.messages {
        for value in changes.values_before(message, session_pk) {
            *delta.entry(value).or_default() -= 1;
        }
        for value in of_message(conn, message, session_pk)? {
            *delta.entry(value).or_default() += 1;
        }
        if let Some(place) = prompt_place(conn, message, session_pk)? {
            candidates.push((place, message));
        }
    }
    let mut add = conn.prepare_cached(
        "INSERT INTO session_value(session_pk, field, value, n, changed) VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(session_pk, field, value) DO UPDATE SET n = n + excluded.n, changed = excluded.changed",
    )?;
    // SQLite checks the proposed row of an upsert, so a decrease updates.
    let mut take = conn.prepare_cached(
        "UPDATE session_value SET n = n - ?4, changed = ?5
         WHERE session_pk = ?1 AND field = ?2 AND value = ?3",
    )?;
    for ((field, value), n) in delta {
        if n > 0 {
            add.execute(params![session_pk, field, value, n, counter])?;
        } else if n < 0 {
            let taken = take.execute(params![session_pk, field, value, -n, counter])?;
            if taken != 1 {
                return Err(crate::StoreError::Invalid(format!(
                    "session {session_pk} had no stored {field} {value} to take away"
                )));
            }
        }
    }
    // Unchanged messages keep whether they are a typed prompt and their
    // place, so the first one is the stored one or a changed one, unless the
    // stored one itself changed.
    let next = match first {
        Some(pk) if change.messages.contains(&pk) => first_prompt(conn, session_pk)?,
        _ => {
            let held = first
                .map(|pk| Ok::<_, crate::StoreError>((prompt_place(conn, pk, session_pk)?, pk)))
                .transpose()?
                .and_then(|(place, pk)| place.map(|place| (place, pk)));
            held.into_iter().chain(candidates).min().map(|(_, pk)| pk)
        }
    };
    if next != first {
        set_first_prompt(conn, session_pk, next, counter)?;
    }
    Ok(())
}

/// The session's values computed from all its rows: for a session first
/// seen, or one whose rows changed in a way the summary does not describe.
/// Only counts that differ from the stored ones are written.
pub(crate) fn rebuild(conn: &Connection, session_pk: i64, counter: i64) -> Result<()> {
    let mut values: BTreeMap<Value, i64> = BTreeMap::new();
    let mut now = conn.prepare_cached(
        "SELECT 'model', model, count(*) FROM message
         WHERE session_pk = ?1 AND model IS NOT NULL AND model <> '' GROUP BY model
         UNION ALL SELECT 'tool', tool, count(*) FROM tool_call WHERE session_pk = ?1 GROUP BY tool",
    )?;
    for row in now.query_map([session_pk], |row| {
        Ok(((row.get(0)?, row.get(1)?), row.get::<_, i64>(2)?))
    })? {
        let (value, n) = row?;
        values.insert(value, n);
    }
    let mut held =
        conn.prepare_cached("SELECT field, value, n FROM session_value WHERE session_pk = ?1")?;
    let stored: Vec<(Value, i64)> = held
        .query_map([session_pk], |row| {
            Ok(((row.get(0)?, row.get(1)?), row.get(2)?))
        })?
        .collect::<rusqlite::Result<_>>()?;
    let mut set = conn.prepare_cached(
        "INSERT INTO session_value(session_pk, field, value, n, changed) VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(session_pk, field, value) DO UPDATE SET n = excluded.n, changed = excluded.changed",
    )?;
    for (value, n) in stored {
        let wanted = values.remove(&value).unwrap_or(0);
        if wanted != n {
            set.execute(params![session_pk, value.0, value.1, wanted, counter])?;
        }
    }
    for ((field, value), n) in values {
        set.execute(params![session_pk, field, value, n, counter])?;
    }
    let first: Option<Option<i64>> = conn
        .prepare_cached("SELECT message_pk FROM session_first_prompt WHERE session_pk = ?1")?
        .query_row([session_pk], |row| row.get(0))
        .optional()?;
    let next = first_prompt(conn, session_pk)?;
    if first != Some(next) {
        set_first_prompt(conn, session_pk, next, counter)?;
    }
    Ok(())
}

/// A message's place in session order when it is a typed prompt with text.
fn prompt_place(conn: &Connection, pk: i64, session_pk: i64) -> Result<Option<OrderKey>> {
    Ok(conn
        .prepare_cached(
            "SELECT m.session_order, m.id FROM message m
             WHERE m.pk = ?1 AND m.session_pk = ?2 AND m.written_by = 'human'
               AND m.prompt_kind = 'prose'
               AND EXISTS (SELECT 1 FROM block b JOIN text t ON t.id = b.text_id
                           WHERE b.message_pk = m.pk AND b.kind = 'text')",
        )?
        .query_row([pk, session_pk], |row| Ok((row.get(0)?, row.get(1)?)))
        .optional()?)
}

/// The session's first typed prompt with text, from all its messages.
fn first_prompt(conn: &Connection, session_pk: i64) -> Result<Option<i64>> {
    Ok(conn
        .prepare_cached(
            "SELECT m.pk FROM message m
             WHERE m.session_pk = ?1 AND m.written_by = 'human' AND m.prompt_kind = 'prose'
               AND EXISTS (SELECT 1 FROM block b JOIN text t ON t.id = b.text_id
                           WHERE b.message_pk = m.pk AND b.kind = 'text')
             ORDER BY m.session_order, m.id LIMIT 1",
        )?
        .query_row([session_pk], |row| row.get(0))
        .optional()?)
}

fn set_first_prompt(
    conn: &Connection,
    session_pk: i64,
    message_pk: Option<i64>,
    counter: i64,
) -> Result<()> {
    conn.prepare_cached(
        "INSERT INTO session_first_prompt(session_pk, message_pk, changed) VALUES (?1, ?2, ?3)
         ON CONFLICT(session_pk) DO UPDATE SET message_pk = excluded.message_pk, changed = excluded.changed",
    )?
    .execute(params![session_pk, message_pk, counter])?;
    Ok(())
}

#[cfg(test)]
mod tests;
