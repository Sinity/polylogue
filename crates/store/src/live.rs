//! Live facts (spec model.md, Live state): the latest hook evidence for a
//! session, kept in its live columns in the transaction that stores the
//! hook line. The live state itself is computed when asked, from these
//! columns and the time ([`polylogue_query`'s `live` field]); nothing here
//! looks at the clock.

use crate::Result;
use polylogue_model::{Event, Json};
use rusqlite::{OptionalExtension, Transaction, params};

/// What a session waits on, from a hook event, or `None` when the event
/// ends a wait. `Some(None)` leaves the wait as it was (an event that
/// decides nothing, such as a notification of unknown type).
fn wait_change(kind: &str, data: &Json) -> Option<Option<&'static str>> {
    match kind {
        "hook_permission_request" => Some(Some("permission")),
        "hook_notification" => match data.get("notification_type").and_then(Json::as_str) {
            Some("permission_prompt") => Some(Some("permission")),
            Some("elicitation_dialog") => Some(Some("question")),
            // An idle notification is evidence for `idle`, not a wait.
            Some("idle_prompt") => Some(None),
            _ => None,
        },
        _ => Some(None),
    }
}

/// The in-progress item of a `TodoWrite` call, as text. Never turned into
/// a task identity.
fn todo_in_progress(data: &Json) -> Option<String> {
    if data.get("tool_name").and_then(Json::as_str) != Some("TodoWrite") {
        return None;
    }
    let todos = data.pointer("/tool_input/todos")?.as_array()?;
    todos
        .iter()
        .find(|t| t.get("status").and_then(Json::as_str) == Some("in_progress"))
        .and_then(|t| {
            t.get("activeForm")
                .or_else(|| t.get("content"))
                .and_then(Json::as_str)
        })
        .map(str::to_owned)
}

/// Updates `session_pk`'s live columns from one stored hook event. Only an
/// event at least as new as the stored one changes them, so lines applied
/// out of order, or again, leave the latest facts in place.
pub(crate) fn apply(
    tx: &Transaction<'_>,
    session_pk: i64,
    event: &Event,
    counter: i64,
) -> Result<()> {
    if !event.kind.starts_with("hook_") {
        return Ok(());
    }
    let Some(at) = event.at.map(|t| t.unix_ms()) else {
        // A hook line without a time cannot be ordered against the others.
        return Ok(());
    };
    let last: Option<i64> = tx
        .prepare_cached("SELECT live_last_event_at FROM session WHERE pk = ?1")?
        .query_row([session_pk], |r| r.get(0))
        .optional()?
        .flatten();
    if last.is_some_and(|l| l > at) {
        return Ok(());
    }
    let data = &event.data;
    tx.prepare_cached(
        "UPDATE session SET live_last_event = ?2, live_last_event_at = ?3, live_changed = ?4
         WHERE pk = ?1",
    )?
    .execute(params![session_pk, event.kind, at, counter])?;
    if let Some(wait) = wait_change(&event.kind, data) {
        tx.prepare_cached(
            "UPDATE session SET live_waiting_on = ?2,
                 live_waiting_since = CASE WHEN ?2 IS NULL THEN NULL
                     WHEN live_waiting_on IS ?2 THEN live_waiting_since ELSE ?3 END
             WHERE pk = ?1",
        )?
        .execute(params![session_pk, wait, at])?;
    }
    match event.kind.as_str() {
        "hook_pre_tool_use" => {
            let call = data.get("tool_use_id").and_then(Json::as_str);
            let pk: Option<i64> = match call {
                Some(id) => tx
                    .prepare_cached(
                        "SELECT pk FROM tool_call WHERE session_pk = ?1 AND call_id = ?2",
                    )?
                    .query_row(params![session_pk, id], |r| r.get(0))
                    .optional()?,
                None => None,
            };
            tx.prepare_cached("UPDATE session SET live_open_tool_pk = ?2 WHERE pk = ?1")?
                .execute(params![session_pk, pk])?;
            if let Some(todo) = todo_in_progress(data) {
                tx.prepare_cached("UPDATE session SET live_todo = ?2 WHERE pk = ?1")?
                    .execute(params![session_pk, todo])?;
            }
        }
        "hook_post_tool_use" | "hook_stop" | "hook_session_end" | "hook_user_prompt_submit" => {
            tx.prepare_cached("UPDATE session SET live_open_tool_pk = NULL WHERE pk = ?1")?
                .execute([session_pk])?;
        }
        _ => {}
    }
    Ok(())
}

/// Recompute hook-derived columns after source interpretation changes. Process
/// and task evidence retain their independent derivations.
pub(crate) fn rebuild(tx: &Transaction<'_>, session_pk: i64, counter: i64) -> Result<()> {
    tx.execute(
        "UPDATE session SET live_last_event=NULL,live_last_event_at=NULL,live_open_tool_pk=NULL,
        live_waiting_on=NULL,live_waiting_since=NULL,live_todo=NULL,live_changed=?2 WHERE pk=?1",
        params![session_pk, counter],
    )?;
    let id: String = tx.query_row("SELECT id FROM session WHERE pk=?1", [session_pk], |r| {
        r.get(0)
    })?;
    let session = polylogue_model::SessionId::parse(&id)
        .map_err(|e| crate::StoreError::Invalid(e.to_string()))?;
    let mut statement = tx.prepare("SELECT kind,at,data FROM current_event WHERE session_pk=?1 AND substr(kind,1,5)='hook_' ORDER BY at,seq,id")?;
    let mut rows = statement.query([session_pk])?;
    while let Some(row) = rows.next()? {
        let event = Event {
            session: Some(session.clone()),
            kind: row.get(0)?,
            at: row
                .get::<_, Option<i64>>(1)?
                .and_then(polylogue_model::Timestamp::from_unix_ms),
            after: None,
            range: None,
            data: crate::encoding::unpack_json(&row.get::<_, Vec<u8>>(2)?)?,
        };
        apply(tx, session_pk, &event, counter)?;
    }
    Ok(())
}
