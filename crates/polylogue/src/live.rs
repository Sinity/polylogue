//! Live state (spec model.md, Live state) and the change feed. Both are
//! answers computed when asked: live state from stored live facts and the
//! time, never rewritten by a clock; the feed from the write counters.
//! Evidence only: nothing here coordinates agents (operator 2026-09-01).

use polylogue_query::fields::{live_state, live_until};
use polylogue_store::{Reader, Result, SqlValue};
use serde_json::{Value, json};

/// Sessions with live evidence in the last day, each with its state at
/// `now`, when that state changes without new facts, what it waits on, its
/// task and todo, and its peers. Ended sessions are left out.
pub fn sessions(reader: &Reader<'_>, now: i64) -> Result<Value> {
    let day = polylogue_query::fields::LIVE_DAY_MS;
    let sql = format!(
        "SELECT * FROM (SELECT s.id, {state} AS live, {until} AS until,
             s.live_last_event AS last_event, s.live_last_event_at AS last_event_at,
             s.live_waiting_on AS waiting_on, s.live_waiting_since AS waiting_since,
             s.live_task AS task, s.live_todo AS todo, s.cwd, s.checkout,
             f.repo_root, s.repo
         FROM session s LEFT JOIN session_facts f ON f.session_pk = s.pk
         WHERE MAX(COALESCE(s.live_last_event_at, 0), COALESCE(s.last_activity_at, 0)) >= ?1)
         WHERE live <> 'ended'
         ORDER BY CASE live WHEN 'waiting' THEN 0 WHEN 'working' THEN 1 WHEN 'idle' THEN 2
                  WHEN 'unknown' THEN 3 ELSE 4 END, last_event_at DESC",
        state = live_state("s", now),
        until = live_until("s", now),
    );
    let mut rows = reader.query(&sql, &[SqlValue::Int(now - day)])?;
    let peers = peers(&rows);
    for (row, p) in rows.iter_mut().zip(peers) {
        row["peers"] = json!(p);
    }
    Ok(json!({ "now": now, "sessions": rows }))
}

/// Peers (spec model.md): the other sessions working, waiting or idle that
/// share a repository or checkout, or, when either has no git evidence, a
/// working directory where one lies at or below the other, by whole path
/// components. The relation is symmetric by construction (law 89).
fn peers(rows: &[Value]) -> Vec<Vec<String>> {
    let active = |r: &Value| matches!(r["live"].as_str(), Some("working" | "waiting" | "idle"));
    let text = |r: &Value, k: &str| r[k].as_str().map(str::to_owned);
    rows.iter()
        .map(|a| {
            if !active(a) {
                return vec![];
            }
            rows.iter()
                .filter(|b| !std::ptr::eq(a, *b) && active(b))
                .filter(|b| {
                    let same = |k: &str| text(a, k).is_some() && text(a, k) == text(b, k);
                    let git =
                        |r: &Value| text(r, "repo_root").is_some() || text(r, "checkout").is_some();
                    same("repo_root")
                        || same("checkout")
                        || ((!git(a) || !git(b))
                            && match (text(a, "cwd"), text(b, "cwd")) {
                                (Some(x), Some(y)) => nested(&x, &y) || nested(&y, &x),
                                _ => false,
                            })
                })
                .filter_map(|b| text(b, "id"))
                .collect()
        })
        .collect()
}

/// Whether `inner` is `outer` or lies below it, by whole components.
fn nested(outer: &str, inner: &str) -> bool {
    let outer = outer.trim_end_matches('/');
    inner == outer
        || inner
            .strip_prefix(outer)
            .is_some_and(|r| r.starts_with('/'))
}

/// A change-feed position: the write counters of `index.db` and
/// `archive.db`, which count separately.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Cursor {
    pub index: i64,
    pub archive: i64,
}

impl Cursor {
    pub fn parse(text: &str) -> Option<Self> {
        let (i, a) = text.split_once('.')?;
        Some(Self {
            index: i.parse().ok()?,
            archive: a.parse().ok()?,
        })
    }

    pub fn render(self) -> String {
        format!("{}.{}", self.index, self.archive)
    }
}

/// What changed after `since`: sessions whose records or live facts were
/// written, sessions removed, and notes added; with the cursor to continue
/// from.
pub fn changes(reader: &Reader<'_>, since: Cursor) -> Result<Value> {
    let counter = |db: &str| -> Result<i64> {
        let rows = reader.query(
            &format!(
                "SELECT CAST(value AS INTEGER) AS n FROM {db}meta WHERE key = 'write_counter'"
            ),
            &[],
        )?;
        Ok(rows.first().and_then(|r| r["n"].as_i64()).unwrap_or(0))
    };
    let next = Cursor {
        index: counter("")?,
        archive: counter("a.")?,
    };
    let i = SqlValue::Int(since.index);
    let sessions = reader.query(
        "SELECT id AS session, changed,
             CASE WHEN live_changed > ?1 THEN live_changed END AS live_changed
         FROM session WHERE changed > ?1 OR live_changed > ?1
         ORDER BY MAX(changed, COALESCE(live_changed, 0))",
        std::slice::from_ref(&i),
    )?;
    let removed = reader.query(
        "SELECT session_id AS session, changed FROM removed_session WHERE changed > ?1 ORDER BY changed",
        std::slice::from_ref(&i),
    )?;
    let notes = reader.query(
        "SELECT id AS note, kind, local_seq FROM a.note WHERE local_seq > ?1 ORDER BY local_seq",
        &[SqlValue::Int(since.archive)],
    )?;
    Ok(json!({
        "since": since.render(),
        "next": next.render(),
        "sessions": sessions,
        "removed": removed,
        "notes": notes,
    }))
}

#[cfg(test)]
mod tests;
