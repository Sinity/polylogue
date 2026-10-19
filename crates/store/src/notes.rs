//! Reading notes and assertions from `archive.db` (attached to every read
//! as `a`). Writing goes through the writer ([`crate::user`]).
//!
//! The current revision of a note family is its newest note; a family with
//! a withdrawal is left out unless asked for.

use crate::{Reader, Result};
use polylogue_model::{Author, Note, NoteKind, Relation, Target, Timestamp};
use rusqlite::types::Value;
use serde::{Deserialize, Serialize};

/// Which notes to read. Empty lists mean "any".
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NoteFilter {
    /// Notes with at least one of these targets.
    #[serde(default)]
    pub targets: Vec<Target>,
    #[serde(default)]
    pub kinds: Vec<NoteKind>,
    /// `Some(false)`: operator and script notes only; `Some(true)`: agent
    /// notes only.
    #[serde(default)]
    pub agents: Option<bool>,
    /// Every revision instead of the current one of each family.
    #[serde(default)]
    pub history: bool,
    /// Include withdrawn families.
    #[serde(default)]
    pub withdrawn: bool,
    /// Notes written after this write-counter value (the change feed).
    #[serde(default)]
    pub after_seq: Option<i64>,
    #[serde(default)]
    pub limit: Option<u64>,
}

impl Reader<'_> {
    pub fn notes(&self, f: &NoteFilter) -> Result<Vec<Note>> {
        let mut sql = String::from(
            "SELECT n.id, n.root, n.kind, n.key, n.value, n.author, n.created_at,
                    n.valid_until, n.source
             FROM a.note n
             WHERE n.kind != 'withdrawal'",
        );
        let mut params: Vec<Value> = Vec::new();
        if !f.history {
            sql.push_str(
                " AND n.commit_seq = (SELECT max(m.commit_seq) FROM a.note m WHERE m.root = n.root)",
            );
        }
        if !f.withdrawn {
            sql.push_str(
                " AND NOT EXISTS (SELECT 1 FROM a.note_relation w
                   WHERE w.kind = 'withdraws' AND w.other = 'note:' || n.root)",
            );
        }
        if !f.targets.is_empty() {
            sql.push_str(
                " AND EXISTS (SELECT 1 FROM a.note_target t WHERE t.note_id = n.id AND t.ref IN (",
            );
            push_list(
                &mut sql,
                &mut params,
                f.targets.iter().map(ToString::to_string),
            );
            sql.push_str("))");
        }
        if !f.kinds.is_empty() {
            sql.push_str(" AND n.kind IN (");
            push_list(
                &mut sql,
                &mut params,
                f.kinds.iter().map(|k| k.as_str().to_owned()),
            );
            sql.push(')');
        }
        match f.agents {
            Some(true) => sql.push_str(" AND n.author LIKE 'agent%'"),
            Some(false) => sql.push_str(" AND n.author NOT LIKE 'agent%'"),
            None => {}
        }
        if let Some(after) = f.after_seq {
            params.push(Value::Integer(after));
            sql.push_str(&format!(" AND n.commit_seq > ?{}", params.len()));
        }
        sql.push_str(" ORDER BY n.commit_seq DESC");
        if let Some(limit) = f.limit {
            params.push(Value::Integer(i64::try_from(limit).unwrap_or(i64::MAX)));
            sql.push_str(&format!(" LIMIT ?{}", params.len()));
        }
        let conn = self.conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(params), |r| {
            Ok(RawNote {
                id: r.get(0)?,
                root: r.get(1)?,
                kind: r.get(2)?,
                key: r.get(3)?,
                value: r.get(4)?,
                author: r.get(5)?,
                created_at: r.get(6)?,
                valid_until: r.get(7)?,
                source: r.get(8)?,
            })
        })?;
        let mut out = Vec::new();
        for raw in rows {
            if let Some(note) = self.finish(raw?)? {
                out.push(note);
            }
        }
        Ok(out)
    }

    /// One note by ID, whatever its revision state.
    pub fn note(&self, id: &str) -> Result<Option<Note>> {
        let conn = self.conn();
        let mut stmt = conn.prepare_cached(
            "SELECT id, root, kind, key, value, author, created_at, valid_until, source
             FROM a.note WHERE id = ?1 AND kind != 'withdrawal'",
        )?;
        let mut rows = stmt.query([id])?;
        let Some(r) = rows.next()? else {
            return Ok(None);
        };
        let raw = RawNote {
            id: r.get(0)?,
            root: r.get(1)?,
            kind: r.get(2)?,
            key: r.get(3)?,
            value: r.get(4)?,
            author: r.get(5)?,
            created_at: r.get(6)?,
            valid_until: r.get(7)?,
            source: r.get(8)?,
        };
        drop(rows);
        self.finish(raw)
    }

    /// The archive's write counter: the change feed's position.
    pub fn archive_seq(&self) -> Result<i64> {
        Ok(self
            .conn()
            .query_row(
                "SELECT coalesce(CAST(value AS INTEGER), 0) FROM a.meta WHERE key = 'write_counter'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0))
    }

    /// Turns a row into a note; a row whose stored names this version does
    /// not know is skipped rather than shown wrong.
    fn finish(&self, raw: RawNote) -> Result<Option<Note>> {
        let (Some(kind), Ok(author)) = (NoteKind::parse(&raw.kind), Author::parse(&raw.author))
        else {
            return Ok(None);
        };
        let conn = self.conn();
        let targets = conn
            .prepare_cached("SELECT ref FROM a.note_target WHERE note_id = ?1 ORDER BY ref")?
            .query_map([&raw.id], |r| r.get::<_, String>(0))?
            .filter_map(|r| r.ok().and_then(|t| Target::parse(&t).ok()))
            .collect();
        let relations = conn
            .prepare_cached(
                "SELECT kind, other FROM a.note_relation
                 WHERE note_id = ?1 AND kind NOT IN ('revises', 'withdraws') ORDER BY kind, other",
            )?
            .query_map([&raw.id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })?
            .filter_map(|r| {
                r.ok()
                    .and_then(|(k, o)| Relation::parse(&k).map(|k| (k, o)))
            })
            .collect();
        Ok(Some(Note {
            id: raw.id,
            root: raw.root,
            kind,
            key: raw.key,
            value: raw.value,
            author,
            created_at: Timestamp::from_unix_ms(raw.created_at)
                .unwrap_or_else(|| Timestamp::from_unix_ms(0).expect("epoch")),
            valid_until: raw.valid_until.and_then(Timestamp::from_unix_ms),
            targets,
            relations,
            source: raw
                .source
                .and_then(|s| polylogue_model::MessageId::parse(&s).ok()),
        }))
    }
}

struct RawNote {
    id: String,
    root: String,
    kind: String,
    key: Option<String>,
    value: Option<String>,
    author: String,
    created_at: i64,
    valid_until: Option<i64>,
    source: Option<String>,
}

fn push_list(sql: &mut String, params: &mut Vec<Value>, items: impl Iterator<Item = String>) {
    for (i, item) in items.enumerate() {
        if i > 0 {
            sql.push_str(", ");
        }
        params.push(Value::Text(item));
        sql.push_str(&format!("?{}", params.len()));
    }
}

#[cfg(test)]
mod tests;
