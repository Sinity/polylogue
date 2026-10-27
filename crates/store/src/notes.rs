//! Reading one note from `archive.db` (attached to every read as `a`).
//! Lists of notes are `notes` queries; writing goes through the writer
//! ([`crate::user_writes`]).

use crate::{Reader, Result, StoreError};
use polylogue_model::{Author, Note, NoteKind, Relation, Target, Timestamp};

impl Reader<'_> {
    /// One note by ID, whatever its revision state.
    pub fn note(&self, id: &str) -> Result<Option<Note>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT id, first_note, kind, key, value, author, created_at, valid_until
             FROM a.note WHERE id = ?1 AND kind != 'withdrawal'",
        )?;
        let mut rows = stmt.query([id])?;
        let Some(r) = rows.next()? else {
            return Ok(None);
        };
        let raw = RawNote {
            id: r.get(0)?,
            first_note: r.get(1)?,
            kind: r.get(2)?,
            key: r.get(3)?,
            value: r.get(4)?,
            author: r.get(5)?,
            created_at: r.get(6)?,
            valid_until: r.get(7)?,
        };
        drop(rows);
        self.finish(raw)
    }

    /// Refuse unsupported recorded values instead of inventing note metadata.
    fn finish(&self, raw: RawNote) -> Result<Option<Note>> {
        let kind = NoteKind::parse(&raw.kind)
            .ok_or_else(|| StoreError::Invalid("the note has an unsupported kind".into()))?;
        let author = Author::parse(&raw.author)
            .map_err(|_| StoreError::Invalid("the note has an unsupported author".into()))?;
        let created_at = Timestamp::from_unix_ms(raw.created_at).ok_or_else(|| {
            StoreError::Invalid("the note creation time is outside the supported range".into())
        })?;
        let conn = self.conn()?;
        let targets = conn
            .prepare_cached("SELECT ref FROM a.note_target WHERE note_id = ?1 ORDER BY ref")?
            .query_map([&raw.id], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?
            .into_iter()
            .map(|target| {
                Target::parse(&target)
                    .map_err(|_| StoreError::Invalid("the note has an unsupported target".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let relations = conn
            .prepare_cached(
                "SELECT kind, other FROM a.note_relation
                 WHERE note_id = ?1 AND kind NOT IN ('revises', 'withdraws') ORDER BY kind, other",
            )?
            .query_map([&raw.id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?
            .into_iter()
            .map(|(kind, other)| {
                Relation::parse(&kind)
                    .map(|kind| (kind, other))
                    .ok_or_else(|| {
                        StoreError::Invalid("the note has an unsupported relation".into())
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(Note {
            id: raw.id,
            first_note: raw.first_note,
            kind,
            key: raw.key,
            value: raw.value,
            author,
            created_at,
            valid_until: raw.valid_until.and_then(Timestamp::from_unix_ms),
            targets,
            relations,
        }))
    }
}

struct RawNote {
    id: String,
    first_note: String,
    kind: String,
    key: Option<String>,
    value: Option<String>,
    author: String,
    created_at: i64,
    valid_until: Option<i64>,
}

#[cfg(test)]
mod tests;
