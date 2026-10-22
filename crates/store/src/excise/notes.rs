//! Freeze matched targets before changing canonical notes or their relations.
use super::*;
use rusqlite::Connection;

/// `messages` is the plan's sorted list of exact IDs.
pub(super) fn remove(
    archive: &Transaction<'_>,
    sessions: &[String],
    messages: &[String],
    dry_run: bool,
) -> Result<u64> {
    let mut targets = Targets::default();
    {
        let mut stmt = archive.prepare("SELECT note_id,ref FROM note_target")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let reference: String = row.get(1)?;
            if about(&reference, sessions, messages) {
                targets.insert(&row.get::<_, String>(0)?, &reference)?;
            }
        }
    }
    let mut removed = 0;
    targets.notes(|note, hits| {
        let total: i64 = archive
            .prepare_cached("SELECT count(*) FROM note_target WHERE note_id=?1")?
            .query_row([note], |r| r.get(0))?;
        let whole = hits == total as u64;
        removed += u64::from(whole);
        if !dry_run {
            targets.refs(note, |reference| {
                archive
                    .prepare_cached("DELETE FROM note_target WHERE note_id=?1 AND ref=?2")?
                    .execute(params![note, reference])?;
                Ok(())
            })?;
            if whole {
                archive.execute("DELETE FROM note_relation WHERE note_id=?1", [note])?;
                archive.execute("DELETE FROM note WHERE id=?1", [note])?;
            }
        }
        Ok(())
    })?;
    Ok(removed)
}

fn about(reference: &str, sessions: &[String], messages: &[String]) -> bool {
    reference
        .strip_prefix("session:")
        .is_some_and(|id| sessions.iter().any(|s| s == id))
        || reference
            .strip_prefix("message:")
            .is_some_and(|id| messages.binary_search_by(|m| m.as_str().cmp(id)).is_ok())
        || reference.strip_prefix("tool:").is_some_and(|rest| {
            sessions.iter().any(|session| {
                rest.strip_prefix(session.as_str())
                    .is_some_and(|suffix| suffix.starts_with('#'))
            })
        })
}

struct State {
    conn: Connection,
    _file: tempfile::NamedTempFile,
}

#[derive(Default)]
struct Targets(Option<State>);
impl Targets {
    fn insert(&mut self, note: &str, reference: &str) -> Result<()> {
        if self.0.is_none() {
            let file = tempfile::NamedTempFile::new()?;
            let conn = Connection::open(file.path())?;
            conn.execute_batch("PRAGMA cache_size=-1024; PRAGMA mmap_size=0; PRAGMA temp_store=FILE;
                PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
                CREATE TABLE target(note TEXT NOT NULL,ref TEXT NOT NULL,PRIMARY KEY(note,ref)) WITHOUT ROWID;
                BEGIN;")?;
            self.0 = Some(State { conn, _file: file });
        }
        crate::fault::check("excision_targets.insert", note)?;
        self.0
            .as_ref()
            .unwrap()
            .conn
            .prepare_cached("INSERT INTO target VALUES (?1,?2)")?
            .execute(params![note, reference])?;
        Ok(())
    }

    fn notes(&self, mut f: impl FnMut(&str, u64) -> Result<()>) -> Result<()> {
        if let Some(state) = &self.0 {
            let mut stmt = state
                .conn
                .prepare_cached("SELECT note,count(*) FROM target GROUP BY note ORDER BY note")?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let note: String = row.get(0)?;
                crate::fault::check("excision_targets.read", &note)?;
                f(&note, row.get::<_, i64>(1)? as u64)?;
            }
        }
        Ok(())
    }

    fn refs(&self, note: &str, mut f: impl FnMut(&str) -> Result<()>) -> Result<()> {
        if let Some(state) = &self.0 {
            let mut stmt = state
                .conn
                .prepare_cached("SELECT ref FROM target WHERE note=?1 ORDER BY ref")?;
            let mut rows = stmt.query([note])?;
            while let Some(row) = rows.next()? {
                f(&row.get::<_, String>(0)?)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
