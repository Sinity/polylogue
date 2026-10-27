//! Remove one chain's statements and show the readings that remain.
use super::*;

impl RowWriter<'_, '_> {
    fn touched(&self, table: &str, column: &str) -> Result<Vec<i64>> {
        Ok(self
            .tx
            .prepare_cached(&format!(
                "SELECT {column} FROM {table} WHERE input_pk=?1 AND chain=?2"
            ))?
            .query_map(params![self.input_pk, self.chain], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?)
    }

    pub(super) fn retract(&self) -> Result<()> {
        self.tx.execute_batch(
            "CREATE TEMP TABLE IF NOT EXISTS reparse_session(pk INTEGER PRIMARY KEY)",
        )?;
        self.tx.execute("INSERT OR IGNORE INTO temp.reparse_session SELECT session_pk FROM parsed WHERE input_pk=?1 AND chain=?2",params![self.input_pk,self.chain])?;
        let messages = self.touched("message_source", "message_pk")?;
        let documents = self.touched("document_source", "document_pk")?;
        let events = self.touched("event_source", "event_pk")?;
        let links = self.touched("link_source", "link_pk")?;
        if self.first_build.is_none() {
            let count = messages.len() + documents.len() + events.len() + links.len();
            self.tx.execute(
                "INSERT INTO meta(key,value) VALUES ('reparse_deletions',?1)
                ON CONFLICT(key) DO UPDATE SET value=CAST(CAST(value AS INTEGER)+?1 AS TEXT)",
                [(count as i64).to_string()],
            )?;
        }
        self.tx.execute("UPDATE session SET changed=?3,live_reparse=1 WHERE pk IN (SELECT session_pk FROM parsed WHERE input_pk=?1 AND chain=?2)", params![self.input_pk,self.chain,self.counter])?;
        for table in [
            "message_source",
            "document_source",
            "event_source",
            "link_source",
            "session_record",
            "parsed",
            "parse_state",
            "unused",
        ] {
            self.tx
                .prepare_cached(&format!(
                    "DELETE FROM {table} WHERE input_pk=?1 AND chain=?2"
                ))?
                .execute(params![self.input_pk, self.chain])?;
        }
        self.tx.execute(
            "DELETE FROM repo_lookup WHERE input_pk=?1 AND chain=?2",
            params![self.input_pk, self.chain],
        )?;
        for pk in messages {
            self.refresh_message(pk)?;
        }
        for pk in documents {
            crate::documents::refresh(self.tx, pk, self.counter, self.first_build)?;
        }
        for pk in events {
            self.refresh_event(pk)?;
        }
        for pk in links {
            self.refresh_link(pk)?;
        }
        Ok(())
    }
}

/// Drop sessions whose last statement disappeared, after the replacement rows exist.
pub(super) fn prune_sessions(tx: &Transaction<'_>) -> Result<()> {
    tx.execute_batch("CREATE TEMP TABLE IF NOT EXISTS reparse_session(pk INTEGER PRIMARY KEY)")?;
    let sessions = tx
        .prepare(
            "SELECT s.pk,s.id FROM session s JOIN temp.reparse_session r ON r.pk=s.pk
        WHERE NOT EXISTS(SELECT 1 FROM session_record f WHERE f.session_pk=s.pk)
        AND NOT EXISTS(SELECT 1 FROM parsed p WHERE p.session_pk=s.pk)
        AND NOT EXISTS(SELECT 1 FROM message m WHERE m.session_pk=s.pk)
        AND NOT EXISTS(SELECT 1 FROM message_source v WHERE v.session_pk=s.pk)
        AND NOT EXISTS(SELECT 1 FROM event e WHERE e.session_pk=s.pk)
        AND NOT EXISTS(SELECT 1 FROM link l WHERE l.child_id=s.id OR l.parent_id=s.id)",
        )?
        .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for (pk, id) in sessions {
        crate::session_row::remove_with_references(tx, pk, &id, |_| Ok(()))?;
    }
    tx.execute("DELETE FROM temp.reparse_session", [])?;
    Ok(())
}
