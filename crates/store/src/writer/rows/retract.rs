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
            "CREATE TEMP TABLE IF NOT EXISTS reparse_session(input_pk INTEGER,chain INTEGER,pk INTEGER,PRIMARY KEY(input_pk,chain,pk)) WITHOUT ROWID",
        )?;
        self.tx.execute("INSERT OR IGNORE INTO temp.reparse_session SELECT input_pk,chain,session_pk FROM parsed WHERE input_pk=?1 AND chain=?2",params![self.input_pk,self.chain])?;
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
pub(crate) fn prune_sessions(tx: &Transaction<'_>, input: &str, chain: i64) -> Result<()> {
    tx.execute_batch("CREATE TEMP TABLE IF NOT EXISTS reparse_session(input_pk INTEGER,chain INTEGER,pk INTEGER,PRIMARY KEY(input_pk,chain,pk)) WITHOUT ROWID")?;
    let sessions = tx
        .prepare(
            "SELECT s.pk,s.id FROM session s JOIN temp.reparse_session r ON r.pk=s.pk
        WHERE r.input_pk=(SELECT pk FROM input_ref WHERE id=?1) AND r.chain=?2
        AND NOT EXISTS(SELECT 1 FROM session_record f WHERE f.session_pk=s.pk)
        AND NOT EXISTS(SELECT 1 FROM parsed p WHERE p.session_pk=s.pk)
        AND NOT EXISTS(SELECT 1 FROM message m WHERE m.session_pk=s.pk)
        AND NOT EXISTS(SELECT 1 FROM message_source v WHERE v.session_pk=s.pk)
        AND NOT EXISTS(SELECT 1 FROM event e WHERE e.session_pk=s.pk)
        AND NOT EXISTS(SELECT 1 FROM event_source e WHERE e.session_pk=s.pk)
        AND NOT EXISTS(SELECT 1 FROM link l WHERE l.child_id=s.id OR l.parent_id=s.id)",
        )?
        .query_map(params![input, chain], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for (pk, id) in sessions {
        crate::session_row::remove_with_references(tx, pk, &id, |_| Ok(()))?;
    }
    tx.execute("DELETE FROM temp.reparse_session WHERE input_pk=(SELECT pk FROM input_ref WHERE id=?1) AND chain=?2", params![input,chain])?;
    Ok(())
}

/// Remove an input's statements before excision removes its remaining index rows.
pub(crate) fn retract_input(
    tx: &Transaction<'_>,
    input_pk: i64,
    input_id: &str,
    counter: i64,
) -> Result<()> {
    let chains = tx
        .prepare("SELECT chain FROM parse_state WHERE input_pk=?1 ORDER BY chain")?
        .query_map([input_pk], |r| r.get::<_, i64>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let packed = HashMap::new();
    for chain in chains {
        RowWriter {
            tx,
            input_pk,
            input_id,
            chain,
            repeats_whole_conversations: false,
            rank: 0,
            counter,
            packed: &packed,
            first_build: None,
        }
        .retract()?;
        prune_sessions(tx, input_id, chain)?;
    }
    Ok(())
}
