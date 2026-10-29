//! Remove one chain's statements and show the readings that remain.
use super::*;

impl RowWriter<'_, '_> {
    pub(super) fn retract(&self) -> Result<()> {
        self.tx.execute_batch(
            "CREATE TEMP TABLE IF NOT EXISTS reparse_session(input_pk INTEGER,chain INTEGER,pk INTEGER,PRIMARY KEY(input_pk,chain,pk)) WITHOUT ROWID",
        )?;
        self.tx.execute("INSERT OR IGNORE INTO temp.reparse_session SELECT input_pk,chain,session_pk FROM parsed WHERE input_pk=?1 AND chain=?2",params![self.input_pk,self.chain])?;
        let mut touched = Vec::new();
        for kind in VersionKind::ALL {
            touched.push((kind, kind.retract(self.tx, self.input_pk, self.chain)?));
        }
        if self.first_build.is_none() {
            let count: usize = touched.iter().map(|(_, rows)| rows.len()).sum();
            self.tx.execute(
                "INSERT INTO meta(key,value) VALUES ('reparse_deletions',?1)
                ON CONFLICT(key) DO UPDATE SET value=CAST(CAST(value AS INTEGER)+?1 AS TEXT)",
                [(count as i64).to_string()],
            )?;
        }
        self.tx.execute("UPDATE session SET changed=?3,live_reparse=1 WHERE pk IN (SELECT session_pk FROM parsed WHERE input_pk=?1 AND chain=?2)", params![self.input_pk,self.chain,self.counter])?;
        for table in ["session_record", "parsed", "parse_state", "unused"] {
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
