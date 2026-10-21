//! Completion of a newer interpretation of one retained revision chain.
use super::*;

impl Ctx<'_, '_> {
    pub(super) fn finish_reparse(&self, p: &ParseStep, length: u64) -> Result<()> {
        if !p.at_end || p.parsed_upto < length {
            return Ok(());
        }
        let mut obsolete = false;
        for table in [
            "message_source",
            "event_source",
            "link_source",
            "session_record",
            "snapshot",
            "repo_lookup",
            "parsed",
        ] {
            obsolete |= self.tx.prepare_cached(
                &format!("SELECT EXISTS(SELECT 1 FROM {table} WHERE input_pk=?1 AND chain=?2 AND parser_version<?3)"),
            )?.query_row(params![self.input_pk, self.chain, self.parser_version], |r| r.get::<_, bool>(0))?;
        }
        if !obsolete {
            return Ok(());
        }
        // IDs, not bodies, stay on disk while sources are removed and selected
        // projections are rebuilt. The transaction owns this temporary set.
        let affected = Affected::new()?;
        for (kind, query) in [
            ("message", "SELECT message_pk FROM message_source WHERE input_pk=?1 AND chain=?2"),
            ("link", "SELECT link_pk FROM link_source WHERE input_pk=?1 AND chain=?2"),
            ("session", "SELECT session_pk FROM parsed WHERE input_pk=?1 AND chain=?2
                UNION ALL SELECT session_pk FROM session_record WHERE input_pk=?1 AND chain=?2
                UNION ALL SELECT session_pk FROM message_source WHERE input_pk=?1 AND chain=?2
                UNION ALL SELECT e.session_pk FROM event_source s JOIN event e ON e.id=s.event_id WHERE s.input_pk=?1 AND s.chain=?2
                UNION ALL SELECT c.session_pk FROM snapshot s JOIN snapshot_session c ON c.snapshot_pk=s.pk WHERE s.input_pk=?1 AND s.chain=?2
                UNION ALL SELECT session.pk FROM link_source s JOIN link l ON l.pk=s.link_pk JOIN session ON session.id=l.parent_id OR session.id=l.child_id WHERE s.input_pk=?1 AND s.chain=?2
                UNION ALL SELECT session.pk FROM repo_lookup r JOIN session ON session.cwd=r.path OR substr(session.cwd,1,length(r.path)+1)=r.path||'/' WHERE r.input_pk=?1 AND r.chain=?2"),
        ] {
            let mut statement = self.tx.prepare(query)?;
            let mut rows = statement.query(params![self.input_pk,self.chain])?;
            while let Some(row) = rows.next()? {
                affected.db.prepare_cached("INSERT OR IGNORE INTO item VALUES(?1,?2)")?.execute(params![kind,row.get::<_,i64>(0)?])?;
            }
        }
        for table in [
            "message_source",
            "event_source",
            "link_source",
            "session_record",
            "snapshot",
            "repo_lookup",
            "parsed",
        ] {
            self.tx.execute(
                &format!(
                    "DELETE FROM {table} WHERE input_pk=?1 AND chain=?2 AND parser_version<?3"
                ),
                params![self.input_pk, self.chain, self.parser_version],
            )?;
        }
        affected.db.execute_batch("COMMIT")?;
        affected.each("session", |pk| {
            self.tx.execute(
                "UPDATE session SET changed=?2 WHERE pk=?1",
                params![pk, self.counter],
            )?;
            Ok(())
        })?;
        affected.each("session", |pk| {
            self.refresh_sources(pk)?;
            self.rebuild_snapshot_facts(pk)?;
            Ok(())
        })?;
        affected.each("message", |pk| self.promote_current(pk))?;
        affected.each("link", |pk| {
            // A row with no sources has legacy meaning in current_link, so an
            // expired last source must remove that row as well.
            self.tx.execute("DELETE FROM link WHERE pk=?1 AND NOT EXISTS(SELECT 1 FROM link_source WHERE link_pk=?1)",[pk])?;
            self.project_link(pk)
        })?;
        affected.each("session", |pk| {
            crate::live::rebuild(self.tx, pk, self.counter)
        })?;
        affected.each("session", |pk| {
            let no_facts: bool = self.tx.query_row(
                "SELECT NOT EXISTS(SELECT 1 FROM session_record WHERE session_pk=?1)",
                [pk],
                |r| r.get(0),
            )?;
            if no_facts {
                let id: String =
                    self.tx
                        .query_row("SELECT id FROM session WHERE pk=?1", [pk], |r| r.get(0))?;
                let session =
                    SessionId::parse(&id).map_err(|e| StoreError::Invalid(e.to_string()))?;
                crate::session_row::write(
                    self.tx,
                    pk,
                    &Session::new(session, polylogue_model::SessionKind::Main),
                )?;
                self.tx
                    .execute("UPDATE session SET stub=1 WHERE pk=?1", [pk])?;
            }
            let orphan: bool = self.tx.query_row(
                "SELECT NOT EXISTS(SELECT 1 FROM parsed WHERE session_pk=?1)
                AND NOT EXISTS(SELECT 1 FROM session_record WHERE session_pk=?1)
                AND NOT EXISTS(SELECT 1 FROM message_source WHERE session_pk=?1)
                AND NOT EXISTS(SELECT 1 FROM current_event WHERE session_pk=?1)
                AND NOT EXISTS(SELECT 1 FROM link WHERE child_id=session.id OR parent_id=session.id)
                FROM session WHERE pk=?1",
                [pk],
                |r| r.get(0),
            )?;
            if orphan {
                let id: String =
                    self.tx
                        .query_row("SELECT id FROM session WHERE pk=?1", [pk], |r| r.get(0))?;
                crate::session_row::remove(self.tx, pk, &id, self.counter)?;
            }
            Ok(())
        })?;
        Ok(())
    }

    fn rebuild_snapshot_facts(&self, session_pk: i64) -> Result<()> {
        let mut statement = self.tx.prepare("SELECT c.facts FROM snapshot_session c JOIN snapshot s ON s.pk=c.snapshot_pk
            WHERE c.session_pk=?1 AND s.input_pk=?2 AND s.chain=?3 AND s.parser_version=?4 ORDER BY s.source_start")?;
        let mut rows = statement.query(params![
            session_pk,
            self.input_pk,
            self.chain,
            self.parser_version
        ])?;
        let mut facts: Option<Session> = None;
        while let Some(row) = rows.next()? {
            let next: Session = crate::encoding::unpack_json(&row.get::<_, Vec<u8>>(0)?)?;
            if let Some(current) = &mut facts {
                current.merge(&next);
            } else {
                facts = Some(next);
            }
        }
        if let Some(facts) = facts {
            self.tx.execute("INSERT INTO session_record(session_pk,input_pk,chain,parser_version,facts) VALUES(?1,?2,?3,?4,?5)
                ON CONFLICT(session_pk,input_pk,chain,parser_version) DO UPDATE SET facts=excluded.facts",
                params![session_pk,self.input_pk,self.chain,self.parser_version,serde_json::to_string(&facts)?])?;
        }
        Ok(())
    }
}

struct Affected {
    db: Connection,
    _file: tempfile::NamedTempFile,
}

impl Affected {
    fn new() -> Result<Self> {
        let file = tempfile::NamedTempFile::new()?;
        let db = Connection::open(file.path())?;
        db.execute_batch("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-1024;
            PRAGMA mmap_size=0; PRAGMA temp_store=FILE;
            CREATE TABLE item(kind TEXT NOT NULL,id INTEGER NOT NULL,PRIMARY KEY(kind,id)) WITHOUT ROWID; BEGIN;")?;
        Ok(Self { db, _file: file })
    }

    fn each(&self, kind: &str, mut visit: impl FnMut(i64) -> Result<()>) -> Result<()> {
        let mut statement = self
            .db
            .prepare("SELECT id FROM item WHERE kind=?1 ORDER BY id")?;
        let mut rows = statement.query([kind])?;
        while let Some(row) = rows.next()? {
            visit(row.get(0)?)?;
        }
        Ok(())
    }
}
