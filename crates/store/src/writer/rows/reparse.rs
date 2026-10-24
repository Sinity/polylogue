//! Retire old parser contributions once a session's selected inputs finish.
use super::*;

impl Ctx<'_, '_> {
    pub(super) fn finish_reparse(&self, p: &ParseStep, length: u64) -> Result<()> {
        if !p.at_end || p.parsed_upto < length {
            return Ok(());
        }
        let pending: bool = self.tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM parse_state WHERE parser_version>retired_version)",
            [],
            |r| r.get(0),
        )?;
        if !pending {
            return Ok(());
        }
        self.tx.execute(
            "DELETE FROM event_source WHERE input_pk=?1 AND chain=?2 AND parser_version<?3
            AND event_id IN (SELECT id FROM event WHERE session_pk IS NULL)",
            params![self.input_pk, self.chain, self.parser_version],
        )?;
        self.refresh_input_events()?;
        self.tx.execute("DELETE FROM event WHERE session_pk IS NULL AND NOT EXISTS(SELECT 1 FROM event_source WHERE event_id=event.id)",[])?;
        let affected = self.ready_sessions()?;
        let mut scopes=affected.db.prepare("SELECT session_pk,input_pk,chain,version,input_id FROM scope ORDER BY session_pk,input_pk,chain")?;
        let mut rows = scopes.query([])?;
        while let Some(row) = rows.next()? {
            let session_pk: i64 = row.get(0)?;
            let input_id: String = row.get(4)?;
            let scope = Ctx {
                input_pk: row.get(1)?,
                chain: row.get(2)?,
                parser_version: row.get(3)?,
                input_id: &input_id,
                ..*self
            };
            scope.retire_session(session_pk, &affected)?;
        }
        drop(rows);
        drop(scopes);
        self.refresh_retired(&affected)?;
        // Input-wide lookup facts may be replaced once this input's complete
        // interpretation is known. Session-owned records retire independently.
        self.tx.execute(
            "DELETE FROM repo_lookup WHERE input_pk=?1 AND chain=?2 AND parser_version<?3",
            params![self.input_pk, self.chain, self.parser_version],
        )?;
        self.tx.execute("DELETE FROM snapshot WHERE input_pk=?1 AND chain=?2 AND parser_version<?3 AND NOT EXISTS(SELECT 1 FROM snapshot_session c WHERE c.snapshot_pk=snapshot.pk)",params![self.input_pk,self.chain,self.parser_version])?;
        self.tx.execute(&format!("UPDATE parse_state AS p SET retired_version=parser_version
            WHERE parser_version>retired_version AND NOT EXISTS(SELECT 1 FROM ({CONTRIBUTIONS}) c
                WHERE c.input_pk=p.input_pk AND c.chain=p.chain AND c.parser_version<p.parser_version)"),[])?;
        Ok(())
    }

    fn ready_sessions(&self) -> Result<Affected> {
        // Readiness and the affected identities stay in operation-owned SQLite;
        // bodies never accumulate while several inputs finish one session.
        let affected = Affected::new()?;
        let mut sessions = self.tx.prepare(&format!(
            "SELECT session_pk FROM ({CONTRIBUTIONS}) WHERE input_pk=?1 AND chain=?2"
        ))?;
        let mut candidates = sessions.query(params![self.input_pk, self.chain])?;
        while let Some(row) = candidates.next()? {
            let session_pk: i64 = row.get(0)?;
            if !affected.add("candidate", session_pk)? {
                continue;
            }
            if self.session_ready(session_pk)? {
                affected.add("session", session_pk)?;
                let mut scopes = self.tx.prepare(&format!(
                    "SELECT c.input_pk,c.chain,p.parser_version,r.id FROM ({CONTRIBUTIONS}) c
                     JOIN parse_state p ON p.input_pk=c.input_pk AND p.chain=c.chain
                     JOIN input_ref r ON r.pk=c.input_pk
                     WHERE c.session_pk=?1 AND p.parser_version>p.retired_version"
                ))?;
                let mut rows = scopes.query([session_pk])?;
                while let Some(row) = rows.next()? {
                    affected.db.execute(
                        "INSERT OR IGNORE INTO scope VALUES (?1,?2,?3,?4,?5)",
                        params![
                            session_pk,
                            row.get::<_, i64>(0)?,
                            row.get::<_, i64>(1)?,
                            row.get::<_, i64>(2)?,
                            row.get::<_, String>(3)?
                        ],
                    )?;
                }
            }
        }
        drop(candidates);
        drop(sessions);
        Ok(affected)
    }

    fn refresh_retired(&self, affected: &Affected) -> Result<()> {
        affected.each("message", |pk| self.promote_current(pk))?;
        affected.each("link", |pk| {
            let (parent, child): (Option<i64>, Option<i64>) = self.tx.query_row(
                "SELECT p.pk,c.pk FROM link l LEFT JOIN session p ON p.id=l.parent_id LEFT JOIN session c ON c.id=l.child_id WHERE l.pk=?1",
                [pk], |r| Ok((r.get(0)?, r.get(1)?)),
            )?;
            for session in [parent, child].into_iter().flatten() {
                if !self.session_ready(session)? { return Ok(()); }
                affected.add("session", session)?;
            }
            self.tx.execute("DELETE FROM link_source AS s WHERE link_pk=?1 AND EXISTS(SELECT 1 FROM parse_state p WHERE p.input_pk=s.input_pk AND p.chain=s.chain AND s.parser_version<p.parser_version)",[pk])?;
            self.tx.execute("DELETE FROM link WHERE pk=?1 AND NOT EXISTS(SELECT 1 FROM link_source WHERE link_pk=?1)",[pk])?;
            self.project_link(pk)
        })?;
        affected.each("session", |pk| {
            self.tx.execute(
                "UPDATE session SET changed=?2 WHERE pk=?1",
                params![pk, self.counter],
            )?;
            crate::live::rebuild(self.tx, pk, self.counter)?;
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

    fn session_ready(&self, session_pk: i64) -> Result<bool> {
        Ok(self.tx.query_row(&format!("SELECT NOT EXISTS(SELECT 1 FROM ({CONTRIBUTIONS}) c
            JOIN input_ref i ON i.pk=c.input_pk
            LEFT JOIN required_parser r ON r.format=i.format
            LEFT JOIN parse_state p ON p.input_pk=c.input_pk AND p.chain=c.chain
            WHERE c.session_pk=?1 AND i.excised=0 AND (p.input_pk IS NULL OR p.source_eof=0
                OR p.parsed_upto<p.source_length OR p.parser_version<COALESCE(r.version,p.parser_version)))"),[session_pk],|r|r.get(0))?)
    }

    fn retire_session(&self, session_pk: i64, affected: &Affected) -> Result<()> {
        for (kind, query) in [
            (
                "message",
                "SELECT message_pk FROM message_source WHERE session_pk=?4 AND input_pk=?1 AND chain=?2 AND parser_version<?3",
            ),
            (
                "link",
                "SELECT link_pk FROM link_source WHERE input_pk=?1 AND chain=?2 AND parser_version<?3 AND link_pk IN (SELECT pk FROM link WHERE parent_id=(SELECT id FROM session WHERE pk=?4) OR child_id=(SELECT id FROM session WHERE pk=?4))",
            ),
        ] {
            let mut statement = self.tx.prepare(query)?;
            let mut rows = statement.query(params![
                self.input_pk,
                self.chain,
                self.parser_version,
                session_pk
            ])?;
            while let Some(row) = rows.next()? {
                affected.add(kind, row.get(0)?)?;
            }
        }
        for (table, scope) in [
            ("message_source", "session_pk=?4"),
            (
                "event_source",
                "event_id IN (SELECT id FROM event WHERE session_pk=?4)",
            ),
            ("session_record", "session_pk=?4"),
            ("parsed", "session_pk=?4"),
        ] {
            self.tx.execute(&format!("DELETE FROM {table} WHERE input_pk=?1 AND chain=?2 AND parser_version<?3 AND ({scope})"),params![self.input_pk,self.chain,self.parser_version,session_pk])?;
        }
        self.tx.execute("DELETE FROM snapshot_session WHERE session_pk=?4 AND snapshot_pk IN (SELECT pk FROM snapshot WHERE input_pk=?1 AND chain=?2 AND parser_version<?3)",params![self.input_pk,self.chain,self.parser_version,session_pk])?;
        self.refresh_sources(session_pk)?;
        self.rebuild_snapshot_facts(session_pk)
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
}

impl Affected {
    fn new() -> Result<Self> {
        let db = Connection::open("")?;
        db.execute_batch("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-1024;
            PRAGMA mmap_size=0; PRAGMA temp_store=FILE;
            CREATE TABLE item(kind TEXT NOT NULL,id INTEGER NOT NULL,PRIMARY KEY(kind,id)) WITHOUT ROWID;
            CREATE TABLE scope(session_pk INTEGER,input_pk INTEGER,chain INTEGER,version INTEGER,input_id TEXT,PRIMARY KEY(session_pk,input_pk,chain)) WITHOUT ROWID; BEGIN;")?;
        Ok(Self { db })
    }

    fn add(&self, kind: &str, id: i64) -> Result<bool> {
        Ok(self
            .db
            .prepare_cached("INSERT OR IGNORE INTO item VALUES(?1,?2)")?
            .execute(params![kind, id])?
            != 0)
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

// Every relation that can retain an old interpretation of a session.
const CONTRIBUTIONS:&str="SELECT session_pk,input_pk,chain,parser_version FROM parsed
    UNION ALL SELECT session_pk,input_pk,chain,parser_version FROM session_record
    UNION ALL SELECT session_pk,input_pk,chain,parser_version FROM message_source
    UNION ALL SELECT e.session_pk,s.input_pk,s.chain,s.parser_version FROM event_source s JOIN event e ON e.id=s.event_id WHERE e.session_pk IS NOT NULL
    UNION ALL SELECT c.session_pk,s.input_pk,s.chain,s.parser_version FROM snapshot s JOIN snapshot_session c ON c.snapshot_pk=s.pk
    UNION ALL SELECT t.pk,s.input_pk,s.chain,s.parser_version FROM link_source s JOIN link l ON l.pk=s.link_pk JOIN session t ON t.id=l.parent_id OR t.id=l.child_id";

#[cfg(test)]
mod tests;
