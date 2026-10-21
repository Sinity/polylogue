use super::*;
use crate::encoding::{pack_json, unpack_json};
use std::collections::BTreeSet;

impl Ctx<'_, '_> {
    pub(super) fn snapshot_current(&self, session_pk: i64) -> Result<bool> {
        let Some(snapshot) = self.snapshot else {
            return Ok(true);
        };
        let head: Option<(i64,i64)>=self.tx.query_row("SELECT s.chain,s.source_start FROM snapshot s JOIN snapshot_session c ON c.snapshot_pk=s.pk WHERE s.input_pk=?1 AND c.session_pk=?2 ORDER BY s.chain DESC,s.source_start DESC LIMIT 1", params![self.input_pk,session_pk],|row|Ok((row.get(0)?,row.get(1)?))).optional()?;
        Ok(head == Some((self.chain, snapshot.start as i64)))
    }

    pub(super) fn begin_snapshot(&self, p: &ParseStep) -> Result<Vec<i64>> {
        let Some(snapshot) = &p.snapshot else {
            return Ok(Vec::new());
        };
        self.tx.execute("INSERT INTO snapshot(input_pk,chain,source_start,source_end,accounting) VALUES (?1,?2,?3,?4,?5) ON CONFLICT(input_pk,chain,source_start) DO UPDATE SET source_end=excluded.source_end,accounting=excluded.accounting",
            params![self.input_pk,self.chain,snapshot.start as i64,snapshot.end as i64,pack_json(&p.accounting)?])?;
        let snapshot_pk: i64 = self.tx.query_row(
            "SELECT pk FROM snapshot WHERE input_pk=?1 AND chain=?2 AND source_start=?3",
            params![self.input_pk, self.chain, snapshot.start as i64],
            |row| row.get(0),
        )?;
        let mut affected = BTreeSet::new();
        for id in &snapshot.covered {
            let pk = self.session_row(id, false)?;
            let facts = p
                .output
                .sessions
                .iter()
                .find(|session| &session.id == id)
                .cloned()
                .unwrap_or_else(|| Session::new(id.clone(), polylogue_model::SessionKind::Main));
            self.tx.execute("INSERT INTO snapshot_session(snapshot_pk,session_pk,facts,complete) VALUES (?1,?2,?3,?4) ON CONFLICT(snapshot_pk,session_pk) DO UPDATE SET facts=excluded.facts,complete=excluded.complete", params![snapshot_pk,pk,pack_json(&facts)?,snapshot.complete as i64])?;
            let mut stmt = self.tx.prepare("SELECT DISTINCT s.message_pk FROM message_source s WHERE s.session_pk=?1 AND s.input_pk=?2 AND s.current=1")?;
            for row in stmt.query_map(params![pk, self.input_pk], |row| row.get::<_, i64>(0))? {
                affected.insert(row?);
            }
            self.tx.execute(
                "UPDATE session SET changed=?2 WHERE pk=?1",
                params![pk, self.counter],
            )?;
        }
        Ok(affected.into_iter().collect())
    }

    pub(super) fn finish_snapshot(&self, p: &ParseStep, affected: &[i64]) -> Result<()> {
        let Some(snapshot) = &p.snapshot else {
            return Ok(());
        };
        let mut affected: BTreeSet<i64> = affected.iter().copied().collect();
        for id in &snapshot.covered {
            let session_pk = self.session_row(id, false)?;
            self.project_session(id, session_pk, p)?;
            // A complete scope is a deletion barrier. Partial scopes replace
            // only members they observed, independently of replay order.
            for (table, key, scope) in [
                ("message_source", "message_pk", "s.session_pk=?2"),
                (
                    "event_source",
                    "event_id",
                    "s.event_id IN (SELECT id FROM event WHERE session_pk=?2)",
                ),
                (
                    "link_source",
                    "link_pk",
                    "s.link_pk IN (SELECT pk FROM link WHERE parent_id=(SELECT id FROM session WHERE pk=?2) OR child_id=(SELECT id FROM session WHERE pk=?2))",
                ),
            ] {
                let placement = if table == "message_source" {
                    "AND n.session_pk=s.session_pk"
                } else {
                    ""
                };
                let sql = format!(
                    "UPDATE {table} AS s SET current=NOT EXISTS (SELECT 1 FROM snapshot h JOIN snapshot_session c ON c.snapshot_pk=h.pk WHERE h.input_pk=s.input_pk AND c.session_pk=?2 AND c.complete=1 AND (h.chain>s.chain OR (h.chain=s.chain AND h.source_start>s.source_snapshot))) AND NOT EXISTS (SELECT 1 FROM {table} n WHERE n.input_pk=s.input_pk AND n.{key}=s.{key} {placement} AND (n.chain>s.chain OR (n.chain=s.chain AND n.source_snapshot>s.source_snapshot))) WHERE s.input_pk=?1 AND (s.current=1 OR (s.chain=?3 AND s.source_snapshot=?4)) AND ({scope})"
                );
                self.tx.execute(
                    &sql,
                    params![self.input_pk, session_pk, self.chain, snapshot.start as i64],
                )?;
            }
            for row in &p.output.messages {
                if &row.message.session == id {
                    let pk = self
                        .tx
                        .query_row(
                            "SELECT pk FROM message WHERE id=?1",
                            [row.message.id.to_string()],
                            |r| r.get::<_, i64>(0),
                        )
                        .optional()?;
                    affected.extend(pk);
                }
            }
        }
        for pk in affected {
            self.promote_current(pk)?;
        }
        for id in &snapshot.covered {
            let links: Vec<i64> = self
                .tx
                .prepare("SELECT pk FROM link WHERE parent_id=?1 OR child_id=?1")?
                .query_map([id.to_string()], |row| row.get(0))?
                .collect::<rusqlite::Result<_>>()?;
            for pk in links {
                self.project_link(pk)?;
            }
        }
        Ok(())
    }

    fn project_session(
        &self,
        id: &polylogue_model::SessionId,
        session_pk: i64,
        p: &ParseStep,
    ) -> Result<()> {
        if self.snapshot_current(session_pk)? {
            let mut next = p.output.sessions.iter().find(|s| &s.id == id).cloned();
            // Population coverage does not make omitted metadata a clear.
            {
                let old: Option<String> = self
                    .tx
                    .query_row(
                        "SELECT facts FROM session_record WHERE input_pk=?1 AND session_pk=?2",
                        params![self.input_pk, session_pk],
                        |r| r.get(0),
                    )
                    .optional()?;
                if let Some(old) = old {
                    let mut old: Session = serde_json::from_str(&old)?;
                    if let Some(next) = next {
                        old.merge(&next);
                    }
                    next = Some(old);
                }
            }
            if let Some(next) = next {
                self.tx.execute("INSERT INTO session_record(session_pk,input_pk,facts) VALUES (?1,?2,?3) ON CONFLICT(session_pk,input_pk) DO UPDATE SET facts=excluded.facts",params![session_pk,self.input_pk,serde_json::to_string(&next)?])?;
                self.tx.execute(
                    "UPDATE session SET stub=0,changed=?2 WHERE pk=?1",
                    params![session_pk, self.counter],
                )?;
            }
            return Ok(());
        }
        let mut stmt=self.tx.prepare("SELECT c.facts FROM snapshot_session c JOIN snapshot s ON s.pk=c.snapshot_pk WHERE s.input_pk=?1 AND c.session_pk=?2 ORDER BY s.chain,s.source_start")?;
        let mut facts: Option<Session> = None;
        for row in stmt.query_map(params![self.input_pk, session_pk], |r| {
            r.get::<_, Vec<u8>>(0)
        })? {
            let bytes = row?;
            let next: Session = unpack_json(&bytes)?;
            if facts.is_none() {
                facts = Some(next);
            } else if let Some(current) = &mut facts {
                current.merge(&next);
            }
        }
        if let Some(facts) = facts {
            self.tx.execute("INSERT INTO session_record(session_pk,input_pk,facts) VALUES (?1,?2,?3) ON CONFLICT(session_pk,input_pk) DO UPDATE SET facts=excluded.facts", params![session_pk,self.input_pk,serde_json::to_string(&facts)?])?;
            self.tx.execute(
                "UPDATE session SET stub=0,changed=?2 WHERE pk=?1",
                params![session_pk, self.counter],
            )?;
        }
        Ok(())
    }

    pub(super) fn promote_current(&self, pk: i64) -> Result<()> {
        let mut stmt = self.tx.prepare("SELECT s.input_pk,s.chain,s.source_snapshot,s.source_start,s.source_end,s.complete,s.route_rank,s.main,s.version_hash,s.source_order,r.id,s.preferred,m.present,s.session_pk,s.parser_version FROM message_source s JOIN input_ref r ON r.pk=s.input_pk JOIN message m ON m.pk=s.message_pk WHERE s.message_pk=?1 AND s.current=1 ORDER BY r.id,s.chain,s.source_snapshot")?;
        let mut best: Option<(Version, String, bool)> = None;
        let rows = stmt.query_map([pk], |row| {
            Ok((
                Version {
                    input_pk: row.get(0)?,
                    session_pk: row.get(13)?,
                    chain: row.get(1)?,
                    snapshot: row.get(2)?,
                    start: row.get(3)?,
                    end: row.get(4)?,
                    complete: row.get::<_, i64>(5)? != 0,
                    rank: row.get(6)?,
                    main: row.get::<_, i64>(7)? != 0,
                    hash: row.get(8)?,
                    order: row.get(9)?,
                    parser_version: row.get(14)?,
                    current: true,
                },
                row.get::<_, String>(10)?,
                row.get::<_, i64>(11)? != 0 && row.get::<_, i64>(12)? != 0,
            ))
        })?;
        for row in rows {
            let candidate = row?;
            if best
                .as_ref()
                .is_none_or(|(version, _, _)| candidate.0.beats(version))
            {
                best = Some(candidate);
            }
        }
        drop(stmt);
        let Some((version, input_id, already_shown)) = best else {
            self.clear_blocks(pk)?;
            self.tx
                .execute("UPDATE message SET present=0 WHERE pk=?1", [pk])?;
            return Ok(());
        };
        if already_shown {
            return Ok(());
        }
        let message = crate::versions::load(self.tx, &version.hash)?;
        let session_pk: i64 = self.tx.query_row(
            "SELECT pk FROM session WHERE id=?1",
            [message.session.to_string()],
            |row| row.get(0),
        )?;
        let (old_session,old_kind):(i64,String)=self.tx.query_row("SELECT m.session_pk,s.kind FROM message m JOIN session s ON s.pk=m.session_pk WHERE m.pk=?1", [pk],|row|Ok((row.get(0)?,row.get(1)?)))?;
        let session_pk = if old_kind == "main" && !version.main {
            old_session
        } else {
            session_pk
        };
        let scope = (version.snapshot >= 0).then(|| polylogue_model::Snapshot {
            start: version.snapshot as u64,
            end: version.end as u64,
            covered: Vec::new(),
            complete: false,
        });
        let ctx = Ctx {
            tx: self.tx,
            input_pk: version.input_pk,
            input_id: &input_id,
            chain: version.chain,
            snapshot: scope.as_ref(),
            rank: version.rank,
            parser_version: self.parser_version,
            counter: self.counter,
            packed: self.packed,
            bulk: self.bulk,
        };
        let bodies = message
            .blocks
            .iter()
            .map(pack_block)
            .collect::<Result<Vec<_>>>()?;
        self.tx.execute(
            "UPDATE message_source SET preferred=0 WHERE message_pk=?1",
            [pk],
        )?;
        ctx.upsert_source(pk, &version, true)?;
        ctx.clear_blocks(pk)?;
        ctx.update_message(
            pk,
            &message,
            session_pk,
            &version,
            (version.input_pk, version.order),
        )?;
        ctx.insert_blocks(pk, session_pk, &message, &bodies)?;
        self.tx
            .execute("UPDATE message SET present=1 WHERE pk=?1", [pk])?;
        self.tx.execute(
            "UPDATE session SET changed=?2 WHERE pk=?1",
            params![session_pk, self.counter],
        )?;
        Ok(())
    }

    pub(super) fn project_link(&self, pk: i64) -> Result<()> {
        let mut stmt=self.tx.prepare("SELECT s.input_pk,s.facts FROM link_source s JOIN input_ref r ON r.pk=s.input_pk WHERE s.link_pk=?1 AND s.current=1 ORDER BY r.id,s.chain,s.source_snapshot")?;
        let rows = stmt.query_map([pk], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?;
        let mut merged: Option<Link> = None;
        let mut source = None;
        for row in rows {
            let (input, bytes) = row?;
            let facts: Link = unpack_json(&bytes)?;
            if let Some(current) = &mut merged {
                if facts.at.is_some() && (current.at.is_none() || facts.at < current.at) {
                    current.at = facts.at.clone();
                }
                if facts.call_id.is_some()
                    && (current.call_id.is_none() || facts.call_id < current.call_id)
                {
                    current.call_id = facts.call_id.clone();
                }
                if facts.evidence < current.evidence {
                    current.evidence = facts.evidence.clone();
                }
            } else {
                merged = Some(facts.clone());
            }
            if source
                .as_ref()
                .is_none_or(|(_, evidence): &(i64, String)| facts.evidence < *evidence)
            {
                source = Some((input, facts.evidence));
            }
        }
        if let (Some(link), Some((input, _))) = (merged, source) {
            self.tx.execute("UPDATE link SET branch_point=?2,spawning_call=?3,evidence=?4,input_pk=?5 WHERE pk=?1",params![pk,link.at.as_ref().map(ToString::to_string),link.call_id,link.evidence,input])?;
            self.tx.execute(
                "UPDATE session SET changed=?2 WHERE id=?1 OR id=?3",
                params![
                    link.parent.to_string(),
                    self.counter,
                    link.child.to_string()
                ],
            )?;
        } else {
            self.tx.execute("UPDATE session SET changed=?2 WHERE id IN (SELECT parent_id FROM link WHERE pk=?1 UNION SELECT child_id FROM link WHERE pk=?1)",params![pk,self.counter])?;
        }
        Ok(())
    }
}
