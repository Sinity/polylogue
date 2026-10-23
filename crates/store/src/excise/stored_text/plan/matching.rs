use super::{Plan, position};
use crate::excise::text_matches::{HitSink, Scope};
use crate::{Result, StoreError};
use rusqlite::params;

pub(in super::super) struct Hits<'a> {
    plan: &'a Plan,
    input: &'a str,
    chain: i64,
}

impl HitSink for Hits<'_> {
    fn scope(&mut self) -> Result<Scope> {
        let next = self
            .plan
            .next_scope
            .get()
            .checked_add(1)
            .ok_or_else(|| StoreError::Invalid("match scope overflow".into()))?;
        position(next)?;
        self.plan.next_scope.set(next);
        Ok(next)
    }

    fn hit(&mut self, scope: Scope, start: u64, end: u64) -> Result<()> {
        self.plan
            .conn
            .prepare_cached("INSERT OR IGNORE INTO candidate_matches VALUES (?1,?2,?3)")?
            .execute(params![position(scope)?, position(start)?, position(end)?])?;
        Ok(())
    }

    fn accept(&mut self, scope: Scope, parent: Scope) -> Result<()> {
        self.plan
            .conn
            .prepare_cached(
                "INSERT OR IGNORE INTO candidate_matches
            SELECT ?1,start,end FROM candidate_matches WHERE scope=?2",
            )?
            .execute(params![position(parent)?, position(scope)?])?;
        self.discard(scope)
    }

    fn discard(&mut self, scope: Scope) -> Result<()> {
        self.plan
            .conn
            .prepare_cached("DELETE FROM candidate_matches WHERE scope=?1")?
            .execute([position(scope)?])?;
        Ok(())
    }
}

impl Hits<'_> {
    pub(in super::super) fn finish(mut self, scope: Scope) -> Result<()> {
        self.plan
            .conn
            .prepare_cached(
                "INSERT OR IGNORE INTO matches
            SELECT ?1,?2,start,end FROM candidate_matches WHERE scope=?3",
            )?
            .execute(params![self.input, self.chain, position(scope)?])?;
        self.discard(scope)
    }
}

impl Plan {
    pub(in super::super) fn hits<'a>(&'a self, input: &'a str, chain: i64) -> Hits<'a> {
        Hits {
            plan: self,
            input,
            chain,
        }
    }

    /// Counts and observations keep exact hits. Writes use a separate ordered,
    /// nonoverlapping stream so chunk readers need retain at most one interval.
    pub(in super::super) fn normalize(&self, input: &str, chain: i64) -> Result<()> {
        // Header masking keeps scanner state and positions; structural-only or
        // crossing hits must never become excision counts or write ranges.
        self.conn.execute("DELETE FROM matches AS m WHERE input=?1 AND chain=?2
            AND EXISTS(SELECT 1 FROM chains WHERE input=?1 AND chain=?2 AND framed=1)
            AND (start<?3 OR EXISTS(SELECT 1 FROM frames AS f INDEXED BY frames_start WHERE f.input=m.input AND f.chain=m.chain
                AND f.start=(SELECT max(start) FROM frames INDEXED BY frames_start WHERE input=m.input AND chain=m.chain AND start<m.end)
                AND f.start+?4>m.start))",
            params![input,chain,polylogue_model::snapshot_log::MAGIC.len() as i64,polylogue_model::snapshot_log::HEADER_LEN as i64])?;
        self.conn.execute(
            "DELETE FROM write_ranges WHERE input=?1 AND chain=?2",
            params![input, chain],
        )?;
        let mut select = self.conn.prepare_cached(
            "SELECT start,end FROM matches WHERE input=?1 AND chain=?2 ORDER BY start,end",
        )?;
        let mut insert = self
            .conn
            .prepare_cached("INSERT INTO write_ranges VALUES (?1,?2,?3,?4)")?;
        let mut pending: Option<(i64, i64)> = None;
        for row in select.query_map(params![input, chain], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))
        })? {
            let (start, end) = row?;
            match &mut pending {
                Some((_, until)) if start <= *until => *until = (*until).max(end),
                _ => {
                    if let Some((start, end)) = pending.take() {
                        insert.execute(params![input, chain, start, end])?;
                    }
                    pending = Some((start, end));
                }
            }
        }
        if let Some((start, end)) = pending {
            insert.execute(params![input, chain, start, end])?;
        }
        Ok(())
    }
}
