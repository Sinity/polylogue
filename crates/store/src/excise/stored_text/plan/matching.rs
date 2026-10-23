use super::{Plan, position};
use crate::Result;
use crate::excise::text_matches::Matches;
use rusqlite::params;

impl Plan {
    pub(in super::super) fn collect(&self, input: &str, chain: i64, batch: Matches) -> Result<()> {
        let ranges = match batch {
            Matches::Undecided(ranges) => {
                let mut insert = self
                    .conn
                    .prepare_cached("INSERT OR IGNORE INTO raw_matches VALUES (?1,?2,?3,?4)")?;
                for (start, end) in ranges {
                    insert.execute(params![input, chain, position(start)?, position(end)?])?;
                }
                return Ok(());
            }
            Matches::Plain(ranges) => {
                self.conn.execute(
                    "INSERT OR IGNORE INTO matches SELECT * FROM raw_matches WHERE input=?1 AND chain=?2",
                    params![input, chain],
                )?;
                ranges
            }
            Matches::Json(ranges) => ranges,
        };
        self.conn.execute(
            "DELETE FROM raw_matches WHERE input=?1 AND chain=?2",
            params![input, chain],
        )?;
        self.matches(input, chain, &ranges)
    }

    /// Counts and observations keep exact hits. Writes use a separate ordered,
    /// nonoverlapping stream so chunk readers need retain at most one interval.
    pub(in super::super) fn normalize(&self, input: &str, chain: i64) -> Result<()> {
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
