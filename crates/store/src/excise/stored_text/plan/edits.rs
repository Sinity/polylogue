use super::{Plan, position};
use crate::excise::stored_text::Chain;
use crate::{Result, StoreError};
use rusqlite::{OptionalExtension, params};

/// A forward-only blanking cursor carries one normalized interval across chunks.
pub(in super::super) struct Edits<'a> {
    plan: &'a Plan,
    key: &'a Chain,
    next: Option<(u64, u64)>,
    at: u64,
    patch_width: u64,
    #[cfg(test)]
    fetched: u64,
}

impl<'a> Edits<'a> {
    pub(super) fn new(plan: &'a Plan, key: &'a Chain, from: u64) -> Result<Self> {
        let before=plan.conn.query_row("SELECT start,end FROM write_ranges WHERE input=?1 AND chain=?2 AND start<=?3 ORDER BY start DESC LIMIT 1",
            params![key.0,key.1,position(from)?],|r|Ok((r.get::<_,i64>(0)? as u64,r.get::<_,i64>(1)? as u64))).optional()?;
        let next = match before {
            Some(range) if range.1 > from => Some(range),
            _ => Self::after(plan, key, from)?,
        };
        let patch_width = plan
            .conn
            .query_row(
                "SELECT patch_width FROM chains WHERE input=?1 AND chain=?2",
                params![key.0, key.1],
                |r| r.get::<_, i64>(0),
            )
            .optional()?
            .unwrap_or(0) as u64;
        Ok(Self {
            plan,
            key,
            next,
            at: from,
            patch_width,
            #[cfg(test)]
            fetched: 1,
        })
    }

    fn after(plan: &Plan, key: &Chain, start: u64) -> Result<Option<(u64, u64)>> {
        Ok(plan.conn.prepare_cached("SELECT start,end FROM write_ranges WHERE input=?1 AND chain=?2 AND start>?3 ORDER BY start LIMIT 1")?
            .query_row(params![key.0,key.1,position(start)?],|r|Ok((r.get::<_,i64>(0)? as u64,r.get::<_,i64>(1)? as u64))).optional()?)
    }

    pub(in super::super) fn blank(&mut self, at: u64, bytes: &mut [u8], fill: u8) -> Result<()> {
        if at < self.at {
            return Err(StoreError::Invalid(
                "text-excision chunks moved backwards".into(),
            ));
        }
        let end = at
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| StoreError::Invalid("text-excision chunk end overflow".into()))?;
        while let Some((start, until)) = self.next {
            if start >= end {
                break;
            }
            let from = start.max(at);
            let to = until.min(end);
            if from < to {
                for b in &mut bytes[(from - at) as usize..(to - at) as usize] {
                    if *b != b'\n' {
                        *b = fill;
                    }
                }
            }
            if until > end {
                break;
            }
            self.next = Self::after(self.plan, self.key, start)?;
            #[cfg(test)]
            {
                self.fetched += 1;
            }
        }
        self.at = end;
        Ok(())
    }

    pub(in super::super) fn rewrite(&mut self, at: u64, bytes: &mut [u8], fill: u8) -> Result<()> {
        self.blank(at, bytes, fill)?;
        let end = at + bytes.len() as u64;
        if self.patch_width == 0 {
            return Ok(());
        }
        let lower = at.saturating_sub(self.patch_width);
        let mut statement=self.plan.conn.prepare_cached("SELECT start,bytes FROM patches INDEXED BY patches_start WHERE input=?1 AND chain=?2 AND start>=?3 AND start<?4 AND end>?5 ORDER BY ordinal")?;
        let mut rows = statement.query(params![
            self.key.0,
            self.key.1,
            position(lower)?,
            position(end)?,
            position(at)?
        ])?;
        while let Some(row) = rows.next()? {
            let start = row.get::<_, i64>(0)? as u64;
            let value = row.get_ref(1)?;
            let patch = value
                .as_blob()
                .map_err(|e| StoreError::Invalid(e.to_string()))?;
            let from = start.max(at);
            let to = (start + patch.len() as u64).min(end);
            bytes[(from - at) as usize..(to - at) as usize]
                .copy_from_slice(&patch[(from - start) as usize..(to - start) as usize]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
