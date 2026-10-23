use super::{Plan, position};
use crate::excise::stored_text::Chain;
use crate::{Result, StoreError};
use polylogue_model::snapshot_log::Header;
use rusqlite::params;

impl Plan {
    pub(in super::super) fn framed(&self, key: &Chain) -> Result<()> {
        self.conn.execute(
            "INSERT INTO chains(input,chain,framed) VALUES (?1,?2,1)
            ON CONFLICT(input,chain) DO UPDATE SET framed=1",
            params![key.0, key.1],
        )?;
        Ok(())
    }

    pub(in super::super) fn frame(
        &self,
        key: &Chain,
        ordinal: i64,
        at: u64,
        header: &Header,
    ) -> Result<()> {
        self.conn
            .prepare_cached("INSERT INTO frames VALUES (?1,?2,?3,?4,?5)")?
            .execute(params![
                key.0,
                key.1,
                ordinal,
                position(at)?,
                header.encode()
            ])?;
        Ok(())
    }

    pub(in super::super) fn visit_frames(
        &self,
        key: &Chain,
        mut visit: impl FnMut(i64, u64, Header) -> Result<()>,
    ) -> Result<()> {
        let mut statement = self.conn.prepare(
            "SELECT ordinal,start,header FROM frames WHERE input=?1 AND chain=?2 ORDER BY ordinal",
        )?;
        let mut rows = statement.query(params![key.0, key.1])?;
        while let Some(row) = rows.next()? {
            let bytes = row.get_ref(2)?;
            let bytes = bytes
                .as_blob()
                .map_err(|e| StoreError::Invalid(e.to_string()))?;
            visit(
                row.get(0)?,
                row.get::<_, i64>(1)? as u64,
                Header::decode(bytes).map_err(StoreError::Invalid)?,
            )?;
        }
        Ok(())
    }
}
