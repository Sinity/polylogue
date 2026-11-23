//! Recorded revision points used to confirm immutable source copies.
use super::*;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RevisionPoint {
    pub number: i64,
    pub chain: i64,
    pub length: u64,
}
impl Reader<'_> {
    /// Multiple chains of `input`, staged on disk for comparison by chain.
    /// A single chain has no other reading to take its rows from.
    pub fn chain_layouts(&self, input: &str) -> Result<Option<crate::ChainLayouts>> {
        crate::ChainLayouts::load(self.conn()?, input)
    }

    /// Visits known points at exactly this length, at most once per byte chain.
    /// Returning true stops the stream once the caller confirms a byte-for-byte copy.
    pub fn visit_revisions_at_length(
        &self,
        input: &str,
        length: u64,
        mut visit: impl FnMut(RevisionPoint) -> Result<bool>,
    ) -> Result<()> {
        let sql_length = i64::try_from(length)
            .map_err(|_| StoreError::invalid("revision length exceeds archive address space"))?;
        let conn = self.conn()?;
        let mut statement = conn.prepare("SELECT max(number),chain FROM source_revision WHERE input=?1 AND length=?2 GROUP BY chain ORDER BY max(number) DESC")?;
        let mut rows = statement.query(params![input, sql_length])?;
        while let Some(row) = rows.next()? {
            let chain = row.get(1)?;
            if visit(RevisionPoint {
                number: row.get(0)?,
                chain,
                length,
            })? {
                break;
            }
        }
        Ok(())
    }
}
