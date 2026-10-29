//! Recorded revision points used to confirm immutable source copies.
use super::*;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RevisionPoint {
    pub number: i64,
    pub chain: i64,
    pub length: u64,
}
impl Reader<'_> {
    /// Visits known points at exactly this length, at most once per byte chain.
    /// Returning true stops the stream once the caller proves an exact copy.
    pub fn visit_revisions_at_length(
        &self,
        input: &str,
        length: u64,
        mut visit: impl FnMut(RevisionPoint) -> Result<bool>,
    ) -> Result<()> {
        let length = i64::try_from(length).map_err(|_| {
            StoreError::Invalid("revision length exceeds archive address space".into())
        })?;
        let mut statement=self.conn()?.prepare("SELECT max(number),chain FROM source_revision WHERE input=?1 AND length=?2 GROUP BY chain ORDER BY max(number) DESC")?;
        let mut rows = statement.query(params![input, length])?;
        while let Some(row) = rows.next()? {
            if visit(RevisionPoint {
                number: row.get(0)?,
                chain: row.get(1)?,
                length: length as u64,
            })? {
                break;
            }
        }
        Ok(())
    }
}
