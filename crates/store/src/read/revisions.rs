//! Recorded revision points used to confirm immutable source copies.
use super::*;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RevisionPoint {
    pub number: i64,
    pub chain: i64,
    pub length: u64,
}
impl Reader<'_> {
    /// The newest later chain of `input` holding exactly the bytes of
    /// `chain` (the same length and extents), whose reading is that chain's.
    pub fn newest_identical_chain(&self, input: &str, chain: i64) -> Result<Option<i64>> {
        crate::same_bytes::newest_identical(self.conn()?, input, chain)
    }

    /// Visits known points at exactly this length, at most once per byte chain.
    /// Returning true stops the stream once the caller proves an exact copy.
    pub fn visit_revisions_at_length(
        &self,
        input: &str,
        length: u64,
        mut visit: impl FnMut(RevisionPoint) -> Result<bool>,
    ) -> Result<()> {
        self.visit_revision_points(input, length, None, &mut visit)
    }

    /// Uses only bytes and revision declarations admitted before provider I/O.
    pub fn visit_revisions_at_length_before(
        &self,
        boundary: &crate::primary::SourceBoundary,
        input: &str,
        length: u64,
        mut visit: impl FnMut(RevisionPoint) -> Result<bool>,
    ) -> Result<()> {
        let through = boundary.through(self.conn()?, &self.pool.archive)?;
        self.visit_revision_points(input, length, Some(through), &mut visit)
    }

    fn visit_revision_points(
        &self,
        input: &str,
        length: u64,
        through: Option<u64>,
        visit: &mut impl FnMut(RevisionPoint) -> Result<bool>,
    ) -> Result<()> {
        let length = i64::try_from(length).map_err(|_| {
            StoreError::Invalid("revision length exceeds archive address space".into())
        })?;
        let through = through.map(i64::try_from).transpose().map_err(|_| {
            StoreError::Invalid("source admission fence exceeds archive address space".into())
        })?;
        let conn = self.conn()?;
        let mut statement = conn.prepare("SELECT max(number),chain FROM source_revision WHERE input=?1 AND length=?2 AND (?3 IS NULL OR commit_seq<=?3) GROUP BY chain ORDER BY max(number) DESC")?;
        let mut rows = statement.query(params![input, length, through])?;
        while let Some(row) = rows.next()? {
            let chain = row.get(1)?;
            if let Some(through) = through {
                let admitted =
                    crate::source::Source::admitted_through(conn, input, chain, length as u64)?;
                if admitted.is_none_or(|position| position > through as u64) {
                    continue;
                }
            }
            if visit(RevisionPoint {
                number: row.get(0)?,
                chain,
                length: length as u64,
            })? {
                break;
            }
        }
        Ok(())
    }
}
