//! Recorded revision points used to confirm immutable source copies.
use super::*;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RevisionPoint {
    pub number: i64,
    pub chain: i64,
    pub length: u64,
}
row! {
    /// A chain's latest revision at a given length.
    struct ChainAtLength {
        number: i64,
        chain: i64,
    }
    statements {
        const LATEST = "SELECT max(number) AS number, chain FROM source_revision
            WHERE input=?1 AND length=?2 GROUP BY chain ORDER BY max(number) DESC";
    }
}

impl Reader<'_> {
    /// Visits known points at exactly this length, at most once per byte chain.
    /// Returning true stops the stream once the caller confirms a byte-for-byte copy.
    pub fn visit_revisions_at_length(
        &self,
        input: &str,
        length: u64,
        mut visit: impl FnMut(RevisionPoint) -> Result<bool>,
    ) -> Result<()> {
        let sql_length = i64::try_from(length).map_err(|_| {
            StoreError::stored_data("revision length exceeds archive address space")
        })?;
        let _ = self.conn()?.each(
            &ChainAtLength::LATEST,
            params![input, sql_length],
            |point| {
                Ok(
                    if visit(RevisionPoint {
                        number: point.number,
                        chain: point.chain,
                        length,
                    })? {
                        std::ops::ControlFlow::Break(())
                    } else {
                        std::ops::ControlFlow::Continue(())
                    },
                )
            },
        )?;
        Ok(())
    }
}

/// The rows this module declares statements for, for the every-statement test.
#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<ChainAtLength>();
}
