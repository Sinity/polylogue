//! Export order is metadata over the full canonical own session, before any
//! query, parent prefix or view subset. Source coordinates remain separate.
use super::*;

#[derive(Clone, Copy)]
pub(super) enum RecordOrder {
    Skip,
    OwnSession,
    Known(i64),
}

pub(super) fn canonical_rows(session: &str) -> String {
    format!("SELECT pk,row_number() OVER (ORDER BY order_input_id,seq,record_position,id)-1 AS record_order
        FROM message WHERE session_pk={session} AND duplicate_of IS NULL")
}

pub(super) fn checked(value: i64) -> Result<u64> {
    u64::try_from(value)
        .map_err(|_| StoreError::Invalid("negative message snapshot ordinal".into()))
}

impl Reader<'_> {
    /// The export snapshot ordinal in this message's entire canonical own
    /// session. It is independent of query order and inherited parent prefixes.
    pub fn message_order(&self, id: &MessageId) -> Result<Option<u64>> {
        let sql = format!(
            "WITH canonical_order AS ({}) SELECT record_order FROM canonical_order
            WHERE pk=(SELECT pk FROM message WHERE id=?1)",
            canonical_rows("(SELECT session_pk FROM message WHERE id=?1)")
        );
        self.conn()?
            .query_row(&sql, [id.to_string()], |row| row.get::<_, i64>(0))
            .optional()?
            .map(checked)
            .transpose()
    }
}
