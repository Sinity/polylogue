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
    format!(
        "SELECT pk,row_number() OVER (ORDER BY session_order,id)-1 AS record_order
        FROM message WHERE session_pk={session} AND duplicate_of IS NULL"
    )
}

pub(super) fn checked(value: i64) -> Result<u64> {
    u64::try_from(value)
        .map_err(|_| StoreError::Invalid("negative message snapshot ordinal".into()))
}

impl Reader<'_> {
    /// The canonical physical owner is metadata; resolving it never hydrates
    /// an endpoint body or expands inherited parents.
    pub fn message_session(&self, id: &MessageId) -> Result<Option<SessionId>> {
        let id = crate::parser_context::canonical_message(self.conn()?, id)?;
        let value = self.conn()?.query_row(
            "SELECT s.id FROM message m JOIN session s ON s.pk=m.session_pk WHERE m.id=?1 AND m.duplicate_of IS NULL",
            [id.to_string()], |row| crate::encoding::Columns::id::<SessionId>(row, 0),
        ).optional()?;
        Ok(value)
    }

    /// The export snapshot ordinal in this message's entire canonical own
    /// session. It is independent of query order and inherited parent prefixes.
    /// The count walks the session's reading-order index up to the message.
    pub fn message_order(&self, id: &MessageId) -> Result<Option<u64>> {
        let id = crate::parser_context::canonical_message(self.conn()?, id)?;
        self.conn()?
            .prepare_cached(
                "SELECT (SELECT count(*) FROM message o WHERE o.session_pk=m.session_pk AND o.duplicate_of IS NULL
                   AND (o.session_order,o.id)<(m.session_order,m.id))
                 FROM message m WHERE m.id=?1 AND m.duplicate_of IS NULL",
            )?
            .query_row([id.to_string()], |row| row.get::<_, i64>(0))
            .optional()?
            .map(checked)
            .transpose()
    }

    /// The messages of `first..last` in reading order, starting at ordinal
    /// `from` of the range. Only identities are read.
    pub fn visit_message_range(
        &self,
        first: &MessageId,
        last: &MessageId,
        from: u64,
        mut visit: impl FnMut(MessageId) -> Result<std::ops::ControlFlow<()>>,
    ) -> Result<()> {
        let mut params = Vec::new();
        let members = crate::query::references::membership(
            polylogue_query::ast::Unit::Messages,
            &polylogue_model::Reference::MessageRange {
                first: first.clone(),
                last: last.clone(),
            },
            &mut |value| {
                params.push(value);
                format!("?{}", params.len())
            },
        )
        .ok_or_else(|| StoreError::Invalid("a message range has message membership".into()))?;
        let from = i64::try_from(from).map_err(|_| {
            StoreError::Invalid("range ordinal exceeds SQLite's integer range".into())
        })?;
        let conn = self.conn()?;
        let mut statement = conn.prepare_cached(&format!(
            "SELECT m.id FROM message m WHERE m.pk IN ({members}) AND m.duplicate_of IS NULL
             ORDER BY m.session_order,m.id LIMIT -1 OFFSET {from}"
        ))?;
        let mut rows = statement.query(rusqlite::params_from_iter(&params))?;
        while let Some(row) = rows.next()? {
            self.check()?;
            let id: MessageId = crate::encoding::Columns::id(row, 0)?;
            if visit(id)?.is_break() {
                break;
            }
        }
        self.check()
    }
}

#[cfg(test)]
mod tests;
