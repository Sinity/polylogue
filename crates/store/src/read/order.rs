//! Export order is metadata over the full own session, before any
//! query, parent prefix or view subset. Source coordinates remain separate.
use super::*;
use crate::OptionExt as _;

#[derive(Clone, Copy)]
pub(super) enum RecordOrder {
    Skip,
    OwnSession,
    Known(i64),
}

/// `WITH record_ranks AS (...)`: each message of the session whose key is
/// parameter 1 (`pk`), with its place in the session's own reading order
/// (`record_order`).
macro_rules! record_ranks {
    () => {
        "WITH record_ranks AS (SELECT pk,row_number() OVER (ORDER BY session_order,id)-1 AS record_order
        FROM message WHERE session_pk=?1 AND duplicate_of IS NULL) "
    };
}
pub(super) use record_ranks;

pub(super) fn checked(value: i64) -> Result<u64> {
    u64::try_from(value).map_err(|_| StoreError::stored(0, "negative message snapshot ordinal"))
}

row! {
    /// A message of a range, by ID.
    struct RangeMember {
        id: MessageId,
    }
}

impl Reader<'_> {
    /// The message's physical session is metadata; resolving it never loads
    /// an endpoint body or expands inherited parents.
    pub fn message_session(&self, id: &MessageId) -> Result<Option<SessionId>> {
        let id = crate::chain_names::resolved_message(self.conn()?, id)?;
        self.conn()?.value(
            "SELECT s.id FROM message m JOIN session s ON s.pk=m.session_pk WHERE m.id=?1 AND m.duplicate_of IS NULL",
            [id.to_string()],
        )
    }

    /// The export snapshot ordinal in the whole physical session that
    /// holds this message. It is independent of query order and inherited parent prefixes.
    /// The count walks the session's reading-order index up to the message.
    pub fn message_order(&self, id: &MessageId) -> Result<Option<u64>> {
        let id = crate::chain_names::resolved_message(self.conn()?, id)?;
        self.conn()?
            .value(
                "SELECT (SELECT count(*) FROM message o WHERE o.session_pk=m.session_pk AND o.duplicate_of IS NULL
                   AND (o.session_order,o.id)<(m.session_order,m.id))
                 FROM message m WHERE m.id=?1 AND m.duplicate_of IS NULL",
                [id.to_string()],
            )?
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
        .or_stored("a message range has message membership")?;
        let from = i64::try_from(from)
            .map_err(|_| StoreError::stored_data("range ordinal exceeds SQLite's integer range"))?;
        let statement = Statement::<RangeMember>::built(format!(
            "SELECT m.id FROM message m WHERE m.pk IN ({members}) AND m.duplicate_of IS NULL
             ORDER BY m.session_order,m.id LIMIT -1 OFFSET {from}"
        ));
        let _ = self
            .conn()?
            .each(&statement, rusqlite::params_from_iter(&params), |member| {
                self.check()?;
                visit(member.id)
            })?;
        self.check()
    }
}

#[cfg(test)]
mod tests;
