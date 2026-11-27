//! Recorded outcomes for tool calls visible on a content page.
use super::*;

row! {
    /// A call block's original position and the call's recorded outcome.
    struct CallOutcomeRow {
        block: String,
        outcome: polylogue_model::CallOutcome,
    }
    statements {
        const OF_MESSAGE = "SELECT coalesce(b.source_position,CAST(b.position AS TEXT)) AS block, t.outcome
            FROM message m JOIN block b ON b.message_pk=m.pk JOIN tool_call t ON t.call_block_pk=b.pk
            WHERE m.id=?1 ORDER BY b.position";
    }
}

impl Reader<'_> {
    pub fn tool_outcomes(
        &self,
        message: &MessageId,
    ) -> Result<Vec<(usize, polylogue_model::CallOutcome)>> {
        let mut outcomes = Vec::new();
        for CallOutcomeRow { block, outcome } in self
            .conn()?
            .all(&CallOutcomeRow::OF_MESSAGE, [message.to_string()])?
        {
            let block = block.parse::<usize>().map_err(|_| {
                StoreError::stored(0, "a tool call has an invalid source block index")
            })?;
            outcomes.push((block, outcome));
        }
        Ok(outcomes)
    }
}

/// The rows this module declares statements for, for the every-statement test.
#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<CallOutcomeRow>();
}
