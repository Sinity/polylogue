//! Recorded outcomes for tool calls visible on a content page.
use super::*;

impl Reader<'_> {
    pub fn tool_outcomes(
        &self,
        message: &MessageId,
    ) -> Result<Vec<(usize, polylogue_model::CallOutcome)>> {
        let mut statement=self.conn()?.prepare_cached("SELECT coalesce(b.ordinal,CAST(b.idx AS TEXT)),t.outcome FROM message m JOIN block b ON b.message_pk=m.pk JOIN tool_call t ON t.call_block_pk=b.pk WHERE m.id=?1 ORDER BY b.idx")?;
        let rows = statement.query_map([message.to_string()], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut outcomes = Vec::new();
        for row in rows {
            let (block, outcome) = row?;
            let outcome = parse_name(&outcome).ok_or_else(|| {
                StoreError::Invalid("a tool call has an invalid recorded outcome".into())
            })?;
            let block = block.parse::<usize>().map_err(|_| {
                StoreError::Invalid("a tool call has an invalid source block index".into())
            })?;
            outcomes.push((block, outcome));
        }
        Ok(outcomes)
    }
}
