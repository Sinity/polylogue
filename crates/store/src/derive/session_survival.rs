//! The per-session facts the survival measures read: when the session's
//! messages start and end, when its first tool call failed, how many prose
//! prompts a human typed before that, and whether its last prompt went
//! unanswered. Whether an unanswered session is over depends on the time
//! of reading, so the measure decides that, not this.

use super::{Derivation, Version};
use crate::Result;
use rusqlite::Transaction;

pub(super) const DERIVATION: Derivation = Derivation {
    name: "session_survival",
    version: Version::Rules(1),
    after: &[],
    run: |tx, pk, _| run(tx, pk),
    finish: None,
};

fn run(tx: &Transaction<'_>, session_pk: i64) -> Result<()> {
    tx.prepare_cached("DELETE FROM session_survival WHERE session_pk = ?1")?
        .execute([session_pk])?;
    tx.prepare_cached(
        "WITH s AS (
           SELECT min(at) AS first_at, max(at) AS last_at, count(*) AS messages
           FROM message WHERE session_pk = ?1),
         f AS (
           SELECT min(coalesce(t.at, m.at)) AS failure_at
           FROM tool_call t JOIN block b ON b.pk = t.call_block_pk
           JOIN message m ON m.pk = b.message_pk
           WHERE t.session_pk = ?1 AND t.outcome = 'error')
         INSERT INTO session_survival(session_pk, first_at, last_at, first_failure_at,
                                   prompts_to_failure, prompted, last_prompt_unanswered)
         SELECT ?1, s.first_at, s.last_at, f.failure_at,
           (SELECT count(*) FROM message h WHERE h.session_pk = ?1 AND h.written_by = 'human'
              AND h.prompt_kind = 'prose' AND h.at <= coalesce(f.failure_at, s.last_at)),
           EXISTS (SELECT 1 FROM message h WHERE h.session_pk = ?1
                   AND h.written_by = 'human' AND h.prompt_kind = 'prose'),
           NOT EXISTS (
             SELECT 1 FROM message a WHERE a.session_pk = ?1 AND a.role = 'assistant'
               AND (a.order_input_id, a.seq, a.record_position) > (
                 SELECT h.order_input_id, h.seq, h.record_position FROM message h WHERE h.session_pk = ?1
                   AND h.written_by = 'human' AND h.prompt_kind = 'prose'
                 ORDER BY h.order_input_id DESC, h.seq DESC, h.record_position DESC LIMIT 1))
         FROM s, f WHERE s.messages > 0",
    )?
    .execute([session_pk])?;
    Ok(())
}
