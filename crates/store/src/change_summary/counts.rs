//! The transaction's count and change-stamp contributions, applied once after
//! selected rows, tool pairing and source claims agree.
use super::*;
use crate::derive::counts::{self, Counts};

impl ChangeSummary {
    pub(crate) fn installed_message(
        &self,
        pk: i64,
        session: i64,
        message: &polylogue_model::Message,
        words: i64,
        author: polylogue_model::WrittenBy,
    ) -> Result<()> {
        use polylogue_model::{Block, PromptKind, WrittenBy};
        let human = author == WrittenBy::Human;
        let model = author == WrittenBy::Model;
        let counts = Counts {
            messages: 1,
            human_prompts: i64::from(human && message.prompt_kind == Some(PromptKind::Prose)),
            human_commands: i64::from(human && message.prompt_kind == Some(PromptKind::Command)),
            words: if human || model { words } else { 0 },
            human_words: if human { words } else { 0 },
            model_words: if model { words } else { 0 },
            thinking: i64::from(
                message
                    .blocks
                    .iter()
                    .any(|block| matches!(block, Block::Thinking { .. })),
            ),
            ..Counts::default()
        };
        let mut changed = self.changed.borrow_mut();
        let change = changed.messages.get_mut(&pk).ok_or_else(|| {
            crate::StoreError::Invalid("installed message has no transaction owner".into())
        })?;
        change.installed = Some((
            session,
            counts,
            message
                .blocks
                .iter()
                .any(|block| matches!(block, Block::ToolCall { .. })),
        ));
        Ok(())
    }

    pub(crate) fn replaced_message(&self, pk: i64) -> Result<()> {
        let mut changed = self.changed.borrow_mut();
        let change = changed.messages.get_mut(&pk).ok_or_else(|| {
            crate::StoreError::Invalid("changed message has no transaction owner".into())
        })?;
        change.installed = None;
        Ok(())
    }
    pub(crate) fn session(&self, conn: &Connection, session: i64) -> Result<()> {
        if !self.changed.borrow().sessions.contains_key(&session)
            && let Some((script, counts_known)) = counts::session_state(conn, session)?
        {
            self.changed.borrow_mut().sessions.insert(
                session,
                SessionBefore {
                    script,
                    counts_known,
                },
            );
        }
        Ok(())
    }

    /// A result can change the count of a call in a different message.
    pub(crate) fn call(&self, conn: &Connection, call: i64) -> Result<()> {
        let message = conn.prepare_cached(
            "SELECT b.message_pk FROM tool_call t JOIN block b ON b.pk=t.call_block_pk WHERE t.pk=?1",
        )?.query_row([call], |row| row.get(0)).optional()?;
        if let Some(message) = message {
            self.record_message(conn, message)?;
        }
        Ok(())
    }

    pub(crate) fn call_result(&self, conn: &Connection, session: i64, id: &str) -> Result<()> {
        let calls = conn
            .prepare_cached("SELECT pk FROM tool_call WHERE session_pk=?1 AND call_id=?2")?
            .query_map(rusqlite::params![session, id], |row| row.get::<_, i64>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for call in calls {
            self.call(conn, call)?;
        }
        Ok(())
    }

    pub(crate) fn remove_results(&self, conn: &Connection, message: i64) -> Result<()> {
        let calls = conn
            .prepare_cached(
                "SELECT pk FROM tool_call WHERE result_block_pk IN
            (SELECT pk FROM block WHERE message_pk=?1)",
            )?
            .query_map([message], |row| row.get::<_, i64>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for call in calls {
            self.call(conn, call)?;
        }
        Ok(())
    }

    pub(crate) fn finish_summaries(&self, conn: &Connection, counter: Option<i64>) -> Result<()> {
        let changed = self.changed.borrow();
        let mut deltas = BTreeMap::<i64, Counts>::new();
        for (&pk, change) in &changed.messages {
            if let Some(before) = &change.before {
                deltas
                    .entry(before.session)
                    .or_default()
                    .accumulate(&before.counts, -1)?;
            }
            let after = match &change.installed {
                Some((session, counts, has_calls)) => {
                    let mut counts = counts.clone();
                    if *has_calls {
                        (counts.tool_calls, counts.errors) = counts::message_actions(conn, pk)?;
                    }
                    Some((*session, counts))
                }
                None => counts::message(conn, pk)?,
            };
            if let Some((session, after)) = after {
                deltas.entry(session).or_default().accumulate(&after, 1)?;
            }
        }
        for (&pk, before) in &changed.events {
            if let Some((session, compaction)) = before {
                deltas.entry(*session).or_default().compactions -= i64::from(*compaction);
            }
            let after = conn
                .prepare_cached("SELECT session_pk,kind='compaction' FROM event WHERE pk=?1")?
                .query_row([pk], |row| {
                    Ok((row.get::<_, Option<i64>>(0)?, row.get::<_, bool>(1)?))
                })
                .optional()?;
            if let Some((Some(session), compaction)) = after {
                deltas.entry(session).or_default().compactions += i64::from(compaction);
            }
        }
        for &session in changed.sessions.keys() {
            deltas.entry(session).or_default();
        }
        for &session in &changed.whole {
            deltas.entry(session).or_default();
        }
        for (session, delta) in deltas {
            if changed.whole.contains(&session)
                || changed
                    .sessions
                    .get(&session)
                    .is_some_and(|before| !before.counts_known)
            {
                // A caller explicitly replaced an unenumerated session. Its
                // existing whole-session boundary owns the complete fold.
                counts::apply_values(conn, session, &counts::compute(conn, session)?, counter)?;
                continue;
            }
            let actions = match changed.sessions.get(&session) {
                Some(before) if before.script != counts::has_script(conn, session)? => {
                    Some(counts::actions(conn, session)?)
                }
                _ => None,
            };
            counts::apply_delta(conn, session, &delta, actions, counter)?;
        }
        Ok(())
    }
}
