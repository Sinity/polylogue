//! The transaction's count and change-stamp contributions, applied once after
//! selected rows, tool pairing and stated sources agree.
use super::*;
use crate::derive::counts::{self, Counts};

impl ChangeSummary {
    pub(crate) fn installed_message(
        &self,
        pk: i64,
        session: i64,
        parent: Option<String>,
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
            crate::StoreError::Invalid(
                "installed message is not in the transaction's changes".into(),
            )
        })?;
        change.installed = Some(Installed {
            session,
            parent,
            counts,
            has_calls: message
                .blocks
                .iter()
                .any(|block| matches!(block, Block::ToolCall { .. })),
        });
        Ok(())
    }

    pub(crate) fn replaced_message(&self, pk: i64) -> Result<()> {
        let mut changed = self.changed.borrow_mut();
        let change = changed.messages.get_mut(&pk).ok_or_else(|| {
            crate::StoreError::Invalid("changed message is not in the transaction's changes".into())
        })?;
        change.installed = None;
        Ok(())
    }

    /// A session row is being removed with its messages: what messages
    /// inserted into it in this transaction held is no longer their row.
    pub(crate) fn removed_session(&self, session: i64) {
        for change in self.changed.borrow_mut().messages.values_mut() {
            if change
                .installed
                .as_ref()
                .is_some_and(|installed| installed.session == session)
            {
                change.installed = None;
            }
        }
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

    /// A result can change the count of a call in a different message: the
    /// messages holding the calls `sql` selects (with `params`) are recorded.
    fn call_messages(
        &self,
        conn: &Connection,
        sql: &str,
        params: impl rusqlite::Params,
    ) -> Result<()> {
        let messages = conn
            .prepare_cached(sql)?
            .query_map(params, |row| row.get::<_, i64>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for message in messages {
            self.record_message(conn, message)?;
        }
        Ok(())
    }

    pub(crate) fn call_result(&self, conn: &Connection, session: i64, id: &str) -> Result<()> {
        self.call_messages(
            conn,
            "SELECT b.message_pk FROM tool_call t JOIN block b ON b.pk=t.call_block_pk
             WHERE t.session_pk=?1 AND t.call_id=?2",
            rusqlite::params![session, id],
        )
    }

    pub(crate) fn remove_results(&self, conn: &Connection, message: i64) -> Result<()> {
        self.call_messages(
            conn,
            "SELECT b.message_pk FROM tool_call t JOIN block b ON b.pk=t.call_block_pk
             WHERE t.result_block_pk IN (SELECT pk FROM block WHERE message_pk=?1)",
            [message],
        )
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
                Some(installed) => {
                    let mut counts = installed.counts.clone();
                    if installed.has_calls {
                        (counts.tool_calls, counts.errors) = counts::message_actions(conn, pk)?;
                    }
                    Some((installed.session, counts))
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
                .query_row([pk], |row| <(Option<i64>, bool)>::try_from(row))
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
                // counts are recomputed for the whole session.
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
