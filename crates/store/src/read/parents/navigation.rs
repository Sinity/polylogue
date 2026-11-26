//! Metadata-only positions for the shared reader.
use super::*;
use crate::ResultExt as _;

fn message_count(count: i64) -> Result<usize> {
    usize::try_from(count).or_internal("the message count exceeds address space")
}

row! {
    /// A message that may be the latest visible one.
    struct Candidate {
        pk: MessagePk,
        id: polylogue_model::MessageId,
    }
}

impl MessageSelection<'_, '_> {
    /// Seek this selection's metadata order without hydrating skipped bodies.
    /// Physical selections share the session order used by message bundles.
    pub fn locate_ordinal(&self, ordinal: u64) -> Result<Option<MessagePosition>> {
        self.reader.check()?;
        let Ok(message) = usize::try_from(ordinal) else {
            return Ok(None);
        };
        if let Some(keys) = &self.with_parents {
            return Ok(keys.get(message).map(|key| MessagePosition {
                message,
                key: Some(key.message_pk),
            }));
        }
        let Some((session, _)) = &self.own else {
            return Ok(None);
        };
        let Ok(ordinal) = i64::try_from(ordinal) else {
            return Ok(None);
        };
        let key: Option<MessagePk> = self.reader.conn()?.value(
            concat!(
                super::super::order::record_ranks!(),
                "SELECT pk FROM record_ranks WHERE record_order=?2"
            ),
            params![session, ordinal],
        )?;
        Ok(key.map(|key| MessagePosition {
            message,
            key: Some(key),
        }))
    }

    pub fn len(&self) -> Result<usize> {
        if let Some(keys) = &self.with_parents {
            return Ok(keys.len());
        }
        match &self.own {
            Some((pk, _)) => message_count(self.reader.conn()?.scalar(
                "SELECT count(*) FROM message WHERE session_pk=?1 AND duplicate_of IS NULL",
                [pk],
            )?),
            None => Ok(0),
        }
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    pub fn message_key(&self, id: &polylogue_model::MessageId) -> Result<Option<MessagePk>> {
        let id = crate::chain_names::resolved_message(self.reader.conn()?, id)?.to_string();
        if let Some(keys) = &self.with_parents {
            return Ok(keys
                .iter()
                .find(|key| key.id == id)
                .map(|key| key.message_pk));
        }
        let Some((pk, _)) = &self.own else {
            return Ok(None);
        };
        self.reader.conn()?.scalar_optional(
            "SELECT pk FROM message WHERE session_pk=?1 AND id=?2 AND duplicate_of IS NULL",
            params![pk, id],
        )
    }

    /// A known message, or the last selected message, without reading bodies.
    /// A missing reference remains distinguishable from an empty session.
    pub fn locate(
        &self,
        id: Option<&polylogue_model::MessageId>,
    ) -> Result<Option<MessagePosition>> {
        if let Some(keys) = &self.with_parents {
            let id = id
                .map(|id| crate::chain_names::resolved_message(self.reader.conn()?, id))
                .transpose()?
                .map(|id| id.to_string());
            return Ok(match &id {
                Some(id) => keys.iter().enumerate().find(|(_, key)| key.id == *id),
                None => keys.iter().enumerate().next_back(),
            }
            .map(|(message, key)| MessagePosition {
                message,
                key: Some(key.message_pk),
            }));
        }
        let Some((pk, _)) = &self.own else {
            return Ok(None);
        };
        let conn = self.reader.conn()?;
        let key=match id {
            Some(id)=>self.message_key(id)?,
            None=>conn.scalar_optional::<MessagePk>("SELECT pk FROM message WHERE session_pk=?1 AND duplicate_of IS NULL ORDER BY session_order DESC,id DESC LIMIT 1",[pk])?,
        };
        let Some(key) = key else {
            return Ok(None);
        };
        let count = conn.scalar::<i64>(
            "SELECT count(*) FROM message m JOIN message target ON target.pk=?2
            WHERE m.session_pk=?1 AND m.duplicate_of IS NULL
            AND (m.session_order,m.id) < (target.session_order,target.id)",
            params![pk, key],
        )?;
        Ok(Some(MessagePosition {
            message: message_count(count)?,
            key: Some(key),
        }))
    }

    /// Resolve the final visible message from block kinds, without decoding
    /// any body. Prompt selection follows the source role and prompt category.
    pub fn latest(
        &self,
        kinds: &[polylogue_model::BlockKind],
        prompts: bool,
    ) -> Result<Option<MessagePosition>> {
        if kinds.is_empty() {
            return Ok(None);
        }
        let kinds = block_kind_list(kinds);
        let sql = format!(
            "SELECT m.pk,m.id FROM message m WHERE m.session_pk=?1 AND m.duplicate_of IS NULL
            AND (?2=0 OR (pl_is_prompt(m.role,m.prompt_kind)))
            AND EXISTS(SELECT 1 FROM block b WHERE b.message_pk=m.pk AND b.kind IN ({kinds}))
            ORDER BY m.session_order DESC,m.id DESC"
        );
        let statement = Statement::<Candidate>::built(sql);
        let conn = self.reader.conn()?;
        for (session, members) in self.sessions().into_iter().rev() {
            let members = members.map(|keys| keys.into_iter().collect::<HashSet<_>>());
            let mut found = None;
            let _ = conn.each(&statement, params![session, prompts], |candidate| {
                if members
                    .as_ref()
                    .is_some_and(|members| !members.contains(&candidate.pk))
                {
                    return Ok(ControlFlow::Continue(()));
                }
                found = Some(candidate.id);
                Ok(ControlFlow::Break(()))
            })?;
            if let Some(id) = found {
                return self.locate(Some(&id));
            }
        }
        Ok(None)
    }

    /// The closest preceding visible message, seeking backwards in source order.
    pub fn previous(
        &self,
        from: MessagePosition,
        kinds: &[polylogue_model::BlockKind],
        prompts: bool,
    ) -> Result<Option<MessagePosition>> {
        let Some(key) = from.key else { return Ok(None) };
        let conn = self.reader.conn()?;
        let kinds = block_kind_list(kinds);
        if let Some(keys) = &self.with_parents {
            let at = from.message.min(keys.len());
            for (index, item) in keys[..at].iter().enumerate().rev() {
                if self.visible_key(item.message_pk, &kinds, prompts)? {
                    return Ok(Some(MessagePosition {
                        message: index,
                        key: Some(item.message_pk),
                    }));
                }
            }
            return Ok(None);
        }
        let Some((session, _)) = &self.own else {
            return Ok(None);
        };
        let sql = format!(
            "SELECT m.pk FROM message m WHERE m.session_pk=?1 AND m.duplicate_of IS NULL
            AND (m.session_order,m.id)<(SELECT session_order,id FROM message WHERE pk=?2)
            AND (?3=0 OR (pl_is_prompt(m.role,m.prompt_kind)))
            AND EXISTS(SELECT 1 FROM block WHERE message_pk=m.pk AND kind IN ({kinds}))
            ORDER BY m.session_order DESC,m.id DESC LIMIT 1"
        );
        let previous = conn.scalar_optional::<MessagePk>(&sql, params![session, key, prompts])?;
        previous
            .map(|previous| {
                let stepped: i64 = conn.scalar(
                    "SELECT count(*) FROM message m
                WHERE m.session_pk=?1 AND m.duplicate_of IS NULL
                AND (m.session_order,m.id)>=(SELECT session_order,id FROM message WHERE pk=?2)
                AND (m.session_order,m.id)<(SELECT session_order,id FROM message WHERE pk=?3)",
                    params![session, previous, key],
                )?;
                Ok(MessagePosition {
                    message: from.message.saturating_sub(message_count(stepped)?),
                    key: Some(previous),
                })
            })
            .transpose()
    }

    fn visible_key(&self, key: MessagePk, kinds: &str, prompts: bool) -> Result<bool> {
        self.reader.conn()?.exists(&format!("SELECT EXISTS(SELECT 1 FROM message m JOIN block b ON b.message_pk=m.pk WHERE m.pk=?1 AND b.kind IN ({kinds}) AND (?2=0 OR (pl_is_prompt(m.role,m.prompt_kind))))"), params![key,prompts])
    }
}
