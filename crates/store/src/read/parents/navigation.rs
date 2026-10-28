//! Metadata-only positions and outline entries for the shared reader.
use super::*;

#[derive(Debug, Clone, Serialize)]
pub struct OutlineEntry {
    #[serde(skip)]
    pub key: i64,
    pub message: polylogue_model::MessageId,
    pub session: SessionId,
    pub at: Option<Timestamp>,
    pub kinds: Vec<String>,
    pub events: Vec<String>,
}

fn message_count(count: i64) -> Result<usize> {
    usize::try_from(count)
        .map_err(|_| StoreError::Invalid("the message count exceeds address space".into()))
}

impl MessageSelection<'_, '_> {
    pub fn len(&self) -> Result<usize> {
        if let Some(keys) = &self.with_parents {
            return Ok(keys.len());
        }
        match &self.own {
            Some((pk, _)) => message_count(self.reader.conn()?.query_row(
                "SELECT count(*) FROM message WHERE session_pk=?1 AND duplicate_of IS NULL",
                [pk],
                |r| r.get::<_, i64>(0),
            )?),
            None => Ok(0),
        }
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    pub fn message_key(&self, id: &polylogue_model::MessageId) -> Result<Option<i64>> {
        let id = id.to_string();
        if let Some(keys) = &self.with_parents {
            return Ok(keys
                .iter()
                .find(|key| key.id == id)
                .map(|key| key.message_pk));
        }
        let Some((pk, _)) = &self.own else {
            return Ok(None);
        };
        Ok(self
            .reader
            .conn()?
            .query_row(
                "SELECT pk FROM message WHERE session_pk=?1 AND id=?2 AND duplicate_of IS NULL",
                params![pk, id],
                |r| r.get(0),
            )
            .optional()?)
    }

    /// A known message, or the last selected message, without reading bodies.
    /// A missing reference remains distinguishable from an empty session.
    pub fn locate(&self, id: Option<&polylogue_model::MessageId>) -> Result<Option<VisitStart>> {
        if let Some(keys) = &self.with_parents {
            let id = id.map(ToString::to_string);
            return Ok(match &id {
                Some(id) => keys.iter().enumerate().find(|(_, key)| key.id == *id),
                None => keys.iter().enumerate().next_back(),
            }
            .map(|(message, key)| VisitStart {
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
            None=>conn.query_row("SELECT pk FROM message WHERE session_pk=?1 AND duplicate_of IS NULL ORDER BY order_input_id DESC,seq DESC,record_position DESC,id DESC LIMIT 1",[pk],|r|r.get::<_,i64>(0)).optional()?,
        };
        let Some(key) = key else {
            return Ok(None);
        };
        let count=conn.query_row("SELECT count(*) FROM message m JOIN message anchor ON anchor.pk=?2
            WHERE m.session_pk=?1 AND m.duplicate_of IS NULL
            AND (m.order_input_id,m.seq,m.record_position,m.id) < (anchor.order_input_id,anchor.seq,anchor.record_position,anchor.id)",params![pk,key],|r|r.get::<_,i64>(0))?;
        Ok(Some(VisitStart {
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
    ) -> Result<Option<VisitStart>> {
        if kinds.is_empty() {
            return Ok(None);
        }
        let kinds = kinds
            .iter()
            .map(|kind| format!("'{}'", kind.as_str()))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT m.pk,m.id FROM message m WHERE m.session_pk=?1 AND m.duplicate_of IS NULL
            AND (?2=0 OR (pl_is_prompt(m.role,m.prompt_kind)))
            AND EXISTS(SELECT 1 FROM block b WHERE b.message_pk=m.pk AND b.kind IN ({kinds}))
            ORDER BY m.order_input_id DESC,m.seq DESC,m.record_position DESC,m.id DESC"
        );
        let conn = self.reader.conn()?;
        let mut statement = conn.prepare_cached(&sql)?;
        for (session, members) in self.sessions().into_iter().rev() {
            let members = members.map(|keys| keys.into_iter().collect::<HashSet<_>>());
            let mut rows = statement.query(params![session, prompts])?;
            while let Some(row) = rows.next()? {
                let pk: i64 = row.get(0)?;
                if members
                    .as_ref()
                    .is_some_and(|members| !members.contains(&pk))
                {
                    continue;
                }
                let id: String = row.get(1)?;
                let id = polylogue_model::MessageId::parse(&id)
                    .map_err(|e| StoreError::Invalid(e.to_string()))?;
                return self.locate(Some(&id));
            }
        }
        Ok(None)
    }

    /// Continue after an order key, within the selected parent-session order.
    pub fn outline(&self, after: Option<i64>, size: usize) -> Result<Vec<OutlineEntry>> {
        let conn = self.reader.conn()?;
        let mut entries = Vec::new();
        let sessions = self.sessions();
        let mut begun = after.is_none();
        for (session, members) in sessions {
            let members = members.map(|keys| keys.into_iter().collect::<HashSet<_>>());
            let anchor = if !begun {
                let belongs = match &members {
                    Some(keys) => keys.contains(&after.unwrap()),
                    None => conn.query_row(
                        "SELECT EXISTS(SELECT 1 FROM message WHERE pk=?1 AND session_pk=?2)",
                        params![after, session],
                        |row| row.get(0),
                    )?,
                };
                if !belongs {
                    continue;
                }
                begun = true;
                after
            } else {
                None
            };
            // Candidate sets are computed once, rather than looked up again for
            // every message. The message order index seeks past the anchor.
            let seek = if anchor.is_some() {
                "AND (m.order_input_id,m.seq,m.record_position,m.id) > (SELECT order_input_id,seq,record_position,id FROM message WHERE pk=?2)"
            } else {
                ""
            };
            let sql = format!("WITH failures AS MATERIALIZED (
                SELECT DISTINCT b.message_pk FROM tool_call t JOIN block b ON b.pk=t.call_block_pk WHERE t.session_pk=?1 AND t.outcome='error'),
                compactions AS MATERIALIZED (SELECT after_message_pk FROM event WHERE session_pk=?1 AND kind IN ('compaction','microcompaction'))
                SELECT m.pk,m.id,s.id,m.at,pl_is_prompt(m.role,m.prompt_kind),m.pk IN failures,EXISTS(SELECT 1 FROM link WHERE branch_point=m.id)
                FROM message m JOIN session s ON s.pk=m.session_pk
                WHERE m.session_pk=?1 AND m.duplicate_of IS NULL {seek} AND
                (pl_is_prompt(m.role,m.prompt_kind) OR m.pk IN failures OR EXISTS(SELECT 1 FROM link WHERE branch_point=m.id) OR m.pk IN compactions)
                ORDER BY m.order_input_id,m.seq,m.record_position,m.id");
            let mut statement = conn.prepare_cached(&sql)?;
            let mut rows = if anchor.is_some() {
                statement.query(params![session, anchor])?
            } else {
                statement.query([session])?
            };
            while let Some(row) = rows.next()? {
                let pk: i64 = row.get(0)?;
                if members
                    .as_ref()
                    .is_some_and(|members| !members.contains(&pk))
                {
                    continue;
                }
                let mut kinds = Vec::new();
                if row.get::<_, bool>(4)? {
                    kinds.push("prompt".into());
                }
                if row.get::<_, bool>(5)? {
                    kinds.push("tool failure".into());
                }
                if row.get::<_, bool>(6)? {
                    kinds.push("branch point".into());
                }
                let events=conn.prepare_cached("SELECT id FROM event WHERE after_message_pk=?1 AND kind IN ('compaction','microcompaction') ORDER BY seq,id")?.query_map([pk],|r|r.get::<_,String>(0))?.collect::<rusqlite::Result<Vec<_>>>()?;
                if !events.is_empty() {
                    kinds.push("compaction".into());
                }
                entries.push(OutlineEntry {
                    key: pk,
                    message: polylogue_model::MessageId::parse(&row.get::<_, String>(1)?)
                        .map_err(|e| StoreError::Invalid(e.to_string()))?,
                    session: SessionId::parse(&row.get::<_, String>(2)?)
                        .map_err(|e| StoreError::Invalid(e.to_string()))?,
                    at: ts(row.get(3)?),
                    kinds,
                    events,
                });
                if entries.len() == size {
                    return Ok(entries);
                }
            }
        }
        Ok(entries)
    }

    /// The closest preceding visible message, seeking backwards in source order.
    pub fn previous(
        &self,
        from: VisitStart,
        kinds: &[polylogue_model::BlockKind],
        prompts: bool,
    ) -> Result<Option<VisitStart>> {
        let Some(key) = from.key else { return Ok(None) };
        let conn = self.reader.conn()?;
        if let Some(keys) = &self.with_parents {
            let at = from.message.min(keys.len());
            for (index, item) in keys[..at].iter().enumerate().rev() {
                if self.visible_key(item.message_pk, kinds, prompts)? {
                    return Ok(Some(VisitStart {
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
        let kinds = kinds
            .iter()
            .map(|kind| format!("'{}'", kind.as_str()))
            .collect::<Vec<_>>()
            .join(",");
        let sql=format!("SELECT m.pk FROM message m WHERE m.session_pk=?1 AND m.duplicate_of IS NULL
            AND (m.order_input_id,m.seq,m.record_position,m.id)<(SELECT order_input_id,seq,record_position,id FROM message WHERE pk=?2)
            AND (?3=0 OR (pl_is_prompt(m.role,m.prompt_kind)))
            AND EXISTS(SELECT 1 FROM block WHERE message_pk=m.pk AND kind IN ({kinds}))
            ORDER BY m.order_input_id DESC,m.seq DESC,m.record_position DESC,m.id DESC LIMIT 1");
        let previous = conn
            .query_row(&sql, params![session, key, prompts], |row| {
                row.get::<_, i64>(0)
            })
            .optional()?;
        previous.map(|previous| {
            let stepped: i64 = conn.query_row("SELECT count(*) FROM message m
                WHERE m.session_pk=?1 AND m.duplicate_of IS NULL
                AND (m.order_input_id,m.seq,m.record_position,m.id)>=(SELECT order_input_id,seq,record_position,id FROM message WHERE pk=?2)
                AND (m.order_input_id,m.seq,m.record_position,m.id)<(SELECT order_input_id,seq,record_position,id FROM message WHERE pk=?3)", params![session,previous,key], |row| row.get(0))?;
            Ok(VisitStart { message: from.message.saturating_sub(message_count(stepped)?), key: Some(previous) })
        }).transpose()
    }

    fn visible_key(
        &self,
        key: i64,
        kinds: &[polylogue_model::BlockKind],
        prompts: bool,
    ) -> Result<bool> {
        let kinds = kinds
            .iter()
            .map(|kind| format!("'{}'", kind.as_str()))
            .collect::<Vec<_>>()
            .join(",");
        Ok(self.reader.conn()?.query_row(&format!("SELECT EXISTS(SELECT 1 FROM message m JOIN block b ON b.message_pk=m.pk WHERE m.pk=?1 AND b.kind IN ({kinds}) AND (?2=0 OR (pl_is_prompt(m.role,m.prompt_kind))))"),params![key,prompts],|row|row.get(0))?)
    }
}
