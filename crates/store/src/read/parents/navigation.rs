//! Metadata-only positions and outline entries for the shared reader.
use super::*;

#[derive(Debug, Clone, Serialize)]
pub struct OutlineEntry {
    pub message: String,
    pub session: String,
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
        let sql=format!("SELECT m.pk,m.id FROM message m WHERE m.session_pk=?1 AND m.duplicate_of IS NULL
            AND (?2=0 OR (m.role='user' AND (m.prompt_kind IS NULL OR m.prompt_kind NOT IN ('harness','model'))))
            AND EXISTS(SELECT 1 FROM block b WHERE b.message_pk=m.pk AND b.kind IN ({kinds}))
            ORDER BY m.order_input_id DESC,m.seq DESC,m.record_position DESC,m.id DESC");
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

    /// The continuation counts outline entries, not transcript messages.
    /// Each inherited session chunk retains the selected ancestry order.
    pub fn outline(&self, start: usize, size: usize) -> Result<Vec<OutlineEntry>> {
        let conn = self.reader.conn()?;
        let mut entries = Vec::new();
        let mut skipped = 0;
        for (session, members) in self.sessions() {
            let members = members.map(|keys| keys.into_iter().collect::<HashSet<_>>());
            let mut statement=conn.prepare_cached("SELECT m.pk,m.id,s.id,m.at,m.written_by='human',
                EXISTS(SELECT 1 FROM block b JOIN tool_call t ON t.call_block_pk=b.pk WHERE b.message_pk=m.pk AND t.outcome='error'),
                m.id IN (SELECT branch_point FROM link WHERE branch_point IS NOT NULL)
                FROM message m JOIN session s ON s.pk=m.session_pk
                WHERE m.session_pk=?1 AND m.duplicate_of IS NULL AND
                (m.written_by='human' OR EXISTS(SELECT 1 FROM block b JOIN tool_call t ON t.call_block_pk=b.pk WHERE b.message_pk=m.pk AND t.outcome='error')
                 OR m.id IN (SELECT branch_point FROM link WHERE branch_point IS NOT NULL)
                 OR EXISTS(SELECT 1 FROM event e WHERE e.after_message_pk=m.pk AND e.kind LIKE '%compact%'))
                ORDER BY m.order_input_id,m.seq,m.record_position,m.id")?;
            let mut rows = statement.query([session])?;
            while let Some(row) = rows.next()? {
                let pk: i64 = row.get(0)?;
                if members
                    .as_ref()
                    .is_some_and(|members| !members.contains(&pk))
                {
                    continue;
                }
                if skipped < start {
                    skipped += 1;
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
                let events=conn.prepare_cached("SELECT id FROM event WHERE after_message_pk=?1 AND kind LIKE '%compact%' ORDER BY seq,id")?.query_map([pk],|r|r.get::<_,String>(0))?.collect::<rusqlite::Result<Vec<_>>>()?;
                if !events.is_empty() {
                    kinds.push("compaction".into());
                }
                entries.push(OutlineEntry {
                    message: row.get(1)?,
                    session: row.get(2)?,
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
}
