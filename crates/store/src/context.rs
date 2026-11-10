//! Reads that context assembly needs beyond the scope index: what a listed
//! session did, whether sessions are hidden, a session's own prompts, what
//! it left open, earlier deliveries and settings.

use crate::encoding::Columns;
use crate::{Reader, Result};
use polylogue_model::Timestamp;

use rusqlite::OptionalExtension;
use serde::Serialize;

/// A delivery as stored, without its text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeliveryRow {
    pub id: String,
    pub event: String,
    pub sender: String,
    pub recipient_id: Option<String>,
    pub at: Timestamp,
    pub bytes: u64,
    /// The archive write counter when it was prepared: notes newer than
    /// this are the delta for the next brief.
    pub seq: i64,
    /// What happened to it, in order: delivered, failed, not_sent, used.
    pub events: Vec<String>,
}

/// What a session left open: the evidence a resumable session is ranked
/// on, rather than how its text ended.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct OpenItems {
    /// The last message was a person's prompt with no answer after it.
    pub unanswered_prompt: bool,
    /// A tool call started and never finished.
    pub open_tool_call: bool,
    /// The harness's in-progress todo item.
    pub todo: Option<String>,
    /// What the session's hooks say it waits on: permission, question, plan, input.
    pub waiting_on: Option<String>,
}

impl OpenItems {
    pub fn any(&self) -> bool {
        self.unanswered_prompt
            || self.open_tool_call
            || self.todo.is_some()
            || self.waiting_on.is_some()
    }
}

/// One prompt a person typed, for re-grounding after compaction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Prompt {
    /// Recorded classification, not independently verified person identity.
    pub written_by: String,
    pub message_id: String,
    pub at: Option<Timestamp>,
    pub text: String,
}

/// What a session did, as the derive step keeps it for briefs: the outcome
/// a brief reports, rather than its title.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct SessionSummary {
    /// The text of its last model message that has text.
    pub last_answer: Option<String>,
    /// How many files it wrote or edited successfully.
    pub files_changed: u64,
    /// The most recently changed of them, newest first.
    pub recent_files: Vec<String>,
    /// Commits (hashes) and pull requests (`owner/repo#n`) its commands made.
    pub commits: Vec<String>,
    pub pull_requests: Vec<String>,
    /// Its last message is a person's prompt with no answer after it.
    pub unanswered_prompt: bool,
    /// Its newest tool call has no result.
    pub unfinished_tool_call: bool,
}

impl Reader<'_> {
    /// What the session with row `pk` did, as the derive step last stored
    /// it; `None` before the derive step has summarized it.
    pub fn session_summary(&self, pk: i64) -> Result<Option<SessionSummary>> {
        let conn = self.conn()?;
        let row = conn
            .prepare_cached(
                "SELECT last_answer_pk, files_changed, recent_files_json, commits_json,
                        pull_requests_json, unanswered_prompt, unfinished_tool_call
                   FROM session_brief WHERE session_pk = ?1",
            )?
            .query_row([pk], |r| {
                Ok((
                    r.get::<_, Option<i64>>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, bool>(5)?,
                    r.get::<_, bool>(6)?,
                ))
            })
            .optional()?;
        let Some((answer, files, recent, commits, pull_requests, unanswered, unfinished)) = row
        else {
            return Ok(None);
        };
        let last_answer = match answer {
            Some(message) => {
                let mut parts = Vec::new();
                let mut stmt = conn.prepare_cached(
                    "SELECT t.body FROM block b JOIN text t ON t.id = b.text_id
                      WHERE b.message_pk = ?1 AND b.kind = 'text' ORDER BY b.idx",
                )?;
                for body in stmt.query_map([message], |r| r.get::<_, Vec<u8>>(0))? {
                    parts.push(crate::encoding::unpack_text(&body?)?);
                }
                let text = parts.join("\n");
                (!text.trim().is_empty()).then_some(text)
            }
            None => None,
        };
        Ok(Some(SessionSummary {
            last_answer,
            files_changed: u64::try_from(files).unwrap_or(0),
            recent_files: serde_json::from_str(&recent)?,
            commits: serde_json::from_str(&commits)?,
            pull_requests: serde_json::from_str(&pull_requests)?,
            unanswered_prompt: unanswered,
            unfinished_tool_call: unfinished,
        }))
    }

    /// Which of these sessions a current `hidden` note hides.
    pub fn hidden_sessions(&self, ids: &[String]) -> Result<Vec<String>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn()?;
        let mut stmt = conn.prepare_cached(&format!(
            "SELECT {} FROM (SELECT ?1 AS id) s",
            crate::query::fields::hidden("s")
        ))?;
        let mut out = Vec::new();
        for id in ids {
            if stmt.query_row([id], |r| r.get::<_, bool>(0))? {
                out.push(id.clone());
            }
        }
        Ok(out)
    }

    /// Prompts people typed in a session, oldest first (harness text and
    /// slash commands left out).
    pub fn prompts(&self, session: &str) -> Result<Vec<Prompt>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT m.id, m.at, t.body
               FROM session s
               JOIN message m ON m.session_pk = s.pk
               JOIN block b ON b.message_pk = m.pk AND b.kind = 'text'
               JOIN text t ON t.id = b.text_id
              WHERE s.id = ?1 AND m.written_by = 'human'
                AND coalesce(m.prompt_kind, 'prose') = 'prose'
              ORDER BY m.order_input_id, m.seq, m.record_position, m.id, b.idx",
        )?;
        let rows = stmt.query_map([session], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.optional_time_ms(1)?,
                r.get::<_, Vec<u8>>(2)?,
            ))
        })?;
        let mut out: Vec<Prompt> = Vec::new();
        for row in rows {
            let (message_id, at, body) = row?;
            let text = crate::encoding::unpack_text(&body)?;
            match out.last_mut() {
                // Several text blocks of one message are one prompt.
                Some(last) if last.message_id == message_id => {
                    last.text.push('\n');
                    last.text.push_str(&text);
                }
                _ => out.push(Prompt {
                    written_by: "human".into(),
                    message_id,
                    at,
                    text,
                }),
            }
        }
        Ok(out)
    }

    /// Latest observed request on the session's own active path. The ordering
    /// lookup uses the existing session-message index; only the selected
    /// message's text blocks are read. Harness injections and commands are not
    /// requests. A user role with unknown authorship stays explicitly unknown.
    pub fn latest_prompt(&self, session: &str) -> Result<Option<Prompt>> {
        if !self.hidden_sessions(&[session.to_owned()])?.is_empty() {
            return Ok(None);
        }
        let conn = self.conn()?;
        let selected = conn.prepare_cached(
            "SELECT m.pk,m.id,m.at,m.written_by FROM session s JOIN message m INDEXED BY message_session ON m.session_pk=s.pk WHERE s.id=?1 AND m.duplicate_of IS NULL AND COALESCE(m.on_active_path,1)=1 AND (m.written_by='human' OR (m.role='user' AND m.written_by='unknown')) AND COALESCE(m.prompt_kind,'prose')='prose' ORDER BY m.order_input_id DESC,m.seq DESC,m.record_position DESC,m.id DESC LIMIT 1"
        )?.query_row([session], |row| Ok((row.get::<_,i64>(0)?,row.get::<_,String>(1)?,row.optional_time_ms(2)?,row.get::<_,String>(3)?))).optional()?;
        let Some((pk, message_id, at, written_by)) = selected else {
            return Ok(None);
        };
        let mut parts = Vec::new();
        let mut statement = conn.prepare_cached("SELECT t.body FROM block b JOIN text t ON t.id=b.text_id WHERE b.message_pk=?1 AND b.kind='text' ORDER BY b.idx")?;
        for body in statement.query_map([pk], |row| row.get::<_, Vec<u8>>(0))? {
            self.check()?;
            parts.push(crate::encoding::unpack_text(&body?)?);
        }
        self.check()?;
        Ok(Some(Prompt {
            written_by,
            message_id,
            at,
            text: parts.join("\n"),
        }))
    }

    /// What a session left open; `None` for an unknown session.
    pub fn open_items(&self, session: &str) -> Result<Option<OpenItems>> {
        let conn = self.conn()?;
        let row = conn
            .prepare_cached(
                "SELECT json_extract(s.live_facts, '$.open_call') IS NOT NULL,
                        json_extract(s.live_facts, '$.todo'),
                        json_extract(s.live_facts, '$.hook_waiting_on.value'),
                        (SELECT m.written_by FROM message m WHERE m.session_pk = s.pk AND m.duplicate_of IS NULL
                          ORDER BY m.order_input_id DESC, m.seq DESC, m.record_position DESC, m.id DESC LIMIT 1),
                        (SELECT m.prompt_kind FROM message m WHERE m.session_pk = s.pk AND m.duplicate_of IS NULL
                          ORDER BY m.order_input_id DESC, m.seq DESC, m.record_position DESC, m.id DESC LIMIT 1),
                        coalesce((SELECT c.result_block_pk IS NULL FROM tool_call c
                                   WHERE c.session_pk = s.pk ORDER BY c.pk DESC LIMIT 1), 0)
                   FROM session s WHERE s.id = ?1",
            )?
            .query_row([session], |r| {
                Ok((
                    r.get::<_, bool>(0)?,
                    r.get::<_, Option<String>>(1)?,
                    r.get::<_, Option<String>>(2)?,
                    r.get::<_, Option<String>>(3)?,
                    r.get::<_, Option<String>>(4)?,
                    r.get::<_, bool>(5)?,
                ))
            })
            .optional()?;
        Ok(row.map(
            |(live_tool, todo, waiting_on, last_by, last_kind, unpaired)| OpenItems {
                unanswered_prompt: last_by.as_deref() == Some("human")
                    && last_kind.as_deref() != Some("command"),
                open_tool_call: live_tool || unpaired,
                todo,
                waiting_on,
            },
        ))
    }

    /// Deliveries prepared for a session, newest first.
    pub fn deliveries_to(&self, session: &str) -> Result<Vec<DeliveryRow>> {
        self.deliveries("WHERE d.recipient_id = ?1", session)
    }

    pub fn delivery(&self, id: &str) -> Result<Option<(DeliveryRow, String)>> {
        let Some(row) = self.deliveries("WHERE d.id = ?1", id)?.pop() else {
            return Ok(None);
        };
        let body: Vec<u8> = self.conn()?.query_row(
            "SELECT body FROM source_delivery WHERE id = ?1",
            [id],
            |r| r.get(0),
        )?;
        let text = crate::encoding::unpack_text(&body)?;
        Ok(Some((row, text)))
    }

    /// Exact references recorded with a delivery, without reading its body.
    pub fn delivery_refs(&self, id: &str) -> Result<Option<Vec<String>>> {
        let refs: Option<String> = self
            .conn()?
            .query_row(
                "SELECT refs_json FROM source_delivery WHERE id=?1",
                [id],
                |row| row.get(0),
            )
            .optional()?;
        refs.map(|refs| serde_json::from_str(&refs).map_err(crate::StoreError::from))
            .transpose()
    }

    fn deliveries(&self, filter: &str, arg: &str) -> Result<Vec<DeliveryRow>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare_cached(&format!(
            "SELECT d.id, d.event, d.sender, d.recipient_id, d.at, d.bytes, d.commit_seq,
                    (SELECT group_concat(e.what, ',') FROM
                        (SELECT what FROM source_delivery_event e WHERE e.delivery = d.id ORDER BY at) e)
               FROM source_delivery d {filter}
              ORDER BY d.commit_seq DESC"
        ))?;
        let rows = stmt.query_map([arg], |r| {
            Ok(DeliveryRow {
                id: r.get(0)?,
                event: r.get(1)?,
                sender: r.get(2)?,
                recipient_id: r.get(3)?,
                at: r.time_ms(4)?,
                bytes: r.get(5)?,
                seq: r.get(6)?,
                events: r
                    .get::<_, Option<String>>(7)?
                    .map(|s| s.split(',').map(str::to_owned).collect())
                    .unwrap_or_default(),
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Pings and notices to `session` with no outcome yet, oldest first,
    /// with their text: what its next context-accepting hook event claims.
    pub fn pending_hook_deliveries(&self, session: &str) -> Result<Vec<(String, String)>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT d.id, d.body FROM source_delivery d
              WHERE d.recipient_id = ?1 AND d.event IN ('ping', 'notice')
                AND NOT EXISTS (SELECT 1 FROM source_delivery_event e WHERE e.delivery = d.id)
              ORDER BY d.at, d.id",
        )?;
        let rows = stmt.query_map([session], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (id, body) = row?;
            out.push((id, crate::encoding::unpack_text(&body)?));
        }
        Ok(out)
    }

    /// The sessions Polylogue started whose first turn (a `pack`
    /// delivery) refers to `reference`, such as `role:reviewer`, newest
    /// first: each with the directory it was started in and when.
    pub fn sessions_started_with(
        &self,
        reference: &str,
    ) -> Result<Vec<(String, Option<String>, i64)>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT d.recipient_id, d.cwd, d.at FROM source_delivery d
              WHERE d.event = 'pack' AND d.recipient_id IS NOT NULL
                AND EXISTS (SELECT 1 FROM json_each(d.refs_json) r WHERE r.value = ?1)
              ORDER BY d.at DESC, d.id DESC",
        )?;
        Ok(stmt
            .query_map([reference], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// The sessions with pings or notices that have no outcome yet.
    pub fn sessions_with_pending_hook_deliveries(&self) -> Result<Vec<String>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT DISTINCT d.recipient_id FROM source_delivery d
              WHERE d.recipient_id IS NOT NULL AND d.event IN ('ping', 'notice')
                AND NOT EXISTS (SELECT 1 FROM source_delivery_event e WHERE e.delivery = d.id)
              ORDER BY d.recipient_id",
        )?;
        Ok(stmt
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// One catalog-checked value. Defaults and primary ownership are resolved
    /// by the declared catalog, rather than by each consumer.
    pub fn setting_value(
        &self,
        key: &str,
        machine: &str,
    ) -> Result<Option<polylogue_model::settings::Value>> {
        let current = self.setting_current(key, machine)?;
        if current.owner == "sinex" {
            return Err(crate::StoreError::Invalid(
                polylogue_model::settings::Error::SinexOwned(key.into()).to_string(),
            ));
        }
        if !current.conflicts.is_empty() {
            return Err(crate::StoreError::Ambiguous {
                reference: format!("setting {key}"),
                candidates: current.conflicts.iter().map(ToString::to_string).collect(),
            });
        }
        Ok(current.value)
    }

    pub fn settings_catalog(
        &self,
        machine: &str,
    ) -> Result<Vec<polylogue_model::settings::Current>> {
        let mut keys: Vec<String> = polylogue_model::settings::CATALOG
            .iter()
            .filter(|spec| !spec.key.ends_with('*'))
            .map(|spec| spec.key.to_string())
            .collect();
        let rows = self
            .conn()?
            .prepare("SELECT DISTINCT key FROM source_assertion WHERE kind='setting' ORDER BY key")?
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for key in rows {
            if polylogue_model::settings::declaration(&key).is_ok() && !keys.contains(&key) {
                keys.push(key);
            }
        }
        keys.sort();
        keys.iter()
            .map(|key| self.setting_current(key, machine))
            .collect()
    }

    pub fn settings_snapshot(&self, machine: &str) -> Result<polylogue_model::settings::Snapshot> {
        Ok(polylogue_model::settings::Snapshot::new(
            self.settings_catalog(machine)?,
        ))
    }

    /// The text of a message's text blocks, joined.
    pub fn message_text(&self, message: &str) -> Result<Option<String>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT t.body FROM message m JOIN block b ON b.message_pk = m.pk AND b.kind = 'text'
               JOIN text t ON t.id = b.text_id
              WHERE m.id = ?1 ORDER BY b.idx",
        )?;
        let mut parts = Vec::new();
        for body in stmt.query_map([message], |r| r.get::<_, Vec<u8>>(0))? {
            parts.push(crate::encoding::unpack_text(&body?)?);
        }
        Ok((!parts.is_empty()).then(|| parts.join("\n")))
    }

    /// Unacknowledged signal metadata in this directory scope or addressed
    /// to this session. No delivery body is loaded until its entry is fitted.
    /// Any outcome excludes a notice, including an uncertain or failed send.
    /// The second value counts records needing scope/effect-state inspection.
    /// NULL scope never proves a global target. Historical signal-effect
    /// outboxes keep attempt/unknown state with their owner; they are not
    /// inferred to be unsent context notices from missing delivery ACKs.
    pub fn pending_notice_candidates(
        &self,
        directories: &[String],
        session: Option<&str>,
    ) -> Result<(Vec<DeliveryRow>, u64)> {
        let selection = serde_json::to_string(
            &serde_json::json!({"directories":directories,"session":session}),
        )?;
        let candidates = self.deliveries(
            "WHERE d.sender='signal' AND d.signal IS NULL AND NOT EXISTS (SELECT 1 FROM source_delivery_event e WHERE e.delivery=d.id) AND (d.recipient_id=json_extract(?1,'$.session') OR (d.recipient_id IS NULL AND d.cwd IS NOT NULL AND (EXISTS (SELECT 1 FROM json_each(?1,'$.directories') scope WHERE (scope.value='/' AND substr(d.cwd,1,1)='/') OR d.cwd=rtrim(scope.value,'/') OR substr(d.cwd,1,length(rtrim(scope.value,'/'))+1)=rtrim(scope.value,'/')||'/'))))",
            &selection,
        )?;
        let inspect_count: i64 = self.conn()?.query_row("SELECT count(*) FROM source_delivery d WHERE d.sender='signal' AND ((d.recipient_id IS NULL AND d.cwd IS NULL) OR d.signal IS NOT NULL) AND NOT EXISTS (SELECT 1 FROM source_delivery_event e WHERE e.delivery=d.id)", [], |row| row.get(0))?;
        self.check()?;
        Ok((
            candidates,
            u64::try_from(inspect_count)
                .map_err(|_| crate::StoreError::Invalid("negative notice count".into()))?,
        ))
    }

    /// Signal notices recorded since `since` (Unix ms), newest first, with
    /// their text and what happened to each.
    pub fn notices_since(&self, since: i64) -> Result<Vec<(DeliveryRow, String)>> {
        let conn = self.conn()?;
        let ids: Vec<String> = conn
            .prepare_cached(
                "SELECT id FROM source_delivery WHERE sender = 'signal' AND at >= ?1
                  ORDER BY at DESC, commit_seq DESC",
            )?
            .query_map([since], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        let mut out = Vec::new();
        for id in ids {
            if let Some(found) = self.delivery(&id)? {
                out.push(found);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
