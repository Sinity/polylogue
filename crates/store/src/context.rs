//! Reads that context assembly needs beyond the scope index: what a listed
//! session did, whether sessions are hidden, a session's own prompts, what
//! it left open, earlier deliveries and settings.

use crate::ConnExt as _;
use crate::encoding::Columns;
use crate::{Reader, Result};
use polylogue_model::Timestamp;

use rusqlite::OptionalExtension;
use serde::Serialize;

/// The bodies of a message's text blocks in order; `check` runs once per
/// block so a caller can cancel a long read.
fn message_text_parts(
    conn: &rusqlite::Connection,
    message_pk: i64,
    mut check: impl FnMut() -> Result<()>,
) -> Result<Vec<String>> {
    let mut stmt = conn.prepare_cached(
        "SELECT g.body, t.start, t.length FROM block b JOIN text t ON t.id = b.text_id
           JOIN content_group g ON g.id = t.group_id
          WHERE b.message_pk = ?1 AND b.kind = 'text' ORDER BY b.position",
    )?;
    let mut parts = Vec::new();
    for text in stmt.query_and_then([message_pk], |r| crate::text_rows::joined(r, 0))? {
        check()?;
        parts.push(text?);
    }
    Ok(parts)
}

/// A delivery as stored, without its text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeliveryRow {
    pub id: String,
    pub event: polylogue_model::DeliveryEvent,
    pub sender: polylogue_model::DeliverySender,
    pub recipient_id: Option<String>,
    pub at: Timestamp,
    pub bytes: u64,
    /// The archive write counter when it was prepared: notes newer than
    /// this are the delta for the next brief.
    pub seq: i64,
    /// What happened to it, in order: delivered, failed, not_sent, used.
    pub events: Vec<String>,
}

/// What a session left open: what a resumable session is ranked
/// on, rather than how its text ended.
#[derive(
    Debug, Clone, Default, PartialEq, Eq, Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
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

    /// What is open, one phrase per item; every surface words it the same way.
    pub fn phrases(&self) -> Vec<String> {
        let mut open = Vec::new();
        if self.unanswered_prompt {
            open.push("the last prompt has no answer".to_owned());
        }
        if self.open_tool_call {
            open.push("a tool call never finished".to_owned());
        }
        if let Some(todo) = &self.todo {
            open.push(format!("todo in progress: {todo}"));
        }
        if let Some(waiting) = &self.waiting_on {
            open.push(format!("waiting on {waiting}"));
        }
        open
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

/// How many changed files a brief names; the rest are counted.
const RECENT_FILES: usize = 4;

/// What a session did, for briefs: the outcome a brief reports, rather than
/// its title. Composed when read, from the session's rows.
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
    /// What the session with row `pk` did; `None` for no such session.
    pub fn session_summary(&self, pk: i64) -> Result<Option<SessionSummary>> {
        let conn = self.conn()?;
        let exists: bool = conn
            .prepare_cached("SELECT EXISTS(SELECT 1 FROM session WHERE pk = ?1)")?
            .query_row([pk], |r| r.get(0))?;
        if !exists {
            return Ok(None);
        }
        let last_answer: Option<i64> = conn
            .prepare_cached(
                "SELECT m.pk FROM message m
                  WHERE m.session_pk = ?1 AND m.written_by = 'model' AND m.duplicate_of IS NULL
                    AND EXISTS (SELECT 1 FROM block b WHERE b.message_pk = m.pk AND b.kind = 'text')
                  ORDER BY m.session_order DESC, m.id DESC
                  LIMIT 1",
            )?
            .query_row([pk], |r| r.get(0))
            .optional()?;
        let files: Vec<String> = conn
            .prepare_cached(
                "SELECT path FROM tool_call
                  WHERE session_pk = ?1 AND action IN ('file_write', 'file_edit')
                    AND outcome = 'ok' AND path IS NOT NULL
                  GROUP BY path ORDER BY max(at) DESC, path",
            )?
            .query_map([pk], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        let commits: Vec<String> = conn
            .prepare_cached(
                "SELECT commit_hash FROM commit_link WHERE session_pk = ?1 ORDER BY commit_hash",
            )?
            .query_map([pk], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        let pull_requests: Vec<String> = conn
            .prepare_cached(
                "SELECT repo, number FROM pull_request_link WHERE session_pk = ?1 ORDER BY repo, number",
            )?
            .query_map([pk], |r| {
                Ok(format!(
                    "{}#{}",
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?
                ))
            })?
            .collect::<rusqlite::Result<_>>()?;
        let (last_by, last_kind): (Option<String>, Option<String>) = conn
            .prepare_cached(
                "SELECT m.written_by, m.prompt_kind FROM message m
                  WHERE m.session_pk = ?1 AND m.duplicate_of IS NULL
                  ORDER BY m.session_order DESC, m.id DESC
                  LIMIT 1",
            )?
            .query_row([pk], |r| <(_, _)>::try_from(r))
            .optional()?
            .unwrap_or_default();
        let unfinished_tool_call: bool = conn
            .prepare_cached(
                "SELECT coalesce((SELECT c.result_message_id IS NULL FROM tool_call c
                                   WHERE c.session_pk = ?1 ORDER BY c.pk DESC LIMIT 1), 0)",
            )?
            .query_row([pk], |r| r.get(0))?;
        let last_answer = match last_answer {
            Some(message) => {
                let text = message_text_parts(conn, message, || Ok(()))?.join("\n");
                (!text.trim().is_empty()).then_some(text)
            }
            None => None,
        };
        Ok(Some(SessionSummary {
            last_answer,
            files_changed: files.len() as u64,
            recent_files: files.into_iter().take(RECENT_FILES).collect(),
            commits,
            pull_requests,
            unanswered_prompt: last_by.as_deref() == Some("human")
                && last_kind.as_deref() != Some("command"),
            unfinished_tool_call,
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
            "SELECT m.id, m.at, g.body, t.start, t.length
               FROM session s
               JOIN message m ON m.session_pk = s.pk
               JOIN block b ON b.message_pk = m.pk AND b.kind = 'text'
               JOIN text t ON t.id = b.text_id
               JOIN content_group g ON g.id = t.group_id
              WHERE s.id = ?1 AND m.written_by = 'human'
                AND coalesce(m.prompt_kind, 'prose') = 'prose'
              ORDER BY m.session_order, m.id, b.position",
        )?;
        let rows = stmt.query_and_then([session], |r| {
            Ok::<_, crate::StoreError>((
                r.get::<_, String>(0)?,
                r.optional_time_ms(1)?,
                crate::text_rows::joined(r, 2)?,
            ))
        })?;
        let mut out: Vec<Prompt> = Vec::new();
        for row in rows {
            let (message_id, at, text) = row?;
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
            "SELECT m.pk,m.id,m.at,m.written_by FROM session s JOIN message m INDEXED BY message_session ON m.session_pk=s.pk WHERE s.id=?1 AND m.duplicate_of IS NULL AND COALESCE(m.on_active_path,1)=1 AND (m.written_by='human' OR (m.role='user' AND m.written_by='unknown')) AND COALESCE(m.prompt_kind,'prose')='prose' ORDER BY m.session_order DESC,m.id DESC LIMIT 1"
        )?.query_row([session], |row| Ok((row.get::<_,i64>(0)?,row.get::<_,String>(1)?,row.optional_time_ms(2)?,row.get::<_,String>(3)?))).optional()?;
        let Some((pk, message_id, at, written_by)) = selected else {
            return Ok(None);
        };
        let parts = message_text_parts(conn, pk, || self.check())?;
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
                          ORDER BY m.session_order DESC, m.id DESC LIMIT 1),
                        (SELECT m.prompt_kind FROM message m WHERE m.session_pk = s.pk AND m.duplicate_of IS NULL
                          ORDER BY m.session_order DESC, m.id DESC LIMIT 1),
                        coalesce((SELECT c.result_message_id IS NULL FROM tool_call c
                                   WHERE c.session_pk = s.pk ORDER BY c.pk DESC LIMIT 1), 0)
                   FROM session s WHERE s.id = ?1",
            )?
            .query_row([session], |r| <(bool, Option<String>, Option<String>, Option<String>, Option<String>, bool)>::try_from(r))
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
        let body: Vec<u8> = self
            .conn()?
            .scalar("SELECT body FROM source_delivery WHERE id = ?1", [id])?;
        let text = crate::encoding::unpack_text(&body)?;
        Ok(Some((row, text)))
    }

    /// The references recorded with a delivery, without reading its body.
    pub fn delivery_refs(&self, id: &str) -> Result<Option<Vec<String>>> {
        let refs: Option<String> = self
            .conn()?
            .scalar_optional("SELECT refs_json FROM source_delivery WHERE id=?1", [id])?;
        refs.map(|refs| serde_json::from_str(&refs).map_err(crate::StoreError::from))
            .transpose()
    }

    fn deliveries(&self, filter: &str, arg: &str) -> Result<Vec<DeliveryRow>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare_cached(&format!(
            "SELECT d.id, d.event, d.sender, d.recipient_id, d.at, d.bytes, d.commit_seq,
                    (SELECT group_concat(e.outcome, ',') FROM
                        (SELECT outcome FROM source_delivery_event e WHERE e.delivery = d.id ORDER BY at) e)
               FROM source_delivery d {filter}
              ORDER BY d.commit_seq DESC"
        ))?;
        let rows = stmt.query_map([arg], |r| {
            Ok(DeliveryRow {
                id: r.get(0)?,
                event: r.name(1)?,
                sender: r.name(2)?,
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

    /// Messages and notices to `session` with no outcome yet, oldest first,
    /// with their text: what its next context-accepting hook event takes.
    pub fn pending_hook_deliveries(&self, session: &str) -> Result<Vec<(String, String)>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT d.id, d.body FROM source_delivery d
              WHERE d.recipient_id = ?1 AND d.event IN ('message', 'notice')
                AND NOT EXISTS (SELECT 1 FROM source_delivery_event e WHERE e.delivery = d.id)
              ORDER BY d.at, d.id",
        )?;
        let rows = stmt.query_map([session], |r| <(String, Vec<u8>)>::try_from(r))?;
        let mut out = Vec::new();
        for row in rows {
            let (id, body) = row?;
            out.push((id, crate::encoding::unpack_text(&body)?));
        }
        Ok(out)
    }

    /// The sessions Polylogue started whose first turn (a `first_turn`
    /// delivery) refers to `reference`, such as `role:reviewer`, newest
    /// first: each with the directory it was started in and when.
    pub fn sessions_started_with(
        &self,
        reference: &str,
    ) -> Result<Vec<(String, Option<String>, i64)>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT d.recipient_id, d.cwd, d.at FROM source_delivery d
              WHERE d.event = 'first_turn' AND d.recipient_id IS NOT NULL
                AND EXISTS (SELECT 1 FROM json_each(d.refs_json) r WHERE r.value = ?1)
              ORDER BY d.at DESC, d.id DESC",
        )?;
        Ok(stmt
            .query_map([reference], |r| <(_, _, _)>::try_from(r))?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// The sessions with messages or notices that have no outcome yet.
    pub fn sessions_with_pending_hook_deliveries(&self) -> Result<Vec<String>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT DISTINCT d.recipient_id FROM source_delivery d
              WHERE d.recipient_id IS NOT NULL AND d.event IN ('message', 'notice')
                AND NOT EXISTS (SELECT 1 FROM source_delivery_event e WHERE e.delivery = d.id)
              ORDER BY d.recipient_id",
        )?;
        Ok(stmt
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// One catalog-checked value. Defaults and the mode a setting applies in are resolved
    /// by the declared catalog, rather than by each consumer.
    pub fn setting_value(
        &self,
        key: &str,
        machine: &str,
    ) -> Result<Option<polylogue_model::settings::Value>> {
        let current = self.setting_current(key, machine)?;
        if current.supplier == "sinex" {
            return Err(polylogue_model::Reason::SettingOwnedBySinex { key: key.into() }.into());
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
        let stored_keys = self
            .conn()?
            .prepare(
                "SELECT DISTINCT key FROM source_annotation WHERE kind='setting' ORDER BY key",
            )?
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for key in stored_keys {
            if polylogue_model::settings::declaration(&key).is_ok()
                && polylogue_model::settings::historical_hint(&key).is_none()
                && !keys.contains(&key)
            {
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
        let Some(pk) =
            conn.scalar_optional::<i64>("SELECT pk FROM message WHERE id = ?1", [message])?
        else {
            return Ok(None);
        };
        let parts = message_text_parts(conn, pk, || Ok(()))?;
        Ok((!parts.is_empty()).then(|| parts.join("\n")))
    }

    /// Unacknowledged signal metadata in this directory scope or addressed
    /// to this session. No delivery body is loaded until its entry is fitted.
    /// Any outcome excludes a notice, including an uncertain or failed send.
    /// The second value counts records needing scope/effect-state inspection.
    /// NULL scope never proves a global target. Pending deliveries of
    /// older signal effects keep their attempt/unknown state; they are not
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
        let inspect_count: i64 = self.conn()?.scalar("SELECT count(*) FROM source_delivery d WHERE d.sender='signal' AND ((d.recipient_id IS NULL AND d.cwd IS NULL) OR d.signal IS NOT NULL) AND NOT EXISTS (SELECT 1 FROM source_delivery_event e WHERE e.delivery=d.id)", [])?;
        self.check()?;
        Ok((
            candidates,
            u64::try_from(inspect_count)
                .map_err(|_| crate::StoreError::stored(0, "negative inspection count"))?,
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
