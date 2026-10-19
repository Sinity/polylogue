//! Reads that context assembly needs: sessions by working directory, a
//! session's own prompts, earlier deliveries and settings.

use crate::read::{SessionSummary, row_summary};
use crate::{Reader, Result};
use polylogue_model::Timestamp;
use rusqlite::params;
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
    /// What the session is waiting on: permission, question, notification.
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
    pub message_id: String,
    pub at: Option<Timestamp>,
    pub text: String,
}

impl Reader<'_> {
    /// Sessions whose working directory is one of `dirs` or below one,
    /// newest activity first, leaving out `exclude`.
    pub fn sessions_under(
        &self,
        dirs: &[String],
        exclude: Option<&str>,
        limit: u64,
    ) -> Result<Vec<SessionSummary>> {
        let mut out: Vec<SessionSummary> = Vec::new();
        let conn = self.conn();
        let mut stmt = conn.prepare_cached(
            "SELECT s.id, s.kind, s.title, s.started_at, s.last_activity_at,
                    (SELECT count(*) FROM message m WHERE m.session_pk = s.pk), s.cwd
               FROM session s
              WHERE s.kind = 'main' AND s.stub = 0
                AND (s.cwd = ?1 OR (s.cwd >= ?1 || '/' AND s.cwd < ?1 || '0'))
                AND s.id IS NOT ?2
              ORDER BY s.last_activity_at DESC
              LIMIT ?3",
        )?;
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        for dir in dirs {
            for row in stmt.query_map(params![dir, exclude, limit], row_summary)? {
                if let Some(s) = row?
                    && !out.iter().any(|o| o.id == s.id)
                {
                    out.push(s);
                }
            }
        }
        out.sort_by_key(|s| std::cmp::Reverse(s.last_activity));
        out.truncate(usize::try_from(limit).unwrap_or(usize::MAX));
        Ok(out)
    }

    /// The repositories recorded for sessions in these directories.
    pub fn repos_under(&self, dirs: &[String]) -> Result<Vec<String>> {
        let conn = self.conn();
        let mut stmt = conn.prepare_cached(
            "SELECT DISTINCT repo FROM session
              WHERE repo IS NOT NULL
                AND (cwd = ?1 OR (cwd >= ?1 || '/' AND cwd < ?1 || '0'))",
        )?;
        let mut out = Vec::new();
        for dir in dirs {
            for r in stmt.query_map([dir], |r| r.get::<_, String>(0))? {
                let r = r?;
                if !out.contains(&r) {
                    out.push(r);
                }
            }
        }
        Ok(out)
    }

    /// Prompts people typed in a session, oldest first (harness text and
    /// slash commands left out).
    pub fn prompts(&self, session: &str) -> Result<Vec<Prompt>> {
        let conn = self.conn();
        let mut stmt = conn.prepare_cached(
            "SELECT m.id, m.at, t.body
               FROM session s
               JOIN message m ON m.session_pk = s.pk
               JOIN block b ON b.message_pk = m.pk AND b.kind = 'text'
               JOIN text t ON t.id = b.text_id
              WHERE s.id = ?1 AND m.written_by = 'human'
                AND coalesce(m.prompt_kind, 'prose') = 'prose'
              ORDER BY m.order_input_pk, m.seq, b.idx",
        )?;
        let rows = stmt.query_map([session], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<i64>>(1)?,
                r.get::<_, Vec<u8>>(2)?,
            ))
        })?;
        let mut out: Vec<Prompt> = Vec::new();
        for row in rows {
            let (message_id, at, body) = row?;
            let text = crate::writer::unpack_text(&body)?;
            match out.last_mut() {
                // Several text blocks of one message are one prompt.
                Some(last) if last.message_id == message_id => {
                    last.text.push('\n');
                    last.text.push_str(&text);
                }
                _ => out.push(Prompt {
                    message_id,
                    at: at.and_then(Timestamp::from_unix_ms),
                    text,
                }),
            }
        }
        Ok(out)
    }

    /// What a session left open; `None` for an unknown session.
    pub fn open_items(&self, session: &str) -> Result<Option<OpenItems>> {
        use rusqlite::OptionalExtension;
        let conn = self.conn();
        let row = conn
            .prepare_cached(
                "SELECT s.live_open_tool_pk IS NOT NULL, s.live_todo, s.live_waiting_on,
                        (SELECT m.written_by FROM message m WHERE m.session_pk = s.pk
                          ORDER BY m.order_input_pk DESC, m.seq DESC LIMIT 1),
                        (SELECT m.prompt_kind FROM message m WHERE m.session_pk = s.pk
                          ORDER BY m.order_input_pk DESC, m.seq DESC LIMIT 1),
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
        let body: Vec<u8> =
            self.conn()
                .query_row("SELECT body FROM a.delivery WHERE id = ?1", [id], |r| {
                    r.get(0)
                })?;
        let text = String::from_utf8(zstd::stream::decode_all(body.as_slice())?)
            .map_err(|e| crate::StoreError::Io(std::io::Error::other(e)))?;
        Ok(Some((row, text)))
    }

    fn deliveries(&self, filter: &str, arg: &str) -> Result<Vec<DeliveryRow>> {
        let conn = self.conn();
        let mut stmt = conn.prepare_cached(&format!(
            "SELECT d.id, d.event, d.sender, d.recipient_id, d.at, d.bytes, d.commit_seq,
                    (SELECT group_concat(e.what, ',') FROM
                        (SELECT what FROM a.delivery_event e WHERE e.delivery = d.id ORDER BY at) e)
               FROM a.delivery d {filter}
              ORDER BY d.commit_seq DESC"
        ))?;
        let rows = stmt.query_map([arg], |r| {
            Ok(DeliveryRow {
                id: r.get(0)?,
                event: r.get(1)?,
                sender: r.get(2)?,
                recipient_id: r.get(3)?,
                at: Timestamp::from_unix_ms(r.get(4)?)
                    .unwrap_or_else(|| Timestamp::from_unix_ms(0).expect("epoch")),
                bytes: u64::try_from(r.get::<_, i64>(5)?).unwrap_or(0),
                seq: r.get(6)?,
                events: r
                    .get::<_, Option<String>>(7)?
                    .map(|s| s.split(',').map(str::to_owned).collect())
                    .unwrap_or_default(),
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// The current value of a setting: this machine's row, else the
    /// archive-wide one.
    pub fn setting(&self, key: &str, machine: &str) -> Result<Option<String>> {
        use rusqlite::OptionalExtension;
        Ok(self
            .conn()
            .query_row(
                "SELECT value FROM a.setting
                  WHERE key = ?1 AND machine IN (?2, '*')
                  ORDER BY machine = '*', set_at DESC, id DESC
                  LIMIT 1",
                params![key, machine],
                |r| r.get(0),
            )
            .optional()?)
    }
}
