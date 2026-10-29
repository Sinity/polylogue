//! Reads that context assembly needs: sessions by working directory, what
//! each did, open Beads and failing commands there, a session's own
//! prompts, earlier deliveries and settings.

use crate::read::{parse_name, ts};
use crate::{Reader, Result};
use polylogue_model::{SessionId, SessionKind, Timestamp};
use rusqlite::params;
use serde::Serialize;

/// A session that worked in or below a directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NearbySession {
    pub id: SessionId,
    pub kind: SessionKind,
    pub title: Option<String>,
    pub started: Option<Timestamp>,
    pub last_activity: Option<Timestamp>,
    pub messages: u64,
    pub cwd: Option<String>,
}

/// A nearby session from a row of (id, kind, title, started_at,
/// last_activity_at, message count, cwd). A row whose id does not parse is
/// `None`.
fn nearby_session(r: &rusqlite::Row<'_>) -> rusqlite::Result<Option<NearbySession>> {
    let Ok(id) = SessionId::parse(&r.get::<_, String>(0)?) else {
        return Ok(None);
    };
    Ok(Some(NearbySession {
        id,
        kind: parse_name(&r.get::<_, String>(1)?).unwrap_or(SessionKind::Main),
        title: r.get(2)?,
        started: ts(r.get(3)?),
        last_activity: ts(r.get(4)?),
        messages: u64::try_from(r.get::<_, i64>(5)?).unwrap_or(0),
        cwd: r.get(6)?,
    }))
}

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

/// What a session did, from its parsed rows and work graph: the outcome a
/// brief reports, rather than its title.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct SessionWork {
    /// The text of its last model message.
    pub last_answer: Option<String>,
    /// Files it wrote or edited successfully, most recent first.
    pub files_changed: Vec<String>,
    /// Commits (short hashes) and pull requests (`owner/repo#n`) its
    /// commands' output names.
    pub commits: Vec<String>,
    pub pull_requests: Vec<String>,
    /// Test, build and lint commands that passed and failed.
    pub checks_passed: u64,
    pub checks_failed: u64,
    /// Beads it claimed, each with the state its own commands left:
    /// `open`, `closed` or `released`.
    pub beads: Vec<(String, String)>,
    /// Its todo or plan items: how many are completed, and the others, in
    /// progress first.
    pub plan_done: u64,
    pub plan_open: Vec<String>,
}

/// A verification that failed under a directory since its last pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FailingCommand {
    pub command: String,
    pub latest: Option<String>,
    pub exit_code: Option<i64>,
    pub at: Option<Timestamp>,
    pub session: String,
    pub times: u64,
    pub sessions: u64,
    pub first: Option<Timestamp>,
    /// The latest failure's source location and time.
    pub state: String,
}

/// A Bead whose latest claim under a directory is still open.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OpenBead {
    pub id: String,
    pub session: String,
    pub since: Option<Timestamp>,
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
    ) -> Result<Vec<NearbySession>> {
        let mut out: Vec<NearbySession> = Vec::new();
        let conn = self.conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT s.id, s.kind, coalesce(s.title, s.derived_title), s.started_at, s.last_activity_at,
                    (SELECT count(*) FROM message m WHERE m.session_pk = s.pk AND m.duplicate_of IS NULL), s.cwd
               FROM session s
              WHERE s.kind = 'main' AND s.stub = 0
                AND (s.cwd = ?1 OR (s.cwd >= ?1 || '/' AND s.cwd < ?1 || '0'))
                AND s.id IS NOT ?2
              ORDER BY s.last_activity_at DESC
              LIMIT ?3",
        )?;
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        for dir in dirs {
            for row in stmt.query_map(params![dir, exclude, limit], nearby_session)? {
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
        let conn = self.conn()?;
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
                r.get::<_, Option<i64>>(1)?,
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
        let conn = self.conn()?;
        let row = conn
            .prepare_cached(
                "SELECT s.live_open_tool_pk IS NOT NULL, s.live_todo, s.live_waiting_on,
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
            .conn()?
            .query_row(
                "SELECT value FROM source_setting
                  WHERE key = ?1 AND machine IN (?2, '*')
                  ORDER BY machine = '*', set_at DESC, id DESC
                  LIMIT 1",
                params![key, machine],
                |r| r.get(0),
            )
            .optional()?)
    }

    /// One catalog-checked value. Defaults and primary ownership are resolved
    /// by the declared catalog, rather than by each consumer.
    pub fn setting_value(
        &self,
        key: &str,
        machine: &str,
    ) -> Result<Option<polylogue_model::settings::Value>> {
        Ok(self.setting_current(key, machine)?.value)
    }

    pub fn setting_current(
        &self,
        key: &str,
        machine: &str,
    ) -> Result<polylogue_model::settings::Current> {
        use polylogue_model::settings::{self, Owner};
        use rusqlite::OptionalExtension;
        let spec = settings::declaration(key)
            .map_err(|error| crate::StoreError::Invalid(error.to_string()))?;
        let primary = matches!(
            crate::source::Source::of(self.conn()?)?,
            crate::source::Source::Cache
        );
        if primary && spec.standalone_only {
            return Ok(settings::Current {
                key: key.into(),
                kind: spec.kind,
                scope: spec.scope,
                owner: "sinex".into(),
                default: None,
                value: None,
                authored: None,
            });
        }
        let row = self.conn()?.query_row("SELECT id,machine,value,author,set_at FROM source_setting WHERE key=?1 AND machine IN (?2,'*') ORDER BY machine='*',set_at DESC,id DESC LIMIT 1", params![key,machine], |row| Ok((row.get::<_,String>(0)?,row.get::<_,String>(1)?,row.get::<_,String>(2)?,row.get::<_,String>(3)?,row.get::<_,i64>(4)?))).optional()?;
        let owner = if primary {
            Owner::Primary
        } else {
            Owner::Standalone
        };
        let value = settings::current(key, row.as_ref().map(|row| row.2.as_str()), owner)
            .map_err(|error| crate::StoreError::Invalid(error.to_string()))?;
        let default = spec
            .default
            .map(|value| settings::parse(key, value, Owner::Standalone))
            .transpose()
            .map_err(|error| crate::StoreError::Invalid(error.to_string()))?;
        let authored = row
            .map(
                |(id, machine, _, author, at)| -> Result<polylogue_model::settings::Authored> {
                    Ok(settings::Authored {
                        id,
                        machine,
                        author: parse_name(&author).ok_or_else(|| {
                            crate::StoreError::Invalid("invalid setting author".into())
                        })?,
                        at: Timestamp::from_unix_ms(at).ok_or_else(|| {
                            crate::StoreError::Invalid("invalid setting timestamp".into())
                        })?,
                    })
                },
            )
            .transpose()?;
        Ok(settings::Current {
            key: key.into(),
            kind: spec.kind,
            scope: spec.scope,
            owner: "polylogue".into(),
            default,
            value,
            authored,
        })
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
            .prepare("SELECT DISTINCT key FROM source_setting ORDER BY key")?
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

    /// What a session did; `None` when there is no such session.
    pub fn session_work(&self, session: &str) -> Result<Option<SessionWork>> {
        use rusqlite::OptionalExtension;
        let conn = self.conn()?;
        let Some(pk) = conn
            .prepare_cached("SELECT pk FROM session WHERE id = ?1")?
            .query_row([session], |r| r.get::<_, i64>(0))
            .optional()?
        else {
            return Ok(None);
        };
        let mut work = SessionWork::default();
        let last: Option<i64> = conn
            .prepare_cached(
                "SELECT m.pk FROM message m
                  WHERE m.session_pk = ?1 AND m.written_by = 'model' AND m.duplicate_of IS NULL
                    AND EXISTS (SELECT 1 FROM block b WHERE b.message_pk = m.pk AND b.kind = 'text')
                  ORDER BY m.order_input_id DESC, m.seq DESC, m.record_position DESC, m.id DESC
                  LIMIT 1",
            )?
            .query_row([pk], |r| r.get(0))
            .optional()?;
        if let Some(message) = last {
            let mut parts = Vec::new();
            let mut stmt = conn.prepare_cached(
                "SELECT t.body FROM block b JOIN text t ON t.id = b.text_id
                  WHERE b.message_pk = ?1 AND b.kind = 'text' ORDER BY b.idx",
            )?;
            for body in stmt.query_map([message], |r| r.get::<_, Vec<u8>>(0))? {
                parts.push(crate::encoding::unpack_text(&body?)?);
            }
            let text = parts.join("\n");
            work.last_answer = (!text.trim().is_empty()).then_some(text);
        }
        let mut stmt = conn.prepare_cached(
            "SELECT path FROM tool_call
              WHERE session_pk = ?1 AND action IN ('file_write', 'file_edit')
                AND outcome = 'ok' AND path IS NOT NULL
              GROUP BY path ORDER BY max(at) DESC, path",
        )?;
        for path in stmt.query_map([pk], |r| r.get::<_, String>(0))? {
            work.files_changed.push(path?);
        }
        let mut stmt = conn.prepare_cached(
            "SELECT key, title FROM work_node
              WHERE session_pk = ?1 AND kind = 'artifact' ORDER BY at, pk",
        )?;
        for row in stmt.query_map([pk], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })? {
            let (key, title) = row?;
            if key.starts_with("commit:") {
                work.commits.push(title.chars().take(10).collect());
            } else {
                work.pull_requests.push(title);
            }
        }
        let mut stmt = conn.prepare_cached(
            "SELECT state, count(*) FROM work_node
              WHERE session_pk = ?1 AND kind = 'verification' GROUP BY state",
        )?;
        for row in stmt.query_map([pk], |r| {
            Ok((r.get::<_, Option<String>>(0)?, r.get::<_, i64>(1)?))
        })? {
            let (state, n) = row?;
            let n = u64::try_from(n).unwrap_or(0);
            match state.as_deref() {
                Some("ok") => work.checks_passed += n,
                Some("error") => work.checks_failed += n,
                _ => {}
            }
        }
        let mut stmt = conn.prepare_cached(
            "SELECT g.key, g.title, coalesce(a.state, '') FROM work_node a
               JOIN work_edge e ON e.from_pk = a.pk AND e.kind = 'pursues'
               JOIN work_node g ON g.pk = e.to_pk
              WHERE a.session_pk = ?1 AND a.kind = 'attempt'
              ORDER BY a.at, a.pk",
        )?;
        let mut in_progress = Vec::new();
        let mut pending = Vec::new();
        for row in stmt.query_map([pk], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })? {
            let (key, title, state) = row?;
            if key.starts_with("bead:") {
                work.beads.push((title, state));
            } else if key.starts_with("todo:") {
                match state.as_str() {
                    "completed" => work.plan_done += 1,
                    "open" => in_progress.push(title),
                    _ => pending.push(title),
                }
            }
        }
        in_progress.extend(pending);
        work.plan_open = in_progress;
        Ok(Some(work))
    }

    /// Verifications that failed since `since` under `dirs`, counting only
    /// failures after the last pass of the same verification there.
    /// A title keeps advice on the verification index's title and time range.
    pub fn failing_commands(
        &self,
        dirs: &[String],
        since: i64,
        title: Option<&str>,
    ) -> Result<Vec<FailingCommand>> {
        let conn = self.conn()?;
        let title_condition = if title.is_some() {
            "AND v.title = ?3"
        } else {
            ""
        };
        let sql = format!(
            "SELECT v.title, t.command, t.exit_code, v.at, s.id, v.state,
                    json_array(s.id, m.id, b.idx, v.at)
               FROM work_node v JOIN session s ON s.pk = v.session_pk
               LEFT JOIN tool_call t ON t.pk = v.tool_call_pk
               LEFT JOIN block b ON b.pk = t.call_block_pk
               LEFT JOIN message m ON m.pk = b.message_pk
              WHERE v.kind = 'verification' AND v.at >= ?2
                {title_condition}
                AND EXISTS (SELECT 1 FROM json_each(?1) d WHERE s.cwd = d.value
                            OR (s.cwd >= d.value || '/' AND s.cwd < d.value || '0'))
              ORDER BY v.at DESC, s.id DESC, m.seq DESC, m.id DESC, b.idx DESC",
        );
        let mut stmt = conn.prepare_cached(&sql)?;
        let mut passed = std::collections::HashSet::new();
        let mut sessions =
            std::collections::HashMap::<String, std::collections::HashSet<String>>::new();
        let mut out: Vec<FailingCommand> = Vec::new();
        let read = |r: &rusqlite::Row<'_>| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<String>>(1)?,
                r.get::<_, Option<i64>>(2)?,
                r.get::<_, Option<i64>>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, String>(5)?,
                r.get::<_, String>(6)?,
            ))
        };
        let dirs = serde_json::to_string(dirs)?;
        let rows = match title {
            Some(title) => stmt.query_map(params![dirs, since, title], read)?,
            None => stmt.query_map(params![dirs, since], read)?,
        };
        for row in rows {
            let (command, latest, exit_code, at, session, state, key) = row?;
            if state == "ok" {
                passed.insert(command);
                continue;
            }
            if state != "error" || passed.contains(&command) {
                continue;
            }
            let seen = sessions.entry(command.clone()).or_default();
            seen.insert(session.clone());
            let index = match out.iter().position(|f| f.command == command) {
                Some(i) => i,
                None => {
                    out.push(FailingCommand {
                        command,
                        latest,
                        exit_code,
                        at: ts(at),
                        session,
                        times: 0,
                        sessions: 0,
                        first: None,
                        state: key,
                    });
                    out.len() - 1
                }
            };
            let f = &mut out[index];
            f.times += 1;
            f.sessions = seen.len() as u64;
            f.first = ts(at);
        }
        Ok(out)
    }

    /// Beads claimed since `since` whose latest claim by a session under
    /// `dirs` is still open in that session, most recent first.
    pub fn open_beads(&self, dirs: &[String], since: i64) -> Result<Vec<OpenBead>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT g.title, a.state, a.at, s.id FROM work_node a
               JOIN work_edge e ON e.from_pk = a.pk AND e.kind = 'pursues'
               JOIN work_node g ON g.pk = e.to_pk
               JOIN session s ON s.pk = a.session_pk
              WHERE a.kind = 'attempt' AND g.key >= 'bead:' AND g.key < 'bead;' AND a.at >= ?2
                AND EXISTS (SELECT 1 FROM json_each(?1) d WHERE s.cwd = d.value
                            OR (s.cwd >= d.value || '/' AND s.cwd < d.value || '0'))
              ORDER BY a.at IS NULL, a.at DESC, a.pk DESC",
        )?;
        let dirs = serde_json::to_string(dirs)?;
        let mut seen: Vec<String> = Vec::new();
        let mut out = Vec::new();
        for row in stmt.query_map(params![dirs, since], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<String>>(1)?,
                r.get::<_, Option<i64>>(2)?,
                r.get::<_, String>(3)?,
            ))
        })? {
            let (id, state, at, session) = row?;
            // A claim names a Bead by its ID; a shell variable or other
            // text in its place names none.
            let plain = id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "-._".contains(c));
            if !plain || seen.contains(&id) {
                continue;
            }
            seen.push(id.clone());
            if state.as_deref() == Some("open") {
                out.push(OpenBead {
                    id,
                    session,
                    since: ts(at),
                });
            }
        }
        Ok(out)
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
