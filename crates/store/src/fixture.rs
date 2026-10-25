//! Archives for tests, enabled with the `fixture` feature; never part of
//! the product.
//!
//! - [`Fixture`]: a small archive written row by row, for tests of what
//!   reads it (the query language, rendering, the daemon). Rows go straight
//!   into the tables, the way the writer leaves them, so a test states the
//!   facts it needs without a parser or an input file.
//! - [`writer::WriterWorld`]: the real writer over a temporary archive, for
//!   tests of what the writer and the derive step make of parsed records.
//! - [`text_search`]: whether a text is still stored or indexed anywhere.

pub mod text_search;
pub mod writer;

use crate::archive::{Archive, ArchiveLock, open_write};
use crate::{ReadPool, Result};
use rusqlite::{Connection, params};
use std::path::Path;

pub struct Fixture {
    pub archive: Archive,
    _lock: ArchiveLock,
    index: Connection,
    notes: Connection,
    input_pk: i64,
    seq: i64,
    commit_seq: i64,
}

/// A session's facts; everything not given is unknown.
#[derive(Debug, Clone, Default)]
pub struct SessionFacts<'a> {
    pub id: &'a str,
    pub kind: &'a str,
    pub title: Option<&'a str>,
    pub cwd: Option<&'a str>,
    pub repo: Option<&'a str>,
    pub started: Option<i64>,
    pub last: Option<i64>,
}

/// A message's facts.
#[derive(Debug, Clone, Default)]
pub struct MessageFacts<'a> {
    pub id: &'a str,
    pub role: &'a str,
    pub written_by: &'a str,
    pub prompt_kind: Option<&'a str>,
    pub at: Option<i64>,
    pub model: Option<&'a str>,
    pub text: Option<&'a str>,
    pub thinking: Option<&'a str>,
    pub cost_nusd: Option<i64>,
    pub inherited_from: Option<&'a str>,
    pub on_active_path: Option<bool>,
}

/// A tool call's facts, with its result when `outcome` is not `unknown`.
#[derive(Debug, Clone, Default)]
pub struct CallFacts<'a> {
    pub tool: &'a str,
    pub action: &'a str,
    pub command: Option<&'a str>,
    pub path: Option<&'a str>,
    pub outcome: &'a str,
    pub exit_code: Option<i64>,
    pub at: Option<i64>,
    pub output: Option<&'a str>,
}

impl Fixture {
    pub fn new(root: &Path) -> Result<Self> {
        let (archive, lock) = Archive::open_or_create(root)?;
        let index = open_write(&archive.index_db(), crate::archive::cache::SETUP)?;
        let notes = open_write(&archive.archive_db(), crate::archive::cache::SETUP)?;
        index.execute("INSERT INTO input_ref(id) VALUES ('fixture')", [])?;
        let input_pk = index.last_insert_rowid();
        Ok(Self {
            archive,
            _lock: lock,
            index,
            notes,
            input_pk,
            seq: 0,
            commit_seq: 0,
        })
    }

    /// Queues search work without committing it, for reader and index tests.
    pub fn queue_search_text(&self, body: &str) -> Result<i64> {
        let tx = self.index.unchecked_transaction()?;
        let id = crate::text_rows::intern(&tx, body, 1, None, false)?;
        tx.commit()?;
        Ok(id)
    }

    /// Commits queued search work through the normal bounded indexer.
    pub fn commit_search(&self) -> Result<u64> {
        crate::text_rows::index_pending(&self.index)
    }

    /// Runs all derivations over the raw rows written so far.
    pub fn derive(&mut self) -> Result<u64> {
        crate::derive::run(&mut self.index)
    }

    pub fn reads(&self) -> Result<ReadPool> {
        ReadPool::open(&self.archive)
    }

    pub fn session(&mut self, s: SessionFacts<'_>) -> Result<i64> {
        let (origin, native) = s.id.split_once(':').unwrap_or(("test", s.id));
        let kind = if s.kind.is_empty() { "main" } else { s.kind };
        self.index.execute(
            "INSERT INTO session(id, origin, native_id, kind, title, cwd, cwd_json, repo, record_repo,
                 started_at, last_activity_at, changed)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8, ?9, ?10, 1)",
            params![
                s.id,
                origin,
                native,
                kind,
                s.title,
                s.cwd,
                s.cwd.map(|c| serde_json::json!([c]).to_string()),
                s.repo,
                s.started,
                s.last
            ],
        )?;
        Ok(self.index.last_insert_rowid())
    }

    /// Sets a session's live facts and git identity: `event` is the latest
    /// hook event (`hook_stop`, …) at `at`; `waiting_on` marks a wait.
    pub fn live(
        &mut self,
        session_pk: i64,
        event: &str,
        at: i64,
        waiting_on: Option<&str>,
        repo_root: Option<&str>,
    ) -> Result<()> {
        self.index.execute(
            "UPDATE session SET live_last_event = ?2, live_last_event_at = ?3,
                 live_waiting_on = ?4, live_waiting_since = CASE WHEN ?4 IS NULL THEN NULL ELSE ?3 END,
                 repo_root = ?5
             WHERE pk = ?1",
            params![session_pk, event, at, waiting_on, repo_root],
        )?;
        Ok(())
    }

    fn text(&self, body: &str, bit: i64) -> Result<i64> {
        let tx = rusqlite::Transaction::new_unchecked(
            &self.index,
            rusqlite::TransactionBehavior::Immediate,
        )?;
        let id = crate::text_rows::intern(&tx, body, bit, None, false)?;
        tx.commit()?;
        crate::text_rows::index_pending(&self.index)?;
        Ok(id)
    }

    fn block(
        &self,
        message_pk: i64,
        session_pk: i64,
        idx: i64,
        kind: &str,
        text_id: Option<i64>,
        body: serde_json::Value,
    ) -> Result<i64> {
        let packed = crate::encoding::pack_json(&body)?;
        self.index.execute(
            "INSERT INTO block(message_pk, session_pk, idx, kind, text_id, body)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![message_pk, session_pk, idx, kind, text_id, packed],
        )?;
        Ok(self.index.last_insert_rowid())
    }

    pub fn message(&mut self, session_pk: i64, m: MessageFacts<'_>) -> Result<i64> {
        self.seq += 1;
        let words = m.text.map_or(0, |t| t.split_whitespace().count() as i64);
        self.index.execute(
            "INSERT INTO message(id, session_pk, order_input_pk, seq, role, written_by, prompt_kind,
                 model, at, input_pk, cost_nusd, inherited_from, on_active_path, words, order_input_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?3, ?10, ?11, ?12, ?13, (SELECT id FROM input_ref WHERE pk = ?3))",
            params![
                m.id,
                session_pk,
                self.input_pk,
                self.seq,
                m.role,
                m.written_by,
                m.prompt_kind,
                m.model,
                m.at,
                m.cost_nusd,
                m.inherited_from,
                m.on_active_path.map(i64::from),
                words
            ],
        )?;
        let pk = self.index.last_insert_rowid();
        // As the writer does: a new message changes its session, so its
        // derived rows are made again.
        self.index.execute(
            "UPDATE session SET changed = changed + 1 WHERE pk = ?1",
            [session_pk],
        )?;
        let mut idx = 0;
        if let Some(t) = m.thinking {
            let id = self.text(t, 1)?;
            self.block(
                pk,
                session_pk,
                idx,
                "thinking",
                Some(id),
                serde_json::json!({"type": "thinking", "text": null}),
            )?;
            idx += 1;
        }
        if let Some(t) = m.text {
            let id = self.text(t, 1)?;
            self.block(
                pk,
                session_pk,
                idx,
                "text",
                Some(id),
                serde_json::json!({"type": "text", "text": ""}),
            )?;
        }
        Ok(pk)
    }

    /// A tool call in its own message, with its result when it has one.
    pub fn tool_call(&mut self, session_pk: i64, message_pk: i64, t: CallFacts<'_>) -> Result<i64> {
        let idx: i64 = self.index.query_row(
            "SELECT COALESCE(max(idx), -1) + 1 FROM block WHERE message_pk = ?1",
            [message_pk],
            |r| r.get(0),
        )?;
        let call = self.block(
            message_pk,
            session_pk,
            idx,
            "tool_call",
            None,
            serde_json::json!({"type": "tool_call", "call_id": format!("c{message_pk}-{idx}"),
                               "tool": t.tool, "input": {"command": t.command, "path": t.path}}),
        )?;
        let result = match t.output {
            Some(out) => {
                let id = self.text(out, 2)?;
                Some(self.block(
                    message_pk,
                    session_pk,
                    idx + 1,
                    "tool_result",
                    Some(id),
                    serde_json::json!({"type": "tool_result", "outcome": {"kind": t.outcome}}),
                )?)
            }
            None => None,
        };
        self.index.execute(
            "INSERT INTO tool_call(session_pk, call_block_pk, call_id, result_block_pk, at, tool,
                 action, command, command_norm, path, outcome, exit_code)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8, ?9, ?10, ?11)",
            params![
                session_pk,
                call,
                format!("c{message_pk}-{idx}"),
                result,
                t.at,
                t.tool,
                t.action,
                t.command,
                t.path,
                if t.outcome.is_empty() {
                    "unknown"
                } else {
                    t.outcome
                },
                t.exit_code
            ],
        )?;
        let pk = self.index.last_insert_rowid();
        self.index.execute(
            "INSERT INTO tool_call_trigram(rowid, command, path) VALUES (?1, ?2, ?3)",
            params![pk, t.command, t.path],
        )?;
        Ok(pk)
    }

    pub fn link(&mut self, parent: &str, child: &str, kind: &str, how: &str) -> Result<()> {
        self.index.execute(
            "INSERT INTO link(parent_id, child_id, child_pk, kind, how, evidence, input_pk)
             VALUES (?1, ?2, (SELECT pk FROM session WHERE id = ?2), ?3, ?4, 'fixture', ?5)",
            params![parent, child, kind, how, self.input_pk],
        )?;
        Ok(())
    }

    /// Something a parser did not know, counted for the fixture's input.
    pub fn unrecognized(
        &mut self,
        format: &str,
        what: &str,
        count: i64,
        first_at: i64,
        counted_only: bool,
    ) -> Result<()> {
        self.index.execute(
            "INSERT INTO unrecognized(input_pk, chain, format, what, count, counted_only, first_at,
                 last_at, first_offset)
             VALUES (?1, 1, ?2, ?3, ?4, ?5, ?6, ?6, 0)",
            params![
                self.input_pk,
                format,
                what,
                count,
                i64::from(counted_only),
                first_at
            ],
        )?;
        Ok(())
    }

    pub fn event(&mut self, session_pk: i64, kind: &str, at: Option<i64>) -> Result<()> {
        self.seq += 1;
        let data = crate::encoding::pack_json(&serde_json::json!({}))?;
        let id = format!("e{}", self.seq);
        self.index.execute(
            "INSERT INTO event(id,session_pk,seq,kind,at,input_pk,data) VALUES (?1,?2,?3,?4,?5,?6,?7)",
            params![id, session_pk, self.seq, kind, at, self.input_pk, data],
        )?;
        Ok(())
    }

    /// Stores raw token counts and quota. Call `derive` before reading catalog
    /// costs; direct `MessageFacts` costs remain useful for read-only fixtures.
    pub fn usage(
        &mut self,
        message_pk: i64,
        usage: &polylogue_model::Usage,
        quota: Option<serde_json::Value>,
    ) -> Result<()> {
        let n = |x: Option<u64>| x.map(|x| x as i64);
        let attrs = quota.map(|q| serde_json::json!({"production": {"quota": q}}).to_string());
        self.index.execute(
            "UPDATE message SET input_tokens = ?2, cache_read_tokens = ?3, cache_write_tokens = ?4,
                 output_tokens = ?5, reasoning_tokens = ?6, attrs_json = ?7,
                 cache_write_5m_tokens = ?8, cache_write_1h_tokens = ?9,
                 service_tier = ?10, speed = ?11
             WHERE pk = ?1",
            params![
                message_pk,
                n(usage.input),
                n(usage.cached_input),
                n(usage.cache_write),
                n(usage.output),
                n(usage.reasoning),
                attrs,
                usage
                    .cache_write_1h
                    .and(usage.cache_write)
                    .map(|w| w.saturating_sub(usage.cache_write_1h.unwrap_or(0)) as i64),
                n(usage.cache_write_1h),
                usage.service_tier,
                usage.speed,
            ],
        )?;
        self.index.execute(
            "UPDATE session SET changed = changed + 1
             WHERE pk = (SELECT session_pk FROM message WHERE pk = ?1)",
            [message_pk],
        )?;
        Ok(())
    }

    /// A note on `target` (`session:<id>`, `message:<id>`, `note:<root>`).
    /// Returns its ID, which is also its family's root.
    /// A kept result, as the writer stores one.
    pub fn kept_result(&mut self, id: &str, query: &str, rows_json: &str) -> Result<()> {
        self.notes.execute(
            "INSERT INTO kept_result(id, query, ran_at, view, rows_json) VALUES (?1, ?2, 0, '{}', ?3)",
            params![id, query, rows_json],
        )?;
        Ok(())
    }

    pub fn note(
        &mut self,
        kind: &str,
        key: Option<&str>,
        value: Option<&str>,
        target: &str,
    ) -> Result<String> {
        self.commit_seq += 1;
        let id = format!("n{}", self.commit_seq);
        self.notes.execute(
            "INSERT INTO note(id, root, kind, key, value, author, machine, created_at, commit_seq)
             VALUES (?1, ?1, ?2, ?3, ?4, 'operator', 'test', ?5, ?5)",
            params![id, kind, key, value, self.commit_seq],
        )?;
        self.notes.execute(
            "INSERT INTO note_target(note_id, ref) VALUES (?1, ?2)",
            params![id, target],
        )?;
        Ok(id)
    }
    /// An excision over this fixture's synthetic rows, naming the sessions
    /// and messages it removed.
    pub fn excision(
        &mut self,
        id: &str,
        scope: &str,
        sessions: &[&str],
        messages: &[&str],
    ) -> Result<()> {
        self.commit_seq += 1;
        self.notes.execute(
            "INSERT INTO excision(id, scope, created_at, commit_seq) VALUES (?1, ?2, ?3, ?3)",
            params![id, scope, self.commit_seq],
        )?;
        for (kind, values) in [("session", sessions), ("message", messages)] {
            for value in values {
                self.notes.execute(
                    "INSERT INTO excised(excision_id, kind, value) VALUES (?1, ?2, ?3)",
                    params![id, kind, value],
                )?;
            }
        }
        crate::excise::mark_applied(&self.notes, &self.index)
    }
}
