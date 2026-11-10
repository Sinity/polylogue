//! Archives for tests, enabled with the `fixture` feature; never part of
//! the product.
//!
//! - [`Fixture`]: a small archive written row by row, for tests of what
//!   reads it (the query language, rendering, the daemon). Rows go straight
//!   into the tables, the way the writer leaves them, so a test states the
//!   facts it needs without a parser or an input file.
//! - [`writer::WriterArchive`]: the real writer over a temporary archive, for
//!   tests of what the writer and the derive step make of parsed records.
//! - [`text_search`]: whether a text is still stored or indexed anywhere.

pub mod authored;
pub mod signals;
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
    pub on_active_path: Option<bool>,
    /// The message's place in session order; by default after every
    /// message written before it.
    pub seq: Option<i64>,
}

/// What [`Fixture::many_sessions`] writes.
#[derive(Debug, Clone, Copy)]
pub struct ManySessions<'a> {
    /// The directory every session works under.
    pub prefix: &'a str,
    pub sessions: usize,
    /// Messages per session.
    pub messages: usize,
    /// How many of the newest sessions end on a model answer with text.
    pub answered: usize,
    /// The newest session's last activity, Unix ms.
    pub last: i64,
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
        index.execute(
            "INSERT INTO input_ref(id,format) VALUES ('fixture','fixture')",
            [],
        )?;
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
        let id = crate::text_rows::intern(&tx, body, 1, None, None, None)?;
        tx.commit()?;
        Ok(id)
    }

    /// Commits queued search work through the normal bounded indexer.
    pub fn commit_search(&self) -> Result<u64> {
        crate::text_rows::index_pending(&self.index, || ())
    }

    /// Runs all derivations over the raw rows written so far.
    pub fn derive(&mut self) -> Result<u64> {
        crate::derive::run(&mut self.index, None)
    }

    /// Runs the ordinary derivations with a test's declared model prices.
    #[cfg(test)]
    pub fn derive_with_catalog(&mut self, catalog: &polylogue_model::Catalog) -> Result<u64> {
        crate::derive::run_with_catalog(&mut self.index, catalog)
    }

    /// The writer indexes each session's items as it commits; fixture rows
    /// are written directly, so sessions they changed are indexed before
    /// reads open.
    pub fn reads(&self) -> Result<ReadPool> {
        crate::items::refresh_stale(&self.index)?;
        ReadPool::open(&self.archive)
    }

    pub fn session(&mut self, s: SessionFacts<'_>) -> Result<i64> {
        let (origin, native) = s.id.split_once(':').unwrap_or(("test", s.id));
        let kind = if s.kind.is_empty() { "main" } else { s.kind };
        // Fixture titles explicitly represent synthetic harness-generated titles.
        let title_source =
            serde_json::to_value(s.title.map(|_| polylogue_model::TitleSource::Generated))?;
        self.index.execute(
            "INSERT INTO session(id, origin, native_id, kind, title, cwd, cwd_json, repo, record_repo,
                 started_at, last_activity_at, changed, started_time, title_source)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8, ?9, ?10, 1, ?11, ?12)",
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
                s.last,
                fixture_time(s.started)?,
                title_source.as_str(),
            ],
        )?;
        Ok(self.index.last_insert_rowid())
    }

    /// Record source time evidence rather than treating every fixture time as exact.
    pub fn session_time(
        &mut self,
        session_pk: i64,
        at: Option<polylogue_model::Timestamp>,
        evidence: polylogue_model::TimeEvidence,
    ) -> Result<()> {
        evidence
            .validate(at)
            .map_err(polylogue_model::bundle::BundleError::Time)?;
        let bytes = crate::encoding::time_bytes(at, evidence)?;
        self.index.execute(
            "UPDATE session SET started_at=?2, started_time=?3, changed=changed+1 WHERE pk=?1",
            params![session_pk, at.map(|time| time.unix_ms()), bytes],
        )?;
        Ok(())
    }

    /// Record a message's source time and precision as the writer does.
    pub fn message_time(
        &mut self,
        message_pk: i64,
        at: Option<polylogue_model::Timestamp>,
        evidence: polylogue_model::TimeEvidence,
    ) -> Result<()> {
        evidence
            .validate(at)
            .map_err(polylogue_model::bundle::BundleError::Time)?;
        let bytes = crate::encoding::time_bytes(at, evidence)?;
        self.index.execute(
            "UPDATE message SET at=?2, at_time=?3 WHERE pk=?1",
            params![message_pk, at.map(|time| time.unix_ms()), bytes],
        )?;
        self.index.execute(
            "UPDATE session SET changed=changed+1, last_activity_at=(SELECT max(at) FROM message WHERE session_pk=session.pk) WHERE pk=(SELECT session_pk FROM message WHERE pk=?1)",
            [message_pk],
        )?;
        crate::items::refresh_stale(&self.index)?;
        Ok(())
    }

    /// The actual source-order coordinate installed for a fixture message.
    pub fn message_order(&self, message_pk: i64) -> Result<i64> {
        Ok(self
            .index
            .query_row("SELECT seq FROM message WHERE pk=?1", [message_pk], |row| {
                row.get(0)
            })?)
    }

    /// Sets a session's live facts and git identity: the facts are the fold
    /// of one live evidence line of `kind` (`hook_stop`, …) at `at`, with no
    /// process identity; `waiting_on` adds the hook line that starts that
    /// wait at the same instant.
    pub fn live(
        &mut self,
        session_pk: i64,
        kind: &str,
        at: i64,
        waiting_on: Option<&str>,
        repo_root: Option<&str>,
    ) -> Result<()> {
        use polylogue_model::live::{LiveFacts, fold};
        let mut facts = LiveFacts::default();
        let none = serde_json::json!({});
        fold(&mut facts, kind, at, &none);
        match waiting_on {
            Some("permission") => {
                fold(&mut facts, "hook_permission_request", at, &none);
            }
            Some("question") => {
                fold(&mut facts, "hook_elicitation", at, &none);
            }
            _ => {}
        }
        self.live_facts(session_pk, &facts)?;
        self.index.execute(
            "UPDATE session SET repo_root = ?2 WHERE pk = ?1",
            params![session_pk, repo_root],
        )?;
        Ok(())
    }

    /// Explicit live evidence already folded by the writer.
    pub fn live_facts(
        &mut self,
        session_pk: i64,
        facts: &polylogue_model::live::LiveFacts,
    ) -> Result<()> {
        self.index.execute(
            "UPDATE session SET live_facts = ?2, live_last_at = ?3 WHERE pk = ?1",
            params![session_pk, serde_json::to_string(facts)?, facts.last_at],
        )?;
        Ok(())
    }

    fn text(&self, body: &str, bit: i64) -> Result<i64> {
        let tx = rusqlite::Transaction::new_unchecked(
            &self.index,
            rusqlite::TransactionBehavior::Immediate,
        )?;
        let id = crate::text_rows::intern(&tx, body, bit, None, None, None)?;
        tx.commit()?;
        crate::text_rows::index_pending(&self.index, || ())?;
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
        let block = self.index.last_insert_rowid();
        Ok(block)
    }

    pub fn message(&mut self, session_pk: i64, m: MessageFacts<'_>) -> Result<i64> {
        self.seq = m.seq.unwrap_or(self.seq + 1);
        let words = m.text.map_or(0, |t| t.split_whitespace().count() as i64);
        self.index.execute(
            "INSERT INTO message(id, session_pk, order_input_pk, seq, role, written_by, prompt_kind,
                 model, at, input_pk, cost_nusd, on_active_path, words, order_input_id, at_time)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?3, ?10, ?11, ?12, (SELECT id FROM input_ref WHERE pk = ?3), ?13)",
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
                m.on_active_path.map(i64::from),
                words,
                fixture_time(m.at)?,
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
        crate::items::refresh_stale(&self.index)?;
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
        self.index.execute(
            "UPDATE session SET changed=changed+1 WHERE pk=?1",
            [session_pk],
        )?;
        crate::items::refresh_stale(&self.index)?;
        Ok(pk)
    }

    /// Many main sessions in one transaction, for tests that bound how much
    /// of the archive a read touches. Session `n` (from 0) is
    /// `test:many-<n>`, works in `<prefix>/d<n mod 97>/w<n mod 13>`, and was
    /// last active `last - (sessions - 1 - n)` minutes ago, so the highest
    /// numbers are the newest. Each has `messages` messages alternating a
    /// person's prompt and a model answer; the newest `answered` end on a
    /// model answer with text. Counts and brief summaries are stored as the
    /// derive step leaves them. Returns the session row keys in order.
    pub fn many_sessions(&mut self, spec: ManySessions<'_>) -> Result<Vec<i64>> {
        let tx = rusqlite::Transaction::new_unchecked(
            &self.index,
            rusqlite::TransactionBehavior::Immediate,
        )?;
        let mut pks = Vec::with_capacity(spec.sessions);
        {
            let mut session = tx.prepare(
                "INSERT INTO session(id, origin, native_id, kind, cwd, cwd_json, started_at,
                     last_activity_at, message_count, changed, started_time)
                 VALUES (?1, 'test', ?2, 'main', ?3, ?4, ?5, ?6, ?7, 1, ?8)",
            )?;
            let mut message = tx.prepare(
                "INSERT INTO message(id, session_pk, order_input_pk, order_input_id, seq, role,
                     written_by, prompt_kind, at, input_pk, on_active_path, words)
                 VALUES (?1, ?2, ?3, 'fixture', ?4, ?5, ?6, ?7, ?8, ?3, 1, 3)",
            )?;
            let mut block = tx.prepare(
                "INSERT INTO block(message_pk, session_pk, idx, kind, text_id, body)
                 VALUES (?1, ?2, 0, 'text', ?3, ?4)",
            )?;
            let text_body =
                crate::encoding::pack_json(&serde_json::json!({"type": "text", "text": ""}))?;
            for n in 0..spec.sessions {
                let native = format!("many-{n}");
                let cwd = format!("{}/d{}/w{}", spec.prefix, n % 97, n % 13);
                let last = spec.last - 60_000 * (spec.sessions - 1 - n) as i64;
                let started = last - 30_000;
                session.execute(params![
                    format!("test:{native}"),
                    native,
                    cwd,
                    serde_json::json!([cwd]).to_string(),
                    started,
                    last,
                    spec.messages as i64,
                    fixture_time(Some(started))?,
                ])?;
                let pk = tx.last_insert_rowid();
                let answered = n + spec.answered >= spec.sessions;
                for m in 0..spec.messages {
                    // The last message is always a model answer.
                    let model = (spec.messages - 1 - m).is_multiple_of(2);
                    self.seq += 1;
                    message.execute(params![
                        format!("many-{n}-{m}"),
                        pk,
                        self.input_pk,
                        self.seq,
                        if model { "assistant" } else { "user" },
                        if model { "model" } else { "human" },
                        (!model).then_some("prose"),
                        started + m as i64,
                    ])?;
                    let message_pk = tx.last_insert_rowid();
                    if answered && m + 1 == spec.messages {
                        let text = crate::text_rows::intern(
                            &tx,
                            &format!("Answer {n}: the synthetic work is finished."),
                            1,
                            None,
                            None,
                            None,
                        )?;
                        block.execute(params![message_pk, pk, text, text_body])?;
                    }
                }
                pks.push(pk);
            }
        }
        tx.commit()?;
        crate::text_rows::index_pending(&self.index, || ())?;
        Ok(pks)
    }

    pub fn link(&mut self, parent: &str, child: &str, kind: &str, how: &str) -> Result<()> {
        self.index.execute(
            "INSERT INTO link(parent_id, child_id, child_pk, kind, how, evidence, input_pk)
             VALUES (?1, ?2, (SELECT pk FROM session WHERE id = ?2), ?3, ?4, 'fixture', ?5)",
            params![parent, child, kind, how, self.input_pk],
        )?;
        crate::items::refresh_stale(&self.index)?;
        Ok(())
    }

    /// Something a parser did not know, counted for the fixture's input.
    pub fn unrecognized(
        &mut self,
        format: &str,
        what: &str,
        count: i64,
        counted_only: bool,
    ) -> Result<()> {
        self.index.execute(
            "INSERT INTO unused(input_pk, chain, format, what, count, counted_only)
             VALUES (?1, 1, ?2, ?3, ?4, ?5)",
            params![self.input_pk, format, what, count, i64::from(counted_only)],
        )?;
        Ok(())
    }

    pub fn event(&mut self, session_pk: i64, kind: &str, at: Option<i64>) -> Result<()> {
        self.seq += 1;
        let data = crate::encoding::pack_json(&serde_json::json!({}))?;
        let id = format!("e{}", self.seq);
        let session: String =
            self.index
                .query_row("SELECT id FROM session WHERE pk=?1", [session_pk], |r| {
                    r.get(0)
                })?;
        // Fixture `at` declares a native instant, like MessageFacts::at.
        let native_at = at.and_then(polylogue_model::Timestamp::from_unix_ms);
        let evidence = if native_at.is_some() {
            polylogue_model::TimeEvidence::intrinsic(polylogue_model::Precision::Instant)
        } else {
            polylogue_model::TimeEvidence::UNKNOWN
        };
        let fields = crate::encoding::pack_json(&polylogue_model::Event {
            at_quality: evidence.quality,
            at_precision: evidence.precision,
            at_until: evidence.until,
            session: Some(
                polylogue_model::SessionId::parse(&session)
                    .map_err(|error| crate::StoreError::Invalid(error.to_string()))?,
            ),
            kind: kind.into(),
            at: native_at,
            after: None,
            range: None,
            data: serde_json::Value::Null,
        })?;
        self.index.execute(
            "INSERT INTO event(id,session_pk,seq,kind,at,input_pk,data,fields,source_chain) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,1)",
            params![id, session_pk, self.seq, kind, at, self.input_pk, data,fields],
        )?;
        let hash = *blake3::hash(b"{}").as_bytes();
        self.index.execute(
            "INSERT OR IGNORE INTO event_value(hash,body) VALUES (?1,?2)",
            params![hash.as_slice(), data],
        )?;
        self.index.execute("INSERT INTO event_source(event_pk,input_pk,chain,value_pk,route,session_pk,fields,imported_provenance)
            SELECT e.pk,?2,1,v.pk,'live',?3,?4,'[]' FROM event e,event_value v WHERE e.id=?1 AND v.hash=?5",
            params![id,self.input_pk,session_pk,fields,hash.as_slice()])?;
        self.index.execute(
            "UPDATE session SET changed=changed+1 WHERE pk=?1",
            [session_pk],
        )?;
        crate::items::refresh_stale(&self.index)?;
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
        // The message's cost is a row value, written with its usage by the
        // function the writer uses.
        let model: Option<String> = self.index.query_row(
            "SELECT model FROM message WHERE pk = ?1",
            [message_pk],
            |r| r.get(0),
        )?;
        self.index.execute(
            "UPDATE message SET cost_nusd = ?2 WHERE pk = ?1",
            params![
                message_pk,
                polylogue_model::Catalog::builtin().cost_nusd(model.as_deref(), usage)
            ],
        )?;
        self.index.execute(
            "UPDATE session SET changed = changed + 1
             WHERE pk = (SELECT session_pk FROM message WHERE pk = ?1)",
            [message_pk],
        )?;
        crate::items::refresh_stale(&self.index)?;
        Ok(())
    }

    /// A kept result, as the writer stores one.
    pub fn kept_result(&mut self, id: &str, query: &str, rows_json: &str) -> Result<()> {
        crate::kept::validate_body(id, rows_json)?;
        self.notes.execute(
            "INSERT INTO kept_result(id, query, ran_at, read_at, rows_json) VALUES (?1, ?2, 0, '{}', ?3)",
            params![id, query, rows_json],
        )?;
        Ok(())
    }

    /// Replace a stored kept body with deliberately damaged bytes for read tests.
    pub fn corrupt_kept_body(&mut self, id: &str, body: &str) -> Result<()> {
        self.notes.execute(
            "UPDATE kept_result SET rows_json=?1 WHERE id=?2",
            params![body, id],
        )?;
        Ok(())
    }

    /// A catalog setting, as the operator would set it.
    pub fn setting(&mut self, key: &str, value: &str) -> Result<()> {
        let at = polylogue_model::Timestamp::from_unix_ms(self.commit_seq + 1)
            .expect("synthetic fixture counter is a timestamp");
        self.setting_at(key, value, at).map(|_| ())
    }

    /// A setting accepted at an explicit synthetic time, through the same
    /// catalog, slot policy and frozen-effect capture as the Writer.
    pub fn setting_at(
        &mut self,
        key: &str,
        value: &str,
        at: polylogue_model::Timestamp,
    ) -> Result<String> {
        self.user_at(
            polylogue_model::user::UserWrite::SetSetting {
                key: key.into(),
                value: value.into(),
                machine: None,
            },
            at,
        )
    }

    /// An operator's write through the same write path a command takes:
    /// uniqueness slots, stale bases and revisions are checked.
    pub fn user(&mut self, write: polylogue_model::user::UserWrite) -> Result<String> {
        let at = polylogue_model::Timestamp::from_unix_ms(self.commit_seq + 1)
            .expect("synthetic fixture counter is a timestamp");
        self.user_at(write, at)
    }

    fn user_at(
        &mut self,
        write: polylogue_model::user::UserWrite,
        at: polylogue_model::Timestamp,
    ) -> Result<String> {
        self.commit_seq += 1;
        let id = format!("s{}", self.commit_seq);
        let request = polylogue_model::user::UserRequest {
            id: id.clone(),
            machine: "fixture-machine".into(),
            at,
            authorship: polylogue_model::authorship::Authorship {
                author: polylogue_model::Author::Operator,
                via: polylogue_model::authorship::Channel::Cli,
            },
            write,
        };
        let original = self.index.unchecked_transaction()?;
        let tx = self.notes.transaction()?;
        crate::user_writes::apply_resolving(
            &tx,
            &original,
            &request,
            self.commit_seq,
            crate::user_writes::Preparation::native(None, None),
        )?;
        tx.commit()?;
        Ok(id)
    }

    /// Inserts a canonical fixture assertion with a text or absent value.
    /// Kind and key follow the same checked shapes as stored assertion facts.
    pub fn note(
        &mut self,
        kind: &str,
        key: Option<&str>,
        value: Option<&str>,
        target: &str,
    ) -> Result<String> {
        use polylogue_model::{
            Author, Reference, Timestamp,
            assertion::{AssertionFact, AssertionKind, AssertionLink, LinkRole},
            authorship::Channel,
        };
        self.commit_seq += 1;
        let id = format!("n{}", self.commit_seq);
        let kind = AssertionKind::from_name(kind)
            .ok_or_else(|| crate::StoreError::Invalid("unknown fixture assertion kind".into()))?;
        let target = Reference::parse(target)
            .map_err(|_| crate::StoreError::Invalid("invalid fixture assertion target".into()))?;
        let fact = AssertionFact {
            id: id.clone(),
            root: id.clone(),
            kind,
            key: key.map(str::to_owned),
            value: value.map(|value| polylogue_model::Json::String(value.into())),
            author: Author::Operator,
            via: Channel::Cli,
            machine: "test".into(),
            created_at: Timestamp::from_unix_ms(self.commit_seq)
                .expect("synthetic fixture counter is a timestamp"),
            valid_until: None,
            links: vec![AssertionLink {
                role: LinkRole::About,
                reference: target,
                seen_hash: None,
                quote: None,
            }],
        };
        let tx = self.notes.transaction()?;
        crate::assertions::apply(&tx, std::slice::from_ref(&fact), self.commit_seq)?;
        tx.commit()?;
        Ok(id)
    }

    /// A current saved definition from the same real prepared Plan used by
    /// query execution. Independent roots permit imported-name conflict tests.
    pub fn saved_query(
        &mut self,
        name: &str,
        plan: &polylogue_query::plan::Plan,
    ) -> Result<String> {
        use polylogue_model::{
            Author, Timestamp,
            assertion::{AssertionFact, AssertionKind},
            authorship::Channel,
        };
        let object = plan.query_object().map_err(crate::StoreError::Query)?;
        self.commit_seq += 1;
        let id = format!("n{}", self.commit_seq);
        let fact = AssertionFact {
            id: id.clone(),
            root: id.clone(),
            kind: AssertionKind::SavedQuery,
            key: Some(name.into()),
            value: Some(serde_json::json!({
                "text": plan.text(), "hash":object.id.to_string(), "denominator":null,
            })),
            author: Author::Operator,
            via: Channel::Cli,
            machine: "test".into(),
            created_at: Timestamp::from_unix_ms(self.commit_seq).unwrap(),
            valid_until: None,
            links: vec![],
        };
        let tx = self.notes.transaction()?;
        crate::query_objects::apply_object(&tx, &object, self.commit_seq)?;
        crate::assertions::apply(&tx, &[fact], self.commit_seq)?;
        tx.commit()?;
        Ok(id)
    }
}

/// Marks one origin's readings as older, in an isolated test or benchmark archive.
/// The caller stops its writer first and keeps the archive lock.
pub fn older_reading(archive: &Archive, origin: &str) -> Result<(u64, u64)> {
    let conn = open_write(&archive.index_db(), crate::archive::cache::SETUP)?;
    conn.execute(
        "ATTACH DATABASE ?1 AS a",
        [archive.archive_db().to_string_lossy().as_ref()],
    )?;
    let size=conn.query_row("SELECT count(*),coalesce(sum((SELECT max(length) FROM a.revision r WHERE r.input=i.id AND r.chain=p.chain)),0)
        FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.format=?1",[origin],|r|Ok((r.get::<_,i64>(0)? as u64,r.get::<_,i64>(1)? as u64)))?;
    conn.execute("UPDATE parse_state SET reading_version='older reading' WHERE input_pk IN (SELECT pk FROM input_ref WHERE format=?1)",[origin])?;
    Ok(size)
}

fn fixture_time(at: Option<i64>) -> Result<Vec<u8>> {
    let at = at.and_then(polylogue_model::Timestamp::from_unix_ms);
    crate::encoding::time_bytes(
        at,
        if at.is_some() {
            polylogue_model::TimeEvidence::intrinsic(polylogue_model::Precision::Instant)
        } else {
            polylogue_model::TimeEvidence::UNKNOWN
        },
    )
}

#[cfg(test)]
mod tests;
