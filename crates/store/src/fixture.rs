//! Archives for tests, enabled with the `fixture` feature; never part of
//! the product.
//!
//! - [`Fixture`]: a small archive written row by row, for tests of what
//!   reads it (the query language, rendering, the daemon). Rows go straight
//!   into the tables, the way the writer leaves them, so a test states the
//!   facts it needs without a parser or an input file.
//! - [`writer::WriterArchive`]: the real writer over a temporary archive, for
//!   tests of what the writer and the derive step make of parsed records.

pub mod authored;
pub mod signals;
pub mod writer;

use crate::ConnExt as _;
use crate::archive::{Archive, ArchiveLock, open_write};
use crate::{OptionExt as _, ResultExt as _};
use crate::{ReadPool, Result};
use rusqlite::{Connection, params};
use std::path::Path;

pub struct Fixture {
    pub archive: Archive,
    _lock: ArchiveLock,
    index: Connection,
    archive_conn: Connection,
    input_pk: i64,
    seq: i64,
    commit_seq: i64,
    /// Each session's `changed` when its transcript lines were last written.
    lines_written: std::cell::RefCell<std::collections::HashMap<i64, i64>>,
}

/// A file's bytes, or `None` for a directory, at a path under an archive root.
type ArchiveFile = (std::path::PathBuf, Option<Vec<u8>>);

/// The files and directories of a new archive whose empty build is
/// installed, parents first, made once per process and read once its
/// writer has stopped. The lock file and the databases' shared-memory
/// files belong to that archive's open connections and are left out.
fn empty_archive() -> Result<&'static [ArchiveFile]> {
    static FILES: std::sync::OnceLock<Vec<ArchiveFile>> = std::sync::OnceLock::new();
    if let Some(files) = FILES.get() {
        return Ok(files);
    }
    let dir = tempfile::tempdir()?;
    let root = dir.path().join("archive");
    {
        let (archive, _lock) = Archive::open_or_create(&root)?;
        crate::Writer::start(&archive)?.build(Vec::new())?;
    }
    let mut files = Vec::new();
    let mut pending = vec![root.clone()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(&directory)? {
            let entry = entry?;
            let path = entry.path();
            let relative = path
                .strip_prefix(&root)
                .map_err(|_| crate::StoreError::internal("fixture file outside its archive"))?
                .to_owned();
            if entry.file_type()?.is_dir() {
                files.push((relative, None));
                pending.push(path);
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == "lock" || name.ends_with("-shm") {
                continue;
            }
            files.push((relative, Some(std::fs::read(&path)?)));
        }
    }
    Ok(FILES.get_or_init(|| files))
}

/// A session's facts; everything not given is unknown, except that an empty
/// `kind` is `main` and an id without a colon gets origin `test`.
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
        // Fixture rows are written directly into an installed index: the
        // new archive's empty build is installed first, as a daemon's first
        // pass over nothing does. That build is the same for every fixture,
        // so it is made once per process and its files written out here.
        std::fs::create_dir_all(root)?;
        for (path, bytes) in empty_archive()? {
            let path = root.join(path);
            match bytes {
                Some(bytes) => std::fs::write(path, bytes)?,
                None => std::fs::create_dir_all(path)?,
            }
        }
        // Each fixture has its own index generation and cursor key, as a
        // fresh build gives, so a cursor from one fixture is foreign to
        // another.
        crate::archive::generation::rotate(&open_write(
            &root.join("index.db"),
            crate::archive::cache::SETUP,
        )?)?;
        let (archive, lock) = Archive::open_or_create(root)?;
        let index = open_write(&archive.index_db(), crate::archive::cache::SETUP)?;
        // Fixture rows count their changes from 1, after a write position of
        // 0; the empty build's own commits are not part of the fixture.
        index.execute("DELETE FROM meta WHERE key='write_counter'", [])?;
        // Fixture rows are written outside the writer, which marks a session
        // due when it changes the session's rows: here each change of a
        // session's own rows marks it due from its first row.
        index.execute_batch(
            "CREATE TEMP TRIGGER fixture_session_due AFTER INSERT ON main.session BEGIN
               INSERT INTO summary_due(session_pk, from_order) VALUES (NEW.pk, X'')
                 ON CONFLICT(session_pk) DO UPDATE SET from_order = X'', marked = marked + 1;
             END;
             CREATE TEMP TRIGGER fixture_rows_due AFTER UPDATE OF changed ON main.session
             WHEN NEW.changed IS NOT OLD.changed BEGIN
               INSERT INTO summary_due(session_pk, from_order) VALUES (NEW.pk, X'')
                 ON CONFLICT(session_pk) DO UPDATE SET from_order = X'', marked = marked + 1;
             END;",
        )?;
        let archive_conn = open_write(&archive.archive_db(), crate::archive::cache::SETUP)?;
        index.execute(
            "INSERT INTO input_ref(id,format) VALUES ('fixture','fixture')",
            [],
        )?;
        let input_pk = index.last_insert_rowid();
        Ok(Self {
            archive,
            _lock: lock,
            index,
            archive_conn,
            input_pk,
            seq: 0,
            commit_seq: 0,
            lines_written: Default::default(),
        })
    }

    fn refresh_counts(&self, session: i64) -> Result<()> {
        crate::summary::counts::apply(
            &self.index,
            session,
            &crate::summary::counts::compute(&self.index, session)?,
        )
    }

    /// Queues search work without committing it, for reader and index tests.
    pub fn queue_search_text(&self, body: &str) -> Result<i64> {
        let tx = self.index.unchecked_transaction()?;
        let id = crate::text_rows::intern(&tx, body)?;
        crate::text_rows::mark_searched(&tx, id, 1)?;
        tx.commit()?;
        Ok(id)
    }

    /// Commits queued search work through the normal bounded indexer.
    pub fn commit_search(&self) -> Result<u64> {
        crate::text_rows::index_pending(&self.index)
    }

    /// Runs all derivations over the raw rows written so far.
    pub fn derive(&mut self) -> Result<u64> {
        crate::summary::run(&mut self.index)
    }

    /// Runs the derivations with a test's declared model prices.
    #[cfg(test)]
    pub fn derive_with_catalog(&mut self, catalog: &polylogue_model::Catalog) -> Result<u64> {
        crate::summary::run_with_catalog(&mut self.index, catalog)
    }

    /// The summary worker writes each session's transcript lines once it
    /// is due; fixture rows are written directly and mark nothing, so the
    /// lines of every session changed since they were written are written
    /// before reads open.
    pub fn reads(&self) -> Result<ReadPool> {
        crate::summary::lines::write_fixture_lines(
            &self.index,
            &mut self.lines_written.borrow_mut(),
        )?;
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
                 started_at, last_activity_at, changed, started_time, title_source,
                 started_precision, last_active_precision)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8, ?9, ?10, 1, ?11, ?12,
                 iif(?9 IS NULL, NULL, 'instant'), iif(?10 IS NULL, NULL, 'instant'))",
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

    /// Record source time accuracy rather than treating every fixture time as exact.
    pub fn session_time(
        &mut self,
        session_pk: i64,
        at: Option<polylogue_model::Timestamp>,
        accuracy: polylogue_model::TimeAccuracy,
    ) -> Result<()> {
        accuracy
            .validate(at)
            .map_err(polylogue_model::bundle::BundleError::Time)?;
        let bytes = crate::encoding::time_bytes(at, accuracy)?;
        let time = crate::encoding::TimeColumns::of(at, accuracy);
        self.index.execute(
            "UPDATE session SET started_at=?2, started_time=?3, started_until=?4, started_precision=?5,
                 changed=changed+1 WHERE pk=?1",
            params![session_pk, time.at, bytes, time.until, time.precision_name()],
        )?;
        Ok(())
    }

    /// Record a message's source time and precision as the writer does.
    pub fn message_time(
        &mut self,
        message_pk: i64,
        at: Option<polylogue_model::Timestamp>,
        accuracy: polylogue_model::TimeAccuracy,
    ) -> Result<()> {
        accuracy
            .validate(at)
            .map_err(polylogue_model::bundle::BundleError::Time)?;
        let bytes = crate::encoding::time_bytes(at, accuracy)?;
        let time = crate::encoding::TimeColumns::of(at, accuracy);
        self.index.execute(
            "UPDATE message SET at=?2, at_time=?3, at_until=?4, at_precision=?5 WHERE pk=?1",
            params![
                message_pk,
                time.at,
                bytes,
                time.until,
                time.precision_name()
            ],
        )?;
        // As the session summary does: the latest of its messages' times.
        let session: i64 = self
            .index
            .scalar("SELECT session_pk FROM message WHERE pk=?1", [message_pk])?;
        let mut latest = crate::summary::session::Latest::default();
        let mut times = self.index.prepare(
            "SELECT at, at_until, at_precision FROM message WHERE session_pk=?1 AND at IS NOT NULL",
        )?;
        let rows = times.query_map([session], |r| <(i64, Option<i64>, String)>::try_from(r))?;
        for row in rows {
            let (at, until, precision) = row?;
            latest.add(at, until, &precision);
        }
        drop(times);
        let (last, until, precision) = latest.columns();
        self.index.execute(
            "UPDATE session SET changed=changed+1, last_activity_at=?2, last_active_until=?3,
                 last_active_precision=?4 WHERE pk=?1",
            params![session, last, until, precision],
        )?;
        crate::summary::lines::write_fixture_lines(
            &self.index,
            &mut self.lines_written.borrow_mut(),
        )?;
        Ok(())
    }

    /// The sequence a fixture message's stored session order holds: the
    /// eight bytes before its record position.
    pub fn message_order(&self, message_pk: i64) -> Result<i64> {
        let order: Vec<u8> = self.index.scalar(
            "SELECT session_order FROM message WHERE pk=?1",
            [message_pk],
        )?;
        let seq = order
            .len()
            .checked_sub(16)
            .and_then(|start| order.get(start..start + 8))
            .and_then(|bytes| <[u8; 8]>::try_from(bytes).ok())
            .map(u64::from_be_bytes)
            .ok_or_else(|| crate::StoreError::internal("a session order without a sequence"))?;
        Ok((seq ^ (1 << 63)) as i64)
    }

    /// Sets a session's live facts and git identity: the facts are computed
    /// from one live event line of `kind` (`hook_stop`, …) at `at`, with no
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
        use polylogue_model::live::{LiveFacts, apply_live_event};
        let mut facts = LiveFacts::default();
        let none = serde_json::json!({});
        apply_live_event(&mut facts, kind, at, &none);
        match waiting_on {
            Some("permission") => {
                apply_live_event(&mut facts, "hook_permission_request", at, &none);
            }
            Some("question") => {
                apply_live_event(&mut facts, "hook_elicitation", at, &none);
            }
            None => {}
            Some(other) => {
                return Err(crate::StoreError::internal(format!(
                    "unsupported waiting_on {other:?}"
                )));
            }
        }
        self.live_facts(session_pk, &facts)?;
        self.index.execute(
            "UPDATE session SET repo_root = ?2 WHERE pk = ?1",
            params![session_pk, repo_root],
        )?;
        Ok(())
    }

    /// Live facts as the writer computes them from live events.
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

    fn block(
        &self,
        message_pk: i64,
        session_pk: i64,
        idx: i64,
        kind: &str,
        text_id: Option<i64>,
        body: serde_json::Value,
    ) -> Result<i64> {
        let piece = crate::content::store_piece(&self.index, &serde_json::to_vec(&body)?, None)?;
        self.index.execute(
            "INSERT INTO block(message_pk, session_pk, idx, kind, text_id, content_group,
                 content_start, content_length)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                message_pk,
                session_pk,
                idx,
                kind,
                text_id,
                piece.group,
                piece.start,
                piece.length
            ],
        )?;
        let block = self.index.last_insert_rowid();
        // The block's text joins its search index as it is written.
        crate::text_rows::index_pending(&self.index)?;
        Ok(block)
    }

    pub fn message(&mut self, session_pk: i64, m: MessageFacts<'_>) -> Result<i64> {
        self.seq = m.seq.unwrap_or(self.seq + 1);
        let words = m.text.map_or(0, |t| t.split_whitespace().count() as i64);
        self.index.execute(
            "INSERT INTO message(id, session_pk, input_pk, session_order, role, written_by, prompt_kind,
                 model, at, cost_nusd, on_active_path, words, at_time, at_precision)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, iif(?9 IS NULL, NULL, 'instant'))",
            params![
                m.id,
                session_pk,
                self.input_pk,
                crate::encoding::session_order(None, "fixture", self.seq, 0),
                // Facts that leave the role or the author unstated store
                // what the writer stores for them.
                if m.role.is_empty() { "user" } else { m.role },
                if m.written_by.is_empty() {
                    "unknown"
                } else {
                    m.written_by
                },
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
            let id = crate::text_rows::intern(&self.index, t)?;
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
            let id = crate::text_rows::intern(&self.index, t)?;
            self.block(
                pk,
                session_pk,
                idx,
                "text",
                Some(id),
                serde_json::json!({"type": "text", "text": ""}),
            )?;
        }
        self.refresh_counts(session_pk)?;
        crate::summary::lines::write_fixture_lines(
            &self.index,
            &mut self.lines_written.borrow_mut(),
        )?;
        Ok(pk)
    }

    /// A tool call appended as the next block of `message_pk`, followed by its
    /// result block when `output` is given.
    pub fn tool_call(&mut self, session_pk: i64, message_pk: i64, t: CallFacts<'_>) -> Result<i64> {
        let idx: i64 = self.index.scalar(
            "SELECT COALESCE(max(idx), -1) + 1 FROM block WHERE message_pk = ?1",
            [message_pk],
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
                let id = crate::text_rows::intern(&self.index, out)?;
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
            "INSERT INTO tool_call(session_pk, call_block_pk, call_id, result_message_id, result_idx, at, tool,
                 action, command, command_norm, path, outcome, exit_code)
             VALUES (?1, ?2, ?3, (SELECT id FROM message WHERE pk=?4), ?12, ?5, ?6, ?7, ?8, ?8, ?9, ?10, ?11)",
            params![
                session_pk,
                call,
                format!("c{message_pk}-{idx}"),
                result.map(|_| message_pk),
                t.at,
                t.tool,
                (!t.action.is_empty()).then_some(t.action),
                t.command,
                t.path,
                if t.outcome.is_empty() {
                    "unknown"
                } else {
                    t.outcome
                },
                t.exit_code,
                result.map(|_| idx + 1),
            ],
        )?;
        let pk = self.index.last_insert_rowid();
        self.index.execute(
            "UPDATE session SET changed=changed+1 WHERE pk=?1",
            [session_pk],
        )?;
        self.refresh_counts(session_pk)?;
        crate::summary::lines::write_fixture_lines(
            &self.index,
            &mut self.lines_written.borrow_mut(),
        )?;
        Ok(pk)
    }

    /// Many main sessions in one transaction, for tests that bound how much
    /// of the archive a read touches. Session `n` (from 0) is
    /// `test:many-<n>`, works in `<prefix>/d<n mod 97>/w<n mod 13>`, and was
    /// last active `last - (sessions - 1 - n)` minutes ago, so the highest
    /// numbers are the newest. Each has `messages` messages alternating a
    /// person's prompt and a model answer; the newest `answered` end on a
    /// model answer with text. The fixture sums each session's counts once;
    /// brief summaries are stored as the derive step leaves them. Returns the session row keys in order.
    pub fn many_sessions(&mut self, spec: ManySessions<'_>) -> Result<Vec<i64>> {
        let tx = rusqlite::Transaction::new_unchecked(
            &self.index,
            rusqlite::TransactionBehavior::Immediate,
        )?;
        let mut pks = Vec::with_capacity(spec.sessions);
        // As a build does, the rows are loaded first and the message
        // table's indexes created after them, a sort each instead of a
        // million inserts into each.
        let indexes: Vec<(String, String)> = tx
            .prepare(
                "SELECT name, sql FROM sqlite_master
                 WHERE type = 'index' AND tbl_name = 'message' AND sql IS NOT NULL",
            )?
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        for (name, _) in &indexes {
            tx.execute_batch(&format!("DROP INDEX \"{name}\""))?;
        }
        {
            let mut session = tx.prepare(
                "INSERT INTO session(id, origin, native_id, kind, cwd, cwd_json, started_at,
                     last_activity_at, changed, started_time, started_precision, last_active_precision)
                 VALUES (?1, 'test', ?2, 'main', ?3, ?4, ?5, ?6, 1, ?7, 'instant', 'instant')",
            )?;
            let mut message = tx.prepare(
                "INSERT INTO message(id, session_pk, input_pk, session_order, role,
                     written_by, prompt_kind, at, on_active_path, words, at_precision)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1, 3, 'instant')",
            )?;
            let mut block = tx.prepare(
                "INSERT INTO block(message_pk, session_pk, idx, kind, text_id, content_group,
                     content_start, content_length)
                 VALUES (?1, ?2, 0, 'text', ?3, ?4, ?5, ?6)",
            )?;
            let text_body = crate::content::store_piece(
                &tx,
                &serde_json::to_vec(&serde_json::json!({"type": "text", "text": ""}))?,
                None,
            )?;
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
                    fixture_time(Some(started))?,
                ])?;
                let pk = tx.last_insert_rowid();
                let answered = n + spec.answered >= spec.sessions;
                for m in 0..spec.messages {
                    // The last message is always a model answer.
                    let model = (spec.messages - 1 - m).is_multiple_of(2);
                    self.seq += 1;
                    let id = format!("test:many-{n}-{m}");
                    message.execute(params![
                        id,
                        pk,
                        self.input_pk,
                        crate::encoding::session_order(None, "fixture", self.seq, 0),
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
                        )?;
                        block.execute(params![
                            message_pk,
                            pk,
                            text,
                            text_body.group,
                            text_body.start,
                            text_body.length
                        ])?;
                    }
                }
                pks.push(pk);
            }
        }
        for (_, sql) in &indexes {
            tx.execute_batch(sql)?;
        }
        for &pk in &pks {
            crate::summary::counts::apply(&tx, pk, &crate::summary::counts::compute(&tx, pk)?)?;
        }
        tx.commit()?;
        crate::text_rows::index_pending(&self.index)?;
        Ok(pks)
    }

    pub fn link(&mut self, parent: &str, child: &str, kind: &str, how: Option<&str>) -> Result<()> {
        self.index.execute(
            "INSERT INTO link(parent_id, child_id, kind, how, evidence, input_pk)
             VALUES (?1, ?2, ?3, ?4, 'fixture', ?5)",
            params![parent, child, kind, how, self.input_pk],
        )?;
        crate::summary::lines::write_fixture_lines(
            &self.index,
            &mut self.lines_written.borrow_mut(),
        )?;
        Ok(())
    }

    /// A link whose child repeats its parent's prefix.
    pub fn copying_link(&mut self, parent: &str, child: &str, kind: &str) -> Result<()> {
        self.index.execute(
            "INSERT INTO link(parent_id, child_id, kind, copies_prefix, evidence, input_pk)
             VALUES (?1, ?2, ?3, 1, 'fixture', ?4)",
            params![parent, child, kind, self.input_pk],
        )?;
        crate::summary::lines::write_fixture_lines(
            &self.index,
            &mut self.lines_written.borrow_mut(),
        )?;
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
        let id = format!("e{}", self.seq);
        let session: String =
            self.index
                .query_row("SELECT id FROM session WHERE pk=?1", [session_pk], |r| {
                    r.get(0)
                })?;
        // Fixture `at` declares a source instant, like MessageFacts::at.
        let source_at = at.and_then(polylogue_model::Timestamp::from_unix_ms);
        let accuracy = if source_at.is_some() {
            polylogue_model::TimeAccuracy::intrinsic(polylogue_model::Precision::Instant)
        } else {
            polylogue_model::TimeAccuracy::UNKNOWN
        };
        let fields = serde_json::to_vec(&polylogue_model::Event {
            at_quality: accuracy.quality,
            at_precision: accuracy.precision,
            at_until: accuracy.until,
            session: Some(
                polylogue_model::SessionId::parse(&session).map_err(crate::StoreError::internal)?,
            ),
            kind: kind.into(),
            at: source_at,
            after: None,
            range: None,
            data: serde_json::Value::Null,
        })?;
        let piece = crate::content::store_piece(&self.index, &fields, Some(b"{}"))?;
        self.index.execute(
            "INSERT INTO event(id,session_pk,seq,kind,at,input_pk,source_chain,content_group,content_start,
                 content_length,data_length,session_order,at_precision)
                 VALUES (?1,?2,?3,?4,?5,?6,1,?7,?8,?9,?10,?11,iif(?5 IS NULL, NULL, 'instant'))",
            params![id, session_pk, self.seq, kind, at, self.input_pk, piece.group, piece.start,
                piece.length, piece.data_length,
                crate::encoding::session_order(at, "fixture", self.seq, 0)],
        )?;
        let hash = *blake3::hash(b"{}").as_bytes();
        self.index.execute(
            "INSERT INTO event_parse(input_pk,chain,event_id,session_id,route,seq,kind,at,hash,
                imported_sources,content_group,content_start,content_length,data_length)
            SELECT ?2,1,?1,s.id,'live',?4,?5,?6,?3,'[]',?8,?9,?10,?11 FROM session s WHERE s.pk=?7",
            params![
                id,
                self.input_pk,
                hash.as_slice(),
                self.seq,
                kind,
                at,
                session_pk,
                piece.group,
                piece.start,
                piece.length,
                piece.data_length
            ],
        )?;
        self.index.execute(
            "UPDATE session SET changed=changed+1 WHERE pk=?1",
            [session_pk],
        )?;
        self.refresh_counts(session_pk)?;
        crate::summary::lines::write_fixture_lines(
            &self.index,
            &mut self.lines_written.borrow_mut(),
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
        crate::summary::lines::write_fixture_lines(
            &self.index,
            &mut self.lines_written.borrow_mut(),
        )?;
        Ok(())
    }

    /// A kept result, as the writer stores one.
    pub fn kept_result(&mut self, id: &str, query: &str, rows_json: &str) -> Result<()> {
        crate::kept::validate_body(id, rows_json)?;
        self.archive_conn.execute(
            "INSERT INTO kept_result(id, query, ran_at, read_at, rows_json) VALUES (?1, ?2, 0, '{}', ?3)",
            params![id, query, rows_json],
        )?;
        Ok(())
    }

    /// Replace a stored kept body with deliberately damaged bytes for read tests.
    pub fn corrupt_kept_body(&mut self, id: &str, body: &str) -> Result<()> {
        self.archive_conn.execute(
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
    /// catalog, slot policy and saved-effect capture as the Writer.
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

    /// [`Self::user`] at a chosen request time.
    pub fn user_at(
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
        let tx = self.archive_conn.transaction()?;
        crate::user_writes::apply_resolving(
            &tx,
            &original,
            &request,
            self.commit_seq,
            crate::user_writes::Preparation::plain(None),
        )?;
        tx.commit()?;
        Ok(id)
    }

    /// Inserts a fixture annotation with a text or absent value.
    /// Kind and key follow the same checked shapes as stored annotation facts.
    pub fn note(
        &mut self,
        kind: &str,
        key: Option<&str>,
        value: Option<&str>,
        target: &str,
    ) -> Result<String> {
        use polylogue_model::{
            Author, Reference, Timestamp,
            annotation::{AnnotationFact, AnnotationKind, AnnotationLink, LinkRole},
            authorship::Channel,
        };
        self.commit_seq += 1;
        let id = format!("n{}", self.commit_seq);
        let kind =
            AnnotationKind::from_name(kind).or_internal("unknown fixture annotation kind")?;
        let target = Reference::parse(target).or_internal("invalid fixture annotation target")?;
        let fact = AnnotationFact {
            id: id.clone(),
            root: id.clone(),
            kind,
            key: key.map(str::to_owned),
            value: value.map(|value| polylogue_model::Json::String(value.into())),
            author: Author::Operator,
            via: Channel::Cli,
            machine: "test".into(),
            created_at: Some(
                Timestamp::from_unix_ms(self.commit_seq)
                    .expect("synthetic fixture counter is a timestamp"),
            ),
            valid_until: None,
            links: vec![AnnotationLink {
                role: LinkRole::About,
                reference: target,
                seen_hash: None,
                quote: None,
            }],
        };
        let tx = self.archive_conn.transaction()?;
        crate::annotations::apply(&tx, std::slice::from_ref(&fact), self.commit_seq)?;
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
        self.saved_query_revision(name, plan, None)
    }

    /// A new revision of the saved query whose family root and current
    /// revision are `revises`, or a new family.
    pub fn saved_query_revision(
        &mut self,
        name: &str,
        plan: &polylogue_query::plan::Plan,
        revises: Option<(&str, &str)>,
    ) -> Result<String> {
        use polylogue_model::{
            Author, Reference, Timestamp,
            annotation::{AnnotationFact, AnnotationKind, AnnotationLink, LinkRole},
            authorship::Channel,
        };
        let object = plan.query_object().map_err(crate::StoreError::Query)?;
        self.commit_seq += 1;
        let id = format!("n{}", self.commit_seq);
        let links = match revises {
            Some((root, revision)) => vec![AnnotationLink {
                role: LinkRole::Revises,
                reference: Reference::parse(&format!("annotation:{root}@{revision}"))
                    .map_err(|error| crate::StoreError::Invalid(error.to_string()))?,
                seen_hash: None,
                quote: None,
            }],
            None => vec![],
        };
        let fact = AnnotationFact {
            id: id.clone(),
            root: revises.map_or_else(|| id.clone(), |(root, _)| root.to_owned()),
            kind: AnnotationKind::SavedQuery,
            key: Some(name.into()),
            value: Some(serde_json::json!({
                "text": plan.text(), "hash":object.id.to_string(), "denominator":null,
            })),
            author: Author::Operator,
            via: Channel::Cli,
            machine: "test".into(),
            created_at: Some(Timestamp::from_unix_ms(self.commit_seq).unwrap()),
            valid_until: None,
            links,
        };
        let tx = self.archive_conn.transaction()?;
        crate::query_objects::apply_object(&tx, &object, self.commit_seq)?;
        crate::annotations::apply(&tx, &[fact], self.commit_seq)?;
        tx.commit()?;
        Ok(id)
    }
}

/// The parser version the writer fixtures parse `claude-code` chains under,
/// as the reading contract status and completeness judge them by.
pub fn synthetic_contracts() -> Vec<crate::ReadingContract> {
    vec![crate::ReadingContract::Expected {
        origin: "claude-code".into(),
        version: "synthetic=1".into(),
    }]
}

/// Marks one origin's readings as older, in an isolated test or benchmark archive.
/// The caller serializes writes and keeps the archive lock.
pub fn older_reading(archive: &Archive, origin: &str) -> Result<(u64, u64)> {
    let conn = open_write(&archive.index_db(), crate::archive::cache::SETUP)?;
    conn.execute(
        "ATTACH DATABASE ?1 AS a",
        [archive.archive_db().to_string_lossy().as_ref()],
    )?;
    crate::source::views(&conn)?;
    let size=conn.query_row("SELECT count(*),coalesce(sum((SELECT max(length) FROM a.revision r WHERE r.input=i.id AND r.chain=p.chain)),0)
        FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.format=?1",[origin],|r|Ok((r.get::<_,i64>(0)? as u64,r.get::<_,i64>(1)? as u64)))?;
    let tx = conn.unchecked_transaction()?;
    tx.execute("UPDATE parse_state SET parser_version='older reading' WHERE input_pk IN (SELECT pk FROM input_ref WHERE format=?1)",[origin])?;
    crate::archive::bump_counter(&tx)?;
    tx.commit()?;
    Ok(size)
}

fn fixture_time(at: Option<i64>) -> Result<Vec<u8>> {
    let at = at.and_then(polylogue_model::Timestamp::from_unix_ms);
    crate::encoding::time_bytes(
        at,
        if at.is_some() {
            polylogue_model::TimeAccuracy::intrinsic(polylogue_model::Precision::Instant)
        } else {
            polylogue_model::TimeAccuracy::UNKNOWN
        },
    )
}

#[cfg(test)]
mod tests;
