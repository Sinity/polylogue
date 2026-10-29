//! A writer over a temporary archive, for tests of what the writer and the
//! derive step make from parsed records: a test hands it messages or a
//! parser's output and reads back the rows. Its synthetic record bytes use
//! the production source store; no provider file or parser is involved.

use crate::writer::{Batch, InputKey, Observation, ParseStep};
use crate::{Archive, ReadPool, Writer};
use polylogue_model::{
    Block, ErrorReason, Json, Message, MessageId, MessageRow, Output, PromptKind, RecordCounts,
    RecordOutcome, Role, Session, SessionId, SessionKind, Timestamp, ToolAction, ToolOutcome,
    Usage,
};

/// An archive in a temporary directory with its writer running.
pub struct WriterArchive {
    pub writer: Writer,
    pub archive: Archive,
    _lock: crate::ArchiveLock,
    offset: u64,
    positions: std::collections::BTreeMap<String, u64>,
    /// Links sent with the next write.
    pub links: Vec<polylogue_model::Link>,
    /// Repository lookups sent with the next write.
    pub lookups: Vec<polylogue_model::RepoLookup>,
    _dir: tempfile::TempDir,
}

impl Default for WriterArchive {
    fn default() -> Self {
        Self::new()
    }
}

pub fn sid(n: &str) -> SessionId {
    SessionId::parse(&format!("claude-code:{n}")).unwrap()
}

pub fn message(id: &str, session: &str, role: Role, at: i64, blocks: Vec<Block>) -> Message {
    let at = Timestamp::from_unix_ms(at);
    let evidence = at.map_or(polylogue_model::TimeEvidence::UNKNOWN, |_| {
        polylogue_model::TimeEvidence::intrinsic(polylogue_model::Precision::Instant)
    });
    Message {
        at_quality: evidence.quality,
        at_precision: evidence.precision,
        at_until: evidence.until,
        id: MessageId::parse(&format!("claude-code:{session}-{id}")).unwrap(),
        session: sid(session),
        parent: None,
        role,
        at,
        model: None,
        prompt_kind: (role == Role::User).then_some(PromptKind::Prose),
        blocks,
        usage: Usage::default(),
        stop_reason: None,
        reported_cost_usd: None,
        duration_ms: None,
        production: Default::default(),
        attrs: Default::default(),
    }
}

/// One tool call and its result, as two messages. `result` is the output,
/// `Ok` for a successful call and `Err` for a failed one.
pub fn call(
    session: &str,
    n: u32,
    at: i64,
    tool: &str,
    command: Option<&str>,
    input: Json,
    result: Result<&str, &str>,
) -> Vec<Message> {
    let (ok, output) = match result {
        Ok(o) => (true, o),
        Err(o) => (false, o),
    };
    let call_id = format!("t{n}");
    let c = message(
        &format!("c{n}"),
        session,
        Role::Assistant,
        at,
        vec![Block::ToolCall {
            run_by_script: false,
            call_id: Some(call_id.clone()),
            tool: Some(tool.into()),
            // A parser decides the action from its tool names; here a call
            // with a command is a shell call.
            action: if command.is_some() {
                ToolAction::Shell
            } else {
                ToolAction::Other
            },
            input,
            command: command.map(Into::into),
            path: None,
        }],
    );
    let outcome = if ok {
        ToolOutcome::Ok
    } else {
        ToolOutcome::Error {
            reason: Some(ErrorReason::ExitCode(1)),
        }
    };
    let r = message(
        &format!("r{n}"),
        session,
        Role::User,
        at + 1,
        vec![Block::ToolResult {
            reported_duration_ms: None,
            call_id: Some(call_id),
            outcome,
            output: Some(output.into()),
            data: None,
            attachments: vec![],
        }],
    );
    vec![c, r]
}

pub fn bash(session: &str, n: u32, at: i64, command: &str, ok: bool, output: &str) -> Vec<Message> {
    let result = if ok { Ok(output) } else { Err(output) };
    call(session, n, at, "Bash", Some(command), Json::Null, result)
}

pub fn prose(session: &str, n: u32, at: i64, text: &str) -> Vec<Message> {
    vec![message(
        &format!("p{n}"),
        session,
        Role::User,
        at,
        vec![Block::Text { text: text.into() }],
    )]
}

impl WriterArchive {
    pub fn new() -> Self {
        let dir = tempfile::tempdir().expect("temporary directory");
        let (archive, lock) =
            Archive::open_or_create(&dir.path().join("archive")).expect("open the archive");
        let writer = Writer::start(&archive).expect("start the writer");
        Self {
            _dir: dir,
            archive,
            _lock: lock,
            writer,
            offset: 0,
            positions: Default::default(),
            links: vec![],
            lookups: vec![],
        }
    }

    /// Writes `messages` as one appended step of the session's input.
    pub fn write(&mut self, session: &str, messages: Vec<Message>) {
        self.write_session(Session::new(sid(session), SessionKind::Main), messages);
    }

    /// As [`WriterArchive::write`], with the session's own facts.
    pub fn write_session(&mut self, s: Session, messages: Vec<Message>) {
        let input = format!("input-{}", s.id.native().as_str());
        self.write_from(&input, s, messages);
    }

    /// As [`WriterArchive::write_session`], from the named input.
    pub fn write_from(&mut self, input: &str, s: Session, messages: Vec<Message>) {
        let path = format!("/p/{}", s.id.native().as_str());
        let start = self.positions.get(input).copied().unwrap_or(0);
        let rows: Vec<MessageRow> = messages
            .into_iter()
            .enumerate()
            .map(|(i, message)| MessageRow {
                offset: start + i as u64,
                end: start + i as u64 + 1,
                message,
            })
            .collect();
        let records = rows.len() as u64;
        let output = Output {
            sessions: vec![s],
            messages: rows,
            links: std::mem::take(&mut self.links),
            repo_lookups: std::mem::take(&mut self.lookups),
            ..Output::default()
        };
        self.write_output(input, "claude-code", &path, output, records);
        self.writer.derive().expect("derive");
    }

    /// The last written input's next byte offset.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Writes `output` as one appended step of `input` (in `format`, found
    /// at `path`) holding `records` records, each one byte at the input's
    /// next offsets, and waits until it is committed.
    pub fn write_output(
        &mut self,
        input: &str,
        format: &str,
        path: &str,
        output: Output,
        records: u64,
    ) {
        let start = self.positions.get(input).copied().unwrap_or(0);
        let mut bytes = vec![b'r'; usize::try_from(records).expect("synthetic record count")];
        bytes.push(b'\n');
        let mut batch = step(input, format, path, output, start, records);
        let mut raw = batch.clone();
        raw.parse.clear();
        let prepared = self
            .archive
            .prepare_bytes(&bytes[..])
            .expect("prepare source bytes");
        let stored = self
            .writer
            .store_bytes(prepared, raw)
            .expect("commit source bytes");
        assert_eq!(stored.length, batch.length);
        batch.change = crate::Change::Reparse {
            chain: stored.chain,
        };
        batch.observation = None;
        self.writer.send(batch).expect("send the reading");
        let report = self.writer.flush().expect("flush");
        assert!(report.errors.is_empty(), "{report:?}");
        self.offset = stored.length;
        self.positions.insert(input.into(), stored.length);
    }

    pub fn rows(&self, sql: &str) -> Vec<Json> {
        let pool = ReadPool::open(&self.archive).unwrap();
        pool.reader().unwrap().query(sql, &[]).unwrap()
    }

    pub fn live_task(&self, session: &str) -> Json {
        self.rows(&format!(
            "SELECT live_task, live_task_evidence FROM session WHERE id = 'claude-code:{session}'"
        ))[0]
            .clone()
    }
}

/// One appended step of `input` (in `format`, found at `path`): `records`
/// records of one byte each from offset `start`, parsed into `output`.
pub fn step(
    input: &str,
    format: &str,
    path: &str,
    output: Output,
    start: u64,
    records: u64,
) -> Batch {
    let length = start + records + 1;
    let mut counts = RecordCounts {
        records,
        ..RecordCounts::default()
    };
    for _ in 0..records {
        counts.add("record", &RecordOutcome::Used);
    }
    Batch {
        input: InputKey {
            id: input.to_owned(),
            format: format.into(),
            arrival_rank: 0,
        },
        change: if start == 0 {
            crate::Change::First
        } else {
            crate::Change::Appended { from: start }
        },
        observation: Some(Observation {
            machine: "m".into(),
            host: "h".into(),
            path: path.into(),
            facts: None,
            seen_at_ms: 1,
            import: None,
            container: None,
        }),
        continues: false,
        new_blob: None,
        blobs: vec![],
        length,
        release: Default::default(),
        parse: vec![{
            let mut step = ParseStep::new(output, counts, start..length, true);
            step.reading_version = "synthetic=1".into();
            step
        }],
    }
}

#[cfg(test)]
mod tests;
