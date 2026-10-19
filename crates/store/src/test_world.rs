//! A writer over a temporary archive, for tests of what the derive step
//! makes from written records.

use crate::writer::{Batch, InputKey, Observation, ParseStep};
use crate::{Archive, ReadPool, Writer};
use polylogue_model::{
    Accounting, Block, Disposition, ErrorReason, Json, Message, MessageId, MessageRow, Output,
    PromptKind, Role, Session, SessionId, SessionKind, Timestamp, ToolOutcome, Usage,
};

pub(crate) struct World {
    _dir: tempfile::TempDir,
    pub(crate) archive: Archive,
    _lock: crate::ArchiveLock,
    pub(crate) writer: Writer,
    offset: u64,
    /// Links sent with the next write.
    pub(crate) links: Vec<polylogue_model::Link>,
    /// Repository lookups sent with the next write.
    pub(crate) lookups: Vec<polylogue_model::RepoLookup>,
}

pub(crate) fn world() -> World {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let writer = Writer::start(&archive).unwrap();
    World {
        _dir: dir,
        archive,
        _lock: lock,
        writer,
        offset: 0,
        links: vec![],
        lookups: vec![],
    }
}

pub(crate) fn sid(n: &str) -> SessionId {
    SessionId::parse(&format!("claude-code:{n}")).unwrap()
}

pub(crate) fn message(id: &str, session: &str, role: Role, at: i64, blocks: Vec<Block>) -> Message {
    Message {
        id: MessageId::parse(&format!("claude-code:{session}-{id}")).unwrap(),
        session: sid(session),
        parent: None,
        role,
        at: Timestamp::from_unix_ms(at),
        model: None,
        prompt_kind: (role == Role::User).then_some(PromptKind::Prose),
        blocks,
        usage: Usage::default(),
        stop_reason: None,
        reported_cost_usd: None,
        duration_ms: None,
        production: Default::default(),
    }
}

/// One tool call and its result, as two messages. `result` is the output,
/// `Ok` for a successful call and `Err` for a failed one.
pub(crate) fn call(
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
            call_id: Some(call_id.clone()),
            tool: Some(tool.into()),
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
            call_id: Some(call_id),
            outcome,
            output: Some(output.into()),
            data: None,
            attachments: vec![],
        }],
    );
    vec![c, r]
}

pub(crate) fn bash(
    session: &str,
    n: u32,
    at: i64,
    command: &str,
    ok: bool,
    output: &str,
) -> Vec<Message> {
    let result = if ok { Ok(output) } else { Err(output) };
    call(session, n, at, "Bash", Some(command), Json::Null, result)
}

pub(crate) fn prose(session: &str, n: u32, at: i64, text: &str) -> Vec<Message> {
    vec![message(
        &format!("p{n}"),
        session,
        Role::User,
        at,
        vec![Block::Text { text: text.into() }],
    )]
}

impl World {
    /// Writes `messages` as one appended step of the session's input.
    pub(crate) fn write(&mut self, session: &str, messages: Vec<Message>) {
        self.write_session(Session::new(sid(session), SessionKind::Main), messages);
    }

    /// As [`World::write`], with the session's own facts.
    pub(crate) fn write_session(&mut self, s: Session, messages: Vec<Message>) {
        let input = format!("input-{}", s.id.native().as_str());
        self.write_from(&input, s, messages);
    }

    /// As [`World::write_session`], from the named input.
    pub(crate) fn write_from(&mut self, input: &str, s: Session, messages: Vec<Message>) {
        let session = s.id.native().as_str().to_owned();
        let session = session.as_str();
        let start = self.offset;
        let rows: Vec<MessageRow> = messages
            .into_iter()
            .enumerate()
            .map(|(i, message)| MessageRow {
                offset: start + i as u64,
                end: start + i as u64 + 1,
                message,
            })
            .collect();
        let n = rows.len() as u64;
        self.offset += n + 1;
        let mut accounting = Accounting {
            framed: n,
            ..Accounting::default()
        };
        for _ in 0..n {
            accounting.add("record", &Disposition::Materialized);
        }
        let batch = Batch {
            input: InputKey {
                id: input.to_owned(),
                format: "claude-code".into(),
                route_rank: 0,
            },
            change: if start == 0 {
                crate::Change::First
            } else {
                crate::Change::Appended { from: start }
            },
            observation: Some(Observation {
                machine: "m".into(),
                host: "h".into(),
                path: format!("/p/{session}"),
                facts: None,
                seen_at_ms: 1,
            }),
            continues: false,
            piece: None,
            blobs: vec![],
            length: self.offset,
            release: Default::default(),
            parse: Some(ParseStep {
                output: Output {
                    sessions: vec![s],
                    messages: rows,
                    links: std::mem::take(&mut self.links),
                    repo_lookups: std::mem::take(&mut self.lookups),
                    ..Output::default()
                },
                accounting,
                start,
                parsed_upto: self.offset,
                parser_version: 1,
                hashes: vec![],
                packed: Default::default(),
                bodies: vec![],
                events: vec![],
            }),
        };
        self.writer.send(batch).unwrap();
        self.writer.flush().unwrap();
        self.writer.derive().unwrap();
    }

    pub(crate) fn rows(&self, sql: &str) -> Vec<Json> {
        let pool = ReadPool::open(&self.archive).unwrap();
        pool.reader().unwrap().query(sql, &[]).unwrap()
    }

    pub(crate) fn live_task(&self, session: &str) -> Json {
        self.rows(&format!(
            "SELECT live_task, live_task_evidence FROM session WHERE id = 'claude-code:{session}'"
        ))[0]
            .clone()
    }
}
