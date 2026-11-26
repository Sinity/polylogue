//! What the harnesses did, described once in their own terms, so a test can
//! write the files each tool would have written ([`Activity::claude_code_files`])
//! and, separately, the view Polylogue shows for them ([`Activity::expected`],
//! [`View`](crate::View)). Neither shares code with the parsers or the store:
//! the encoder is written from the format as the golden inputs show it, and
//! the expected view from `docs/sessions.md`.
//!
//! Claude Code is described as it writes `projects/<project>/<session>.jsonl`:
//! one JSON record per line, a response written as several records that
//! share `message.id`, `parentUuid` naming the record replied to, and a resume
//! that begins its file with records copied from another session's file.

mod claude_code;
mod expected;

pub use claude_code::File;

/// Sessions as the harnesses wrote them, in the order their files were
/// started. A resume copies from a session earlier in the list.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Activity {
    pub sessions: Vec<Transcript>,
}

/// One Claude Code session file: the session it starts, the directory it
/// ran in, the records it copied from another file, and its own entries in
/// file order.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    /// The session's native ID, which is also the file name's stem.
    pub session: String,
    pub cwd: String,
    /// The leading entries of another session's file, copied before this
    /// file's own entries, as a resume writes them.
    pub copy: Option<Copy>,
    pub entries: Vec<Entry>,
}

/// A resume's copy of the start of another file.
#[derive(Debug, Clone, PartialEq)]
pub struct Copy {
    /// The copied file's index in [`Activity::sessions`], lower than the
    /// copying one's. Copying a file that itself begins with a copy is a
    /// chained resume.
    pub from: usize,
    /// How many of that file's lines are copied, its own copied lines
    /// included. Whole entries only, and none of its sidechain entries or
    /// those after the first one.
    pub lines: usize,
    pub change: Option<Change>,
}

/// One copied entry written differently from the original: a response the
/// copy caught while it was being written, so it has no stop reason. The
/// target is a one-record response with a text part; the original, which
/// finished, is the one shown (`docs/sessions.md`, one conversation through
/// several routes, rule 4).
#[derive(Debug, Clone, PartialEq)]
pub struct Change {
    /// Index among the copied lines.
    pub line: usize,
    pub how: ChangeHow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeHow {
    /// The first text part says something else.
    Edited,
    /// The first text part ends halfway.
    CutShort,
}

/// One message as the harness wrote it: one record, or several records of
/// one response. Every record of an entry has the entry's time.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    /// The first record's `uuid`, which names the message.
    pub uuid: String,
    /// The record `parentUuid` names. A fork is two entries naming one.
    pub parent: Option<String>,
    /// Milliseconds since the Unix epoch.
    pub at: i64,
    /// Written with `isSidechain: true`: a side conversation kept in the
    /// session's file, a session of its own (`<session>/sidechain`).
    pub sidechain: bool,
    pub body: Body,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Body {
    /// A person's prompt: a user record whose content is the text.
    Prompt(String),
    /// A model response: assistant records sharing `message.id` (`key`),
    /// each holding some of its parts, the first one's `uuid` being the
    /// entry's. More than one record is a split response.
    Response {
        key: String,
        records: Vec<ResponseRecord>,
    },
    /// A user record holding only tool results.
    Results(Vec<ToolResult>),
    /// A compaction: a `compact_boundary` record whose `logicalParentUuid`
    /// is the entry's parent, then the summary record (the entry's `uuid`)
    /// whose `parentUuid` is the boundary.
    Compaction { boundary: String, summary: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResponseRecord {
    pub uuid: String,
    pub parts: Vec<Part>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Part {
    Text(String),
    Thinking(String),
    /// A `Bash` call running `command`.
    Call {
        id: String,
        command: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolResult {
    /// The call's ID.
    pub call: String,
    pub error: bool,
    pub output: String,
}

/// One line group of a file: an entry and the session it belongs to, with
/// the change a copy made to it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Line<'a> {
    pub owner: &'a str,
    pub cwd: &'a str,
    pub entry: &'a Entry,
    pub change: Option<ChangeHow>,
}

impl Activity {
    /// The entries of `index`'s file in file order: its copied lines, then
    /// its own entries.
    pub(crate) fn lines(&self, index: usize) -> Vec<Line<'_>> {
        let transcript = &self.sessions[index];
        let mut lines = match &transcript.copy {
            Some(copy) => {
                assert!(copy.from < index, "a resume copies an earlier file");
                let mut copied = self.lines(copy.from);
                copied.truncate(copy.lines);
                if let Some(change) = &copy.change {
                    copied[change.line].change = Some(change.how);
                }
                copied
            }
            None => Vec::new(),
        };
        lines.extend(transcript.entries.iter().map(|entry| Line {
            owner: &transcript.session,
            cwd: &transcript.cwd,
            entry,
            change: None,
        }));
        lines
    }
}

impl Transcript {
    pub fn new(session: &str, cwd: &str) -> Self {
        Self {
            session: session.into(),
            cwd: cwd.into(),
            copy: None,
            entries: Vec::new(),
        }
    }

    /// A resume of file `from`, copying its first `lines` lines.
    pub fn resuming(session: &str, cwd: &str, from: usize, lines: usize) -> Self {
        Self {
            copy: Some(Copy {
                from,
                lines,
                change: None,
            }),
            ..Self::new(session, cwd)
        }
    }

    /// Adds an entry replying to `parent`; returns it for further changes.
    pub fn push(&mut self, uuid: &str, parent: Option<&str>, at: i64, body: Body) -> &mut Entry {
        self.entries.push(Entry {
            uuid: uuid.into(),
            parent: parent.map(str::to_owned),
            at,
            sidechain: false,
            body,
        });
        self.entries.last_mut().expect("just pushed")
    }

    pub fn prompt(&mut self, uuid: &str, parent: Option<&str>, at: i64, text: &str) -> &mut Entry {
        self.push(uuid, parent, at, Body::Prompt(text.into()))
    }

    /// A response written as one record.
    pub fn response(
        &mut self,
        uuid: &str,
        parent: Option<&str>,
        at: i64,
        parts: Vec<Part>,
    ) -> &mut Entry {
        self.split_response(parent, at, vec![(uuid, parts)])
    }

    /// A response written as one record per `(uuid, parts)`.
    pub fn split_response(
        &mut self,
        parent: Option<&str>,
        at: i64,
        records: Vec<(&str, Vec<Part>)>,
    ) -> &mut Entry {
        let uuid = records.first().expect("a response has a record").0;
        let body = Body::Response {
            key: format!("msg_{uuid}"),
            records: records
                .into_iter()
                .map(|(uuid, parts)| ResponseRecord {
                    uuid: uuid.into(),
                    parts,
                })
                .collect(),
        };
        self.push(uuid, parent, at, body)
    }

    pub fn results(
        &mut self,
        uuid: &str,
        parent: Option<&str>,
        at: i64,
        results: Vec<ToolResult>,
    ) -> &mut Entry {
        self.push(uuid, parent, at, Body::Results(results))
    }
}

impl Entry {
    /// The `uuid` of the entry's last record: what a reply names.
    pub fn last_record(&self) -> &str {
        match &self.body {
            Body::Response { records, .. } => &records.last().expect("a record").uuid,
            _ => &self.uuid,
        }
    }

    /// The record uuids this entry writes, the boundary of a compaction
    /// included.
    pub fn record_uuids(&self) -> Vec<&str> {
        match &self.body {
            Body::Response { records, .. } => records.iter().map(|r| r.uuid.as_str()).collect(),
            Body::Compaction { boundary, .. } => vec![boundary, &self.uuid],
            _ => vec![&self.uuid],
        }
    }
}

impl Part {
    pub fn text(text: &str) -> Self {
        Self::Text(text.into())
    }

    pub fn call(id: &str, command: &str) -> Self {
        Self::Call {
            id: id.into(),
            command: command.into(),
        }
    }
}

impl ToolResult {
    pub fn ok(call: &str, output: &str) -> Self {
        Self {
            call: call.into(),
            error: false,
            output: output.into(),
        }
    }

    pub fn error(call: &str, output: &str) -> Self {
        Self {
            error: true,
            ..Self::ok(call, output)
        }
    }
}

/// The text a [`Change`] gives a copied entry's first text part.
pub(crate) fn changed_text(text: &str, how: ChangeHow) -> String {
    match how {
        ChangeHow::Edited => format!("{text} (rewritten)"),
        ChangeHow::CutShort => {
            let keep = text.chars().count().div_ceil(2);
            text.chars().take(keep).collect()
        }
    }
}
