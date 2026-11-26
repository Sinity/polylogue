//! What Polylogue shows of an archive's sessions, as a plain value a test
//! compares for equality: [`Activity::expected`](crate::Activity::expected)
//! writes it from the docs, [`View::stored`] reads it from an archive
//! through the public reader and query text.

use crate::Home;
use polylogue_model::{Block, CallOutcome, LinkKind, PromptKind, Role, SessionId, SessionKind};

#[derive(Debug, Clone, PartialEq)]
pub struct View {
    /// Every physical session, by ID.
    pub sessions: Vec<SessionView>,
    /// Every link between sessions, each once, by parent, child and kind.
    pub links: Vec<LinkView>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionView {
    pub id: String,
    pub kind: SessionKind,
    pub cwd: Vec<String>,
    /// The messages stored in this session, in session order.
    pub messages: Vec<MessageView>,
    /// The IDs of the messages a read of the session shows with the history
    /// it copied from other sessions (`Reader::messages` with parents).
    pub read: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MessageView {
    pub id: String,
    pub role: Role,
    /// Milliseconds since the Unix epoch.
    pub at: Option<i64>,
    pub parent: Option<String>,
    /// The summary at a compaction boundary.
    pub compaction_summary: bool,
    pub blocks: Vec<BlockView>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BlockView {
    Text(String),
    Thinking(String),
    /// A call with the outcome its stored result relation gives it.
    Call {
        id: String,
        tool: String,
        command: Option<String>,
        outcome: Option<CallOutcome>,
    },
    Result {
        call: String,
        error: bool,
        output: String,
    },
    /// A block of a kind the view does not describe, by its kind.
    Other(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinkView {
    pub kind: LinkKind,
    pub parent: String,
    pub child: String,
    pub copies_prefix: bool,
}

impl View {
    /// The view of `home`'s archive.
    pub fn stored(home: &Home) -> Self {
        let ids: Vec<SessionId> = query_ids(home, "sessions where physical:true")
            .iter()
            .map(|id| SessionId::parse(id).expect("a session ID"))
            .collect();
        home.read(|reader| {
            let mut sessions = Vec::new();
            let mut links = Vec::new();
            for id in &ids {
                let session = reader
                    .session(id)?
                    .ok_or_else(|| polylogue_store::StoreError::NotFound(id.to_string()))?;
                let mut messages = Vec::new();
                for message in reader.messages(id, false)? {
                    let mut outcomes = reader.tool_outcomes(&message.id)?.into_iter();
                    let blocks = message
                        .blocks
                        .iter()
                        .map(|block| match block {
                            Block::Text { text } => BlockView::Text(text.clone()),
                            Block::Thinking { text, .. } => {
                                BlockView::Thinking(text.clone().unwrap_or_default())
                            }
                            Block::ToolCall {
                                call_id,
                                tool,
                                command,
                                ..
                            } => BlockView::Call {
                                id: call_id.clone().unwrap_or_default(),
                                tool: tool.clone().unwrap_or_default(),
                                command: command.clone(),
                                outcome: outcomes.next().map(|(_, outcome)| outcome),
                            },
                            Block::ToolResult {
                                call_id,
                                outcome,
                                output,
                                ..
                            } => BlockView::Result {
                                call: call_id.clone().unwrap_or_default(),
                                error: outcome.stored() == CallOutcome::Error,
                                output: output.clone().unwrap_or_default(),
                            },
                            other => BlockView::Other(other.kind().as_str().to_owned()),
                        })
                        .collect();
                    messages.push(MessageView {
                        id: message.id.to_string(),
                        role: message.role,
                        at: message.at.map(|at| at.unix_ms()),
                        parent: message.parent.as_ref().map(ToString::to_string),
                        compaction_summary: message.prompt_kind
                            == Some(PromptKind::CompactionSummary),
                        blocks,
                    });
                }
                let read = reader
                    .messages(id, true)?
                    .iter()
                    .map(|message| message.id.to_string())
                    .collect();
                for link in reader.links(id)? {
                    links.push(LinkView {
                        kind: link.kind,
                        parent: link.parent.to_string(),
                        child: link.child.to_string(),
                        copies_prefix: link.copies_prefix,
                    });
                }
                sessions.push(SessionView {
                    id: id.to_string(),
                    kind: session.kind,
                    cwd: session.cwd,
                    messages,
                    read,
                });
            }
            Ok(Self::sorted(sessions, links))
        })
    }

    /// Sessions by ID and links by parent, child and kind, each link once.
    pub fn sorted(mut sessions: Vec<SessionView>, mut links: Vec<LinkView>) -> Self {
        sessions.sort_by(|a, b| a.id.cmp(&b.id));
        let key = |l: &LinkView| {
            (
                l.parent.clone(),
                l.child.clone(),
                l.kind.as_str(),
                l.copies_prefix,
            )
        };
        links.sort_by_key(key);
        links.dedup_by_key(|l| key(l));
        Self { sessions, links }
    }
}

/// The `id` column of every row `text` answers, run as query text.
pub fn query_ids(home: &Home, text: &str) -> Vec<String> {
    let options = polylogue_store::query::Options {
        clock: query_language::time::Clock::new(
            "2026-10-07T10:00:00Z".parse().expect("a time"),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::UTC,
                jiff::civil::time(0, 0, 0, 0),
            )
            .expect("a day setting"),
        ),
        origins: polylogue_sources::ORIGINS
            .iter()
            .map(|origin| origin.name.to_owned())
            .collect(),
        page_size: None,
        random_seed: Some(1),
        embeddings: false,
        self_session: None,
        here: None,
        seen_marks_of: None,
        saved: Default::default(),
        fields: vec!["id".into()],
        kept: Default::default(),
        provider_catalog: None,
    };
    let program = polylogue_store::query::compile(text, &options).expect("compile the query");
    let pool = home.reads();
    let reader = pool.reader().expect("a reader");
    reader
        .query_rows(&program)
        .expect("run the query")
        .into_iter()
        .map(|row| row["id"].as_str().expect("an `id` column").to_owned())
        .collect()
}
