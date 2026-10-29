//! Claude Code session files under its configuration directory
//! (`$CLAUDE_CONFIG_DIR`, default `~/.claude`):
//!
//! - `projects/<project>/<session>.jsonl`, the main transcript, and a copy
//!   some versions keep at `<project>/<session>/<session>.jsonl`;
//! - subagent transcripts in `<project>/<session>/subagents/agent-<id>.jsonl`,
//!   Workflow agents one level deeper in `subagents/workflows/<run>/`, and
//!   the older layout `<project>/agent-<id>.jsonl`;
//! - a Workflow run's journal, `subagents/workflows/<run>/journal.jsonl`.
//!
//! One line is one record. An assistant response arrives as several
//! records sharing `message.id`, one per content block; they become one
//! message. A message copied from another session (a resume replays the
//! earlier history) keeps its ID and its original session, so it is stored
//! once, and the copying session gets a `copy` link.

/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

mod blocks;
mod fields;
pub mod side;

use crate::lines::Record;
use crate::origin::Root;
use crate::parse::{BackgroundUpdate, MessageRow, Output, Parse, Parser};
use crate::time;
use crate::{RecordOutcome, Records};
use polylogue_model::{
    ApiError, Attribution, Block, ErrorReason, Event, Json, Link, LinkKind, Message, MessageId,
    NativeId, Origin, Production, PromptKind, QuotaLimit, QuotaReading, QuotaWindow, Role, Session,
    SessionId, SessionKind, Timestamp, TitleSource, ToolOutcome, Usage,
};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;

pub const ROOT: Root = Root {
    env: Some("CLAUDE_CONFIG_DIR"),
    default: ".claude",
};

pub const LOCATIONS: &[&str] = &[
    "projects/*/*.jsonl",
    "projects/*/*/*.jsonl",
    "projects/*/*/subagents/agent-*.jsonl",
    "projects/*/*/subagents/workflows/*/agent-*.jsonl",
    "projects/*/*/subagents/workflows/*/journal.jsonl",
];

/// How many closed responses stay reopenable. A continuation record can
/// arrive after another message (3,117 of 1.2M continuations on the real
/// archive); a few slots cover them without an unbounded ID set.
const REOPEN_WINDOW: usize = 4;

pub fn recognize(path: &Path, head: &[u8]) -> bool {
    File::from_path(path).is_some()
        && head
            .iter()
            .find(|b| !b.is_ascii_whitespace())
            .is_none_or(|&b| b == b'{')
}

pub fn open(path: &Path) -> Option<Parse> {
    File::from_path(path).map(|file| Parse::Lines(Box::new(ClaudeCode { file })))
}

/// `claude --resume <id>`. Subagents and sidechains are not resumable on
/// their own.
pub fn resume(native: &str) -> Option<Vec<String>> {
    if native.starts_with("agent-") || native.ends_with("/sidechain") {
        return None;
    }
    Some(vec!["claude".into(), "--resume".into(), native.into()])
}

/// Which kind of file is being read, from its place in the directory tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum File {
    /// A session's own transcript. `stem` is the session ID the file name
    /// carries (an `.orphaned-…` suffix removed).
    Main { stem: String },
    /// A subagent's transcript. `stem` is the file name (`agent-<id>`).
    Subagent { stem: String },
    /// A Workflow run's journal: which agents it started and what each
    /// returned. `session` is the session that ran the workflow.
    Journal { session: String, run: String },
}

impl File {
    pub fn from_path(path: &Path) -> Option<Self> {
        if path.extension()? != "jsonl" {
            return None;
        }
        let name = path.file_stem()?.to_str()?;
        let dirs: Vec<&str> = path
            .ancestors()
            .skip(1)
            .take(4)
            .map(|p| p.file_name().and_then(|n| n.to_str()).unwrap_or(""))
            .collect();
        let dir = |i: usize| dirs.get(i).copied().unwrap_or("");
        // `<session>/subagents/workflows/<run>/…`
        let in_workflow = dir(1) == "workflows" && dir(2) == "subagents";
        if in_workflow && name == "journal" {
            return Some(Self::Journal {
                session: dir(3).to_owned(),
                run: dir(0).to_owned(),
            });
        }
        let agent = name.starts_with("agent-");
        if agent && (dir(0) == "subagents" || in_workflow) {
            return Some(Self::Subagent {
                stem: name.to_owned(),
            });
        }
        if dir(0) == "subagents" || in_workflow {
            return None;
        }
        if agent {
            // The older layout kept subagent transcripts beside the session.
            return Some(Self::Subagent {
                stem: name.to_owned(),
            });
        }
        let stem = name.split('.').next().unwrap_or(name);
        // Inside the tool's own tree the layout decides: a copy of the
        // transcript in the session's own directory reads as another input
        // of the same session, and any other file there is not a
        // transcript. An exported file brought in by `import` can sit
        // anywhere.
        let in_tree = path
            .ancestors()
            .skip(1)
            .any(|p| p.file_name().is_some_and(|n| n == "projects"));
        if in_tree && dir(1) != "projects" && dir(0) != stem {
            return None;
        }
        Some(Self::Main {
            stem: stem.to_owned(),
        })
    }

    fn stem(&self) -> &str {
        match self {
            Self::Main { stem } | Self::Subagent { stem } => stem,
            Self::Journal { session, .. } => session,
        }
    }
}

pub struct ClaudeCode {
    pub file: File,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParserState {
    /// Session facts gathered so far, by native ID. A file touches a few
    /// sessions: its own, the ones it copies from, its sidechain.
    sessions: BTreeMap<String, Session>,
    /// Embedded transcripts use the same response assembly as separate files.
    #[serde(default)]
    agents: BTreeMap<String, Box<ParserState>>,
    /// The assistant response still being assembled.
    open: Option<Response>,
    /// Recently closed responses that a late continuation may reopen.
    recent: VecDeque<Response>,
    /// Links already sent, as `kind parent child`.
    links: BTreeSet<String>,
    /// The latest message, for placing events on the timeline.
    last: Option<MessageId>,
    /// Record uuids that are not a message's own: a split response's later
    /// records (its message carries the first one's uuid) and event records
    /// (compaction boundaries, attachments, progress), each mapped to the
    /// message a child naming it as parent belongs under. Recent ones only:
    /// children follow their parents closely.
    #[serde(default)]
    aliases: VecDeque<(String, Option<MessageId>)>,
    /// The session-wide names (plan `slug`, `teamName`) last recorded, by
    /// kind and session, so each event is sent once per name, not with
    /// every record.
    #[serde(default)]
    names: BTreeMap<String, String>,
}

/// How many record aliases a parser keeps.
const ALIASES: usize = 4096;

impl ParserState {
    fn alias(&mut self, uuid: &str, target: Option<MessageId>) {
        if self.aliases.len() == ALIASES {
            self.aliases.pop_front();
        }
        self.aliases.push_back((uuid.to_owned(), target));
    }

    /// The message a record's parent pointer names, through aliases.
    fn parent(&self, v: &Json) -> Option<MessageId> {
        self.resolve(str_field(v, "parentUuid")?)
    }

    fn resolve(&self, uuid: &str) -> Option<MessageId> {
        match self.aliases.iter().rev().find(|(u, _)| u == uuid) {
            Some((_, target)) => target.clone(),
            None => message_id(uuid),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Response {
    key: String,
    /// Where the response's first record starts and its latest one ends.
    offset: u64,
    end: u64,
    message: Message,
}

fn origin() -> Origin {
    Origin::new("claude-code").expect("constant origin")
}

fn str_field<'a>(v: &'a Json, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Json::as_str).filter(|s| !s.is_empty())
}

fn session_id(native: &str) -> Option<SessionId> {
    Some(SessionId::new(origin(), NativeId::new(native).ok()?))
}

fn message_id(native: &str) -> Option<MessageId> {
    Some(MessageId::new(origin(), NativeId::new(native).ok()?))
}

/// The session a record belongs to and its kind.
struct RecordSession {
    native: String,
    kind: SessionKind,
}

impl ClaudeCode {
    fn record_session(&self, v: &Json) -> RecordSession {
        match &self.file {
            File::Subagent { stem } => {
                let native = str_field(v, "agentId")
                    .map(|a| format!("agent-{a}"))
                    .unwrap_or_else(|| stem.clone());
                // The call that compacted the parent's history runs as an
                // agent named `acompact-…`.
                let kind = if native.starts_with("agent-acompact") {
                    SessionKind::Compaction
                } else {
                    SessionKind::Subagent
                };
                RecordSession { native, kind }
            }
            File::Journal { session, .. } => RecordSession {
                native: session.clone(),
                kind: SessionKind::Main,
            },
            File::Main { stem } => {
                let base = str_field(v, "sessionId").unwrap_or(stem).to_owned();
                if v.get("isSidechain").and_then(Json::as_bool) == Some(true) {
                    RecordSession {
                        native: format!("{base}/sidechain"),
                        kind: SessionKind::Sidechain,
                    }
                } else {
                    RecordSession {
                        native: base,
                        kind: SessionKind::Main,
                    }
                }
            }
        }
    }

    /// The session's facts entry, created on first sight, with the
    /// record's envelope facts offered to it. Also sends the link that
    /// explains why a record belongs to a session other than the file's
    /// own.
    fn session<'c>(
        &self,
        parser_state: &'c mut ParserState,
        v: &Json,
        out: &mut Output,
    ) -> Option<&'c mut Session> {
        let record_session = self.record_session(v);
        let id = session_id(&record_session.native)?;
        let parent = str_field(v, "sessionId").and_then(session_id);
        match (&self.file, record_session.kind) {
            (File::Main { stem }, SessionKind::Main) if record_session.native != *stem => {
                if let Some(child) = session_id(stem) {
                    link(
                        parser_state,
                        out,
                        LinkKind::Copy,
                        id.clone(),
                        child,
                        "sessionId",
                    );
                }
            }
            (_, SessionKind::Sidechain)
            | (File::Subagent { .. }, SessionKind::Subagent | SessionKind::Compaction) => {
                if let Some(parent) = parent.filter(|p| *p != id) {
                    let evidence = match record_session.kind {
                        SessionKind::Sidechain => "isSidechain",
                        SessionKind::Compaction => "agentId acompact",
                        _ => "agentId",
                    };
                    link(
                        parser_state,
                        out,
                        LinkKind::Spawn,
                        parent,
                        id.clone(),
                        evidence,
                    );
                }
            }
            _ => {}
        }
        let s = parser_state
            .sessions
            .entry(record_session.native)
            .or_insert_with(|| Session::new(id, record_session.kind));
        if let Some(cwd) = str_field(v, "cwd") {
            s.offer_cwd(cwd);
        }
        if let Some(branch) = str_field(v, "gitBranch") {
            s.git_branch = Some(branch.to_owned());
        }
        if let Some(version) = str_field(v, "version") {
            s.tool_version = Some(version.to_owned());
        }
        if let Some(at) = v.get("timestamp").and_then(time::rfc3339) {
            s.offer_start(at);
        }
        if let Some(mode) = str_field(v, "sessionKind") {
            s.mode = Some(mode.to_owned());
        }
        Some(s)
    }

    /// The file's own session, for records that name none.
    fn own_session(&self, parser_state: &mut ParserState) -> Option<SessionId> {
        let stem = self.file.stem();
        let kind = match self.file {
            File::Main { .. } | File::Journal { .. } => SessionKind::Main,
            File::Subagent { .. } if stem.starts_with("agent-acompact") => SessionKind::Compaction,
            File::Subagent { .. } => SessionKind::Subagent,
        };
        let id = session_id(stem)?;
        parser_state
            .sessions
            .entry(stem.to_owned())
            .or_insert_with(|| Session::new(id.clone(), kind));
        Some(id)
    }

    /// Session-wide names carried on every record: the plan file the
    /// session writes (`slug`, as `plans/<slug>.md`) and the agent team it
    /// belongs to (`teamName`). Each is sent as an event when it first
    /// appears or changes, so readers find it from the session.
    fn names(
        &self,
        parser_state: &mut ParserState,
        record: Record<'_>,
        v: &Json,
        out: &mut Output,
    ) {
        if let Some(slug) = str_field(v, "slug") {
            let data = serde_json::json!({ "slug": slug, "file": format!("plans/{slug}.md") });
            self.name_event(parser_state, record, v, out, "plan", slug, data);
        }
        if let Some(team) = str_field(v, "teamName") {
            let data = serde_json::json!({ "team": team });
            self.name_event(parser_state, record, v, out, "team", team, data);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn name_event(
        &self,
        parser_state: &mut ParserState,
        record: Record<'_>,
        v: &Json,
        out: &mut Output,
        kind: &str,
        name: &str,
        data: Json,
    ) {
        let Some(stem) = str_field(v, "sessionId").or(match &self.file {
            File::Main { stem } | File::Subagent { stem } => Some(stem.as_str()),
            File::Journal { .. } => None,
        }) else {
            return;
        };
        let key = format!("{kind}\u{0}{stem}");
        if parser_state.names.get(&key).map(String::as_str) == Some(name) {
            return;
        }
        parser_state.names.insert(key, name.to_owned());
        if let Some(session) = session_id(stem) {
            out.events.push(Event {
                session: Some(session),
                kind: kind.to_owned(),
                at: v.get("timestamp").and_then(time::rfc3339),
                after: parser_state.last.clone(),
                range: Some(range(record)),
                data,
            });
        }
    }

    fn event(
        &self,
        parser_state: &mut ParserState,
        record: Record<'_>,
        v: &Json,
        kind: &str,
        out: &mut Output,
    ) -> RecordOutcome {
        let session = match v.get("sessionId").or(v.get("agentId")) {
            Some(_) => self.session(parser_state, v, out).map(|s| s.id.clone()),
            None => self.own_session(parser_state),
        };
        let Some(session) = session else {
            return RecordOutcome::unreadable("session ID is empty");
        };
        out.events.push(Event {
            session: Some(session),
            kind: kind.to_owned(),
            at: v.get("timestamp").and_then(time::rfc3339),
            after: parser_state.last.clone(),
            range: Some(range(record)),
            data: v.clone(),
        });
        RecordOutcome::Used
    }

    fn hook_summary(
        &self,
        parser_state: &mut ParserState,
        record: Record<'_>,
        v: &Json,
        summary: (Vec<Block>, Option<&'static str>),
        out: &mut Output,
    ) -> RecordOutcome {
        let outcome = self.event(parser_state, record, v, "stop_hook_summary", out);
        if outcome != RecordOutcome::Used {
            return outcome;
        }
        let (blocks, invalid) = summary;
        if !blocks.is_empty() {
            let Some(session) = self.session(parser_state, v, out).map(|s| s.id.clone()) else {
                return RecordOutcome::unreadable("session ID is empty");
            };
            let native = str_field(v, "uuid")
                .map(str::to_owned)
                .unwrap_or_else(|| format!("{}@{}", self.file.stem(), record.offset));
            let Some(id) = message_id(&native) else {
                return RecordOutcome::unreadable("message ID is empty");
            };
            // The summary records supplied feedback, including when the
            // hook prevented continuation. It is not a delivery receipt.
            close(parser_state, out);
            out.messages.push(MessageRow {
                offset: record.offset,
                end: range(record).1,
                message: Message {
                    id: id.clone(),
                    session,
                    parent: parser_state.parent(v),
                    role: Role::System,
                    at: v.get("timestamp").and_then(time::rfc3339),
                    model: None,
                    prompt_kind: Some(PromptKind::Harness),
                    blocks,
                    usage: Usage::default(),
                    stop_reason: None,
                    reported_cost_usd: None,
                    duration_ms: None,
                    attrs: Default::default(),
                    production: Production::default(),
                },
            });
            parser_state.last = Some(id);
        }
        match invalid {
            Some(reason) => RecordOutcome::unreadable(reason),
            None => outcome,
        }
    }

    fn message(
        &self,
        parser_state: &mut ParserState,
        record: Record<'_>,
        v: &Json,
        role: Role,
        out: &mut Output,
        records: &mut Records,
    ) -> RecordOutcome {
        let kind = self.kind(v);
        records.fields(&kind, v, &fields::MESSAGE_FIELDS);
        let Some(session) = self.session(parser_state, v, out).map(|s| s.id.clone()) else {
            return RecordOutcome::unreadable("session ID is empty");
        };
        let native = str_field(v, "uuid")
            .map(str::to_owned)
            .unwrap_or_else(|| format!("{}@{}", self.file.stem(), record.offset));
        let Some(id) = message_id(&native) else {
            return RecordOutcome::unreadable("message ID is empty");
        };
        let msg = v.get("message").unwrap_or(&Json::Null);
        let content = msg.get("content").unwrap_or(&Json::Null);
        let blocks = blocks::blocks(v, content, &mut out.blobs);
        let usage = msg.get("usage").map(usage).unwrap_or_default();
        let stop_reason = str_field(msg, "stop_reason").map(str::to_owned);
        let model = str_field(msg, "model");
        // This reserved value marks a runtime-authored notice.
        let harness = role == Role::Assistant && model == Some("<synthetic>");
        let model = model.filter(|_| !harness).map(str::to_owned);
        let cost = v.get("costUSD").and_then(Json::as_number).cloned();
        let duration = v.get("durationMs").and_then(Json::as_u64);
        let production = production(v, msg);
        let attrs = message_attrs(v, msg, &blocks);
        let end = range(record).1;
        if let Some(fork) = v.get("forkedFrom")
            && let (Some(parent), Some(at)) = (
                str_field(fork, "sessionId").and_then(session_id),
                str_field(fork, "messageUuid").and_then(message_id),
            )
            && parent != session
        {
            link_at(
                parser_state,
                out,
                LinkKind::Fork,
                parent,
                session.clone(),
                Some(at),
                None,
                "forkedFrom",
            );
        }
        // An agent call's result names the agent it started: the parent's
        // side of the spawn link, with the call.
        if let Some(agent) = v
            .get("toolUseResult")
            .and_then(|r| r.get("agentId"))
            .and_then(Json::as_str)
            && let Some(child) = session_id(&format!("agent-{agent}"))
            && child != session
        {
            let call = content.as_array().and_then(|items| {
                let mut results = items
                    .iter()
                    .filter(|i| str_field(i, "type") == Some("tool_result"));
                let only = results.next();
                only.filter(|_| results.next().is_none())
                    .and_then(|i| str_field(i, "tool_use_id"))
                    .map(str::to_owned)
            });
            link_at(
                parser_state,
                out,
                LinkKind::Spawn,
                session.clone(),
                child,
                None,
                call,
                "toolUseResult.agentId",
            );
        }

        if role == Role::Assistant && !harness {
            let key = str_field(msg, "id")
                .or_else(|| str_field(v, "requestId"))
                .unwrap_or(&native)
                .to_owned();
            let same = |r: &Response| r.key == key && r.message.session == session;
            if !parser_state.open.as_ref().is_some_and(same) {
                let reopened = parser_state
                    .recent
                    .iter()
                    .position(same)
                    .and_then(|i| parser_state.recent.remove(i));
                close(parser_state, out);
                parser_state.open = reopened;
            }
            match &mut parser_state.open {
                Some(open) => {
                    let target = Some(open.message.id.clone());
                    if native != open.message.id.native().as_str() {
                        parser_state.alias(&native, target);
                    }
                    let Some(open) = &mut parser_state.open else {
                        unreachable!("the open response was just matched")
                    };
                    open.end = open.end.max(end);
                    let m = &mut open.message;
                    let block_offset = m.blocks.len();
                    m.blocks.extend(blocks);
                    if !usage.is_empty() {
                        m.usage = usage;
                    }
                    m.stop_reason = stop_reason.or(m.stop_reason.take());
                    m.model = m.model.take().or(model);
                    m.reported_cost_usd = cost.or(m.reported_cost_usd.take());
                    m.duration_ms = duration.or(m.duration_ms);
                    m.production.merge(production);
                    merge_attrs(&mut m.attrs, attrs, block_offset);
                }
                None => {
                    parser_state.open = Some(Response {
                        key,
                        offset: record.offset,
                        end,
                        message: Message {
                            id: id.clone(),
                            session,
                            parent: parser_state.parent(v),
                            role,
                            at: v.get("timestamp").and_then(time::rfc3339),
                            model,
                            prompt_kind: None,
                            blocks,
                            usage,
                            stop_reason,
                            reported_cost_usd: cost,
                            duration_ms: duration,
                            attrs,
                            production,
                        },
                    });
                    parser_state.last = Some(id);
                }
            }
        } else {
            if harness {
                // Runtime notices keep their own record identity.
                close(parser_state, out);
            }
            let message = Message {
                id: id.clone(),
                session,
                parent: parser_state.parent(v),
                role,
                at: v.get("timestamp").and_then(time::rfc3339),
                model,
                prompt_kind: Some(if harness {
                    PromptKind::Harness
                } else {
                    self.prompt_kind(v, content)
                }),
                blocks,
                usage,
                stop_reason,
                reported_cost_usd: cost,
                duration_ms: duration,
                attrs,
                production,
            };
            out.messages.push(MessageRow {
                offset: record.offset,
                end,
                message,
            });
            parser_state.last = Some(id);
        }
        RecordOutcome::Used
    }

    /// A Workflow journal record: an agent the run started, what it
    /// returned, or the run's launch. Each spawned agent is linked to the
    /// session that ran the workflow and named by its label.
    fn journal(
        &self,
        parser_state: &mut ParserState,
        record: Record<'_>,
        v: &Json,
        ty: &str,
        out: &mut Output,
    ) -> RecordOutcome {
        let kind = match ty {
            "started" => "workflow_agent_started",
            "result" => "workflow_agent_result",
            "launched" => "workflow_launched",
            _ => return RecordOutcome::skipped("unrecognized journal record"),
        };
        let Some(parent) = self.own_session(parser_state) else {
            return RecordOutcome::unreadable("session ID is empty");
        };
        if let Some(agent) = str_field(v, "agentId").map(|a| format!("agent-{a}"))
            && let Some(child) = session_id(&agent)
        {
            link(
                parser_state,
                out,
                LinkKind::Spawn,
                parent,
                child.clone(),
                "workflow journal",
            );
            if let Some(label) = str_field(v, "label") {
                parser_state
                    .sessions
                    .entry(agent)
                    .or_insert_with(|| Session::new(child, SessionKind::Subagent))
                    .offer_title(label, TitleSource::AgentName);
            }
        }
        self.event(parser_state, record, v, kind, out)
    }

    /// Who wrote a user record. A subagent transcript's root record (no
    /// parent) is the task its parent model wrote; a compaction call's is
    /// the harness's summarizing instruction.
    fn prompt_kind(&self, v: &Json, content: &Json) -> PromptKind {
        let kind = blocks::prompt_kind(v, content);
        match &self.file {
            File::Subagent { stem }
                if kind == PromptKind::Prose && str_field(v, "parentUuid").is_none() =>
            {
                if stem.starts_with("agent-acompact") {
                    PromptKind::Harness
                } else {
                    PromptKind::Model
                }
            }
            _ => kind,
        }
    }

    fn title(
        &self,
        parser_state: &mut ParserState,
        v: &Json,
        field: &str,
        source: TitleSource,
    ) -> RecordOutcome {
        let Some(text) = str_field(v, field) else {
            return RecordOutcome::skipped(format!("no {field}"));
        };
        let Some(id) = self.own_session(parser_state) else {
            return RecordOutcome::unreadable("session ID is empty");
        };
        if let Some(s) = parser_state.sessions.get_mut(id.native().as_str()) {
            s.offer_title(text, source);
        }
        RecordOutcome::Used
    }
}

fn link(
    parser_state: &mut ParserState,
    out: &mut Output,
    kind: LinkKind,
    parent: SessionId,
    child: SessionId,
    evidence: &str,
) {
    link_at(parser_state, out, kind, parent, child, None, None, evidence);
}

#[allow(clippy::too_many_arguments)]
fn link_at(
    parser_state: &mut ParserState,
    out: &mut Output,
    kind: LinkKind,
    parent: SessionId,
    child: SessionId,
    at: Option<MessageId>,
    call_id: Option<String>,
    evidence: &str,
) {
    // A later record may add the branch point; the writer merges them.
    let key = format!("{kind:?} {parent} {child} {}", at.is_some());
    if parser_state.links.insert(key) {
        out.links.push(Link {
            kind,
            parent,
            child,
            at,
            call_id,
            evidence: evidence.to_owned(),
        });
    }
}

/// Moves the open response to the output and keeps it reopenable.
fn close(parser_state: &mut ParserState, out: &mut Output) {
    if let Some(open) = parser_state.open.take() {
        out.messages.push(MessageRow {
            offset: open.offset,
            end: open.end,
            message: open.message.clone(),
        });
        parser_state.recent.push_back(open);
        while parser_state.recent.len() > REOPEN_WINDOW {
            parser_state.recent.pop_front();
        }
    }
}

/// Where a record starts and ends in its input.
fn range(record: Record<'_>) -> (u64, u64) {
    (record.offset, record.offset + record.bytes.len() as u64)
}

/// Claude Code's subscription windows: five hours from the first response
/// after the last window ended, and a week on a fixed schedule. They exist
/// whether or not a turn was ever refused.
pub const QUOTA_LIMITS: &[QuotaLimit] = &[
    QuotaLimit {
        limit: Cow::Borrowed("five_hour"),
        minutes: 300,
        from_first_response: true,
    },
    QuotaLimit {
        limit: Cow::Borrowed("seven_day"),
        minutes: 10_080,
        from_first_response: false,
    },
];

/// `quotaLimits`, which Claude Code records only when it refuses a turn:
/// `{status, resetsAt, rateLimitType}`.
fn quota(q: &Json) -> Option<QuotaReading> {
    let kind = str_field(q, "rateLimitType")?;
    let limit = QUOTA_LIMITS
        .iter()
        .find(|l| kind == l.limit || (l.minutes == 10_080 && kind.starts_with("seven_day")))?;
    let status = str_field(q, "status").map(str::to_owned);
    Some(QuotaReading {
        windows: vec![QuotaWindow {
            limit: QuotaLimit {
                limit: Cow::Owned(kind.to_owned()),
                ..limit.clone()
            },
            // A refused turn means the window is used up; other states do
            // not say how much.
            used_percent: (status.as_deref() == Some("rejected")).then_some(100.0),
            resets_at: q
                .get("resetsAt")
                .and_then(Json::as_i64)
                .and_then(|s| s.checked_mul(1000))
                .and_then(Timestamp::from_unix_ms),
            status,
        }],
        ..QuotaReading::default()
    })
}

/// How the harness produced a message: settings, attribution and the
/// API's diagnostics (`Production`).
fn production(v: &Json, msg: &Json) -> Production {
    let s = |k: &str| str_field(v, k).map(str::to_owned);
    let kept = |x: Option<&Json>| x.filter(|x| !x.is_null()).cloned();
    Production {
        server_tool_use: msg
            .pointer("/usage/server_tool_use")
            .and_then(|value| serde_json::from_value(value.clone()).ok()),
        iterations: msg.pointer("/usage/iterations").and_then(Json::as_u64),
        phase: None,
        response_boundary_unknown: None,
        effort: s("effort").or_else(|| s("perTurnEffort")),
        requested_model: s("requestedModel"),
        advisor_model: s("advisorModel"),
        attribution: Attribution {
            agent: s("attributionAgent"),
            skill: s("attributionSkill"),
            plugin: s("attributionPlugin"),
            mcp_server: s("attributionMcpServer"),
            mcp_tool: s("attributionMcpTool"),
        },
        thinking_ms: v.get("thinkingDurationMs").and_then(Json::as_u64),
        first_token_ms: msg.get("ttftMs").and_then(Json::as_u64),
        api_error: s("error").map(|kind| ApiError {
            kind,
            status: v
                .get("apiErrorStatus")
                .and_then(Json::as_u64)
                .and_then(|n| u16::try_from(n).ok()),
        }),
        cache_miss: kept(
            msg.get("diagnostics")
                .and_then(|d| d.get("cache_miss_reason")),
        ),
        refusal: kept(msg.get("stop_details")),
        permission: kept(v.get("permissionDecision")),
        request: s("requestRef"),
        request_id: s("requestId"),
        turn_origin: s("turnOrigin"),
        quota: v.get("quotaLimits").and_then(quota),
    }
}

fn message_attrs(record: &Json, message: &Json, blocks: &[Block]) -> serde_json::Map<String, Json> {
    let mut attrs = crate::fields::attributes(record, &fields::MESSAGE_FIELDS);
    if let Some(usage) = message.get("usage").and_then(Json::as_object) {
        let details: serde_json::Map<String, Json> = usage
            .iter()
            .filter(|(key, _)| {
                !matches!(
                    key.as_str(),
                    "input_tokens"
                        | "output_tokens"
                        | "cache_read_input_tokens"
                        | "cache_creation_input_tokens"
                        | "service_tier"
                )
            })
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        if !details.is_empty() {
            attrs.insert("usage".into(), Json::Object(details));
        }
    }
    if let Some(result) = record.get("toolUseResult") {
        let in_result = blocks.iter().any(|block| matches!(block, Block::ToolResult { data: Some(data), .. } if data == result || data.get("toolUseResult") == Some(result)));
        if in_result {
            attrs.remove("toolUseResult");
        } else {
            attrs.insert("toolUseResult".into(), result.clone());
        }
    }
    let details: Vec<Json> = message
        .get("content")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(position, block)| {
            let fields: serde_json::Map<String, Json> = block
                .as_object()?
                .iter()
                .filter(|(key, _)| {
                    matches!(
                        key.as_str(),
                        "citations" | "summaries" | "start_timestamp" | "stop_timestamp"
                    )
                })
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            (!fields.is_empty()).then(|| serde_json::json!({"position":position,"fields":fields}))
        })
        .collect();
    if !details.is_empty() {
        attrs.insert("content".into(), Json::Array(details));
    }
    attrs
}

fn merge_attrs(
    into: &mut serde_json::Map<String, Json>,
    later: serde_json::Map<String, Json>,
    block_offset: usize,
) {
    for (key, mut value) in later {
        if key == "content" {
            if let Some(details) = value.as_array_mut() {
                for detail in details {
                    if let Some(position) = detail.get("position").and_then(Json::as_u64) {
                        detail["position"] = Json::from(position + block_offset as u64);
                    }
                }
            }
            if let Some(Json::Array(existing)) = into.get_mut(&key) {
                if let Json::Array(additional) = value {
                    existing.extend(additional);
                }
                continue;
            }
        }
        into.insert(key, value);
    }
}

fn usage(u: &Json) -> Usage {
    let n = |k: &str| u.get(k).and_then(Json::as_u64);
    Usage {
        input: n("input_tokens"),
        cached_input: n("cache_read_input_tokens"),
        cache_write: n("cache_creation_input_tokens"),
        output: n("output_tokens"),
        reasoning: u.get("thinking_tokens").and_then(Json::as_u64).or_else(|| {
            u.pointer("/output_tokens_details/thinking_tokens")
                .and_then(Json::as_u64)
        }),
        cache_write_1h: u
            .get("cache_creation")
            .and_then(|c| c.get("ephemeral_1h_input_tokens"))
            .and_then(Json::as_u64),
        service_tier: str_field(u, "service_tier").map(str::to_owned),
        speed: str_field(u, "speed").map(str::to_owned),
    }
}

/// Polylogue labels typed diagnostics for readers; the error and reason
/// strings stay unchanged. The complete native summary remains in its event.
fn hook_text(v: &Json) -> (Vec<Block>, Option<&'static str>) {
    let mut blocks = Vec::new();
    let mut invalid = None;
    if let Some(errors) = v.get("hookErrors") {
        match errors.as_array() {
            Some(errors) => {
                for error in errors {
                    match error.as_str() {
                        Some(error) => blocks.push(Block::Text {
                            text: format!("Stop hook error:\n{error}"),
                        }),
                        None => invalid = Some("hookErrors contains a non-string value"),
                    }
                }
            }
            None => invalid = Some("hookErrors is not an array of strings"),
        }
    }
    let prevented = v.get("preventedContinuation");
    if prevented.is_some_and(|value| !value.is_boolean()) {
        invalid = Some("preventedContinuation is not a boolean");
    }
    let reason = v.get("stopReason");
    if reason.is_some_and(|value| !value.is_string()) {
        invalid = Some("stopReason is not a string");
    }
    if prevented.and_then(Json::as_bool) == Some(true) {
        blocks.push(Block::Text {
            text: "Stop hook prevented continuation.".into(),
        });
        if let Some(reason) = reason.and_then(Json::as_str) {
            blocks.push(Block::Text {
                text: format!("Stop hook reason:\n{reason}"),
            });
        }
    }
    if let Some(context) = v.get("hookAdditionalContext") {
        match context.as_array() {
            Some(context) => {
                for text in context {
                    match text.as_str() {
                        Some(text) => blocks.push(Block::Text {
                            text: text.to_owned(),
                        }),
                        None => invalid = Some("hookAdditionalContext contains a non-string value"),
                    }
                }
            }
            None => invalid = Some("hookAdditionalContext is not an array of strings"),
        }
    }
    (blocks, invalid)
}

/// A background task's completion, from the notification the harness
/// queues for the model.
fn background_update(v: &Json) -> Option<(String, ToolOutcome, Option<String>, Option<String>)> {
    let content = v.get("content").and_then(Json::as_str)?;
    let tag = |name: &str| {
        let open = format!("<{name}>");
        let start = content.find(&open)? + open.len();
        let end = content[start..].find(&format!("</{name}>"))?;
        Some(content[start..start + end].trim().to_owned())
    };
    let task = tag("task-id")?;
    let exit = tag("summary").and_then(|s| {
        let at = s.find("exit code")? + "exit code".len();
        s[at..]
            .trim()
            .split(|c: char| !c.is_ascii_digit() && c != '-')
            .next()?
            .parse()
            .ok()
    });
    let outcome = match tag("status")?.as_str() {
        "completed" => ToolOutcome::Ok,
        "failed" => ToolOutcome::Error {
            reason: exit.map(ErrorReason::ExitCode),
        },
        "killed" | "stopped" => ToolOutcome::Error {
            reason: Some(ErrorReason::Interrupted),
        },
        _ => return None,
    };
    Some((task, outcome, tag("tool-use-id"), tag("output-file")))
}

impl Parser for ClaudeCode {
    type ParserState = ParserState;

    fn kind(&self, v: &Json) -> String {
        let ty = str_field(v, "type").unwrap_or("?");
        let sub = match ty {
            "system" => str_field(v, "subtype"),
            "progress" => v.get("data").and_then(|d| str_field(d, "type")),
            _ => None,
        };
        match sub {
            Some(sub) => format!("{ty}/{sub}"),
            None => ty.to_owned(),
        }
    }

    fn parse_record(
        &self,
        parser_state: &mut ParserState,
        record: Record<'_>,
        v: &Json,
        out: &mut Output,
        records: &mut Records,
    ) -> RecordOutcome {
        if !v.is_object() {
            return RecordOutcome::unreadable("record is not a JSON object");
        }
        let Some(ty) = str_field(v, "type") else {
            return RecordOutcome::skipped("no type field");
        };
        if let File::Journal { .. } = self.file {
            return self.journal(parser_state, record, v, ty, out);
        }
        let hook = (ty == "system" && str_field(v, "subtype") == Some("stop_hook_summary"))
            .then(|| hook_text(v));
        let message_record = (matches!(ty, "user" | "assistant")
            && v.get("isCompactSummary").and_then(Json::as_bool) != Some(true))
            || hook.as_ref().is_some_and(|(blocks, _)| !blocks.is_empty());
        if !message_record && let Some(uuid) = str_field(v, "uuid") {
            // A child naming this record as its parent belongs under the
            // nearest message above it; a compaction boundary points there
            // through `logicalParentUuid`.
            let target = str_field(v, "logicalParentUuid")
                .or_else(|| str_field(v, "parentUuid"))
                .and_then(|p| parser_state.resolve(p));
            parser_state.alias(uuid, target);
        }
        self.names(parser_state, record, v, out);
        if matches!(ty, "summary" | "custom-title" | "ai-title") {
            records.fields(ty, v, &fields::TITLE_FIELDS);
        }
        if ty == "continued-in" {
            records.fields(ty, v, &fields::CONTINUATION_FIELDS);
        }
        match ty {
            "user" if v.get("isCompactSummary").and_then(Json::as_bool) == Some(true) => {
                self.event(parser_state, record, v, "compaction_summary", out)
            }
            "user" => self.message(parser_state, record, v, Role::User, out, records),
            "assistant" => self.message(parser_state, record, v, Role::Assistant, out, records),
            "system" => match str_field(v, "subtype") {
                Some("stop_hook_summary") => self.hook_summary(
                    parser_state,
                    record,
                    v,
                    hook.expect("recognized stop-hook summary"),
                    out,
                ),
                Some("compact_boundary") => self.event(parser_state, record, v, "compaction", out),
                Some("microcompact_boundary") => {
                    self.event(parser_state, record, v, "microcompaction", out)
                }
                Some(sub) => self.event(parser_state, record, v, &sub.replace('-', "_"), out),
                None => RecordOutcome::skipped("system record without a subtype"),
            },
            "attachment" => self.event(parser_state, record, v, "attachment", out),
            "pr-link"
            | "permission-mode"
            | "last-prompt"
            | "bridge-session"
            | "frame-link"
            | "worktree-state"
            | "artifact-comment-monitor"
            | "artifact-autoreact-ledger" => {
                self.event(parser_state, record, v, &ty.replace('-', "_"), out)
            }
            "file-history-snapshot" | "file-history-delta" => {
                self.event(parser_state, record, v, "file_history", out)
            }
            "cost-state" => self.event(parser_state, record, v, "tool_cost", out),
            // The request the harness sent the model, rebuilt exactly: the
            // system prompt and tools (`shape`) and each message as sent
            // (`blob`) are keyed by hash, and the request lists them.
            "api-request" => self.event(parser_state, record, v, "api_request", out),
            "api-request-shape" => self.event(parser_state, record, v, "api_request_shape", out),
            "api-request-blob" => self.event(parser_state, record, v, "api_request_blob", out),
            "started" => self.event(parser_state, record, v, "subagent_started", out),
            "result" => self.event(parser_state, record, v, "subagent_result", out),
            "queue-operation" => {
                if let Some((task_id, outcome, call_id, output_file)) = background_update(v)
                    && let Some(session) = match v.get("sessionId") {
                        Some(_) => self.session(parser_state, v, out).map(|s| s.id.clone()),
                        None => self.own_session(parser_state),
                    }
                {
                    out.background.push(BackgroundUpdate {
                        result_at: v.get("timestamp").and_then(time::rfc3339),
                        reported_duration_ms: v.get("durationMs").and_then(Json::as_u64),
                        session,
                        task_id,
                        call_id,
                        outcome,
                        output_file,
                    });
                }
                self.event(parser_state, record, v, "queue", out)
            }
            "relocated" => {
                if let Some(dir) = str_field(v, "relocatedCwd")
                    && let Some(id) = self.own_session(parser_state)
                    && let Some(s) = parser_state.sessions.get_mut(id.native().as_str())
                {
                    s.offer_cwd(dir);
                }
                self.event(parser_state, record, v, "relocated", out)
            }
            "agent-name" => {
                let d = self.title(parser_state, v, "agentName", TitleSource::AgentName);
                if d != RecordOutcome::Used {
                    return d;
                }
                self.event(parser_state, record, v, "agent_name", out)
            }
            "custom-title" => self.title(parser_state, v, "customTitle", TitleSource::Custom),
            "ai-title" => self.title(parser_state, v, "aiTitle", TitleSource::Generated),
            "summary" => self.title(parser_state, v, "summary", TitleSource::Summary),
            "continued-in" => {
                let (Some(parent), Some(child)) = (
                    self.own_session(parser_state),
                    str_field(v, "continuedInSessionId").and_then(session_id),
                ) else {
                    return RecordOutcome::skipped("continued-in without a target");
                };
                link(
                    parser_state,
                    out,
                    LinkKind::Continuation,
                    parent,
                    child,
                    "continuedInSessionId",
                );
                RecordOutcome::Used
            }
            "fork-context-ref" => {
                let child = self.session(parser_state, v, out).map(|s| s.id.clone());
                let (Some(parent), Some(child)) =
                    (str_field(v, "parentSessionId").and_then(session_id), child)
                else {
                    return RecordOutcome::skipped("fork-context-ref without a parent");
                };
                // Evidence for the subagent link, with its branch point:
                // never a second parent.
                let at = str_field(v, "parentLastUuid").and_then(message_id);
                link_at(
                    parser_state,
                    out,
                    LinkKind::Spawn,
                    parent,
                    child,
                    at,
                    None,
                    "fork-context-ref",
                );
                self.event(parser_state, record, v, "fork_context", out)
            }
            "progress" => {
                let data = v.get("data").unwrap_or(&Json::Null);
                match str_field(data, "type") {
                    Some("agent_progress") => {
                        records.fields(
                            "progress/agent_progress",
                            v,
                            &fields::AGENT_PROGRESS_FIELDS,
                        );
                        let parent = self.session(parser_state, v, out).map(|s| s.id.clone());
                        let child = str_field(data, "agentId")
                            .and_then(|a| session_id(&format!("agent-{a}")));
                        let (Some(parent), Some(child)) = (parent, child) else {
                            return RecordOutcome::skipped("agent progress without an agent ID");
                        };
                        let call = str_field(v, "parentToolUseID").map(str::to_owned);
                        link_at(
                            parser_state,
                            out,
                            LinkKind::Spawn,
                            parent,
                            child.clone(),
                            None,
                            call,
                            "agent_progress",
                        );
                        if let Some(embedded) = data.get("message").filter(|m| m.is_object()) {
                            let agent = ClaudeCode {
                                file: File::Subagent {
                                    stem: child.native().to_string(),
                                },
                            };
                            let state = parser_state
                                .agents
                                .entry(child.native().to_string())
                                .or_default();
                            let mut embedded = embedded.clone();
                            if let Some(object) = embedded.as_object_mut() {
                                // Some progress records wrap an API message; others hold
                                // the API message itself with its identity beside it.
                                if !object.contains_key("message") {
                                    let api = Json::Object(object.clone());
                                    object.clear();
                                    object.insert("message".into(), api);
                                }
                                for key in ["uuid", "timestamp", "parentUuid"] {
                                    if !object.contains_key(key)
                                        && let Some(value) = data.get(key)
                                    {
                                        object.insert(key.into(), value.clone());
                                    }
                                }
                                if !object.contains_key("type")
                                    && let Some(role) =
                                        object.get("message").and_then(|m| m.get("role")).cloned()
                                {
                                    object.insert("type".into(), role);
                                }
                            }
                            let role = match str_field(&embedded, "type") {
                                Some("assistant") => Role::Assistant,
                                Some("user") => Role::User,
                                _ => {
                                    return self.event(
                                        parser_state,
                                        record,
                                        v,
                                        "agent_progress",
                                        out,
                                    );
                                }
                            };
                            let outcome =
                                agent.message(state, record, &embedded, role, out, records);
                            if !matches!(outcome, RecordOutcome::Used) {
                                return outcome;
                            }
                        }
                        self.event(parser_state, record, v, "agent_progress", out)
                    }
                    Some("hook_progress") => self.event(parser_state, record, v, "hook_run", out),
                    Some(_) => {
                        RecordOutcome::skipped("progress tick; the tool's own result supersedes it")
                    }
                    None => RecordOutcome::skipped("progress without data"),
                }
            }
            "mode" => RecordOutcome::skipped("always \"normal\" in observed data"),
            "atis-latch" => RecordOutcome::skipped("always empty in observed data"),
            "agent-color" => RecordOutcome::skipped("terminal colour only; names no agent"),
            "init" => RecordOutcome::skipped("empty start marker"),
            _ => RecordOutcome::skipped("unrecognized record type"),
        }
    }

    fn flush(&self, parser_state: &mut ParserState, out: &mut Output) {
        let mut agent_sessions = Vec::new();
        for (native, state) in &mut parser_state.agents {
            let agent = ClaudeCode {
                file: File::Subagent {
                    stem: native.clone(),
                },
            };
            agent.flush(state, out);
            agent_sessions.append(&mut out.sessions);
        }
        if let Some(open) = &parser_state.open {
            out.messages.push(MessageRow {
                offset: open.offset,
                end: open.end,
                message: open.message.clone(),
            });
        }
        // A response reopened within one step was sent more than once; keep
        // the last, which is the longest. Rows go out in file order.
        let mut seen = BTreeSet::new();
        let mut kept: Vec<MessageRow> = Vec::with_capacity(out.messages.len());
        for row in out.messages.drain(..).rev() {
            if seen.insert((row.message.id.clone(), row.offset)) {
                kept.push(row);
            }
        }
        kept.sort_by_key(|row| row.offset);
        out.messages = kept;
        out.sessions = parser_state
            .sessions
            .values()
            .cloned()
            .chain(agent_sessions)
            .collect();
    }
}

#[cfg(test)]
mod tests;
