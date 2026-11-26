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
//! once. Copying is recorded independently of explicit provider links.

use polylogue_model::JsonExt;
/// Version of the rows this module produces.
pub const VERSION: u32 = 14;

parser_modules! {
    mod blocks;
    mod fields;
}

pub mod side;

use crate::lines::Record;
use crate::origin::Root;
use crate::parse::{BackgroundUpdate, MessageRow, Output, Parse, Parser};
use crate::time;
use crate::{RecordOutcome, Records};
use polylogue_model::{
    ApiError, Attribution, Block, ErrorReason, Event, IdKind, Json, Link, LinkKind, Message,
    MessageId, NativeId, Origin, Production, PromptKind, QuotaLimit, QuotaReading, QuotaWindow,
    Role, Session, SessionId, SessionKind, Timestamp, TitleSource, ToolOutcome, Usage,
};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::BTreeMap;
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

pub fn recognize(path: &Path, head: &[u8]) -> bool {
    File::from_path(path).is_some() && crate::document::starts_object_lines(head)
}

pub fn open(path: &Path) -> Option<Parse> {
    File::from_path(path).map(|file| Parse::Lines(Box::new(ClaudeCode { file })))
}

/// `claude --resume <id>`. Subagents and sidechains are not resumable on
/// their own.
pub fn resume(native: &str) -> Option<Vec<String>> {
    if native.starts_with("agent-") || native.contains("~side-") {
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
}

pub struct ClaudeCode {
    pub file: File,
}

/// What the parser carries from one record to the next. A response
/// written as several records yields one row per record, and a record that
/// stands for another yields an alias row; both are joined when the chain's
/// reading is formed ([`crate::chain_names`]).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParserState {
    /// The latest message, for placing events on the timeline.
    last: Option<MessageId>,
    /// A compaction boundary waiting for the summary record that follows
    /// it, so the boundary names its summary message. Any other record
    /// sends it as it is. Its range runs to the end of the record that sends
    /// it, since the event depends on the bytes up to there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pending_boundary: Option<Event>,
    #[serde(default)]
    compaction_ownership_reported: bool,
    /// The session-wide names (plan `slug`, `teamName`) last recorded, by
    /// kind and session, so each event is sent once per name, not with
    /// every record. A file names a few sessions.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    names: BTreeMap<String, String>,
    /// The directory each session was last in, by native ID.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    cwd: BTreeMap<String, String>,
    /// One entry per sidechain in the file: its latest record's UUID names
    /// its session. A parent other than a sidechain's latest record starts
    /// a separate sidechain.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    sidechains: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    open_tasks: BTreeMap<String, BTreeMap<String, String>>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    subagent_prompt_seen: bool,
}

impl ParserState {
    fn record_tasks(&mut self, session: &SessionId, blocks: &[Block]) {
        let native = session.native().as_str();
        for block in blocks {
            match block {
                Block::ToolCall {
                    tool: Some(tool),
                    call_id: Some(id),
                    input,
                    ..
                } if tool == "Task" => {
                    if let Some(prompt) = input.str("prompt").filter(|prompt| !prompt.is_empty()) {
                        self.open_tasks
                            .entry(native.to_owned())
                            .or_default()
                            .insert(id.clone(), prompt.to_owned());
                    }
                }
                Block::ToolResult {
                    call_id: Some(id), ..
                } => {
                    if let Some(tasks) = self.open_tasks.get_mut(native) {
                        tasks.remove(id);
                        if tasks.is_empty() {
                            self.open_tasks.remove(native);
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

fn origin() -> Origin {
    Origin::new("claude-code").expect("constant origin")
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
    sidechain_start: bool,
}

impl ClaudeCode {
    fn record_session(&self, parser_state: &ParserState, v: &Json) -> RecordSession {
        match &self.file {
            File::Subagent { stem } => {
                let native = v
                    .str("agentId")
                    .filter(|s| !s.is_empty())
                    .map(|a| format!("agent-{a}"))
                    .unwrap_or_else(|| stem.clone());
                // The call that compacted the parent's history runs as an
                // agent named `acompact-…`.
                let kind = if native.starts_with("agent-acompact") {
                    SessionKind::Compaction
                } else {
                    SessionKind::Subagent
                };
                RecordSession {
                    native,
                    kind,
                    sidechain_start: false,
                }
            }
            File::Journal { session, .. } => RecordSession {
                native: session.clone(),
                kind: SessionKind::Main,
                sidechain_start: false,
            },
            File::Main { stem } => {
                let base = v
                    .str("sessionId")
                    .filter(|s| !s.is_empty())
                    .unwrap_or(stem)
                    .to_owned();
                if v.bool("isSidechain") == Some(true) {
                    let uuid = v
                        .str("uuid")
                        .filter(|uuid| !uuid.is_empty())
                        .map(str::to_owned)
                        .unwrap_or_else(|| crate::content_ids::untagged_hash(v));
                    let earlier = parser_state
                        .sidechains
                        .get(&uuid)
                        .or_else(|| {
                            v.str("parentUuid")
                                .and_then(|parent| parser_state.sidechains.get(parent))
                        })
                        .filter(|session| session.starts_with(&format!("{base}~side-")));
                    let first = format!("{base}~side-{uuid}");
                    RecordSession {
                        sidechain_start: earlier.is_none(),
                        native: earlier.cloned().unwrap_or(first),
                        kind: SessionKind::Sidechain,
                    }
                } else {
                    RecordSession {
                        native: base,
                        kind: SessionKind::Main,
                        sidechain_start: false,
                    }
                }
            }
        }
    }

    /// The record's session, stated with the record's wrapper fields. Also
    /// sends the link that explains why a record belongs to a session other
    /// than the file's own.
    fn session(
        &self,
        parser_state: &mut ParserState,
        v: &Json,
        out: &mut Output,
    ) -> Option<SessionId> {
        let record_session = self.record_session(parser_state, v);
        self.state_session(parser_state, v, record_session, None, out)
    }

    fn state_session(
        &self,
        parser_state: &mut ParserState,
        v: &Json,
        record_session: RecordSession,
        fork: Option<(&SessionId, &MessageId)>,
        out: &mut Output,
    ) -> Option<SessionId> {
        let stated = session_id(&record_session.native)?;
        let id = if let Some((parent, at)) = fork {
            state(out, Session::new(stated.clone(), record_session.kind));
            link_at(
                out,
                LinkKind::Fork,
                parent.clone(),
                stated.clone(),
                Some(at.clone()),
                None,
                "forkedFrom",
            );
            parent.clone()
        } else {
            stated.clone()
        };
        if let Some(parent) = v
            .str("resumeSessionId")
            .filter(|s| !s.is_empty())
            .and_then(session_id)
            && parent != stated
        {
            link(
                out,
                LinkKind::Resume,
                parent,
                stated.clone(),
                "resumeSessionId",
            );
        }
        let copied_into = if fork.is_some() {
            Some(stated)
        } else {
            match (&self.file, record_session.kind) {
                (File::Main { stem }, SessionKind::Main) => session_id(stem),
                (File::Subagent { stem }, SessionKind::Subagent)
                    if !stem.starts_with("agent-acompact") =>
                {
                    session_id(stem)
                }
                _ => None,
            }
        };
        if let Some(child) = copied_into.filter(|child| *child != id) {
            out.prefix_copies.push(polylogue_model::PrefixCopy {
                explicit_members: fork.is_some(),
                parent: id.clone(),
                child,
            });
        }
        let mut s = Session::new(id.clone(), record_session.kind);
        if let Some(cwd) = v.str("cwd").filter(|s| !s.is_empty()) {
            // A step's directories continue from the one the session was
            // last in, so its moves, a move back included, join the earlier
            // ones (`Session::merge`).
            if !out.sessions.iter().any(|stated| stated.id == id)
                && let Some(previous) = parser_state.cwd.get(id.native().as_str())
            {
                s.offer_cwd(previous);
            }
            s.offer_cwd(cwd);
            parser_state
                .cwd
                .insert(id.native().as_str().to_owned(), cwd.to_owned());
        }
        if let Some(branch) = v.str("gitBranch").filter(|s| !s.is_empty()) {
            s.git_branch = Some(branch.to_owned());
        }
        if let Some(version) = v.str("version").filter(|s| !s.is_empty()) {
            s.tool_version = Some(version.to_owned());
        }
        if let Some(at) = v.get("timestamp").and_then(time::rfc3339) {
            s.offer_start(at.timestamp(), at.accuracy())
                .expect("decoded source time is valid");
        }
        if let Some(mode) = v.str("sessionKind").filter(|s| !s.is_empty()) {
            s.mode = Some(mode.to_owned());
        }
        state(out, s);
        Some(id)
    }

    /// The file's own session, for records that name none.
    fn own_session(&self, out: &mut Output) -> Option<SessionId> {
        self.own_session_with(out, |_| {})
    }

    /// The file's own session, stated with what `facts` adds.
    fn own_session_with(
        &self,
        out: &mut Output,
        facts: impl FnOnce(&mut Session),
    ) -> Option<SessionId> {
        let (stem, kind) = match &self.file {
            File::Main { stem } => (stem, SessionKind::Main),
            File::Journal { session, .. } => (session, SessionKind::Main),
            File::Subagent { stem } if stem.starts_with("agent-acompact") => {
                (stem, SessionKind::Compaction)
            }
            File::Subagent { stem } => (stem, SessionKind::Subagent),
        };
        let id = session_id(stem)?;
        let mut s = Session::new(id.clone(), kind);
        facts(&mut s);
        state(out, s);
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
        if let Some(slug) = v.str("slug").filter(|s| !s.is_empty()) {
            let data = serde_json::json!({ "slug": slug, "file": format!("plans/{slug}.md") });
            self.name_event(parser_state, record, v, out, "plan", slug, data);
        }
        if let Some(team) = v.str("teamName").filter(|s| !s.is_empty()) {
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
        let Some(stem) = v
            .str("sessionId")
            .filter(|s| !s.is_empty())
            .or(match &self.file {
                File::Main { stem } | File::Subagent { stem } => Some(stem.as_str()),
                File::Journal { .. } => None,
            })
        else {
            return;
        };
        let key = format!("{kind}\u{0}{stem}");
        if parser_state.names.get(&key).map(String::as_str) == Some(name) {
            return;
        }
        parser_state.names.insert(key, name.to_owned());
        if let Some(session) = session_id(stem) {
            out.events.push({
                let (at, time_accuracy) =
                    crate::time::parts(v.get("timestamp").and_then(time::rfc3339));
                Event {
                    session: Some(session),
                    kind: kind.to_owned(),
                    at,
                    at_quality: time_accuracy.quality,
                    at_precision: time_accuracy.precision,
                    at_until: time_accuracy.until,
                    after: parser_state.last.clone(),
                    range: Some(range(record)),
                    data,
                }
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
            Some(_) => self.session(parser_state, v, out),
            None => self.own_session(out),
        };
        let Some(session) = session else {
            return RecordOutcome::unreadable("session ID is empty");
        };
        out.events.push({
            let (at, time_accuracy) =
                crate::time::parts(v.get("timestamp").and_then(time::rfc3339));
            Event {
                session: Some(session),
                kind: kind.to_owned(),
                at,
                at_quality: time_accuracy.quality,
                at_precision: time_accuracy.precision,
                at_until: time_accuracy.until,
                after: parser_state.last.clone(),
                range: Some(range(record)),
                data: context_boundary(kind, v),
            }
        });
        RecordOutcome::Used
    }

    /// The summary message of a compaction. The boundary record written just
    /// before it is sent now, naming this message; a summary with no such
    /// record (older versions) is a boundary of its own.
    fn compaction_summary(
        &self,
        parser_state: &mut ParserState,
        record: Record<'_>,
        value: &Json,
        out: &mut Output,
        records: &mut Records,
    ) -> RecordOutcome {
        let pending = parser_state.pending_boundary.take();
        let own = if pending.is_none() {
            if self.event(parser_state, record, value, "compaction", out) != RecordOutcome::Used {
                return RecordOutcome::unreadable("compaction has no session");
            }
            out.events.pop()
        } else {
            None
        };
        let outcome = self.message(parser_state, record, value, Role::User, out, records);
        if let Some(mut boundary) = pending.or(own) {
            boundary.range = boundary.range.map(|(start, _)| (start, range(record).1));
            let mut data: polylogue_model::CompactionBoundary =
                serde_json::from_value(boundary.data.clone()).unwrap_or_default();
            data.summary = parser_state.last.clone();
            boundary.data = serde_json::to_value(data).expect("boundary is JSON");
            out.events.push(boundary);
        }
        outcome
    }

    fn hook_summary(
        &self,
        parser_state: &mut ParserState,
        record: Record<'_>,
        v: &Json,
        summary: HookText,
        out: &mut Output,
    ) -> RecordOutcome {
        let outcome = self.event(parser_state, record, v, "stop_hook_summary", out);
        if outcome != RecordOutcome::Used {
            return outcome;
        }
        let HookText {
            blocks,
            context_blocks,
            invalid,
        } = summary;
        if !blocks.is_empty() {
            let Some(session) = self.session(parser_state, v, out) else {
                return RecordOutcome::unreadable("session ID is empty");
            };
            let (id, id_kind) = match v.str("uuid").filter(|s| !s.is_empty()) {
                Some(uuid) => (message_id(uuid), IdKind::Native),
                None => (
                    crate::content_ids::named(&session, Role::System, &blocks, record.offset),
                    IdKind::Content,
                ),
            };
            let Some(id) = id else {
                return RecordOutcome::unreadable("message ID is empty");
            };
            // The summary records supplied feedback, including when the
            // hook prevented continuation. It is not a delivery receipt.
            // This sidecar cites fields as the harness wrote them, not a value
            // stated in the message. The commit decides whether they were observed
            // locally or only declared by an import; neither changes the message hash.
            let injected_context = !context_blocks.is_empty();
            if injected_context {
                out.hook_fields.insert(
                    out.messages.len(),
                    context_blocks
                        .into_iter()
                        .map(|(block, source_index)| polylogue_model::ParsedHookField {
                            block,
                            start: record.offset,
                            end: range(record).1,
                            pointer: format!("/hookAdditionalContext/{source_index}"),
                        })
                        .collect(),
                );
            }
            out.messages.push(MessageRow {
                response: None,
                page_text: false,
                offset: record.offset,
                end: range(record).1,
                id_kind,
                message: {
                    let (at, time_accuracy) =
                        crate::time::parts(v.get("timestamp").and_then(time::rfc3339));
                    Message {
                        id: id.clone(),
                        session,
                        parent: written_parent(v),
                        role: Role::System,
                        at,
                        at_quality: time_accuracy.quality,
                        at_precision: time_accuracy.precision,
                        at_until: time_accuracy.until,
                        model: None,
                        prompt_kind: Some(if injected_context {
                            PromptKind::InjectedContext
                        } else {
                            PromptKind::Harness
                        }),
                        blocks,
                        usage: Usage::default(),
                        stop_reason: None,
                        reported_cost_usd: None,
                        duration_ms: None,
                        attrs: Default::default(),
                        production: Production::default(),
                    }
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
        let record_session = self.record_session(parser_state, v);
        let child = session_id(&record_session.native);
        let fork = v.get("forkedFrom").and_then(|fork| {
            let parent = session_id(fork.str("sessionId")?)?;
            let message = message_id(fork.str("messageUuid")?)?;
            (Some(&parent) != child.as_ref()).then_some((parent, message))
        });
        let session_kind = record_session.kind;
        let sidechain = session_kind == SessionKind::Sidechain;
        let Some(session) = self.state_session(
            parser_state,
            v,
            record_session,
            fork.as_ref().map(|(parent, message)| (parent, message)),
            out,
        ) else {
            return RecordOutcome::unreadable("session ID is empty");
        };
        let msg = v.get("message").unwrap_or(&Json::Null);
        let content = msg.get("content").unwrap_or(&Json::Null);
        let mut blocks = blocks::blocks(v, content, &mut out.blobs);
        let end = range(record).1;
        let (id, id_kind) = if let Some((parent, message)) = fork {
            let child = child.expect("a copy has a named child");
            out.prefix_members.push(polylogue_model::PrefixMember {
                parent,
                child,
                message: message.clone(),
                source: None,
                position: [record.offset, 0],
            });
            if let Some(name) = v.str("uuid").and_then(message_id)
                && name != message
            {
                out.aliases.push(polylogue_model::Alias {
                    offset: record.offset,
                    end,
                    name,
                    target: Some(message.clone()),
                });
            }
            (Some(message), IdKind::Native)
        } else {
            match v.str("uuid").filter(|s| !s.is_empty()) {
                Some(uuid) => (message_id(uuid), IdKind::Native),
                None => (
                    crate::content_ids::named_blocks(&session, role, &mut blocks, record.offset),
                    IdKind::Content,
                ),
            }
        };
        let Some(id) = id else {
            return RecordOutcome::unreadable("message ID is empty");
        };
        if session_kind == SessionKind::Main && matches!(self.file, File::Main { .. }) {
            parser_state.record_tasks(&session, &blocks);
        }
        let prompt_kind = if role == Role::User {
            let own_subagent = session_kind == SessionKind::Subagent
                && matches!(&self.file, File::Subagent { stem } if stem == session.native().as_str());
            let delegated = (sidechain || session_kind == SessionKind::Subagent)
                && written_parent(v).is_none()
                || own_subagent && !parser_state.subagent_prompt_seen;
            let kind = prompt_kind(v, content, session_kind, delegated);
            if own_subagent && kind.is_some_and(|kind| !kind.is_harness()) {
                parser_state.subagent_prompt_seen = true;
            }
            kind
        } else {
            None
        };
        let usage = msg.get("usage").map(usage).unwrap_or_default();
        let stop_reason = msg
            .str("stop_reason")
            .filter(|s| !s.is_empty())
            .map(str::to_owned);
        let model = msg.str("model").filter(|s| !s.is_empty());
        // This reserved value marks a runtime-authored notice.
        let harness = role == Role::Assistant && model == Some("<synthetic>");
        let model = model.filter(|_| !harness).map(str::to_owned);
        let cost = v.get("costUSD").and_then(Json::as_number).cloned();
        let duration = v.u64("durationMs");
        let production = production(v, msg);
        let attrs = message_attrs(v, msg, &blocks);
        // An agent call's result names the agent it started: the parent's
        // side of the spawn link, with the call.
        if let Some(agent) = v.get("toolUseResult").and_then(|r| r.str("agentId"))
            && let Some(child) = session_id(&format!("agent-{agent}"))
            && child != session
        {
            let call = content.as_array().and_then(|items| {
                let mut results = items
                    .iter()
                    .filter(|i| i.str("type").filter(|s| !s.is_empty()) == Some("tool_result"));
                let only = results.next();
                only.filter(|_| results.next().is_none())
                    .and_then(|i| i.str("tool_use_id").filter(|s| !s.is_empty()))
                    .map(str::to_owned)
            });
            link_at(
                out,
                LinkKind::Spawn,
                session.clone(),
                child,
                None,
                call,
                "toolUseResult.agentId",
            );
        }

        // Each record of a response is a row of its own, carrying the key
        // that ties the records together; the reading joins them into one
        // message. Runtime notices keep their own record identity.
        let response = (role == Role::Assistant && !harness)
            .then(|| {
                msg.str("id")
                    .filter(|s| !s.is_empty())
                    .or_else(|| v.str("requestId").filter(|s| !s.is_empty()))
                    .map(str::to_owned)
            })
            .flatten();
        let message = {
            let (at, time_accuracy) =
                crate::time::parts(v.get("timestamp").and_then(time::rfc3339));
            Message {
                id: id.clone(),
                session,
                parent: written_parent(v),
                role,
                at,
                at_quality: time_accuracy.quality,
                at_precision: time_accuracy.precision,
                at_until: time_accuracy.until,
                model,
                prompt_kind: if harness {
                    Some(PromptKind::Harness)
                } else {
                    prompt_kind
                },
                blocks,
                usage,
                stop_reason,
                reported_cost_usd: cost,
                duration_ms: duration,
                attrs,
                production,
            }
        };
        out.messages.push(MessageRow {
            response,
            page_text: false,
            id_kind,
            offset: record.offset,
            end,
            message,
        });
        parser_state.last = Some(id);
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
        let Some(parent) = self.own_session(out) else {
            return RecordOutcome::unreadable("session ID is empty");
        };
        if let Some(agent) = v
            .str("agentId")
            .filter(|s| !s.is_empty())
            .map(|a| format!("agent-{a}"))
            && let Some(child) = session_id(&agent)
        {
            link(
                out,
                LinkKind::Spawn,
                parent,
                child.clone(),
                "workflow journal",
            );
            if let Some(label) = v.str("label").filter(|s| !s.is_empty()) {
                let mut agent = Session::new(child, SessionKind::Subagent);
                agent.offer_title(label, TitleSource::AgentName);
                state(out, agent);
            }
        }
        self.event(parser_state, record, v, kind, out)
    }

    fn title(&self, v: &Json, field: &str, source: TitleSource, out: &mut Output) -> RecordOutcome {
        let Some(text) = v.str(field).filter(|s| !s.is_empty()) else {
            return RecordOutcome::skipped(format!("no {field}"));
        };
        if self
            .own_session_with(out, |s| s.offer_title(text, source))
            .is_none()
        {
            return RecordOutcome::unreadable("session ID is empty");
        }
        RecordOutcome::Used
    }
}

/// Explicit authorship wins over the position of the prompt in the file.
fn prompt_kind(
    v: &Json,
    content: &Json,
    session_kind: SessionKind,
    delegated: bool,
) -> Option<PromptKind> {
    if v.bool("isCompactSummary") == Some(true) {
        return Some(PromptKind::CompactionSummary);
    }
    let kind = blocks::prompt_kind(v, content);
    let human = v.pointer("/origin/kind").and_then(Json::as_str) == Some("human");
    if kind == PromptKind::Prose && session_kind == SessionKind::Compaction && !human {
        return None;
    }
    if delegated && !human && matches!(kind, PromptKind::Prose | PromptKind::Command) {
        return Some(PromptKind::Model);
    }
    Some(kind)
}

/// States a session's facts from one record: merged into what the step
/// already states about it, as the writer merges steps.
fn state(out: &mut Output, mut session: Session) {
    match out.sessions.iter_mut().find(|s| s.id == session.id) {
        Some(stated) => {
            let cwd = std::mem::take(&mut session.cwd);
            stated.merge(&session);
            for dir in &cwd {
                stated.offer_cwd(dir);
            }
        }
        None => out.sessions.push(session),
    }
}

fn link(out: &mut Output, kind: LinkKind, parent: SessionId, child: SessionId, evidence: &str) {
    link_at(out, kind, parent, child, None, None, evidence);
}

fn link_at(
    out: &mut Output,
    kind: LinkKind,
    parent: SessionId,
    child: SessionId,
    at: Option<MessageId>,
    call_id: Option<String>,
    evidence: &str,
) {
    out.links.push(Link {
        kind,
        copies_prefix: evidence == "fork-context-ref",
        how: None,
        parent,
        child,
        branch_point: at,
        call_id,
        evidence: evidence.to_owned(),
    });
}

/// The record's parent as the source names it.
fn written_parent(v: &Json) -> Option<MessageId> {
    v.str("parentUuid")
        .filter(|s| !s.is_empty())
        .and_then(message_id)
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
        minutes: polylogue_model::FIVE_HOUR_MINUTES,
        from_first_response: true,
    },
    QuotaLimit {
        limit: Cow::Borrowed("seven_day"),
        minutes: polylogue_model::WEEK_MINUTES,
        from_first_response: false,
    },
];

/// `quotaLimits`, which Claude Code records only when it refuses a turn:
/// `{status, resetsAt, rateLimitType}`.
fn quota(q: &Json) -> Option<QuotaReading> {
    let kind = q.str("rateLimitType").filter(|s| !s.is_empty())?;
    let limit = QUOTA_LIMITS.iter().find(|l| {
        kind == l.limit
            || (l.minutes == polylogue_model::WEEK_MINUTES && kind.starts_with("seven_day"))
    })?;
    let status = q.str("status").filter(|s| !s.is_empty()).map(str::to_owned);
    Some(QuotaReading {
        windows: vec![QuotaWindow {
            limit: QuotaLimit {
                limit: Cow::Owned(kind.to_owned()),
                ..limit.clone()
            },
            // A refused turn means the window is used up; other states do
            // not say how much.
            used_percent: (status.as_deref() == Some("rejected")).then_some(100.0),
            resets_at: q.i64("resetsAt").and_then(Timestamp::from_unix_seconds),
            status,
        }],
        ..QuotaReading::default()
    })
}

/// How the harness produced a message: settings, attribution and the
/// API's diagnostics (`Production`).
fn production(v: &Json, msg: &Json) -> Production {
    let s = |k: &str| v.str(k).filter(|s| !s.is_empty()).map(str::to_owned);
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
        thinking_ms: v.u64("thinkingDurationMs"),
        first_token_ms: msg.u64("ttftMs"),
        api_error: s("error").map(|kind| ApiError {
            kind,
            status: v.u64("apiErrorStatus").and_then(|n| u16::try_from(n).ok()),
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

fn context_boundary(kind: &str, value: &Json) -> Json {
    let strings = |pointer: &str| {
        value
            .pointer(pointer)
            .and_then(Json::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Json::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default()
    };
    match kind {
        "compaction" => serde_json::to_value(polylogue_model::CompactionBoundary {
            pre_tokens: value
                .pointer("/compactMetadata/preTokens")
                .and_then(Json::as_u64),
            post_tokens: value
                .pointer("/compactMetadata/postTokens")
                .and_then(Json::as_u64),
            trigger: value
                .pointer("/compactMetadata/trigger")
                .and_then(Json::as_str)
                .map(str::to_owned),
            kept: value
                .pointer("/compactMetadata/preservedMessages/uuids")
                .and_then(Json::as_array)
                .into_iter()
                .flatten()
                .filter_map(Json::as_str)
                .filter_map(message_id)
                .collect(),
            kept_range: value
                .pointer("/compactMetadata/preservedSegment")
                .and_then(|segment| {
                    Some(polylogue_model::KeptRange {
                        first: segment.str("headUuid").and_then(message_id)?,
                        last: segment.str("tailUuid").and_then(message_id)?,
                    })
                }),
            kept_after: value
                .pointer("/compactMetadata/preservedMessages/anchorUuid")
                .or_else(|| value.pointer("/compactMetadata/preservedSegment/anchorUuid"))
                .and_then(Json::as_str)
                .and_then(message_id),
            ..Default::default()
        })
        .expect("boundary is JSON"),
        "microcompaction" => serde_json::to_value(polylogue_model::MicrocompactionBoundary {
            tokens_saved: value
                .pointer("/microcompactMetadata/tokensSaved")
                .and_then(Json::as_u64),
            removed_calls: strings("/microcompactMetadata/compactedToolIds"),
            removed_attachments: strings("/microcompactMetadata/clearedAttachmentUUIDs"),
        })
        .expect("boundary is JSON"),
        _ => value.clone(),
    }
}

fn message_attrs(record: &Json, message: &Json, blocks: &[Block]) -> serde_json::Map<String, Json> {
    let mut attrs = crate::fields::attributes(record, &fields::MESSAGE_FIELDS);
    if let Some(usage) = message.object("usage") {
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
        .array("content")
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

fn usage(u: &Json) -> Usage {
    let n = |k: &str| u.u64(k);
    let thinking = n("thinking_tokens").or_else(|| {
        u.pointer("/output_tokens_details/thinking_tokens")
            .and_then(Json::as_u64)
    });
    // `output_tokens` counts thinking too; `Usage.output` excludes it. A
    // record whose thinking exceeds its output leaves output unknown.
    let output = match (n("output_tokens"), thinking) {
        (Some(output), Some(thinking)) => output.checked_sub(thinking),
        (output, None) => output,
        (None, Some(_)) => None,
    };
    Usage {
        input: n("input_tokens"),
        cached_input: n("cache_read_input_tokens"),
        cache_write: n("cache_creation_input_tokens"),
        output,
        reasoning: thinking,
        cache_write_1h: u
            .get("cache_creation")
            .and_then(|c| c.u64("ephemeral_1h_input_tokens")),
        service_tier: u
            .str("service_tier")
            .filter(|s| !s.is_empty())
            .map(str::to_owned),
        speed: u.str("speed").filter(|s| !s.is_empty()).map(str::to_owned),
    }
}

/// Polylogue labels typed diagnostics for readers; the error and reason
/// strings stay unchanged. The complete summary as the harness wrote it remains in its event.
struct HookText {
    blocks: Vec<Block>,
    /// Original block ordinal and the position in the record's own array.
    context_blocks: Vec<(u64, usize)>,
    invalid: Option<&'static str>,
}

fn hook_text(v: &Json) -> HookText {
    let mut blocks = Vec::new();
    let mut context_blocks = Vec::new();
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
                for (source_index, text) in context.iter().enumerate() {
                    match text.as_str() {
                        Some(text) => {
                            context_blocks.push((blocks.len() as u64, source_index));
                            blocks.push(Block::Text {
                                text: text.to_owned(),
                            });
                        }
                        None => invalid = Some("hookAdditionalContext contains a non-string value"),
                    }
                }
            }
            None => invalid = Some("hookAdditionalContext is not an array of strings"),
        }
    }
    HookText {
        blocks,
        context_blocks,
        invalid,
    }
}

/// A background task's completion, from the notification the harness
/// queues for the model.
fn background_update(v: &Json) -> Option<(String, ToolOutcome, Option<String>, Option<String>)> {
    let content = v.str("content")?;
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

    /// Session facts go out in the order of their native IDs.
    fn flush(&self, _: &mut ParserState, out: &mut Output) {
        out.sessions
            .sort_by(|a, b| a.id.native().as_str().cmp(b.id.native().as_str()));
    }

    fn kind(&self, v: &Json) -> String {
        let ty = v.str("type").filter(|s| !s.is_empty()).unwrap_or("?");
        let sub = match ty {
            "system" => v.str("subtype").filter(|s| !s.is_empty()),
            "progress" => v
                .get("data")
                .and_then(|d| d.str("type").filter(|s| !s.is_empty())),
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
        let Some(ty) = v.str("type").filter(|s| !s.is_empty()) else {
            return RecordOutcome::skipped("no type field");
        };
        if let File::Journal { .. } = self.file {
            return self.journal(parser_state, record, v, ty, out);
        }
        let record_session = self.record_session(parser_state, v);
        if record_session.kind == SessionKind::Sidechain {
            let uuid = v
                .str("uuid")
                .filter(|uuid| !uuid.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| crate::content_ids::untagged_hash(v));
            if record_session.sidechain_start
                && let Some(parent) = v.str("sessionId").and_then(session_id)
                && let Some(child) = session_id(&record_session.native)
            {
                let content = v.pointer("/message/content");
                let mut matching = parser_state
                    .open_tasks
                    .get(parent.native().as_str())
                    .into_iter()
                    .flatten()
                    .filter(|(_, prompt)| {
                        ty == "user"
                            && match content {
                                Some(Json::String(text)) => text.contains(prompt.as_str()),
                                Some(Json::Array(items)) => items.iter().any(|item| {
                                    item.str("type") == Some("text")
                                        && item
                                            .str("text")
                                            .is_some_and(|text| text.contains(prompt.as_str()))
                                }),
                                _ => false,
                            }
                    });
                let call = matching
                    .next()
                    .filter(|_| matching.next().is_none())
                    .map(|(id, _)| id.clone());
                link_at(
                    out,
                    LinkKind::Spawn,
                    parent,
                    child,
                    None,
                    call,
                    "isSidechain",
                );
            }
            if let Some(parent) = v.str("parentUuid") {
                parser_state.sidechains.remove(parent);
            }
            parser_state.sidechains.insert(uuid, record_session.native);
        }
        let hook = (ty == "system"
            && v.str("subtype").filter(|s| !s.is_empty()) == Some("stop_hook_summary"))
        .then(|| hook_text(v));
        let message_record = matches!(ty, "user" | "assistant")
            || (ty == "summary" && v.get("message").is_some())
            || hook
                .as_ref()
                .is_some_and(|summary| !summary.blocks.is_empty());
        if !message_record && let Some(uuid) = v.str("uuid").filter(|s| !s.is_empty()) {
            // A child naming this record as its parent belongs under the
            // nearest message above it; a compaction boundary points there
            // through `logicalParentUuid`.
            if let Some(name) = message_id(uuid) {
                let (offset, end) = range(record);
                out.aliases.push(polylogue_model::Alias {
                    offset,
                    end,
                    name,
                    target: v
                        .str("logicalParentUuid")
                        .filter(|s| !s.is_empty())
                        .or_else(|| v.str("parentUuid").filter(|s| !s.is_empty()))
                        .and_then(message_id),
                });
            }
        }
        self.names(parser_state, record, v, out);
        if self.record_session(parser_state, v).kind == SessionKind::Compaction
            && !parser_state.compaction_ownership_reported
        {
            let session = self.own_session(out);
            let (at, accuracy) = crate::time::parts(v.get("timestamp").and_then(time::rfc3339));
            out.events.push(Event {
                session, kind: "compaction_ownership_unknown".into(),
                at, at_quality: accuracy.quality, at_precision: accuracy.precision,
                at_until: accuracy.until, after: None, range: Some(range(record)),
                data: serde_json::json!({"reason":"compactor identity and root carrier do not establish which work session owns the replay"}),
            });
            parser_state.compaction_ownership_reported = true;
        }
        if matches!(ty, "summary" | "custom-title" | "ai-title") {
            records.fields(ty, v, &fields::TITLE_FIELDS);
        }
        if ty == "continued-in" {
            records.fields(ty, v, &fields::CONTINUATION_FIELDS);
        }
        let summary = (ty == "user" && v.bool("isCompactSummary") == Some(true))
            || (ty == "summary" && v.get("message").is_some());
        if !summary && let Some(mut boundary) = parser_state.pending_boundary.take() {
            boundary.range = boundary.range.map(|(start, _)| (start, range(record).1));
            out.events.push(boundary);
        }
        match ty {
            "user" if summary => self.compaction_summary(parser_state, record, v, out, records),
            "user" => self.message(parser_state, record, v, Role::User, out, records),
            "assistant" => self.message(parser_state, record, v, Role::Assistant, out, records),
            "system" => match v.str("subtype").filter(|s| !s.is_empty()) {
                Some("stop_hook_summary") => self.hook_summary(
                    parser_state,
                    record,
                    v,
                    hook.expect("recognized stop-hook summary"),
                    out,
                ),
                Some("compact_boundary") => {
                    records.fields("system/compact_boundary", v, &fields::COMPACTION_FIELDS);
                    let outcome = self.event(parser_state, record, v, "compaction", out);
                    if outcome == RecordOutcome::Used {
                        parser_state.pending_boundary = out.events.pop();
                    }
                    outcome
                }
                Some("microcompact_boundary") => {
                    records.fields(
                        "system/microcompact_boundary",
                        v,
                        &fields::COMPACTION_FIELDS,
                    );
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
                        Some(_) => self.session(parser_state, v, out),
                        None => self.own_session(out),
                    }
                {
                    let (result_at, result_at_evidence) =
                        crate::time::parts(v.get("timestamp").and_then(time::rfc3339));
                    out.background.push(BackgroundUpdate {
                        result_at,
                        result_at_evidence,
                        reported_duration_ms: v.u64("durationMs"),
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
                if let Some(dir) = v.str("relocatedCwd").filter(|s| !s.is_empty()) {
                    self.own_session_with(out, |s| s.offer_cwd(dir));
                }
                self.event(parser_state, record, v, "relocated", out)
            }
            "agent-name" => {
                let d = self.title(v, "agentName", TitleSource::AgentName, out);
                if d != RecordOutcome::Used {
                    return d;
                }
                self.event(parser_state, record, v, "agent_name", out)
            }
            "custom-title" => self.title(v, "customTitle", TitleSource::Custom, out),
            "ai-title" => self.title(v, "aiTitle", TitleSource::Generated, out),
            // A `summary` line with only its text titles the session
            // (`leafUuid` names the message it summarizes). One carrying a
            // `message` is the summary a compaction replay wrote.
            "summary" if v.get("message").is_some() => {
                let mut value = v.clone();
                value["isCompactSummary"] = Json::Bool(true);
                self.compaction_summary(parser_state, record, &value, out, records)
            }
            "summary" => self.title(v, "summary", TitleSource::Summary, out),
            "continued-in" => {
                let (Some(parent), Some(child)) = (
                    self.own_session(out),
                    v.str("continuedInSessionId")
                        .filter(|s| !s.is_empty())
                        .and_then(session_id),
                ) else {
                    return RecordOutcome::skipped("continued-in without a target");
                };
                link(
                    out,
                    LinkKind::Continuation,
                    parent,
                    child,
                    "continuedInSessionId",
                );
                RecordOutcome::Used
            }
            "fork-context-ref" => {
                let child = self.session(parser_state, v, out);
                let (Some(parent), Some(child)) = (
                    v.str("parentSessionId")
                        .filter(|s| !s.is_empty())
                        .and_then(session_id),
                    child,
                ) else {
                    return RecordOutcome::skipped("fork-context-ref without a parent");
                };
                // Evidence for the subagent link, with its branch point:
                // never a second parent.
                let at = v
                    .str("parentLastUuid")
                    .filter(|s| !s.is_empty())
                    .and_then(message_id);
                link_at(
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
                match data.str("type").filter(|s| !s.is_empty()) {
                    Some("agent_progress") => {
                        records.fields(
                            "progress/agent_progress",
                            v,
                            &fields::AGENT_PROGRESS_FIELDS,
                        );
                        let parent = self.session(parser_state, v, out);
                        let child = data
                            .str("agentId")
                            .filter(|s| !s.is_empty())
                            .and_then(|a| session_id(&format!("agent-{a}")));
                        let (Some(parent), Some(child)) = (parent, child) else {
                            return RecordOutcome::skipped("agent progress without an agent ID");
                        };
                        let call = v
                            .str("parentToolUseID")
                            .filter(|s| !s.is_empty())
                            .map(str::to_owned);
                        link_at(
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
                            let role = match embedded.str("type").filter(|s| !s.is_empty()) {
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
                            // An embedded message is named and joined like
                            // one in the agent's own file; nothing carries over.
                            let mut embedded_state = ParserState {
                                subagent_prompt_seen: written_parent(&embedded).is_some(),
                                ..ParserState::default()
                            };
                            let outcome = agent.message(
                                &mut embedded_state,
                                record,
                                &embedded,
                                role,
                                out,
                                records,
                            );
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
}

#[cfg(test)]
mod tests;
