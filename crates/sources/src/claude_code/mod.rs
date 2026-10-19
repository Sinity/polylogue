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

mod blocks;
mod fields;
pub mod side;

use crate::account::{Accounting, Disposition};
use crate::fields::unaccounted;
use crate::frame::Record;
use crate::origin::Root;
use crate::parse::{BackgroundUpdate, MessageRow, Output, Parser, StepParser};
use crate::time;
use polylogue_model::{
    ApiError, Attribution, ErrorReason, Event, Json, Link, LinkKind, Message, MessageId, NativeId,
    Origin, Production, PromptKind, Role, Session, SessionId, SessionKind, TitleSource,
    ToolOutcome, Usage,
};
use serde::{Deserialize, Serialize};
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

pub fn open(path: &Path) -> Option<Box<dyn StepParser>> {
    File::from_path(path).map(|file| Box::new(ClaudeCode { file }) as Box<dyn StepParser>)
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
pub struct Carry {
    /// Session facts gathered so far, by native ID. A file touches a few
    /// sessions: its own, the ones it copies from, its sidechain.
    sessions: BTreeMap<String, Session>,
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
    /// The plan name (`slug`) last recorded per session, so a session's
    /// `plan` event is sent once per name, not with every record.
    #[serde(default)]
    slugs: BTreeMap<String, String>,
}

/// How many record aliases a parser keeps.
const ALIASES: usize = 4096;

impl Carry {
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
struct Route {
    native: String,
    kind: SessionKind,
}

impl ClaudeCode {
    fn route(&self, v: &Json) -> Route {
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
                Route { native, kind }
            }
            File::Journal { session, .. } => Route {
                native: session.clone(),
                kind: SessionKind::Main,
            },
            File::Main { stem } => {
                let base = str_field(v, "sessionId").unwrap_or(stem).to_owned();
                if v.get("isSidechain").and_then(Json::as_bool) == Some(true) {
                    Route {
                        native: format!("{base}/sidechain"),
                        kind: SessionKind::Sidechain,
                    }
                } else {
                    Route {
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
        carry: &'c mut Carry,
        v: &Json,
        out: &mut Output,
    ) -> Option<&'c mut Session> {
        let route = self.route(v);
        let id = session_id(&route.native)?;
        let parent = str_field(v, "sessionId").and_then(session_id);
        match (&self.file, route.kind) {
            (File::Main { stem }, SessionKind::Main) if route.native != *stem => {
                if let Some(child) = session_id(stem) {
                    link(carry, out, LinkKind::Copy, id.clone(), child, "sessionId");
                }
            }
            (_, SessionKind::Sidechain)
            | (File::Subagent { .. }, SessionKind::Subagent | SessionKind::Compaction) => {
                if let Some(parent) = parent.filter(|p| *p != id) {
                    let evidence = match route.kind {
                        SessionKind::Sidechain => "isSidechain",
                        SessionKind::Compaction => "agentId acompact",
                        _ => "agentId",
                    };
                    link(carry, out, LinkKind::Spawn, parent, id.clone(), evidence);
                }
            }
            _ => {}
        }
        let s = carry
            .sessions
            .entry(route.native)
            .or_insert_with(|| Session::new(id, route.kind));
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
    fn own_session(&self, carry: &mut Carry) -> Option<SessionId> {
        let stem = self.file.stem();
        let kind = match self.file {
            File::Main { .. } | File::Journal { .. } => SessionKind::Main,
            File::Subagent { .. } if stem.starts_with("agent-acompact") => SessionKind::Compaction,
            File::Subagent { .. } => SessionKind::Subagent,
        };
        let id = session_id(stem)?;
        carry
            .sessions
            .entry(stem.to_owned())
            .or_insert_with(|| Session::new(id.clone(), kind));
        Some(id)
    }

    /// A session names its plan file (`plans/<slug>.md`) by `slug` on its
    /// records. Sent as a `plan` event when the name first appears or
    /// changes, so readers find the plan from the session.
    fn plan(&self, carry: &mut Carry, record: Record<'_>, v: &Json, out: &mut Output) {
        let Some(slug) = str_field(v, "slug") else {
            return;
        };
        let Some(stem) = str_field(v, "sessionId").or(match &self.file {
            File::Main { stem } | File::Subagent { stem } => Some(stem.as_str()),
            File::Journal { .. } => None,
        }) else {
            return;
        };
        if carry.slugs.get(stem).map(String::as_str) == Some(slug) {
            return;
        }
        carry.slugs.insert(stem.to_owned(), slug.to_owned());
        if let Some(session) = session_id(stem) {
            out.events.push(Event {
                session,
                kind: "plan".to_owned(),
                at: v.get("timestamp").and_then(time::rfc3339),
                after: carry.last.clone(),
                range: Some(range(record)),
                data: serde_json::json!({ "slug": slug, "file": format!("plans/{slug}.md") }),
            });
        }
    }

    fn event(
        &self,
        carry: &mut Carry,
        record: Record<'_>,
        v: &Json,
        kind: &str,
        out: &mut Output,
    ) -> Disposition {
        let session = match v.get("sessionId").or(v.get("agentId")) {
            Some(_) => self.session(carry, v, out).map(|s| s.id.clone()),
            None => self.own_session(carry),
        };
        let Some(session) = session else {
            return Disposition::refused("session ID is empty");
        };
        out.events.push(Event {
            session,
            kind: kind.to_owned(),
            at: v.get("timestamp").and_then(time::rfc3339),
            after: carry.last.clone(),
            range: Some(range(record)),
            data: v.clone(),
        });
        Disposition::Materialized
    }

    fn message(
        &self,
        carry: &mut Carry,
        record: Record<'_>,
        v: &Json,
        role: Role,
        out: &mut Output,
        acc: &mut Accounting,
    ) -> Disposition {
        let kind = self.kind(v);
        for path in unaccounted(v, fields::MESSAGE_FIELDS) {
            acc.unread(&kind, &path);
        }
        let Some(session) = self.session(carry, v, out).map(|s| s.id.clone()) else {
            return Disposition::refused("session ID is empty");
        };
        let native = str_field(v, "uuid")
            .map(str::to_owned)
            .unwrap_or_else(|| format!("{}@{}", self.file.stem(), record.offset));
        let Some(id) = message_id(&native) else {
            return Disposition::refused("message ID is empty");
        };
        let msg = v.get("message").unwrap_or(&Json::Null);
        let content = msg.get("content").unwrap_or(&Json::Null);
        let blocks = blocks::blocks(v, content, &mut out.blobs);
        let usage = msg.get("usage").map(usage).unwrap_or_default();
        let stop_reason = str_field(msg, "stop_reason").map(str::to_owned);
        let model = str_field(msg, "model").map(str::to_owned);
        let cost = v.get("costUSD").and_then(Json::as_number).cloned();
        let duration = v.get("durationMs").and_then(Json::as_u64);
        let production = production(v, msg);
        let end = range(record).1;
        if let Some(fork) = v.get("forkedFrom")
            && let (Some(parent), Some(at)) = (
                str_field(fork, "sessionId").and_then(session_id),
                str_field(fork, "messageUuid").and_then(message_id),
            )
            && parent != session
        {
            link_at(
                carry,
                out,
                LinkKind::Fork,
                parent,
                session.clone(),
                Some(at),
                None,
                "forkedFrom",
            );
        }

        if role == Role::Assistant {
            let key = str_field(msg, "id")
                .or_else(|| str_field(v, "requestId"))
                .unwrap_or(&native)
                .to_owned();
            let same = |r: &Response| r.key == key && r.message.session == session;
            if !carry.open.as_ref().is_some_and(same) {
                let reopened = carry
                    .recent
                    .iter()
                    .position(same)
                    .and_then(|i| carry.recent.remove(i));
                close(carry, out);
                carry.open = reopened;
            }
            match &mut carry.open {
                Some(open) => {
                    let target = Some(open.message.id.clone());
                    if native != open.message.id.native().as_str() {
                        carry.alias(&native, target);
                    }
                    let Some(open) = &mut carry.open else {
                        unreachable!("the open response was just matched")
                    };
                    open.end = open.end.max(end);
                    let m = &mut open.message;
                    m.blocks.extend(blocks);
                    if !usage.is_empty() {
                        m.usage = usage;
                    }
                    m.stop_reason = stop_reason.or(m.stop_reason.take());
                    m.model = m.model.take().or(model);
                    m.reported_cost_usd = cost.or(m.reported_cost_usd.take());
                    m.duration_ms = duration.or(m.duration_ms);
                    m.production.merge(production);
                }
                None => {
                    carry.open = Some(Response {
                        key,
                        offset: record.offset,
                        end,
                        message: Message {
                            id: id.clone(),
                            session,
                            parent: carry.parent(v),
                            role,
                            at: v.get("timestamp").and_then(time::rfc3339),
                            model,
                            prompt_kind: None,
                            blocks,
                            usage,
                            stop_reason,
                            reported_cost_usd: cost,
                            duration_ms: duration,
                            production,
                        },
                    });
                    carry.last = Some(id);
                }
            }
        } else {
            let message = Message {
                id: id.clone(),
                session,
                parent: carry.parent(v),
                role,
                at: v.get("timestamp").and_then(time::rfc3339),
                model,
                prompt_kind: Some(self.prompt_kind(v, content)),
                blocks,
                usage,
                stop_reason,
                reported_cost_usd: cost,
                duration_ms: duration,
                production,
            };
            out.messages.push(MessageRow {
                offset: record.offset,
                end,
                message,
            });
            carry.last = Some(id);
        }
        Disposition::Materialized
    }

    /// A Workflow journal record: an agent the run started, what it
    /// returned, or the run's launch. Each spawned agent is linked to the
    /// session that ran the workflow and named by its label.
    fn journal(
        &self,
        carry: &mut Carry,
        record: Record<'_>,
        v: &Json,
        ty: &str,
        out: &mut Output,
    ) -> Disposition {
        let kind = match ty {
            "started" => "workflow_agent_started",
            "result" => "workflow_agent_result",
            "launched" => "workflow_launched",
            _ => return Disposition::unknown("unrecognized journal record"),
        };
        let Some(parent) = self.own_session(carry) else {
            return Disposition::refused("session ID is empty");
        };
        if let Some(agent) = str_field(v, "agentId").map(|a| format!("agent-{a}"))
            && let Some(child) = session_id(&agent)
        {
            link(
                carry,
                out,
                LinkKind::Spawn,
                parent,
                child.clone(),
                "workflow journal",
            );
            if let Some(label) = str_field(v, "label") {
                carry
                    .sessions
                    .entry(agent)
                    .or_insert_with(|| Session::new(child, SessionKind::Subagent))
                    .offer_title(label, TitleSource::AgentName);
            }
        }
        self.event(carry, record, v, kind, out)
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

    fn title(&self, carry: &mut Carry, v: &Json, field: &str, source: TitleSource) -> Disposition {
        let Some(text) = str_field(v, field) else {
            return Disposition::unknown(format!("no {field}"));
        };
        let Some(id) = self.own_session(carry) else {
            return Disposition::refused("session ID is empty");
        };
        if let Some(s) = carry.sessions.get_mut(id.native().as_str()) {
            s.offer_title(text, source);
        }
        Disposition::Materialized
    }
}

fn link(
    carry: &mut Carry,
    out: &mut Output,
    kind: LinkKind,
    parent: SessionId,
    child: SessionId,
    evidence: &str,
) {
    link_at(carry, out, kind, parent, child, None, None, evidence);
}

#[allow(clippy::too_many_arguments)]
fn link_at(
    carry: &mut Carry,
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
    if carry.links.insert(key) {
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
fn close(carry: &mut Carry, out: &mut Output) {
    if let Some(open) = carry.open.take() {
        out.messages.push(MessageRow {
            offset: open.offset,
            end: open.end,
            message: open.message.clone(),
        });
        carry.recent.push_back(open);
        while carry.recent.len() > REOPEN_WINDOW {
            carry.recent.pop_front();
        }
    }
}

/// Where a record starts and ends in its input.
fn range(record: Record<'_>) -> (u64, u64) {
    (record.offset, record.offset + record.bytes.len() as u64)
}

/// How the harness produced a message: settings, attribution and the
/// API's diagnostics (`Production`).
fn production(v: &Json, msg: &Json) -> Production {
    let s = |k: &str| str_field(v, k).map(str::to_owned);
    let kept = |x: Option<&Json>| x.filter(|x| !x.is_null()).cloned();
    Production {
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
        turn_origin: s("turnOrigin"),
        quota: kept(v.get("quotaLimits")),
    }
}

fn usage(u: &Json) -> Usage {
    let n = |k: &str| u.get(k).and_then(Json::as_u64);
    Usage {
        input: n("input_tokens"),
        cached_input: n("cache_read_input_tokens"),
        cache_write: n("cache_creation_input_tokens"),
        output: n("output_tokens"),
        reasoning: None,
        service_tier: str_field(u, "service_tier").map(str::to_owned),
    }
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
    type Carry = Carry;

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

    fn record(
        &self,
        carry: &mut Carry,
        record: Record<'_>,
        v: &Json,
        out: &mut Output,
        acc: &mut Accounting,
    ) -> Disposition {
        if !v.is_object() {
            return Disposition::refused("record is not a JSON object");
        }
        let Some(ty) = str_field(v, "type") else {
            return Disposition::unknown("no type field");
        };
        if let File::Journal { .. } = self.file {
            return self.journal(carry, record, v, ty, out);
        }
        let message_record = matches!(ty, "user" | "assistant")
            && v.get("isCompactSummary").and_then(Json::as_bool) != Some(true);
        if !message_record && let Some(uuid) = str_field(v, "uuid") {
            // A child naming this record as its parent belongs under the
            // nearest message above it; a compaction boundary points there
            // through `logicalParentUuid`.
            let target = str_field(v, "logicalParentUuid")
                .or_else(|| str_field(v, "parentUuid"))
                .and_then(|p| carry.resolve(p));
            carry.alias(uuid, target);
        }
        self.plan(carry, record, v, out);
        match ty {
            "user" if v.get("isCompactSummary").and_then(Json::as_bool) == Some(true) => {
                self.event(carry, record, v, "compaction_summary", out)
            }
            "user" => self.message(carry, record, v, Role::User, out, acc),
            "assistant" => self.message(carry, record, v, Role::Assistant, out, acc),
            "system" => match str_field(v, "subtype") {
                Some("compact_boundary") => self.event(carry, record, v, "compaction", out),
                Some("microcompact_boundary") => {
                    self.event(carry, record, v, "microcompaction", out)
                }
                Some(sub) => self.event(carry, record, v, &sub.replace('-', "_"), out),
                None => Disposition::unknown("system record without a subtype"),
            },
            "attachment" => self.event(carry, record, v, "attachment", out),
            "pr-link"
            | "permission-mode"
            | "last-prompt"
            | "bridge-session"
            | "frame-link"
            | "worktree-state"
            | "artifact-comment-monitor"
            | "artifact-autoreact-ledger" => {
                self.event(carry, record, v, &ty.replace('-', "_"), out)
            }
            "file-history-snapshot" | "file-history-delta" => {
                self.event(carry, record, v, "file_history", out)
            }
            "cost-state" => self.event(carry, record, v, "tool_cost", out),
            // The request the harness sent the model, rebuilt exactly: the
            // system prompt and tools (`shape`) and each message as sent
            // (`blob`) are keyed by hash, and the request lists them.
            "api-request" => self.event(carry, record, v, "api_request", out),
            "api-request-shape" => self.event(carry, record, v, "api_request_shape", out),
            "api-request-blob" => self.event(carry, record, v, "api_request_blob", out),
            "started" => self.event(carry, record, v, "subagent_started", out),
            "result" => self.event(carry, record, v, "subagent_result", out),
            "queue-operation" => {
                if let Some((task_id, outcome, call_id, output_file)) = background_update(v)
                    && let Some(session) = match v.get("sessionId") {
                        Some(_) => self.session(carry, v, out).map(|s| s.id.clone()),
                        None => self.own_session(carry),
                    }
                {
                    out.background.push(BackgroundUpdate {
                        session,
                        task_id,
                        call_id,
                        outcome,
                        output_file,
                    });
                }
                self.event(carry, record, v, "queue", out)
            }
            "relocated" => {
                if let Some(dir) = str_field(v, "relocatedCwd")
                    && let Some(id) = self.own_session(carry)
                    && let Some(s) = carry.sessions.get_mut(id.native().as_str())
                {
                    s.offer_cwd(dir);
                }
                self.event(carry, record, v, "relocated", out)
            }
            "agent-name" => {
                let d = self.title(carry, v, "agentName", TitleSource::AgentName);
                if d != Disposition::Materialized {
                    return d;
                }
                self.event(carry, record, v, "agent_name", out)
            }
            "custom-title" => self.title(carry, v, "customTitle", TitleSource::Custom),
            "ai-title" => self.title(carry, v, "aiTitle", TitleSource::Generated),
            "summary" => self.title(carry, v, "summary", TitleSource::Summary),
            "continued-in" => {
                let (Some(parent), Some(child)) = (
                    self.own_session(carry),
                    str_field(v, "continuedInSessionId").and_then(session_id),
                ) else {
                    return Disposition::unknown("continued-in without a target");
                };
                link(
                    carry,
                    out,
                    LinkKind::Continuation,
                    parent,
                    child,
                    "continuedInSessionId",
                );
                Disposition::Materialized
            }
            "fork-context-ref" => {
                let child = self.session(carry, v, out).map(|s| s.id.clone());
                let (Some(parent), Some(child)) =
                    (str_field(v, "parentSessionId").and_then(session_id), child)
                else {
                    return Disposition::unknown("fork-context-ref without a parent");
                };
                // Evidence for the subagent link, with its branch point
                // (spec sources/claude-code.md): never a second parent.
                let at = str_field(v, "parentLastUuid").and_then(message_id);
                link_at(
                    carry,
                    out,
                    LinkKind::Spawn,
                    parent,
                    child,
                    at,
                    None,
                    "fork-context-ref",
                );
                self.event(carry, record, v, "fork_context", out)
            }
            "progress" => {
                let data = v.get("data").unwrap_or(&Json::Null);
                match str_field(data, "type") {
                    Some("agent_progress") => {
                        let parent = self.session(carry, v, out).map(|s| s.id.clone());
                        let child = str_field(data, "agentId")
                            .and_then(|a| session_id(&format!("agent-{a}")));
                        let (Some(parent), Some(child)) = (parent, child) else {
                            return Disposition::unknown("agent progress without an agent ID");
                        };
                        let call = str_field(v, "parentToolUseID").map(str::to_owned);
                        link_at(
                            carry,
                            out,
                            LinkKind::Spawn,
                            parent,
                            child,
                            None,
                            call,
                            "agent_progress",
                        );
                        Disposition::Materialized
                    }
                    Some("hook_progress") => self.event(carry, record, v, "hook_run", out),
                    Some(_) => {
                        Disposition::unknown("progress tick; the tool's own result supersedes it")
                    }
                    None => Disposition::unknown("progress without data"),
                }
            }
            "mode" => Disposition::unknown("always \"normal\" in observed data"),
            "atis-latch" => Disposition::unknown("always empty in observed data"),
            "agent-color" => Disposition::unknown("terminal colour only; names no agent"),
            "init" => Disposition::unknown("empty start marker"),
            _ => Disposition::unknown("unrecognized record type"),
        }
    }

    fn flush(&self, carry: &mut Carry, out: &mut Output) {
        if let Some(open) = &carry.open {
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
        out.sessions = carry.sessions.values().cloned().collect();
    }
}

#[cfg(test)]
mod tests;
