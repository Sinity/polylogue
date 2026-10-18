//! Claude Code session files: `~/.claude/projects/<project>/<session>.jsonl`
//! and subagent transcripts in `<project>/<session>/subagents/`.
//!
//! One line is one record. An assistant response arrives as several
//! records sharing `message.id`, one per content block; they become one
//! message. A message copied from another session (a resume replays the
//! earlier history) keeps its ID and its original session, so it is stored
//! once, and the copying session gets a `copy` link.

mod blocks;
mod fields;

use crate::account::{Accounting, Disposition};
use crate::fields::unaccounted;
use crate::frame::Record;
use crate::parse::{BackgroundUpdate, MessageRow, Output, Parser, StepParser};
use crate::time;
use polylogue_model::{
    ErrorReason, Event, Json, Link, LinkKind, Message, MessageId, NativeId, Origin, Role, Session,
    SessionId, SessionKind, TitleSource, ToolOutcome, Usage,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;

pub const LOCATIONS: &[&str] = &[
    ".claude/projects/*/*.jsonl",
    ".claude/projects/*/*/subagents/agent-*.jsonl",
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
    Main { stem: String },
    Subagent { stem: String },
}

impl File {
    pub fn from_path(path: &Path) -> Option<Self> {
        if path.extension()? != "jsonl" {
            return None;
        }
        let stem = path.file_stem()?.to_str()?.to_owned();
        let dir = path.parent()?.file_name()?.to_str()?;
        Some(if dir == "subagents" {
            Self::Subagent { stem }
        } else {
            Self::Main { stem }
        })
    }

    fn stem(&self) -> &str {
        match self {
            Self::Main { stem } | Self::Subagent { stem } => stem,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Response {
    key: String,
    /// Where the response's first record starts.
    offset: u64,
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
            File::Subagent { stem } => Route {
                native: str_field(v, "agentId")
                    .map(|a| format!("agent-{a}"))
                    .unwrap_or_else(|| stem.clone()),
                kind: SessionKind::Subagent,
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
            (_, SessionKind::Sidechain) | (File::Subagent { .. }, SessionKind::Subagent) => {
                if let Some(parent) = parent.filter(|p| *p != id) {
                    let evidence = if route.kind == SessionKind::Sidechain {
                        "isSidechain"
                    } else {
                        "agentId"
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
        Some(s)
    }

    /// The file's own session, for records that name none.
    fn own_session(&self, carry: &mut Carry) -> Option<SessionId> {
        let stem = self.file.stem();
        let kind = match self.file {
            File::Main { .. } => SessionKind::Main,
            File::Subagent { .. } => SessionKind::Subagent,
        };
        let id = session_id(stem)?;
        carry
            .sessions
            .entry(stem.to_owned())
            .or_insert_with(|| Session::new(id.clone(), kind));
        Some(id)
    }

    fn event(&self, carry: &mut Carry, v: &Json, kind: &str, out: &mut Output) -> Disposition {
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
                    let m = &mut open.message;
                    m.blocks.extend(blocks);
                    if !usage.is_empty() {
                        m.usage = usage;
                    }
                    m.stop_reason = stop_reason.or(m.stop_reason.take());
                    m.model = m.model.take().or(model);
                    m.reported_cost_usd = cost.or(m.reported_cost_usd.take());
                    m.duration_ms = duration.or(m.duration_ms);
                }
                None => {
                    carry.open = Some(Response {
                        key,
                        offset: record.offset,
                        message: Message {
                            id: id.clone(),
                            session,
                            parent: str_field(v, "parentUuid").and_then(message_id),
                            role,
                            at: v.get("timestamp").and_then(time::rfc3339),
                            model,
                            prompt_kind: None,
                            blocks,
                            usage,
                            stop_reason,
                            reported_cost_usd: cost,
                            duration_ms: duration,
                        },
                    });
                    carry.last = Some(id);
                }
            }
        } else {
            let message = Message {
                id: id.clone(),
                session,
                parent: str_field(v, "parentUuid").and_then(message_id),
                role,
                at: v.get("timestamp").and_then(time::rfc3339),
                model,
                prompt_kind: Some(blocks::prompt_kind(v, content)),
                blocks,
                usage,
                stop_reason,
                reported_cost_usd: cost,
                duration_ms: duration,
            };
            out.messages.push(MessageRow {
                offset: record.offset,
                message,
            });
            carry.last = Some(id);
        }
        Disposition::Materialized
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
    let key = format!("{kind:?} {parent} {child}");
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
            message: open.message.clone(),
        });
        carry.recent.push_back(open);
        while carry.recent.len() > REOPEN_WINDOW {
            carry.recent.pop_front();
        }
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
        match ty {
            "user" if v.get("isCompactSummary").and_then(Json::as_bool) == Some(true) => {
                self.event(carry, v, "compaction_summary", out)
            }
            "user" => self.message(carry, record, v, Role::User, out, acc),
            "assistant" => self.message(carry, record, v, Role::Assistant, out, acc),
            "system" => match str_field(v, "subtype") {
                Some("compact_boundary") => self.event(carry, v, "compaction", out),
                Some("microcompact_boundary") => self.event(carry, v, "microcompaction", out),
                Some(sub) => self.event(carry, v, &sub.replace('-', "_"), out),
                None => Disposition::unknown("system record without a subtype"),
            },
            "attachment" => self.event(carry, v, "attachment", out),
            "pr-link"
            | "permission-mode"
            | "last-prompt"
            | "bridge-session"
            | "frame-link"
            | "worktree-state"
            | "artifact-comment-monitor"
            | "artifact-autoreact-ledger" => self.event(carry, v, &ty.replace('-', "_"), out),
            "file-history-snapshot" | "file-history-delta" => {
                self.event(carry, v, "file_history", out)
            }
            "cost-state" => self.event(carry, v, "tool_cost", out),
            "started" => self.event(carry, v, "subagent_started", out),
            "result" => self.event(carry, v, "subagent_result", out),
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
                self.event(carry, v, "queue", out)
            }
            "relocated" => {
                if let Some(dir) = str_field(v, "relocatedCwd")
                    && let Some(id) = self.own_session(carry)
                    && let Some(s) = carry.sessions.get_mut(id.native().as_str())
                {
                    s.offer_cwd(dir);
                }
                self.event(carry, v, "relocated", out)
            }
            "agent-name" => {
                let d = self.title(carry, v, "agentName", TitleSource::AgentName);
                if d != Disposition::Materialized {
                    return d;
                }
                self.event(carry, v, "agent_name", out)
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
                let at = str_field(v, "parentLastUuid").and_then(message_id);
                link_at(
                    carry,
                    out,
                    LinkKind::Fork,
                    parent,
                    child,
                    at,
                    None,
                    "fork-context-ref",
                );
                self.event(carry, v, "fork_context", out)
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
                    Some("hook_progress") => self.event(carry, v, "hook_run", out),
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
