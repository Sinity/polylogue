//! Codex: rollouts `<codex>/sessions/YYYY/MM/DD/rollout-<time>-<thread>.jsonl`
//! (and the same under `archived_sessions/`), the prompt history
//! `history.jsonl` and the thread-name index `session_index.jsonl`, where
//! `<codex>` is `$CODEX_HOME`, by default `~/.codex`.
//!
//! A rollout is one thread. Each line is an envelope
//! `{timestamp, type, payload}`; files from mid-2025 write the items bare
//! after a header line. Items arrive complete, so a parser never holds a
//! whole turn: each response item is one message, and only the latest
//! assistant item stays open until the token count that follows it.
//!
//! A subagent's rollout starts with its parent's history copied in
//! (`subagent_history_start_ordinal` says where its own begins). The copy
//! is stored once, in the parent: copied items that carry an ID keep it
//! and their parent session, and the rest is accounted as copied.

pub mod database;
mod fields;
mod index;
mod items;

use crate::fields::unaccounted;
use crate::frame::Record;
use crate::origin::Root;
use crate::parse::{BackgroundUpdate, MessageRow, Output, Parser, StepParser};
use crate::time;
use crate::{Accounting, Disposition};
use items::{Completed, str_field};
use polylogue_model::{
    Block, Event, Json, Link, LinkKind, Message, MessageId, NativeId, Origin, Production,
    PromptKind, Role, Session, SessionId, SessionKind, TitleSource, ToolAction, ToolOutcome, Usage,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::path::Path;

pub const ROOT: Root = Root {
    env: Some("CODEX_HOME"),
    default: ".codex",
};

pub const LOCATIONS: &[&str] = &[
    "sessions/**/rollout-*.jsonl",
    "archived_sessions/**/rollout-*.jsonl",
    "history.jsonl",
    "session_index.jsonl",
];

pub fn recognize(path: &Path, head: &[u8]) -> bool {
    File::from_path(path).is_some()
        && head
            .iter()
            .find(|b| !b.is_ascii_whitespace())
            .is_none_or(|&b| b == b'{')
}

pub fn open(path: &Path) -> Option<Box<dyn StepParser>> {
    Some(match File::from_path(path)? {
        File::Rollout { thread } => Box::new(Rollout { thread }),
        File::History => Box::new(index::History),
        File::SessionIndex => Box::new(index::SessionIndex),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum File {
    /// A thread's rollout; the thread ID from the file name, for files
    /// whose first record does not name it.
    Rollout {
        thread: Option<String>,
    },
    History,
    SessionIndex,
}

impl File {
    pub fn from_path(path: &Path) -> Option<Self> {
        let name = path.file_name()?.to_str()?;
        match name {
            "history.jsonl" => Some(Self::History),
            "session_index.jsonl" => Some(Self::SessionIndex),
            _ => {
                let stem = name.strip_prefix("rollout-")?.strip_suffix(".jsonl")?;
                // `rollout-<time>-<uuid>`: the UUID is the last 36 characters.
                let thread = stem
                    .len()
                    .checked_sub(36)
                    .map(|at| &stem[at..])
                    .filter(|t| t.chars().filter(|&c| c == '-').count() == 4)
                    .map(str::to_owned);
                Some(Self::Rollout { thread })
            }
        }
    }
}

fn origin() -> Origin {
    Origin::new("codex").expect("constant origin")
}

pub fn session_id(native: &str) -> Option<SessionId> {
    Some(SessionId::new(origin(), NativeId::new(native).ok()?))
}

fn message_id(native: &str) -> Option<MessageId> {
    Some(MessageId::new(origin(), NativeId::new(native).ok()?))
}

fn digest(v: &Json) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    v.to_string().hash(&mut h);
    h.finish()
}

pub struct Rollout {
    pub thread: Option<String>,
}

/// A tool call seen in the current turn.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Call {
    tool: String,
    /// The handle a `write_stdin` or `wait` call continues.
    continues: Option<String>,
    answered: bool,
    /// How the harness recorded the command ending, when its
    /// `item_completed` arrived before the output.
    completed: Option<Completed>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Carry {
    session: Option<Session>,
    /// The thread a subagent's copied history came from, and the envelope
    /// ordinal where the subagent's own records begin.
    copied_from: Option<String>,
    copied_until: Option<u64>,
    /// The last copied item, where the child branched.
    branch_point: Option<MessageId>,
    model: Option<String>,
    /// The latest assistant item, open until the token count after it.
    open: Option<MessageRow>,
    calls: BTreeMap<String, Call>,
    /// Commands left running, by handle (`process 123`, `cell 4`), to the
    /// call that started them.
    running: BTreeMap<String, String>,
    links: BTreeSet<String>,
    last: Option<MessageId>,
    /// Digests of the last stored turn settings and rate limits; a repeat
    /// is not stored again.
    settings: Option<u64>,
    limits: Option<u64>,
    /// Reasoning effort the current turn asked for.
    effort: Option<String>,
    /// The service tier the thread runs at (`default`, `flex`, `priority`),
    /// from its settings; prices follow it.
    #[serde(default)]
    tier: Option<String>,
    /// The byte range of the record being read, for events.
    #[serde(skip)]
    current: Option<(u64, u64)>,
    /// The previous record, when it was a stored user prompt: an item or a
    /// `user_message` event. Codex can write one prompt both ways in
    /// adjacent records; the second of the pair is the same prompt.
    #[serde(default)]
    prompt: Option<Prompt>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Prompt {
    text: String,
    from_event: bool,
}

/// One record, unwrapped from its envelope.
struct Rec<'a> {
    /// `session_meta`, `response_item`, `event_msg`, `turn_context`, ...
    ty: &'a str,
    payload: &'a Json,
    at: Option<polylogue_model::Timestamp>,
    ordinal: Option<u64>,
}

const BARE_ITEMS: &[&str] = &[
    "message",
    "reasoning",
    "function_call",
    "function_call_output",
    "custom_tool_call",
    "custom_tool_call_output",
    "local_shell_call",
    "web_search_call",
];

fn unwrap(v: &Json) -> Rec<'_> {
    let at = v.get("timestamp").and_then(time::rfc3339);
    let ordinal = v.get("ordinal").and_then(Json::as_u64);
    if let Some(payload) = v.get("payload").filter(|p| p.is_object()) {
        return Rec {
            ty: str_field(v, "type").unwrap_or("?"),
            payload,
            // Some writers put the time on the item, not the envelope.
            at: at.or_else(|| payload.get("timestamp").and_then(time::rfc3339)),
            ordinal,
        };
    }
    // Mid-2025 rollouts: a header, then bare items and state markers.
    let ty = match str_field(v, "type") {
        Some(t) if BARE_ITEMS.contains(&t) => "response_item",
        Some(_) => "?",
        None if v.get("record_type").is_some() => "state",
        None if v.get("id").is_some() => "legacy_header",
        None => "?",
    };
    Rec {
        ty,
        payload: v,
        at,
        ordinal,
    }
}

impl Rollout {
    fn own(&self, carry: &mut Carry) -> Option<SessionId> {
        if carry.session.is_none() {
            let id = session_id(self.thread.as_deref()?)?;
            carry.session = Some(Session::new(id, SessionKind::Main));
        }
        carry.session.as_ref().map(|s| s.id.clone())
    }

    fn session_meta(&self, carry: &mut Carry, p: &Json, out: &mut Output) -> Disposition {
        let Some(thread) = str_field(p, "id").or(self.thread.as_deref()) else {
            return Disposition::refused("session_meta without a thread ID");
        };
        let Some(id) = session_id(thread) else {
            return Disposition::refused("thread ID is empty");
        };
        if carry.session.as_ref().is_some_and(|s| s.id != id) {
            // A resumed rollout names the thread it came from; this file's
            // thread was already set by an earlier header.
            return self.event(carry, p, "session_meta", None, out);
        }
        let subagent = str_field(p, "thread_source") == Some("subagent")
            || p.get("source").and_then(|s| s.get("subagent")).is_some();
        let kind = if subagent {
            SessionKind::Subagent
        } else {
            SessionKind::Main
        };
        let s = carry
            .session
            .get_or_insert_with(|| Session::new(id.clone(), kind));
        if let Some(at) = p.get("timestamp").and_then(time::rfc3339) {
            s.offer_start(at);
        }
        if let Some(cwd) = str_field(p, "cwd") {
            s.offer_cwd(cwd);
        }
        if let Some(v) = str_field(p, "cli_version") {
            s.tool_version = Some(v.to_owned());
        }
        if let Some(branch) = p.get("git").and_then(|g| str_field(g, "branch")) {
            s.git_branch = Some(branch.to_owned());
        }
        if let Some(url) = p.get("git").and_then(|g| str_field(g, "repository_url")) {
            s.repo = Some(url.to_owned());
        }
        if let Some(name) = str_field(p, "agent_nickname") {
            s.offer_title(name, TitleSource::AgentName);
        }
        let parent = str_field(p, "parent_thread_id").or_else(|| {
            p.get("source")
                .and_then(|s| s.get("subagent"))
                .and_then(|s| s.get("thread_spawn"))
                .and_then(|t| str_field(t, "parent_thread_id"))
        });
        let forked = str_field(p, "forked_from_id");
        if let Some(parent_id) = parent.and_then(session_id) {
            link(
                carry,
                out,
                LinkKind::Spawn,
                parent_id,
                id.clone(),
                None,
                None,
                "parent_thread_id",
            );
        }
        if let Some(fork_id) = forked.filter(|f| Some(*f) != parent).and_then(session_id) {
            link(
                carry,
                out,
                LinkKind::Fork,
                fork_id,
                id.clone(),
                None,
                None,
                "forked_from_id",
            );
        }
        if let Some(start) = p
            .get("subagent_history_start_ordinal")
            .and_then(Json::as_u64)
        {
            carry.copied_until = Some(start);
            carry.copied_from = forked.or(parent).map(str::to_owned);
        }
        Disposition::Materialized
    }

    fn legacy_header(&self, carry: &mut Carry, v: &Json) -> Disposition {
        let Some(id) = str_field(v, "id")
            .or(self.thread.as_deref())
            .and_then(session_id)
        else {
            return Disposition::refused("header without a thread ID");
        };
        let s = carry
            .session
            .get_or_insert_with(|| Session::new(id, SessionKind::Main));
        if let Some(at) = v.get("timestamp").and_then(time::rfc3339) {
            s.offer_start(at);
        }
        if let Some(branch) = v.get("git").and_then(|g| str_field(g, "branch")) {
            s.git_branch = Some(branch.to_owned());
        }
        Disposition::Materialized
    }

    fn event(
        &self,
        carry: &mut Carry,
        data: &Json,
        kind: &str,
        at: Option<polylogue_model::Timestamp>,
        out: &mut Output,
    ) -> Disposition {
        let Some(session) = self.own(carry) else {
            return Disposition::refused("no thread ID in the file name or its header");
        };
        out.events.push(Event {
            session,
            kind: kind.to_owned(),
            at,
            after: carry.last.clone(),
            range: carry.current,
            data: data.clone(),
        });
        Disposition::Materialized
    }

    /// Whether a record belongs to history a subagent copied from its
    /// parent.
    fn copied(carry: &Carry, ordinal: Option<u64>) -> bool {
        matches!((carry.copied_until, ordinal), (Some(until), Some(o)) if o > 1 && o < until)
    }

    /// Sends the open assistant item and keeps nothing open.
    fn close(carry: &mut Carry, out: &mut Output) {
        if let Some(row) = carry.open.take() {
            out.messages.push(row);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn response_item(
        &self,
        carry: &mut Carry,
        record: Record<'_>,
        item: &Json,
        at: Option<polylogue_model::Timestamp>,
        ordinal: Option<u64>,
        out: &mut Output,
        acc: &mut Accounting,
    ) -> Disposition {
        let ty = str_field(item, "type").unwrap_or("?");
        for path in unaccounted(item, fields::for_item(ty)) {
            acc.unread(&format!("response_item/{ty}"), &path);
        }
        Self::close(carry, out);
        let copied = Self::copied(carry, ordinal);
        let session = if copied {
            // Stored once, in the parent; only items with an ID can be
            // recognized there.
            let (Some(parent), Some(_)) = (carry.copied_from.clone(), str_field(item, "id")) else {
                return Disposition::unknown("copied from the parent thread, which stores it");
            };
            match session_id(&parent) {
                Some(s) => s,
                None => return Disposition::refused("parent thread ID is empty"),
            }
        } else {
            match self.own(carry) {
                Some(s) => s,
                None => return Disposition::refused("no thread ID in the file name or its header"),
            }
        };
        let native = str_field(item, "id")
            .map(str::to_owned)
            .unwrap_or_else(|| format!("{}@{}", session.native(), record.offset));
        let Some(id) = message_id(&native) else {
            return Disposition::refused("item ID is empty");
        };
        let Some((role, prompt_kind, blocks)) = self.item_blocks(carry, &session, ty, item, out)
        else {
            return Disposition::unknown("item with no content");
        };
        let message = Message {
            id: id.clone(),
            session,
            parent: None,
            role,
            at,
            model: (role == Role::Assistant)
                .then(|| carry.model.clone())
                .flatten(),
            prompt_kind,
            blocks,
            usage: Usage::default(),
            stop_reason: None,
            reported_cost_usd: None,
            duration_ms: None,
            production: Production {
                effort: (role == Role::Assistant && !copied)
                    .then(|| carry.effort.clone())
                    .flatten(),
                ..Production::default()
            },
        };
        let row = MessageRow {
            offset: record.offset,
            end: record.offset + record.bytes.len() as u64,
            message,
        };
        if copied {
            carry.branch_point = Some(id);
            out.messages.push(row);
            return Disposition::Materialized;
        }
        carry.last = Some(id);
        if role == Role::Assistant {
            carry.open = Some(row);
        } else {
            out.messages.push(row);
        }
        Disposition::Materialized
    }

    /// An item's role, prompt kind and blocks; `None` for an item with
    /// nothing in it.
    fn item_blocks(
        &self,
        carry: &mut Carry,
        session: &SessionId,
        ty: &str,
        item: &Json,
        out: &mut Output,
    ) -> Option<(Role, Option<PromptKind>, Vec<Block>)> {
        let call_id = || {
            str_field(item, "call_id")
                .or(str_field(item, "id"))
                .map(str::to_owned)
        };
        Some(match ty {
            "message" => {
                let content = item.get("content").unwrap_or(&Json::Null);
                let blocks = items::message_blocks(content, &mut out.blobs);
                match str_field(item, "role") {
                    Some("user") => (Role::User, Some(items::prompt_kind(content)), blocks),
                    Some("assistant") => (Role::Assistant, None, blocks),
                    _ => (Role::System, None, blocks),
                }
            }
            "agent_message" => {
                let content = item.get("content").unwrap_or(&Json::Null);
                let mut blocks = items::message_blocks(content, &mut out.blobs);
                blocks.insert(
                    0,
                    Block::Other {
                        type_name: "agent_message".into(),
                        data: serde_json::json!({
                            "author": item.get("author"),
                            "recipient": item.get("recipient"),
                        }),
                    },
                );
                (Role::User, Some(PromptKind::Harness), blocks)
            }
            "reasoning" => (Role::Assistant, None, vec![items::reasoning_block(item)]),
            "function_call" | "custom_tool_call" | "local_shell_call" | "tool_search_call" => {
                let tool = items::tool_name(item)
                    .unwrap_or_else(|| ty.trim_end_matches("_call").to_owned());
                let input = items::call_input(item);
                let continues = match tool.as_str() {
                    "write_stdin" => input
                        .get("session_id")
                        .map(|s| format!("process {}", unquote(s))),
                    "wait" => input.get("cell_id").map(|s| format!("cell {}", unquote(s))),
                    _ => None,
                };
                let id = call_id();
                if let Some(id) = &id {
                    carry.calls.insert(
                        id.clone(),
                        Call {
                            tool: tool.clone(),
                            continues,
                            ..Call::default()
                        },
                    );
                }
                let block = Block::ToolCall {
                    call_id: id,
                    action: items::action(&tool),
                    command: items::call_command(&tool, &input),
                    path: items::call_path(&tool, &input),
                    tool: Some(tool),
                    input,
                };
                (Role::Assistant, None, vec![block])
            }
            "function_call_output" | "custom_tool_call_output" | "tool_search_output" => {
                let id = call_id();
                let raw = item
                    .get("output")
                    .or_else(|| item.get("tools"))
                    .unwrap_or(&Json::Null);
                let (text, attachments) = items::output_parts(raw, &mut out.blobs);
                let (outcome, duration) = self.answer(carry, session, id.as_deref(), &text, out);
                let block = Block::ToolResult {
                    call_id: id,
                    outcome,
                    output: Some(text),
                    data: duration.map(|ms| serde_json::json!({ "duration_ms": ms })),
                    attachments,
                };
                (Role::User, Some(PromptKind::Harness), vec![block])
            }
            "web_search_call" => {
                // One item holds both the search and that it ran.
                let id = call_id();
                let outcome = match str_field(item, "status") {
                    Some("completed") => ToolOutcome::Ok,
                    Some("failed") => ToolOutcome::Error { reason: None },
                    _ => ToolOutcome::Unknown {
                        reason: polylogue_model::UnknownReason::StatusFieldUnreadable,
                    },
                };
                let blocks = vec![
                    Block::ToolCall {
                        call_id: id.clone(),
                        tool: Some("web_search".into()),
                        action: ToolAction::Web,
                        input: item.get("action").cloned().unwrap_or(Json::Null),
                        command: None,
                        path: None,
                    },
                    Block::ToolResult {
                        call_id: id,
                        outcome,
                        output: None,
                        data: None,
                        attachments: Vec::new(),
                    },
                ];
                (Role::Assistant, None, blocks)
            }
            other => (
                Role::Assistant,
                None,
                vec![Block::Other {
                    type_name: other.to_owned(),
                    data: item.clone(),
                }],
            ),
        })
    }

    /// The outcome of a call's output, and its duration when the harness
    /// recorded one. Updates the call a `write_stdin` or `wait` continues,
    /// and links an agent a `spawn_agent` call started.
    fn answer(
        &self,
        carry: &mut Carry,
        session: &SessionId,
        call_id: Option<&str>,
        text: &str,
        out: &mut Output,
    ) -> (ToolOutcome, Option<u64>) {
        let call = call_id
            .and_then(|id| carry.calls.get(id))
            .cloned()
            .unwrap_or_default();
        let tool = if call.tool.is_empty() {
            "?"
        } else {
            call.tool.as_str()
        };
        let (mut outcome, running) = items::output_outcome(tool, text);
        let mut duration = None;
        if let Some(done) = &call.completed {
            outcome = done.outcome.clone();
            duration = done.duration_ms;
        }
        if let (Some(handle), Some(id)) = (running, call_id) {
            carry.running.insert(handle.key(), id.to_owned());
        }
        // A later read of a running command reports how it ended.
        if let Some(handle) = &call.continues
            && !matches!(
                outcome,
                ToolOutcome::Unknown {
                    reason: polylogue_model::UnknownReason::RunningInBackground
                }
            )
            && let Some(original) = carry.running.remove(handle)
        {
            out.background.push(BackgroundUpdate {
                session: session.clone(),
                task_id: handle.clone(),
                call_id: Some(original),
                outcome: outcome.clone(),
                output_file: None,
            });
        }
        if tool.ends_with("spawn_agent")
            && let Ok(Json::Object(reply)) = serde_json::from_str::<Json>(text.trim())
            && let Some(child) = reply
                .get("agent_id")
                .and_then(Json::as_str)
                .and_then(session_id)
        {
            link(
                carry,
                out,
                LinkKind::Spawn,
                session.clone(),
                child,
                None,
                call_id.map(str::to_owned),
                "spawn_agent",
            );
        }
        if let Some(id) = call_id
            && let Some(c) = carry.calls.get_mut(id)
        {
            c.answered = true;
        }
        (outcome, duration)
    }

    /// A thread item the harness recorded when it finished.
    fn item_completed(
        &self,
        carry: &mut Carry,
        record: Record<'_>,
        p: &Json,
        at: Option<polylogue_model::Timestamp>,
        out: &mut Output,
    ) -> Disposition {
        let item = p.get("item").unwrap_or(&Json::Null);
        let ty = str_field(item, "type").unwrap_or("?");
        match ty {
            "AgentMessage" | "UserMessage" | "Reasoning" => {
                Disposition::unknown("mirrors the response item, which is stored")
            }
            "ContextCompaction" => {
                Disposition::unknown("mirrors the compacted record, which is stored")
            }
            "CommandExecution" | "FileChange" | "McpToolCall" | "WebSearch" | "ImageView" => {
                let Some(session) = self.own(carry) else {
                    return Disposition::refused("no thread ID in the file name or its header");
                };
                let id = str_field(item, "id").unwrap_or("");
                if let Some(call) = carry.calls.get_mut(id) {
                    let Some(done) = items::completed(item) else {
                        return Disposition::unknown("call item without a status");
                    };
                    if call.answered {
                        out.background.push(BackgroundUpdate {
                            session,
                            task_id: id.to_owned(),
                            call_id: Some(id.to_owned()),
                            outcome: done.outcome,
                            output_file: None,
                        });
                    } else {
                        call.completed = Some(done);
                    }
                    return Disposition::Materialized;
                }
                // A command a code-mode script ran: no call item of its own.
                let Some(blocks) = items::recovered_call(item, &mut out.blobs) else {
                    return Disposition::unknown("tool item without an ID");
                };
                let Some(mid) = message_id(id) else {
                    return Disposition::refused("item ID is empty");
                };
                Self::close(carry, out);
                let duration = items::completed(item).and_then(|c| c.duration_ms);
                out.messages.push(MessageRow {
                    offset: record.offset,
                    end: record.offset + record.bytes.len() as u64,
                    message: Message {
                        id: mid.clone(),
                        session,
                        parent: None,
                        role: Role::Assistant,
                        at,
                        model: carry.model.clone(),
                        prompt_kind: None,
                        blocks,
                        usage: Usage::default(),
                        stop_reason: None,
                        reported_cost_usd: None,
                        duration_ms: duration,
                        production: Production::default(),
                    },
                });
                carry.last = Some(mid);
                Disposition::Materialized
            }
            "SubAgentActivity" => {
                if let (Some(parent), Some(child)) = (
                    self.own(carry),
                    str_field(item, "agent_thread_id").and_then(session_id),
                ) {
                    let call = str_field(item, "id").map(str::to_owned);
                    link(
                        carry,
                        out,
                        LinkKind::Spawn,
                        parent,
                        child,
                        None,
                        call,
                        "SubAgentActivity",
                    );
                }
                self.event(carry, p, "subagent_activity", at, out)
            }
            other => {
                let kind = format!("item_{}", snake(other));
                self.event(carry, p, &kind, at, out)
            }
        }
    }

    fn usage(carry: &mut Carry, usage: &Json) {
        let n = |k: &str| usage.get(k).and_then(Json::as_u64);
        let (Some(input), cached) = (n("input_tokens"), n("cached_input_tokens")) else {
            return;
        };
        let output = n("output_tokens");
        let reasoning = n("reasoning_output_tokens");
        // OpenAI counts cached input inside input and reasoning inside
        // output; each kind of token is counted once.
        let u = Usage {
            input: Some(input.saturating_sub(cached.unwrap_or(0))),
            cached_input: cached,
            cache_write: n("cache_write_input_tokens").filter(|&w| w > 0),
            output: output.map(|o| o.saturating_sub(reasoning.unwrap_or(0))),
            reasoning,
            cache_write_1h: None,
            service_tier: carry.tier.clone(),
            speed: None,
        };
        if let Some(open) = &mut carry.open {
            open.message.usage = u;
        }
    }

    fn event_msg(
        &self,
        carry: &mut Carry,
        record: Record<'_>,
        p: &Json,
        at: Option<polylogue_model::Timestamp>,
        out: &mut Output,
    ) -> Disposition {
        let sub = str_field(p, "type").unwrap_or("?");
        match sub {
            "token_count" => {
                if let Some(last) = p.get("info").and_then(|i| i.get("last_token_usage")) {
                    Self::usage(carry, last);
                }
                // The subscription window state the API returned with the
                // response this count follows.
                if let (Some(open), Some(limits)) = (
                    &mut carry.open,
                    p.get("rate_limits").filter(|r| !r.is_null()),
                ) {
                    open.message.production.quota = Some(limits.clone());
                }
                Self::close(carry, out);
                match p.get("rate_limits").filter(|r| !r.is_null()) {
                    Some(limits) if carry.limits != Some(digest(limits)) => {
                        carry.limits = Some(digest(limits));
                        self.event(carry, p, "rate_limits", at, out)
                    }
                    _ => Disposition::Materialized,
                }
            }
            "item_completed" => self.item_completed(carry, record, p, at, out),
            "agent_message"
            | "agent_reasoning"
            | "agent_reasoning_raw_content"
            | "agent_reasoning_section_break" => {
                Disposition::unknown("mirrors a response item, which is stored")
            }
            "exec_command_end" | "patch_apply_end" | "mcp_tool_call_end" => {
                // Older rollouts: how a call ended, before item_completed.
                let id = str_field(p, "call_id").unwrap_or("");
                if let (Some(call), Some(done)) = (carry.calls.get_mut(id), items::completed(p)) {
                    if call.answered
                        && let Some(session) = carry.session.as_ref().map(|s| s.id.clone())
                    {
                        out.background.push(BackgroundUpdate {
                            session,
                            task_id: id.to_owned(),
                            call_id: Some(id.to_owned()),
                            outcome: done.outcome,
                            output_file: None,
                        });
                    } else {
                        call.completed = Some(done);
                    }
                }
                self.event(carry, p, sub, at, out)
            }
            "task_complete" | "turn_aborted" => {
                Self::close(carry, out);
                // Calls do not outlive their turn; commands still running do.
                carry.calls.clear();
                self.event(carry, p, sub, at, out)
            }
            "thread_settings" => {
                if let Some(tier) = p
                    .get("thread_settings")
                    .and_then(|s| str_field(s, "service_tier"))
                {
                    carry.tier = Some(tier.to_owned());
                }
                self.event(carry, p, sub, at, out)
            }
            "thread_name_updated" => {
                if let (Some(name), Some(s)) = (str_field(p, "thread_name"), carry.session.as_mut())
                {
                    s.offer_title(name, TitleSource::Generated);
                }
                self.event(carry, p, sub, at, out)
            }
            _ => self.event(carry, p, sub, at, out),
        }
    }
}

/// The text of a user message item, for pairing it with a `user_message`
/// event that records the same prompt.
fn user_text(item: &Json) -> Option<String> {
    if str_field(item, "type") != Some("message") || str_field(item, "role") != Some("user") {
        return None;
    }
    let Json::Array(parts) = item.get("content")? else {
        return None;
    };
    Some(
        parts
            .iter()
            .filter_map(|c| str_field(c, "text"))
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

fn unquote(v: &Json) -> String {
    match v {
        Json::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// `CollabAgentToolCall` as `collab_agent_tool_call`.
fn snake(name: &str) -> String {
    let mut out = String::new();
    for (i, c) in name.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn link(
    carry: &mut Carry,
    out: &mut Output,
    kind: LinkKind,
    parent: SessionId,
    child: SessionId,
    at: Option<MessageId>,
    call_id: Option<String>,
    evidence: &str,
) {
    if parent == child {
        return;
    }
    let key = format!(
        "{kind:?} {parent} {child} {}",
        call_id.as_deref().unwrap_or("")
    );
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

impl Parser for Rollout {
    type Carry = Carry;

    fn kind(&self, v: &Json) -> String {
        let rec = unwrap(v);
        match rec.ty {
            "response_item" | "event_msg" => {
                format!(
                    "{}/{}",
                    rec.ty,
                    str_field(rec.payload, "type").unwrap_or("?")
                )
            }
            ty => ty.to_owned(),
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
        let rec = unwrap(v);
        let p = rec.payload;
        carry.current = Some((record.offset, record.offset + record.bytes.len() as u64));
        if Self::copied(carry, rec.ordinal) && rec.ty != "response_item" {
            return Disposition::unknown("copied from the parent thread, which stores it");
        }
        let prev = carry.prompt.take();
        if rec.ty == "response_item" && str_field(p, "type") == Some("token_count") {
            // Some writers record usage as an item: it belongs to the
            // answer it follows, like an event_msg token count.
            Self::usage(carry, p);
            Self::close(carry, out);
            return Disposition::Materialized;
        }
        let same = |text: &str, from_event: bool| {
            prev.as_ref()
                .is_some_and(|p| p.from_event == from_event && p.text == text)
        };
        if rec.ty == "response_item"
            && let Some(text) = user_text(p)
        {
            if same(&text, true) {
                return Disposition::unknown(
                    "same prompt as the user_message event before it, which is stored",
                );
            }
            let d = self.response_item(carry, record, p, rec.at, rec.ordinal, out, acc);
            if d == Disposition::Materialized {
                carry.prompt = Some(Prompt {
                    text,
                    from_event: false,
                });
            }
            return d;
        }
        if rec.ty == "event_msg" && str_field(p, "type") == Some("user_message") {
            let text = str_field(p, "message").unwrap_or_default().to_owned();
            if same(&text, false) {
                return Disposition::unknown(
                    "same prompt as the user message item before it, which is stored",
                );
            }
            // A prompt only the event recorded: it is the prompt.
            let mut content = vec![serde_json::json!({"type": "input_text", "text": text})];
            if let Some(Json::Array(images)) = p.get("images") {
                content.extend(
                    images
                        .iter()
                        .filter_map(Json::as_str)
                        .map(|url| serde_json::json!({"type": "input_image", "image_url": url})),
                );
            }
            let mut item =
                serde_json::json!({"type": "message", "role": "user", "content": content});
            if let Some(id) = str_field(p, "client_id") {
                item["id"] = Json::String(id.to_owned());
            }
            let d = self.response_item(carry, record, &item, rec.at, rec.ordinal, out, acc);
            if d == Disposition::Materialized {
                carry.prompt = Some(Prompt {
                    text,
                    from_event: true,
                });
            }
            return d;
        }
        match rec.ty {
            "session_meta" => self.session_meta(carry, p, out),
            "legacy_header" => self.legacy_header(carry, p),
            "state" => Disposition::unknown("turn-state marker of 2025 rollouts; holds no content"),
            "response_item" => self.response_item(carry, record, p, rec.at, rec.ordinal, out, acc),
            "event_msg" => self.event_msg(carry, record, p, rec.at, out),
            "turn_context" => {
                if let Some(model) = str_field(p, "model") {
                    carry.model = Some(model.to_owned());
                }
                carry.effort = str_field(p, "effort").map(str::to_owned);
                if let Some(tier) = str_field(p, "service_tier") {
                    carry.tier = Some(tier.to_owned());
                }
                if let (Some(cwd), Some(s)) = (str_field(p, "cwd"), carry.session.as_mut()) {
                    s.offer_cwd(cwd);
                }
                // Settings repeat every turn; only a change is stored.
                let mut settings = p.clone();
                if let Some(m) = settings.as_object_mut() {
                    m.remove("turn_id");
                    m.remove("root_turn_id");
                }
                let d = digest(&settings);
                if carry.settings == Some(d) {
                    return Disposition::Materialized;
                }
                carry.settings = Some(d);
                self.event(carry, p, "turn_context", rec.at, out)
            }
            "token_usage_record" => {
                if let Some(u) = p.get("usage") {
                    Self::usage(carry, u);
                }
                Disposition::Materialized
            }
            "compacted" => {
                // The history the model continues from replaces the old
                // one; its items are already stored, so the event keeps
                // only their count.
                let mut data = p.clone();
                if let Some(m) = data.as_object_mut()
                    && let Some(Json::Array(history)) = m.remove("replacement_history")
                {
                    m.insert("replacement_history_items".into(), history.len().into());
                    // A fork starts from its parent's history: the last
                    // item is where it branched.
                    if carry.last.is_none()
                        && let Some(at) = history.iter().rev().find_map(|i| str_field(i, "id"))
                    {
                        carry.branch_point = message_id(at);
                    }
                }
                self.event(carry, &data, "compaction", rec.at, out)
            }
            "world_state" | "inter_agent_communication_metadata" => {
                self.event(carry, p, rec.ty, rec.at, out)
            }
            _ => Disposition::unknown("unrecognized record type"),
        }
    }

    fn flush(&self, carry: &mut Carry, out: &mut Output) {
        if let Some(open) = &carry.open {
            out.messages.push(open.clone());
        }
        // An item re-sent within one step (the open item, then with its
        // usage) keeps its last form. Rows go out in file order.
        let mut seen = BTreeSet::new();
        let mut kept: Vec<MessageRow> = Vec::with_capacity(out.messages.len());
        for row in out.messages.drain(..).rev() {
            if seen.insert((row.message.id.clone(), row.offset)) {
                kept.push(row);
            }
        }
        kept.sort_by_key(|row| row.offset);
        out.messages = kept;
        // The branch point is known once the copied history has been read.
        if let (Some(at), Some(s)) = (&carry.branch_point, &carry.session) {
            for l in &mut out.links {
                if l.child == s.id && l.at.is_none() && l.kind != LinkKind::Copy {
                    l.at = Some(at.clone());
                }
            }
        }
        out.sessions = carry.session.iter().cloned().collect();
    }
}

#[cfg(test)]
mod tests;
