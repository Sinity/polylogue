//! Native Antigravity CLI step transcripts. Provider step numbers are local
//! to a conversation; stored message identities include that conversation.
use crate::document::{message_id, session_id};
use crate::fields::{ExpectedType as T, Fields, KnownKinds, Rule};
use crate::lines::Record;
use crate::parse::{MessageRow, Output, Parse, Parser};
use crate::{RecordOutcome, Records};
use polylogue_model::{
    Block, Event, Json, JsonExt, Message, MessageId, PromptKind, Role, Session, SessionId,
    SessionKind, ToolAction, ToolOutcome, UnknownReason,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::path::Path;

pub const VERSION: u32 = 3;

parser_modules! {}
pub const LOCATION: &str =
    ".gemini/antigravity-cli/brain/*/.system_generated/logs/transcript_full.jsonl";
const TYPES: &[&str] = &[
    "USER_INPUT",
    "PLANNER_RESPONSE",
    "RUN_COMMAND",
    "VIEW_FILE",
    "GREP_SEARCH",
    "CODE_ACTION",
    "LIST_DIRECTORY",
    "GENERIC",
    "CONVERSATION_HISTORY",
    "CHECKPOINT",
    "ERROR_MESSAGE",
    "SYSTEM_MESSAGE",
];
static FIELDS: Fields = Fields::new(
    "step",
    &[
        ("step_index", Rule::Read(T::Number)),
        ("source", Rule::Read(T::String)),
        ("type", Rule::Read(T::String)),
        ("status", Rule::Read(T::String)),
        ("created_at", Rule::Read(T::String)),
        ("content", Rule::Read(T::String)),
        ("thinking", Rule::Read(T::String)),
        ("tool_calls", Rule::Read(T::Array)),
        ("tool_calls[].name", Rule::Read(T::String)),
        (
            "tool_calls[].args",
            Rule::ReadWhole("structured tool input"),
        ),
        ("error", Rule::Kept("reported step error")),
        ("error_code", Rule::Kept("reported step error")),
    ],
)
.shape(
    &[],
    &[
        KnownKinds {
            path: "source",
            values: &["MODEL", "USER_EXPLICIT", "SYSTEM"],
        },
        KnownKinds {
            path: "type",
            values: TYPES,
        },
    ],
);

fn session_of(path: &Path) -> Option<SessionId> {
    if path.file_name()? != "transcript_full.jsonl" {
        return None;
    }
    let logs = path.parent()?;
    let generated = logs.parent()?;
    let directory = generated.parent()?;
    if logs.file_name()? != "logs"
        || generated.file_name()? != ".system_generated"
        || directory.parent()?.file_name()? != "brain"
    {
        return None;
    }
    directory
        .file_name()?
        .to_str()
        .and_then(|native| session_id(super::ORIGIN, native))
}
pub fn recognize(path: &Path) -> bool {
    session_of(path).is_some()
}
pub fn open(path: &Path) -> Option<Parse> {
    session_of(path).map(|session| Parse::Lines(Box::new(Transcript { session })))
}

struct Transcript {
    session: SessionId,
}
#[derive(Clone, Default, Serialize, Deserialize)]
struct State {
    last: Option<Last>,
    pending: VecDeque<String>,
    assigned: BTreeMap<u64, Assignment>,
}
#[derive(Clone, Serialize, Deserialize)]
struct Last {
    index: u64,
    id: MessageId,
    parent: Option<MessageId>,
}
#[derive(Clone, Serialize, Deserialize)]
struct Assignment {
    call: String,
    parent: Option<MessageId>,
}

impl Parser for Transcript {
    type ParserState = State;
    fn kind(&self, value: &Json) -> String {
        value.str("type").unwrap_or("?").into()
    }
    fn parse_record(
        &self,
        state: &mut State,
        record: Record<'_>,
        value: &Json,
        out: &mut Output,
        records: &mut Records,
    ) -> RecordOutcome {
        let kind = value.str("type").unwrap_or("?");
        records.fields(kind, value, &FIELDS);
        let source = value.str("source").unwrap_or("?");
        if !TYPES.contains(&kind) || !["MODEL", "USER_EXPLICIT", "SYSTEM"].contains(&source) {
            return RecordOutcome::skipped(format!("step type {kind} from {source} has no reader"));
        }
        let Some(index) = value.u64("step_index") else {
            return RecordOutcome::unreadable("step without a nonnegative index");
        };
        let (at, evidence) =
            crate::time::parts(value.get("created_at").and_then(crate::time::rfc3339));
        if out.sessions.is_empty() {
            out.sessions
                .push(Session::new(self.session.clone(), SessionKind::Main));
        }
        if source == "SYSTEM" {
            let kind = match kind {
                "CHECKPOINT" => "antigravity_checkpoint",
                "CONVERSATION_HISTORY" => "antigravity_history",
                "ERROR_MESSAGE" => "antigravity_error",
                "SYSTEM_MESSAGE" => "antigravity_system",
                _ => return RecordOutcome::skipped("system step type has no reader"),
            };
            out.events.push(Event {
                session: Some(self.session.clone()),
                kind: kind.into(),
                at,
                at_quality: evidence.quality,
                at_precision: evidence.precision,
                at_until: evidence.until,
                after: None,
                range: Some((record.offset, record.offset + record.bytes.len() as u64)),
                data: value.clone(),
            });
            return RecordOutcome::Used;
        }
        let id = message_id(
            super::ORIGIN,
            &format!("{}:step-{index}", self.session.native().as_str()),
        )
        .expect("session-qualified step ID");
        let mut parent = state.last.as_ref().and_then(|last| {
            if last.index == index {
                last.parent.clone()
            } else {
                Some(last.id.clone())
            }
        });
        let (role, prompt_kind, blocks) = match (source, kind) {
            ("USER_EXPLICIT", "USER_INPUT") => (
                Role::User,
                Some(PromptKind::Prose),
                value
                    .str("content")
                    .map(|text| Block::Text { text: text.into() })
                    .into_iter()
                    .collect(),
            ),
            ("MODEL", "PLANNER_RESPONSE") => {
                let mut blocks = Vec::new();
                let mut pending = VecDeque::new();
                if let Some(text) = value.str("thinking") {
                    blocks.push(Block::Thinking {
                        text: Some(text.into()),
                        signature: None,
                        redacted: false,
                    });
                }
                if let Some(text) = value.str("content") {
                    blocks.push(Block::Text { text: text.into() });
                }
                for (position, call) in value.array("tool_calls").into_iter().flatten().enumerate()
                {
                    let Some(tool) = call.str("name") else {
                        return RecordOutcome::unreadable("planner tool call without a name");
                    };
                    let call_id = format!("step-{index}-{position}");
                    let input = call.get("args").cloned().unwrap_or(Json::Null);
                    pending.push_back(call_id.clone());
                    blocks.push(Block::ToolCall {
                        call_id: Some(call_id),
                        tool: Some(tool.into()),
                        action: action(tool),
                        command: input.str("CommandLine").map(str::to_owned),
                        path: ["AbsolutePath", "DirectoryPath", "SearchPath"]
                            .into_iter()
                            .find_map(|field| input.str(field).map(str::to_owned)),
                        written_text: None,
                        input,
                        run_by_script: false,
                    });
                }
                state.pending = pending;
                state.assigned.clear();
                (Role::Assistant, None, blocks)
            }
            ("MODEL", _) => {
                let call_id = if let Some(assigned) = state.assigned.get(&index) {
                    parent.clone_from(&assigned.parent);
                    Some(assigned.call.clone())
                } else if let Some(call) = state.pending.pop_front() {
                    state.assigned.insert(
                        index,
                        Assignment {
                            call: call.clone(),
                            parent: parent.clone(),
                        },
                    );
                    Some(call)
                } else {
                    None
                };
                let outcome = if value.get("error").is_some_and(|error| !error.is_null()) {
                    ToolOutcome::Error { reason: None }
                } else {
                    match value.str("status") {
                        Some("DONE") => ToolOutcome::Ok,
                        Some("RUNNING") => ToolOutcome::Unknown {
                            reason: UnknownReason::RunningInBackground,
                        },
                        _ => ToolOutcome::Unknown {
                            reason: UnknownReason::StatusFieldUnreadable,
                        },
                    }
                };
                (
                    Role::User,
                    Some(PromptKind::Harness),
                    vec![Block::ToolResult {
                        reported_duration_ms: None,
                        call_id,
                        outcome,
                        output: value.str("content").map(str::to_owned),
                        data: None,
                        attachments: Vec::new(),
                    }],
                )
            }
            _ => {
                return RecordOutcome::skipped(
                    "source and step type do not name a supported message",
                );
            }
        };
        if state.last.as_ref().is_none_or(|last| last.index < index) {
            state.last = Some(Last {
                index,
                id: id.clone(),
                parent: parent.clone(),
            });
        }
        out.messages.push(MessageRow {
            response: None,
            page_text: false,
            id_kind: polylogue_model::IdKind::Native,
            offset: record.offset,
            end: record.offset + record.bytes.len() as u64,
            message: Message {
                id,
                session: self.session.clone(),
                parent,
                role,
                at,
                at_quality: evidence.quality,
                at_precision: evidence.precision,
                at_until: evidence.until,
                model: None,
                prompt_kind,
                blocks,
                usage: Default::default(),
                stop_reason: None,
                reported_cost_usd: None,
                duration_ms: None,
                production: Default::default(),
                attrs: crate::fields::attributes(value, &FIELDS),
            },
        });
        RecordOutcome::Used
    }
    fn flush(&self, _: &mut State, _: &mut Output) {}
}

fn action(tool: &str) -> ToolAction {
    match tool.to_ascii_lowercase().as_str() {
        "run_command" => ToolAction::Shell,
        "view_file" => ToolAction::FileRead,
        "grep_search" | "list_dir" | "list_directory" => ToolAction::Search,
        "write_to_file" => ToolAction::FileWrite,
        "code_action" | "replace_file_content" | "multi_replace_file_content" => {
            ToolAction::FileEdit
        }
        _ => ToolAction::Other,
    }
}

#[cfg(test)]
mod tests;
