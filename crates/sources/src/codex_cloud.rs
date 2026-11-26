//! Codex Cloud tasks from ChatGPT's codex.json. Task and turn IDs are the
//! provider's task_e_ namespace, separate from local Codex rollouts.
use crate::RecordOutcome;
use crate::document::{Sink, message_id, session_id};
use crate::export::{ExportParser, ExportState};
use crate::fields::{ExpectedType as T, Fields, KnownKinds, Rule};
use crate::parse::{MessageRow, Parse};
use polylogue_model::JsonExt;
use polylogue_model::{
    Block, Event, Json, Message, PromptKind, Role, Session, SessionKind, TitleSource, Usage,
};
use std::path::Path;

pub const VERSION: u32 = 1;

parser_modules! {}
pub const ORIGIN: &str = "codex-cloud";

pub fn recognize(path: &Path, head: &[u8]) -> bool {
    path.file_name().is_some_and(|name| name == "codex.json")
        && matches!(crate::document::first_byte(head), Some(b'[' | b'{'))
}
pub fn open(_: &Path) -> Option<Parse> {
    Some(Parse::Export(Box::new(Cloud)))
}
pub struct Cloud;

static FIELDS: Fields = Fields::new(
    "task",
    &[
        ("id", Rule::Read(T::String)),
        ("title", Rule::Read(T::String)),
        ("archived", Rule::Kept("task archive state")),
        ("turns", Rule::Read(T::Array)),
        ("turns[]", Rule::Read(T::Object)),
        ("turns[].id", Rule::Read(T::String)),
        ("turns[].role", Rule::Read(T::String)),
        ("turns[].previous_turn_id", Rule::Read(T::String)),
        (
            "turns[].custom_instructions",
            Rule::ReadWhole("instructions supplied to the cloud run"),
        ),
        (
            "turns[].branch",
            Rule::ReadWhole("Codex Cloud run metadata"),
        ),
        (
            "turns[].branch_name",
            Rule::ReadWhole("Codex Cloud run metadata"),
        ),
        (
            "turns[].external_pull_request_id",
            Rule::ReadWhole("Codex Cloud pull request reference"),
        ),
        (
            "turns[].pull_request_status",
            Rule::ReadWhole("Codex Cloud run metadata"),
        ),
        (
            "turns[].turn_status",
            Rule::ReadWhole("Codex Cloud run metadata"),
        ),
        ("turns[].input_items", Rule::Read(T::Array)),
        ("turns[].input_items[]", Rule::Read(T::Object)),
        ("turns[].input_items[].type", Rule::Read(T::String)),
        ("turns[].input_items[].role", Rule::Read(T::String)),
        ("turns[].input_items[].content", Rule::Read(T::Array)),
        ("turns[].input_items[].content[]", Rule::Read(T::Object)),
        (
            "turns[].input_items[].content[].content_type",
            Rule::Read(T::String),
        ),
        (
            "turns[].input_items[].content[].text",
            Rule::Read(T::String),
        ),
        ("turns[].output_items", Rule::Read(T::Array)),
        ("turns[].output_items[]", Rule::Read(T::Object)),
        ("turns[].output_items[].type", Rule::Read(T::String)),
        ("turns[].output_items[].role", Rule::Read(T::String)),
        ("turns[].output_items[].content", Rule::Read(T::Array)),
        ("turns[].output_items[].content[]", Rule::Read(T::Object)),
        (
            "turns[].output_items[].content[].content_type",
            Rule::Read(T::String),
        ),
        (
            "turns[].output_items[].content[].text",
            Rule::Read(T::String),
        ),
        (
            "turns[].output_items[].content[].path",
            Rule::Read(T::String),
        ),
        (
            "turns[].output_items[].content[].line_range_start",
            Rule::Read(T::Number),
        ),
        (
            "turns[].output_items[].content[].line_range_end",
            Rule::Read(T::Number),
        ),
    ],
)
.shape(
    &[],
    &[
        KnownKinds {
            path: "turns[].role",
            values: &["user", "assistant"],
        },
        KnownKinds {
            path: "turns[].output_items[].content[].content_type",
            values: &["text", "repo_file_citation"],
        },
        KnownKinds {
            path: "turns[].input_items[].content[].content_type",
            values: &["text"],
        },
    ],
);

pub fn task_shape(task: &Json) -> bool {
    task.get("mapping").is_none()
        && task
            .str("id")
            .filter(|s| !s.is_empty())
            .is_some_and(|id| id.starts_with("task_e_"))
        && task.array("turns").is_some_and(|turns| {
            !turns.is_empty()
                && turns.iter().all(|turn| {
                    turn.str("id").filter(|s| !s.is_empty()).is_some()
                        && matches!(
                            turn.str("role").filter(|s| !s.is_empty()),
                            Some("user" | "assistant")
                        )
                })
        })
}

impl ExportParser for Cloud {
    fn scan(
        &mut self,
        _input: &str,
        reader: &mut dyn crate::export::Input,
        items: &mut dyn crate::export::ItemSink,
    ) -> Result<crate::export::ScanStats, crate::export::ScanError> {
        crate::export::scan_array_or_object(reader, |range, raw| {
            items.push(range, Vec::new(), None, raw.get().as_bytes())
        })
    }

    fn item(
        &mut self,
        bytes: &[u8],
        sink: &mut Sink<'_>,
        _: &mut dyn ExportState,
    ) -> Result<(), String> {
        let task: Json = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
        if !task_shape(&task) {
            sink.records
                .record("task", &RecordOutcome::unreadable("not a Codex Cloud task"));
            return Ok(());
        }
        sink.records.fields("task", &task, &FIELDS);
        let native = task
            .str("id")
            .filter(|s| !s.is_empty())
            .expect("validated task ID");
        let Some(sid) = session_id(ORIGIN, native) else {
            sink.records
                .record("task", &RecordOutcome::unreadable("unusable task ID"));
            return Ok(());
        };
        let mut session = Session::new(sid.clone(), SessionKind::Main);
        if let Some(title) = task.str("title").filter(|s| !s.is_empty()) {
            session.offer_title(title, TitleSource::Generated);
        }
        for turn in task["turns"].as_array().expect("validated turns") {
            let role = if turn["role"] == "user" {
                Role::User
            } else {
                Role::Assistant
            };
            let id = message_id(
                ORIGIN,
                turn.str("id")
                    .filter(|s| !s.is_empty())
                    .expect("validated turn ID"),
            )
            .ok_or("unusable turn ID")?;
            let parent = turn
                .str("previous_turn_id")
                .filter(|s| !s.is_empty())
                .and_then(|id| message_id(ORIGIN, id));
            let mut blocks = Vec::new();
            let items = turn
                .get(if role == Role::User {
                    "input_items"
                } else {
                    "output_items"
                })
                .and_then(Json::as_array);
            for item in items.into_iter().flatten() {
                for part in item.array("content").into_iter().flatten() {
                    match part.str("content_type").filter(|s| !s.is_empty()) {
                        Some("text") => {
                            if let Some(text) = part.str("text") {
                                blocks.push(Block::Text { text: text.into() });
                            }
                        }
                        Some(kind) => blocks.push(Block::Other {
                            type_name: kind.into(),
                            data: part.clone(),
                        }),
                        None => blocks.push(Block::Other {
                            type_name: "cloud_content".into(),
                            data: part.clone(),
                        }),
                    }
                }
            }
            if let Some(instructions) = turn.str("custom_instructions").filter(|s| !s.is_empty()) {
                session.instructions = Some(instructions.into());
            }
            if let Some(branch) = turn
                .str("branch")
                .filter(|s| !s.is_empty())
                .or_else(|| turn.str("branch_name").filter(|s| !s.is_empty()))
            {
                session.git_branch = Some(branch.into());
            }
            let mut attrs = turn.as_object().cloned().unwrap_or_default();
            for field in [
                "id",
                "role",
                "previous_turn_id",
                "input_items",
                "output_items",
            ] {
                attrs.remove(field);
            }
            if let Some(Json::String(value)) = attrs.get("external_pull_request_id")
                && matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "none" | "null" | ""
                )
            {
                attrs.remove("external_pull_request_id");
            }
            if !attrs.is_empty() {
                sink.out.events.push({
                    let (at, time_accuracy) = crate::time::parts(None);
                    Event {
                        session: Some(sid.clone()),
                        kind: "codex_cloud_turn".into(),
                        at,
                        at_quality: time_accuracy.quality,
                        at_precision: time_accuracy.precision,
                        at_until: time_accuracy.until,
                        after: Some(id.clone()),
                        range: None,
                        data: Json::Object(attrs.clone()),
                    }
                });
            }
            let offset = sink.position();
            sink.out.messages.push(MessageRow {
                response: None,
                page_text: false,
                id_kind: polylogue_model::IdKind::Native,
                offset,
                end: offset + 1,
                message: {
                    let (at, time_accuracy) = crate::time::parts(None);
                    Message {
                        id,
                        session: sid.clone(),
                        parent,
                        role,
                        at,
                        at_quality: time_accuracy.quality,
                        at_precision: time_accuracy.precision,
                        at_until: time_accuracy.until,
                        model: None,
                        prompt_kind: (role == Role::User).then_some(PromptKind::Prose),
                        blocks,
                        usage: Usage::default(),
                        stop_reason: None,
                        reported_cost_usd: None,
                        duration_ms: None,
                        production: Default::default(),
                        attrs,
                    }
                },
            });
        }
        let mut metadata = task.as_object().cloned().unwrap_or_default();
        for field in ["id", "title", "turns"] {
            metadata.remove(field);
        }
        if !metadata.is_empty() {
            sink.out.events.push({
                let (at, time_accuracy) = crate::time::parts(None);
                Event {
                    session: Some(sid.clone()),
                    kind: "codex_cloud_task".into(),
                    at,
                    at_quality: time_accuracy.quality,
                    at_precision: time_accuracy.precision,
                    at_until: time_accuracy.until,
                    after: None,
                    range: None,
                    data: Json::Object(metadata),
                }
            });
        }
        sink.out.sessions.push(session);
        sink.records.record("task", &RecordOutcome::Used);
        Ok(())
    }
}
#[cfg(test)]
mod tests;
