//! Codex Cloud tasks from ChatGPT's codex.json. Task and turn IDs are the
//! provider's task_e_ namespace, separate from local Codex rollouts.
use crate::RecordOutcome;
use crate::document::{Sink, message_id, session_id, str_of};
use crate::export::{ExportParser, ExportState};
use crate::fields::{ExpectedType as T, Fields, KnownKinds, Rule};
use crate::parse::{MessageRow, Parse};
use polylogue_model::{
    Block, Event, Json, Message, PromptKind, Role, Session, SessionKind, TitleSource, Usage,
};
use std::path::Path;

pub const VERSION: u32 = 1;
pub const ORIGIN: &str = "codex-cloud";

pub fn recognize(path: &Path, head: &[u8]) -> bool {
    path.file_name().is_some_and(|name| name == "codex.json")
        && matches!(crate::chatgpt::first_byte(head), Some(b'[' | b'{'))
}
pub fn open(_: &Path) -> Option<Parse> {
    Some(Parse::Export(Box::new(Cloud)))
}
pub struct Cloud;

static FIELDS: Fields = Fields::new(&[
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
        Rule::Kept("instructions supplied to the cloud run"),
    ),
    ("turns[].branch", Rule::Kept("native cloud run metadata")),
    (
        "turns[].branch_name",
        Rule::Kept("native cloud run metadata"),
    ),
    (
        "turns[].external_pull_request_id",
        Rule::Kept("native cloud pull request reference"),
    ),
    (
        "turns[].pull_request_status",
        Rule::Kept("native cloud run metadata"),
    ),
    (
        "turns[].turn_status",
        Rule::Kept("native cloud run metadata"),
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
])
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
        && str_of(task, "id").is_some_and(|id| id.starts_with("task_e_"))
        && task
            .get("turns")
            .and_then(Json::as_array)
            .is_some_and(|turns| {
                !turns.is_empty()
                    && turns.iter().all(|turn| {
                        str_of(turn, "id").is_some()
                            && matches!(str_of(turn, "role"), Some("user" | "assistant"))
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
        let native = str_of(&task, "id").expect("validated task ID");
        let Some(sid) = session_id(ORIGIN, native) else {
            sink.records
                .record("task", &RecordOutcome::unreadable("unusable task ID"));
            return Ok(());
        };
        let mut session = Session::new(sid.clone(), SessionKind::Main);
        if let Some(title) = str_of(&task, "title") {
            session.offer_title(title, TitleSource::Generated);
        }
        for turn in task["turns"].as_array().expect("validated turns") {
            let role = if turn["role"] == "user" {
                Role::User
            } else {
                Role::Assistant
            };
            let id = message_id(ORIGIN, str_of(turn, "id").expect("validated turn ID"))
                .ok_or("unusable turn ID")?;
            let parent = str_of(turn, "previous_turn_id").and_then(|id| message_id(ORIGIN, id));
            let mut blocks = Vec::new();
            let items = turn
                .get(if role == Role::User {
                    "input_items"
                } else {
                    "output_items"
                })
                .and_then(Json::as_array);
            for item in items.into_iter().flatten() {
                for part in item
                    .get("content")
                    .and_then(Json::as_array)
                    .into_iter()
                    .flatten()
                {
                    match str_of(part, "content_type") {
                        Some("text") => {
                            if let Some(text) = part.get("text").and_then(Json::as_str) {
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
            if let Some(branch) = clean(turn, "branch").or_else(|| clean(turn, "branch_name")) {
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
                sink.out.events.push(Event {
                    session: Some(sid.clone()),
                    kind: "codex_cloud_turn".into(),
                    at: None,
                    after: Some(id.clone()),
                    range: None,
                    data: Json::Object(attrs.clone()),
                });
            }
            let offset = sink.position();
            sink.out.messages.push(MessageRow {
                offset,
                end: offset + 1,
                message: Message {
                    id,
                    session: sid.clone(),
                    parent,
                    role,
                    at: None,
                    model: None,
                    prompt_kind: (role == Role::User).then_some(PromptKind::Prose),
                    blocks,
                    usage: Usage::default(),
                    stop_reason: None,
                    reported_cost_usd: None,
                    duration_ms: None,
                    production: Default::default(),
                    attrs,
                },
            });
        }
        sink.out.events.push(Event {
            session: Some(sid.clone()),
            kind: "codex_cloud_task".into(),
            at: None,
            after: None,
            range: None,
            data: serde_json::json!({"archived":task.get("archived")}),
        });
        sink.out.sessions.push(session);
        sink.records.record("task", &RecordOutcome::Used);
        Ok(())
    }
}
fn clean<'a>(value: &'a Json, key: &str) -> Option<&'a str> {
    str_of(value, key)
        .map(str::trim)
        .filter(|value| !matches!(value.to_ascii_lowercase().as_str(), "none" | "null" | ""))
}

#[cfg(test)]
mod tests;
