//! Which normalized fields a Codex item actually supplied, with their raw
//! values and origins. A block's `Group` has one `SlotValue` per slot; each
//! owner (item, status, continuation) replaces only its own entries.
use super::origins::Origins;
use super::selection::{FactSelection, PathPart, Transform};
use super::{
    has_nonempty_text, has_recorded_output, is_shell_tool, native_duration, raw, text_parts,
};
use polylogue_model::RawJson as Value;
use polylogue_model::merge::{BlockField as B, MessageField as M, Slot, UsageField as U};
use polylogue_model::{BlockSlot, Timestamp};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Owner {
    Primary,
    Status,
    Continuation,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotValue {
    pub slot: Slot,
    pub canonical: Vec<u8>,
    pub source_time: Option<Timestamp>,
    pub owner: Owner,
    #[serde(default)]
    pub origins: Origins,
    #[serde(default)]
    pub selection: Option<FactSelection>,
}
impl SlotValue {
    fn valid_selection(&self) -> bool {
        let Some(selection) = &self.selection else {
            return true;
        };
        if selection.validate().is_err() {
            return false;
        }
        let expected = match &self.slot {
            Slot::Message(M::At | M::CompletedAt) => Transform::ProviderTimeV1,
            Slot::Message(M::Model) => Transform::JsonStringV1,
            Slot::Block {
                field: B::Text | B::ThinkingText,
                ..
            } => Transform::TextJoinLfV1,
            Slot::Block {
                field: B::ThinkingData,
                ..
            } => Transform::JsonStringV1,
            Slot::Extra { .. } => Transform::JsonValueV1,
            Slot::Usage(U::Input) if selection.transform == Transform::UncachedInputV1 => {
                Transform::UncachedInputV1
            }
            Slot::Usage(_) => Transform::JsonValueV1,
            Slot::Block { field, .. } if matches!(selection.transform,Transform::ToolFieldV1{field:f} if f==*field) => {
                selection.transform
            }
            _ => return false,
        };
        if expected != selection.transform {
            return false;
        }
        let Some(base) = self.origins.selected() else {
            return false;
        };
        if selection.transform == Transform::UncachedInputV1 {
            let mut cached = base.clone();
            match cached.path.last_mut() {
                Some(PathPart::Key(key)) if key == "input_tokens" => {
                    *key = "cached_input_tokens".into();
                }
                _ => return false,
            }
            return selection.inputs == [base, cached];
        }
        selection.inputs.iter().all(|input| {
            input.record == base.record
                && if matches!(selection.transform, Transform::ToolFieldV1 { .. }) {
                    base.path.starts_with(&input.path)
                } else if selection.transform == Transform::TextJoinLfV1 {
                    input.path.starts_with(&base.path)
                } else {
                    input.path == base.path
                }
        })
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Group {
    pub nonempty: bool,
    pub nonempty_text: bool,
    pub fields: Vec<SlotValue>,
}
impl Group {
    pub(super) fn valid(&self) -> bool {
        let mut slots = std::collections::BTreeSet::new();
        (!self.nonempty_text || self.nonempty)
            && self.fields.iter().all(|w| {
                matches!(
                    w.slot,
                    Slot::Block { .. } | Slot::Extra { block: Some(_), .. }
                ) && w.valid_selection()
                    && w.source_time.is_none()
                    && slots.insert(&w.slot)
            })
    }
    pub fn replace(&mut self, owner: Owner, other: Group) {
        if owner == Owner::Primary && !other.nonempty {
            self.fields.clear();
            self.nonempty = false;
            self.nonempty_text = false;
            return;
        }
        self.fields.retain(|w| w.owner != owner);
        for witness in other.fields {
            self.fields.retain(|old| old.slot != witness.slot);
            self.fields.push(witness);
        }
        if owner == Owner::Primary {
            self.nonempty = other.nonempty;
            self.nonempty_text = other.nonempty_text;
        }
    }
}
fn value(raw: &[u8]) -> Option<Value> {
    serde_json::from_slice(raw).ok()
}
fn slot_value(slot: Slot, raw: &[u8], owner: Owner) -> Result<SlotValue, String> {
    Ok(SlotValue {
        slot,
        canonical: polylogue_model::identity::canonical_record(raw, &[])
            .map_err(|e| e.to_string())?,
        source_time: None,
        owner,
        origins: Origins::Unknown,
        selection: None,
    })
}
pub(super) fn message(field: M, raw: &[u8], time: Option<Timestamp>) -> Result<SlotValue, String> {
    let mut w = slot_value(Slot::Message(field), raw, Owner::Primary)?;
    w.source_time = time;
    Ok(w)
}
pub(super) fn usage(field: U, raw: &[u8]) -> Result<SlotValue, String> {
    slot_value(Slot::Usage(field), raw, Owner::Primary)
}
pub(super) fn extra(key: &str, raw: &[u8]) -> Result<SlotValue, String> {
    slot_value(
        Slot::Extra {
            block: None,
            key: key.into(),
        },
        raw,
        Owner::Primary,
    )
}
/// The block kind the item maps to.
#[derive(Clone, Copy)]
pub(super) enum Kind {
    Text,
    Thinking,
    Call,
    Result,
    Other,
}
pub(super) fn block(
    record: &[u8],
    path: &[&str],
    slot: &BlockSlot,
    kind: Kind,
    owner: Owner,
    origins: &Origins,
) -> Result<Group, String> {
    let itemraw = raw::field(record, path)?;
    let item: Value = serde_json::from_slice(itemraw).map_err(|e| e.to_string())?;
    let mut group = Group {
        nonempty: true,
        ..Group::default()
    };
    let mut add = |field: B, fieldraw: &[u8], source_field: &str| -> Result<(), String> {
        let mut w = slot_value(
            Slot::Block {
                block: slot.clone(),
                field,
            },
            fieldraw,
            owner,
        )?;
        w.origins = origins.field(source_field);
        w.selection = match field {
            B::Text | B::ThinkingText => {
                let selected = w.origins.selected();
                selected.and_then(|base| {
                    let inputs: Vec<_> = super::text_parts_with_paths(item.get(source_field))
                        .map(|(path, _)| {
                            let mut selected = base.clone();
                            selected.path.extend(path);
                            selected
                        })
                        .collect();
                    (!inputs.is_empty())
                        .then(|| FactSelection::new(Transform::TextJoinLfV1, inputs))
                })
            }
            B::ThinkingData => w
                .origins
                .selected()
                .map(|raw| FactSelection::new(Transform::JsonStringV1, vec![raw])),
            B::Tool
            | B::CallId
            | B::Input
            | B::Command
            | B::Output
            | B::Outcome
            | B::ExitCode
            | B::DurationMs => origins
                .selected()
                .map(|raw| FactSelection::new(Transform::ToolFieldV1 { field }, vec![raw])),
            _ => None,
        };
        group.fields.retain(|old| old.slot != w.slot);
        group.fields.push(w);
        Ok(())
    };
    let get = |name: &str| raw::field(itemraw, &[name]).ok();
    match kind {
        Kind::Text => {
            let content = item.get("content");
            if !has_nonempty_text(content) {
                return Ok(Group::default());
            }
            group.nonempty_text = true;
            if let Some(raw) = get("content") {
                add(B::Text, raw, "content")?;
            }
        }
        Kind::Thinking => {
            if let Some(raw) = get("summary") {
                if text_parts(item.get("summary")).next().is_some() {
                    add(B::ThinkingText, raw, "summary")?;
                }
                group.nonempty_text = has_nonempty_text(item.get("summary"));
            }
            if let Some(raw) =
                get("encrypted_content").filter(|raw| value(raw).is_some_and(|v| v.is_string()))
            {
                add(B::ThinkingData, raw, "encrypted_content")?;
            }
            // `encrypted` is derived from summary/data, not supplied.
        }
        Kind::Call => {
            for (name, field) in [
                ("name", B::Tool),
                ("call_id", B::CallId),
                ("command", B::Command),
            ] {
                if let Some(raw) = get(name).filter(|raw| value(raw).is_some_and(|v| v.is_string()))
                {
                    add(field, raw, name)?;
                }
            }
            if let Some(raw) = get("arguments").or_else(|| get("input")) {
                add(
                    B::Input,
                    raw,
                    if get("arguments").is_some() {
                        "arguments"
                    } else {
                        "input"
                    },
                )?;
            }
            if item
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(is_shell_tool)
                && let Some(args) = get("arguments").or_else(|| get("input"))
            {
                let parsed = value(args);
                let inner = parsed
                    .as_ref()
                    .and_then(Value::as_str)
                    .map(str::as_bytes)
                    .unwrap_or(args);
                if let Ok(command) =
                    raw::field(inner, &["cmd"]).or_else(|_| raw::field(inner, &["command"]))
                    && value(command).is_some_and(|v| {
                        v.is_string()
                            || v.as_array().is_some_and(|a| a.iter().all(Value::is_string))
                    })
                {
                    add(
                        B::Command,
                        command,
                        if get("arguments").is_some() {
                            "arguments"
                        } else {
                            "input"
                        },
                    )?;
                }
            }
            // Tool, call ID and input are derived from the item; only `command` is
            // supplied.
        }
        Kind::Result => {
            if owner == Owner::Primary
                && let Some(raw) =
                    get("call_id").filter(|raw| value(raw).is_some_and(|v| v.is_string()))
            {
                add(B::CallId, raw, "call_id")?;
            }
            let output = get("aggregated_output").or_else(|| get("output"));
            if let Some(raw) = output.filter(|raw| {
                owner != Owner::Status && value(raw).is_some_and(|v| has_recorded_output(Some(&v)))
            }) {
                add(
                    B::Output,
                    raw,
                    if get("aggregated_output").is_some() {
                        "aggregated_output"
                    } else {
                        "output"
                    },
                )?;
            }
            if let Some(raw) = get("status").filter(|raw| {
                value(raw).is_some_and(|v| {
                    matches!(v.as_str(), Some("completed" | "failed" | "declined"))
                })
            }) {
                add(B::Outcome, raw, "status")?;
            }
            if let Some(raw) =
                get("exit_code").filter(|raw| value(raw).is_some_and(|v| v.as_i64().is_some()))
            {
                add(B::ExitCode, raw, "exit_code")?;
            }
            if native_duration(&item).is_some()
                && let Some(raw) = get("duration_ms")
                    .or_else(|| get("duration"))
                    .or_else(|| get("output"))
            {
                add(
                    B::DurationMs,
                    raw,
                    if get("duration_ms").is_some() {
                        "duration_ms"
                    } else if get("duration").is_some() {
                        "duration"
                    } else {
                        "output"
                    },
                )?;
            }
            // An exit code supplies ExitCode; the Ok/Error derived from it is not
            // a supplied Outcome.
            if let Some(output) = output {
                for (field, raw) in output_status(output)? {
                    add(
                        field,
                        &raw,
                        if get("aggregated_output").is_some() {
                            "aggregated_output"
                        } else {
                            "output"
                        },
                    )?;
                }
            }
        }
        Kind::Other => {
            // An unmapped payload is kept, but its fields are not recognized.
            add(B::OtherData, itemraw, "")?;
            if let Some(w) = group.fields.last_mut() {
                w.origins = origins.clone();
            }
            return Ok(group);
        }
    }
    let native_result = matches!(kind, Kind::Result)
        && matches!(
            item.get("type").and_then(Value::as_str),
            Some("CommandExecution" | "FileChange" | "McpToolCall")
        );
    if owner == Owner::Primary && !native_result {
        let fields: std::collections::BTreeMap<String, &serde_json::value::RawValue> =
            serde_json::from_slice(itemraw).map_err(|e| e.to_string())?;
        for (key, raw) in fields {
            if matches!(
                key.as_str(),
                "id" | "type"
                    | "role"
                    | "timestamp"
                    | "ordinal"
                    | "name"
                    | "call_id"
                    | "arguments"
                    | "input"
                    | "content"
                    | "summary"
                    | "encrypted_content"
                    | "command"
                    | "status"
                    | "exit_code"
                    | "duration"
                    | "duration_ms"
                    | "aggregated_output"
                    | "output"
                    | "phase"
            ) {
                continue;
            }
            let mut w = slot_value(
                Slot::Extra {
                    block: Some(slot.clone()),
                    key,
                },
                raw.get().as_bytes(),
                owner,
            )?;
            if let Slot::Extra { key, .. } = &w.slot {
                w.origins = origins.field(key);
                w.selection = w
                    .origins
                    .selected()
                    .map(|raw| FactSelection::new(Transform::JsonValueV1, vec![raw]));
            }
            group.fields.push(w);
        }
    }
    Ok(group)
}

/// The slot a message-level presence key stands for.
pub fn message_slot(key: &str) -> Option<Slot> {
    Some(match key {
        "at" => Slot::Message(M::At),
        "completed_at" => Slot::Message(M::CompletedAt),
        "model" => Slot::Message(M::Model),
        "usage:input_tokens" => Slot::Usage(U::Input),
        "usage:output_tokens" => Slot::Usage(U::Output),
        "usage:cached_input_tokens" => Slot::Usage(U::CacheRead),
        "usage:cache_write_input_tokens" => Slot::Usage(U::CacheWrite),
        "usage:reasoning_output_tokens" => Slot::Usage(U::Reasoning),
        key if key.starts_with("extra:") => Slot::Extra {
            block: None,
            key: key[6..].into(),
        },
        _ => return None,
    })
}
pub(super) fn matches_message_key(key: &str, w: &SlotValue) -> bool {
    w.valid_selection()
        && message_slot(key).as_ref() == Some(&w.slot)
        && (matches!(w.slot, Slot::Message(M::At | M::CompletedAt)) == w.source_time.is_some())
}

fn output_status(output: &[u8]) -> Result<Vec<(B, Vec<u8>)>, String> {
    let mut code = None;
    let mut is_error = None;
    let mut candidate = |raw: &[u8]| {
        if code.is_none()
            && let Ok(raw) = raw::field(raw, &["exit_code"])
                .or_else(|_| raw::field(raw, &["metadata", "exit_code"]))
            && value(raw).is_some_and(|v| v.as_i64().is_some())
        {
            code = Some(raw.to_vec());
        }
        if is_error.is_none()
            && let Ok(raw) = raw::field(raw, &["is_error"])
                .or_else(|_| raw::field(raw, &["metadata", "is_error"]))
            && value(raw).is_some_and(|v| v.as_bool().is_some())
        {
            is_error = Some(raw.to_vec());
        }
    };
    match output.first() {
        Some(b'"') => {
            if let Ok(inner) = serde_json::from_slice::<String>(output) {
                candidate(inner.as_bytes());
            }
        }
        Some(b'{') => candidate(output),
        Some(b'[') => {
            let items: Vec<&serde_json::value::RawValue> =
                serde_json::from_slice(output).map_err(|e| e.to_string())?;
            for item in items {
                if let Ok(text) = raw::field(item.get().as_bytes(), &["text"])
                    && let Ok(inner) = serde_json::from_slice::<String>(text)
                {
                    candidate(inner.as_bytes());
                }
            }
        }
        _ => {}
    }
    Ok(if let Some(code) = code {
        vec![(B::ExitCode, code)]
    } else if let Some(value) = is_error {
        vec![(B::Outcome, value)]
    } else {
        vec![]
    })
}
