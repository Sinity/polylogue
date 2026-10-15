//! Source-supplied witnesses, captured beside the single mapping core. Values
//! remain canonical raw fields, not normalized DTO values. A group has one entry
//! per semantic slot; replacement removes only that source owner's old fields.
use super::{is_shell_tool, native_duration, raw, recorded_output, texts};
use polylogue_model::merge::{BlockField as B, MessageField as M, Slot, UsageField as U};
use polylogue_model::{BlockSlot, Timestamp};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Owner {
    Primary,
    Status,
    Continuation,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Witness {
    pub slot: Slot,
    pub canonical: Vec<u8>,
    pub source_time: Option<Timestamp>,
    pub owner: Owner,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Group {
    pub nonempty: bool,
    pub nonempty_text: bool,
    pub fields: Vec<Witness>,
}
impl Group {
    pub(super) fn valid(&self) -> bool {
        let mut slots = std::collections::BTreeSet::new();
        (!self.nonempty_text || self.nonempty)
            && self.fields.iter().all(|w| {
                matches!(
                    w.slot,
                    Slot::Block { .. } | Slot::Extra { block: Some(_), .. }
                ) && w.source_time.is_none()
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
fn witness(slot: Slot, raw: &[u8], owner: Owner) -> Result<Witness, String> {
    Ok(Witness {
        slot,
        canonical: polylogue_model::identity::canonical_record(raw, &[])
            .map_err(|e| e.to_string())?,
        source_time: None,
        owner,
    })
}
pub(super) fn message(field: M, raw: &[u8], time: Option<Timestamp>) -> Result<Witness, String> {
    let mut w = witness(Slot::Message(field), raw, Owner::Primary)?;
    w.source_time = time;
    Ok(w)
}
pub(super) fn usage(field: U, raw: &[u8]) -> Result<Witness, String> {
    witness(Slot::Usage(field), raw, Owner::Primary)
}
pub(super) fn extra(key: &str, raw: &[u8]) -> Result<Witness, String> {
    witness(
        Slot::Extra {
            block: None,
            key: key.into(),
        },
        raw,
        Owner::Primary,
    )
}
/// kind is the mapper's target branch, not evidence that a field was supplied.
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
) -> Result<Group, String> {
    let itemraw = raw::field(record, path)?;
    let item: Value = serde_json::from_slice(itemraw).map_err(|e| e.to_string())?;
    let mut group = Group {
        nonempty: true,
        ..Group::default()
    };
    let mut add = |field: B, fieldraw: &[u8]| -> Result<(), String> {
        let w = witness(
            Slot::Block {
                block: slot.clone(),
                field,
            },
            fieldraw,
            owner,
        )?;
        group.fields.retain(|old| old.slot != w.slot);
        group.fields.push(w);
        Ok(())
    };
    let get = |name: &str| raw::field(itemraw, &[name]).ok();
    match kind {
        Kind::Text => {
            let content = item.get("content");
            let text = texts(content).join("\n");
            if text.is_empty() {
                return Ok(Group::default());
            }
            group.nonempty_text = true;
            if let Some(raw) = get("content") {
                add(B::Text, raw)?;
            }
        }
        Kind::Thinking => {
            if let Some(raw) = get("summary") {
                if value(raw).is_some_and(|v| v.is_array() || v.is_string()) {
                    add(B::ThinkingText, raw)?;
                }
                group.nonempty_text = !texts(item.get("summary")).join("\n").is_empty();
            }
            if let Some(raw) =
                get("encrypted_content").filter(|raw| value(raw).is_some_and(|v| v.is_string()))
            {
                add(B::ThinkingData, raw)?;
            }
            // encrypted is derived from summary/data, never a supplied boolean.
        }
        Kind::Call => {
            for (name, field) in [
                ("name", B::Tool),
                ("call_id", B::CallId),
                ("command", B::Command),
            ] {
                if let Some(raw) = get(name).filter(|raw| value(raw).is_some_and(|v| v.is_string()))
                {
                    add(field, raw)?;
                }
            }
            if let Some(raw) = get("arguments").or_else(|| get("input")) {
                add(B::Input, raw)?;
            }
            if item
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(is_shell_tool)
            {
                if let Some(args) = get("arguments").or_else(|| get("input")) {
                    let parsed = value(args);
                    let inner = parsed
                        .as_ref()
                        .and_then(Value::as_str)
                        .map(str::as_bytes)
                        .unwrap_or(args);
                    if let Ok(command) =
                        raw::field(inner, &["cmd"]).or_else(|_| raw::field(inner, &["command"]))
                    {
                        if value(command).is_some_and(|v| {
                            v.is_string()
                                || v.as_array().is_some_and(|a| a.iter().all(Value::is_string))
                        }) {
                            add(B::Command, command)?;
                        }
                    }
                }
            }
            // Native item type -> tool, item ID -> call ID and whole item ->
            // input are mapper-generated. Only explicit command is supplied.
        }
        Kind::Result => {
            if owner == Owner::Primary {
                if let Some(raw) =
                    get("call_id").filter(|raw| value(raw).is_some_and(|v| v.is_string()))
                {
                    add(B::CallId, raw)?;
                }
            }
            let output = get("aggregated_output").or_else(|| get("output"));
            if let Some(raw) = output.filter(|raw| {
                owner != Owner::Status
                    && value(raw).is_some_and(|v| recorded_output(Some(&v)).is_some())
            }) {
                add(B::Output, raw)?;
            }
            if let Some(raw) = get("status").filter(|raw| {
                value(raw).is_some_and(|v| {
                    matches!(v.as_str(), Some("completed" | "failed" | "declined"))
                })
            }) {
                add(B::Outcome, raw)?;
            }
            if let Some(raw) =
                get("exit_code").filter(|raw| value(raw).is_some_and(|v| v.as_i64().is_some()))
            {
                add(B::ExitCode, raw)?;
            }
            if native_duration(&item).is_some() {
                if let Some(raw) = get("duration_ms").or_else(|| get("duration")) {
                    add(B::DurationMs, raw)?;
                }
            }
            // Status embedded in an opaque output may directly supply fields.
            // Textual success/error recognition is an inferred outcome: excluded.
            if let Some(output) = output {
                let parsed = value(output);
                let inner = parsed
                    .as_ref()
                    .and_then(Value::as_str)
                    .map(str::as_bytes)
                    .unwrap_or(output);
                for (name, field) in [("is_error", B::Outcome), ("exit_code", B::ExitCode)] {
                    let raw = raw::field(inner, &[name])
                        .or_else(|_| raw::field(inner, &["metadata", name]));
                    if let Ok(raw) = raw {
                        let valid = value(raw).is_some_and(|v| {
                            if name == "is_error" {
                                v.is_boolean()
                            } else {
                                v.as_i64().is_some()
                            }
                        });
                        if valid {
                            add(field, raw)?;
                        }
                    }
                }
            }
        }
        Kind::Other => {
            // An unmapped payload is retained but is not a claim that all its
            // semantic fields were recognized. The owning evidence stays pending.
            add(B::OtherData, itemraw)?;
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
            group.fields.push(witness(
                Slot::Extra {
                    block: Some(slot.clone()),
                    key,
                },
                raw.get().as_bytes(),
                owner,
            )?);
        }
    }
    Ok(group)
}

pub(super) fn matches_message_key(key: &str, w: &Witness) -> bool {
    match (&w.slot, key) {
        (Slot::Message(M::At), "at") | (Slot::Message(M::CompletedAt), "completed_at") => {
            w.source_time.is_some()
        }
        (Slot::Message(M::Model), "model") => w.source_time.is_none(),
        (Slot::Usage(field), name) => {
            w.source_time.is_none()
                && matches!(
                    (field, name),
                    (U::Input, "usage:input_tokens")
                        | (U::Output, "usage:output_tokens")
                        | (U::CacheRead, "usage:cached_input_tokens")
                        | (U::CacheWrite, "usage:cache_write_input_tokens")
                        | (U::Reasoning, "usage:reasoning_output_tokens")
                )
        }
        (
            Slot::Extra {
                block: None,
                key: extra,
            },
            name,
        ) => w.source_time.is_none() && name.strip_prefix("extra:") == Some(extra.as_str()),
        _ => false,
    }
}
