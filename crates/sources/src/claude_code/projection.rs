//! Provider-owned, source-local output units. Interpretation records facts here
//! before fill/enrichment; projection never scores an enriched Message. The owner
//! retains (contribution, local block) identity and assigns final BlockIds later.
use super::keyed::{Contribution, EvidenceUnit, ResponseCell};
use polylogue_model::RetainedJson as Value;
use polylogue_model::merge::{BlockField, FinishState, MessageField, UsageField};
use polylogue_model::{BlockContent, Message};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Field {
    /// Applies only to contribution zero, never filled from a later record.
    FirstMessage(MessageField),
    SelectedMessage(MessageField),
    /// Fixed provider extra whose one response-level winner is source-owned.
    SelectedExtra {
        key: String,
    },
    Usage(UsageField),
    /// Named structured result member; the container itself has no rank slot.
    ResultData {
        local: u32,
        key: String,
    },
    Block {
        local: u32,
        field: BlockField,
    },
    /// Exact path components distinguish a top-level "message.x" key from x
    /// inside message. Each immediate extra is opaque, including empty keys.
    Extra {
        local: Option<u32>,
        path: Vec<String>,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fact {
    pub field: Field,
    pub source_path: Vec<String>,
    /// File interpretation order of the last supplied fact. Not arrival time,
    /// not part of version evidence, and never a cross-input tie breaker.
    pub supplied_revision: u64,
    pub selection: Option<super::selection::LeafSelection>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CountingObligation {
    MissingText { local: u32 },
    UnmappedField { path: Vec<String> },
    FirstAuthorUpdate,
    ParentResolution,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordFacts {
    pub facts: Vec<Fact>,
    /// Derived auxiliary leaves retained with origins, absent from rank/fill.
    pub retained: Vec<Fact>,
    pub block_counts: Vec<(bool, bool)>,
    pub obligations: Vec<CountingObligation>,
    /// Positive proof on this effective record only. Removing/replacing the
    /// record removes its witness; the owner combines selected group witnesses.
    pub finished_witness: bool,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordCursor {
    pub phase: u8,
    pub position: u32,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Unit {
    Retained {
        item: u64,
        fact: Fact,
        value: Value,
    },
    /// Stored/exported auxiliary value; deliberately absent from ranking facts.
    Signature {
        item: u64,
        local: u32,
        value: String,
        selection: Option<super::selection::LeafSelection>,
    },
    RecordHeader {
        item: u64,
        uuid: String,
        blocks: u32,
        counting_complete: bool,
        finished_witness: bool,
    },
    Block {
        item: u64,
        local: u32,
        content: BlockContent,
        nonempty: bool,
        nonempty_text: bool,
    },
    Fact {
        item: u64,
        fact: Fact,
        value: Value,
    },
    Counting {
        item: u64,
        obligation: CountingObligation,
    },
    Evidence(EvidenceUnit),
    RequiredBodyUnavailable {
        item: u64,
        content_item: u32,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeaderUnit {
    /// Header only: this has no response blocks and its version remains pending.
    pub message: Message,
    pub revision: u64,
    pub ordinal: u64,
    pub author_selection: Option<super::selection::AuthorSelection>,
    pub contributions: u64,
    pub finish: FinishState,
    /// Existing active-path/global operation pairing remain separate owner work.
    pub active_path_unresolved: bool,
    pub pairing_unresolved: bool,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectedUnit {
    pub field: u8,
    pub from_item: Option<u64>,
    pub raw: Option<Vec<u8>>,
    pub facts: Vec<Fact>,
}

impl ResponseCell {
    pub fn output_header(&self) -> Result<HeaderUnit, String> {
        if !self.message.blocks.is_empty() {
            return Err("Claude response header unexpectedly owns historical blocks".into());
        }
        let reason = self.selected[1]
            .as_ref()
            .map(|raw| serde_json::from_slice::<Value>(raw).map_err(|e| e.to_string()))
            .transpose()?;
        let finish = match reason {
            Some(Value::String(reason)) if !reason.text().is_empty() => FinishState::Finished,
            _ => FinishState::Unknown,
        };
        Ok(HeaderUnit {
            message: self.message.clone(),
            revision: self.revision,
            ordinal: self.ordinal,
            author_selection: self.author_selection.clone(),
            contributions: self.items,
            finish,
            active_path_unresolved: self.message.on_active_path.is_none(),
            pairing_unresolved: true,
        })
    }
    /// Four fixed point units: usage, stop_reason, model, completed timestamp.
    /// An owner reduces each semantic slot once, not once per source record.
    pub fn selected_unit(&self, index: u8) -> Option<SelectedUnit> {
        let i = usize::from(index);
        if i >= 6 {
            return None;
        }
        let mut facts = self.selected_facts[i].clone();
        if (i == 3 && self.selected_from[i] == Some(0))
            || (i == 2 && self.message.written_by != polylogue_model::WrittenBy::Model)
        {
            facts.clear();
        } // same raw time already supplies `at`
        Some(SelectedUnit {
            field: index,
            from_item: self.selected_from[i],
            raw: self.selected[i].clone(),
            facts,
        })
    }
}
impl Contribution {
    /// One source-owned, non-ranking auxiliary at an exact contribution-local block.
    pub fn signature_unit(&self, item: u64, local: u32) -> Result<Option<Unit>, String> {
        let Some(BlockContent::Thinking(thinking)) = self.blocks.get(local as usize) else {
            return Ok(None);
        };
        let Some(value) = &thinking.signature else {
            return Ok(None);
        };
        Ok(Some(Unit::Signature {
            item,
            local,
            value: value.clone(),
            selection: super::selection::signature(self, local),
        }))
    }
    /// One loaded/admitted source cell, one output unit per call. The caller
    /// persists this cursor and uses the existing ContributionCursor between
    /// groups. This never walks other records or assigns prefix block offsets.
    pub fn next_output(
        &self,
        item: u64,
        cursor: &mut RecordCursor,
    ) -> Result<Option<Unit>, String> {
        loop {
            let n = usize::try_from(cursor.position)
                .map_err(|_| "Claude projection cursor overflow")?;
            let unit = match cursor.phase {
                0 => Some(Unit::RecordHeader {
                    item,
                    uuid: self.uuid.clone(),
                    blocks: u32::try_from(self.blocks.len())
                        .map_err(|_| "Claude block count overflow")?,
                    counting_complete: self.projection.obligations.is_empty(),
                    finished_witness: self.projection.finished_witness,
                }),
                1 => {
                    if let Some(content) = self.blocks.get(n) {
                        let (nonempty, nonempty_text) = self
                            .projection
                            .block_counts
                            .get(n)
                            .copied()
                            .ok_or("Claude block projection facts are missing")?;
                        Some(Unit::Block {
                            item,
                            local: cursor.position,
                            content: content.clone(),
                            nonempty,
                            nonempty_text,
                        })
                    } else {
                        None
                    }
                }
                2 => {
                    if let Some(fact) = self.projection.facts.get(n) {
                        if item != 0 && matches!(fact.field, Field::FirstMessage(_)) {
                            cursor.position = cursor
                                .position
                                .checked_add(1)
                                .ok_or("Claude projection cursor overflow")?;
                            continue;
                        }
                        let raw = if matches!(
                            fact.selection.as_ref().map(|s| &s.transform),
                            Some(super::selection::Transform::BackgroundV1 { .. })
                        ) {
                            &self
                                .completion
                                .as_ref()
                                .ok_or("background fact lost its notification")?
                                .raw
                        } else {
                            &self.raw
                        };
                        let source: Value =
                            serde_json::from_slice(raw).map_err(|e| e.to_string())?;
                        let value = at(&source, &fact.source_path)
                            .ok_or("Claude presence fact lost its raw value")?
                            .clone();
                        Some(Unit::Fact {
                            item,
                            fact: fact.clone(),
                            value,
                        })
                    } else {
                        None
                    }
                }
                3 => self
                    .projection
                    .obligations
                    .get(n)
                    .map(|obligation| Unit::Counting {
                        item,
                        obligation: obligation.clone(),
                    }),
                4 => Some(Unit::Evidence(self.record_unit(item))),
                5 => match self.bodies.get(n) {
                    Some(body) if body.payload.is_none() => Some(Unit::RequiredBodyUnavailable {
                        item,
                        content_item: body.content_item,
                    }),
                    Some(_) => self.body_unit(item, n)?.map(Unit::Evidence),
                    None => None,
                },
                6 => {
                    if let Some(fact) = self.projection.retained.get(n) {
                        let raw = if matches!(
                            fact.selection.as_ref().map(|s| &s.transform),
                            Some(super::selection::Transform::BackgroundV1 { .. })
                        ) {
                            &self
                                .completion
                                .as_ref()
                                .ok_or("background fact lost its notification")?
                                .raw
                        } else {
                            &self.raw
                        };
                        let source: Value =
                            serde_json::from_slice(raw).map_err(|e| e.to_string())?;
                        let value = at(&source, &fact.source_path)
                            .ok_or("Claude retained leaf lost its raw value")?
                            .clone();
                        Some(Unit::Retained {
                            item,
                            fact: fact.clone(),
                            value,
                        })
                    } else {
                        cursor.phase = 8;
                        cursor.position = 0;
                        continue;
                    }
                }
                7 => Some(
                    self.signature_unit(item, cursor.position)?
                        .ok_or("signature cursor lost its thinking signature")?,
                ),
                8 => self.completion_unit(item).map(Unit::Evidence),
                _ => return Ok(None),
            };
            if cursor.phase == 1
                && matches!(&unit,Some(Unit::Block {content:BlockContent::Thinking(thinking),..}) if thinking.signature.is_some())
            {
                cursor.phase = 7;
            } else if cursor.phase == 7 {
                cursor.phase = 1;
                cursor.position = cursor
                    .position
                    .checked_add(1)
                    .ok_or("Claude signature cursor overflow")?;
            } else if matches!(cursor.phase, 0 | 4 | 8) {
                cursor.phase += 1;
                cursor.position = 0;
            } else if unit.is_some() {
                cursor.position = cursor
                    .position
                    .checked_add(1)
                    .ok_or("Claude projection cursor overflow")?;
            } else {
                cursor.phase += 1;
                cursor.position = 0;
            }
            if unit.is_some() {
                return Ok(unit);
            }
        }
    }
}
fn path(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|s| (*s).into()).collect()
}
fn at<'a>(v: &'a Value, parts: &[String]) -> Option<&'a Value> {
    let mut value = v;
    for part in parts {
        value = if let Some(a) = value.as_array() {
            a.get(part.parse::<usize>().ok()?)?
        } else {
            value.get(part)?
        };
    }
    Some(value)
}
fn add(facts: &mut Vec<Fact>, field: Field, source_path: Vec<String>, revision: u64) {
    facts.push(Fact {
        field,
        source_path,
        supplied_revision: revision,
        selection: None,
    })
}
fn extra(facts: &mut Vec<Fact>, local: Option<u32>, source_path: Vec<String>, revision: u64) {
    add(
        facts,
        Field::Extra {
            local,
            path: source_path.clone(),
        },
        source_path,
        revision,
    )
}

pub(super) fn selected(index: usize, raw: &Value, revision: u64) -> Vec<Fact> {
    let mut facts = Vec::new();
    match index {
        0 => {
            if let Some(object) = raw.as_object() {
                for (native_name, value) in object {
                    let name = native_name.as_str();
                    let field = match name {
                        "input_tokens" => Some(UsageField::Input),
                        "output_tokens" => Some(UsageField::Output),
                        "cache_read_input_tokens" => Some(UsageField::CacheRead),
                        "cache_creation_input_tokens" => Some(UsageField::CacheWrite),
                        "service_tier" => Some(UsageField::ServiceTier),
                        _ => None,
                    };
                    if let Some(field) = field {
                        if (field == UsageField::ServiceTier && value.is_string())
                            || (field != UsageField::ServiceTier && value.as_u64().is_some())
                        {
                            add(
                                &mut facts,
                                Field::Usage(field),
                                path(&["message", "usage", name]),
                                revision,
                            )
                        }
                    } else if matches!(name, "cache_creation" | "output_tokens_details") {
                        if let Some(members) = value.as_object() {
                            for (native_key, value) in members {
                                let key = native_key.as_str();
                                let field = match (name, key) {
                                    ("cache_creation", "ephemeral_5m_input_tokens") => {
                                        Some(UsageField::CacheWrite5m)
                                    }
                                    ("cache_creation", "ephemeral_1h_input_tokens") => {
                                        Some(UsageField::CacheWrite1h)
                                    }
                                    ("output_tokens_details", "thinking_tokens") => {
                                        Some(UsageField::Reasoning)
                                    }
                                    _ => None,
                                };
                                let p = path(&["message", "usage", name, key]);
                                if let Some(field) = field {
                                    if value.as_u64().is_some() {
                                        add(&mut facts, Field::Usage(field), p, revision)
                                    }
                                } else {
                                    extra(&mut facts, None, p, revision)
                                }
                            }
                        }
                    } else {
                        extra(
                            &mut facts,
                            None,
                            path(&["message", "usage", name]),
                            revision,
                        )
                    }
                }
            }
        }
        1 if raw.is_string() => add(
            &mut facts,
            Field::SelectedMessage(MessageField::StopReason),
            path(&["message", "stop_reason"]),
            revision,
        ),
        2 if raw.is_string() => add(
            &mut facts,
            Field::SelectedMessage(MessageField::Model),
            path(&["message", "model"]),
            revision,
        ),
        3 if super::parse_time(Some(raw)).is_some() => add(
            &mut facts,
            Field::SelectedMessage(MessageField::CompletedAt),
            path(&["timestamp"]),
            revision,
        ),
        4 if super::usd_nanos(raw).is_some() => add(
            &mut facts,
            Field::Usage(UsageField::ProviderCostNusd),
            path(&["costUSD"]),
            revision,
        ),
        5 if raw.as_u64().is_some() => add(
            &mut facts,
            Field::SelectedMessage(MessageField::DurationMs),
            path(&["durationMs"]),
            revision,
        ),
        _ => {}
    }
    facts
}

pub(super) fn record(
    v: &Value,
    incoming: &Value,
    blocks: &[BlockContent],
    revision: u64,
    old: Option<&RecordFacts>,
) -> Result<RecordFacts, String> {
    let mut result = RecordFacts {
        finished_witness: v.get("isAbortedMidStream").and_then(Value::as_bool) == Some(true),
        ..RecordFacts::default()
    };
    if super::parse_time(v.get("timestamp")).is_some() {
        add(
            &mut result.facts,
            Field::FirstMessage(MessageField::At),
            path(&["timestamp"]),
            revision,
        )
    }
    if v.get("parentUuid")
        .and_then(Value::native)
        .is_some_and(|s| !s.as_str().is_empty())
    {
        add(
            &mut result.facts,
            Field::FirstMessage(MessageField::Parent),
            path(&["parentUuid"]),
            revision,
        )
    }
    if v.get("origin")
        .and_then(|o| o.get("kind"))
        .and_then(Value::as_str)
        == Some("peer")
        && v.get("origin")
            .and_then(|o| o.get("name"))
            .is_some_and(Value::is_string)
    {
        add(
            &mut result.facts,
            Field::FirstMessage(MessageField::FromName),
            path(&["origin", "name"]),
            revision,
        )
    }
    if let Some(object) = v.as_object() {
        for (native_key, value) in object {
            let key = native_key.as_str();
            match key {
                "type"
                | "uuid"
                | "parentUuid"
                | "sessionId"
                | "requestId"
                | "agentId"
                | "timestamp"
                | "cwd"
                | "gitBranch"
                | "version"
                | "isSidechain"
                | "forkedFrom"
                | "userType"
                | "entrypoint"
                | "promptId"
                | "promptSource"
                | "isMeta"
                | "isVisibleInTranscriptOnly"
                | "message"
                | "isApiErrorMessage"
                | "isAbortedMidStream"
                | "thinkingDurationMs" => {}
                "toolUseResult" | "toolDenialKind" => {} // reduced once on its attributed result below
                "costUSD" if super::usd_nanos(value).is_some() => {}
                "durationMs" if value.as_u64().is_some() => {}
                "origin" => {
                    if let Some(origin) = value.as_object() {
                        for (native_name, value) in origin {
                            let name = native_name.as_str();
                            if name == "kind" {
                                continue;
                            }
                            if name == "name" {
                                if value_of_origin(origin, "kind").and_then(Value::as_str)
                                    != Some("peer")
                                    && value.is_string()
                                {
                                    result.obligations.push(CountingObligation::UnmappedField {
                                        path: path(&["origin", name]),
                                    });
                                }
                            } else {
                                extra(&mut result.facts, None, path(&["origin", name]), revision)
                            }
                        }
                    }
                }
                _ => extra(&mut result.facts, None, path(&[key]), revision),
            }
        }
    }
    if let Some(message) = v.get("message").and_then(Value::as_object) {
        for native_key in message.keys() {
            let key = native_key.as_str();
            if !matches!(
                key,
                "id" | "role" | "type" | "model" | "content" | "usage" | "stop_reason" | "ttftMs"
            ) {
                if key == "stop_sequence" {
                    add(
                        &mut result.facts,
                        Field::SelectedExtra {
                            key: key.to_owned(),
                        },
                        path(&["message", key]),
                        revision,
                    )
                } else {
                    extra(&mut result.facts, None, path(&["message", key]), revision)
                }
            }
        }
    }
    let content = v.get("message").and_then(|m| m.get("content"));
    for (index, block) in blocks.iter().enumerate() {
        let local = u32::try_from(index).map_err(|_| "Claude local block index overflow")?;
        let empty = matches!(block,BlockContent::Text{text} if text.is_empty());
        let nonempty_text = match block {
            BlockContent::Text { text } => !text.is_empty(),
            BlockContent::Thinking(t) => t.text.as_deref().is_some_and(|s| !s.is_empty()),
            _ => false,
        };
        result.block_counts.push((!empty, nonempty_text));
        let base = if content.is_some_and(Value::is_string) {
            path(&["message", "content"])
        } else {
            vec!["message".into(), "content".into(), index.to_string()]
        };
        let raw = content
            .and_then(|c| {
                if c.is_string() {
                    Some(c)
                } else {
                    c.as_array().and_then(|a| a.get(index))
                }
            })
            .unwrap_or(&Value::Null);
        if empty {
            if !raw.is_string() && !raw.get("text").is_some_and(Value::is_string) {
                result
                    .obligations
                    .push(CountingObligation::MissingText { local })
            }
            continue;
        }
        let mut known = vec!["type"];
        let mut leaf = |key: &str, field: BlockField, present: bool| {
            if present {
                let mut p = base.clone();
                if !raw.is_string() && !key.is_empty() {
                    p.push(key.into())
                }
                add(
                    &mut result.facts,
                    Field::Block { local, field },
                    p,
                    revision,
                )
            }
        };
        match block {
            BlockContent::Text { .. } => {
                known.push("text");
                leaf("text", BlockField::Text, true);
            }
            BlockContent::Thinking(t) => {
                known.extend(["thinking", "signature", "encrypted", "data"]);
                leaf("thinking", BlockField::ThinkingText, t.text.is_some());
                leaf("encrypted", BlockField::Encrypted, t.encrypted.is_some());
                leaf("data", BlockField::ThinkingData, t.data.is_some());
            }
            BlockContent::ToolCall(call) => {
                known.extend(["name", "id", "input"]);
                leaf("name", BlockField::Tool, call.tool.is_some());
                leaf("id", BlockField::CallId, call.call_id.is_some());
                leaf("input", BlockField::Input, call.input.is_some());
            }
            BlockContent::ToolResult(output) => {
                known.extend([
                    "tool_use_id",
                    "is_error",
                    "content",
                    "persistedOutputPath",
                    "toolDenialKind",
                    "backgroundTaskId",
                ]);
                leaf("tool_use_id", BlockField::CallId, output.call_id.is_some());
                leaf(
                    "content",
                    BlockField::Output,
                    raw.get("content")
                        .is_some_and(|value| value.is_string() || value.is_array()),
                );
                let decision = super::tool_result::decide(v, index, raw);
                for (field, path) in [
                    (BlockField::Outcome, decision.outcome),
                    (BlockField::ExitCode, decision.exit),
                ] {
                    if let Some(path) = path {
                        add(
                            &mut result.facts,
                            Field::Block { local, field },
                            path,
                            revision,
                        );
                    }
                }
                // A reason derived from the same status witness is retained,
                // not a second supplied semantic status slot.
                if let Some(path) = decision.reason {
                    add(
                        &mut result.retained,
                        Field::Block {
                            local,
                            field: BlockField::OutcomeReason,
                        },
                        path,
                        revision,
                    );
                }
                for (key, path) in decision.data {
                    add(
                        &mut result.facts,
                        Field::ResultData { local, key },
                        path,
                        revision,
                    );
                }
                for path in decision.unresolved {
                    result
                        .obligations
                        .push(CountingObligation::UnmappedField { path });
                }
                if raw
                    .get("content")
                    .and_then(Value::as_array)
                    .is_some_and(|a| {
                        a.iter()
                            .any(|i| i.get("type").and_then(Value::as_str) != Some("text"))
                    })
                {
                    result.obligations.push(CountingObligation::UnmappedField {
                        path: {
                            let mut p = base.clone();
                            p.push("content".into());
                            p
                        },
                    })
                }
            }
            BlockContent::Image(_) | BlockContent::File(_) => {
                known.push("source");
                result
                    .obligations
                    .push(CountingObligation::UnmappedField { path: base.clone() });
            }
            BlockContent::Other { .. } => {
                leaf("", BlockField::OtherData, true);
                result
                    .obligations
                    .push(CountingObligation::UnmappedField { path: base.clone() });
            }
        }
        if let Some(object) = raw
            .as_object()
            .filter(|_| !matches!(block, BlockContent::Other { .. }))
        {
            for native_key in object.keys() {
                let key = native_key.as_str();
                if !known.contains(&key) {
                    let mut p = base.clone();
                    p.push(key.to_owned());
                    extra(&mut result.facts, Some(local), p, revision)
                }
            }
        }
    }
    if v.get("toolUseResult").is_some()
        && !blocks
            .iter()
            .any(|b| matches!(b, BlockContent::ToolResult(_)))
    {
        result.obligations.push(CountingObligation::UnmappedField {
            path: path(&["toolUseResult"]),
        });
    }
    if v.get("toolDenialKind").is_some()
        && !blocks
            .iter()
            .any(|b| matches!(b, BlockContent::ToolResult(_)))
    {
        result.obligations.push(CountingObligation::UnmappedField {
            path: path(&["toolDenialKind"]),
        });
    }
    if let Some(old) = old {
        for fact in result.facts.iter_mut().chain(&mut result.retained) {
            if at(incoming, &fact.source_path).is_none() {
                if let Some(before) = old
                    .facts
                    .iter()
                    .chain(&old.retained)
                    .find(|f| f.field == fact.field)
                {
                    fact.supplied_revision = before.supplied_revision;
                }
            }
        }
    }
    Ok(result)
}

impl SelectedUnit {
    /// Resolve one supplied source fact from this selected raw subtree. This
    /// reads no other contribution and never substitutes an enriched body field.
    pub fn fact_value(&self, fact: &Fact) -> Result<Value, String> {
        let raw = self
            .raw
            .as_ref()
            .ok_or("Claude selected fact has no raw subtree")?;
        let value: Value = serde_json::from_slice(raw).map_err(|e| e.to_string())?;
        let prefix = if self.field == 3 { 1 } else { 2 };
        at(
            &value,
            fact.source_path
                .get(prefix..)
                .ok_or("Claude selected fact path is invalid")?,
        )
        .cloned()
        .ok_or_else(|| "Claude selected fact lost its raw value".into())
    }
}

fn value_of_origin<'a>(
    object: &'a std::collections::BTreeMap<polylogue_model::NativeId, Value>,
    key: &str,
) -> Option<&'a Value> {
    object.get(&polylogue_model::NativeId::unicode(key))
}
