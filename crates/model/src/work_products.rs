//! Canonical work products derived from selected confirmed conversation facts.
//! Inputs retain causal event IDs. The reducer has no clock, storage or transport.
use crate::{
    Block, Message, MessageId, PromptKind, Role, SessionId, Timestamp, ToolAction, ToolOutcome,
    WrittenBy,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ToolCallProduct {
    /// Stable structural identity; display references and native IDs are evidence.
    pub entry_id: String,
    pub call: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_block: Option<u64>,
    pub unavailable: Vec<String>,
    pub session: SessionId,
    pub physical_session: SessionId,
    pub message: MessageId,
    pub tool: Option<String>,
    pub action: ToolAction,
    pub command: Option<String>,
    pub path: Option<String>,
    pub at: Option<Timestamp>,
    pub at_evidence: crate::TimeEvidence,
    pub result: Option<MessageId>,
    pub result_at: Option<Timestamp>,
    pub result_at_evidence: crate::TimeEvidence,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<ToolOutcome>,
    pub outcome_reason: Option<String>,
    pub reported_duration_ms: Option<u64>,
    pub run_by_script: bool,
    pub revision: Option<String>,
    pub check_name: Option<String>,
    pub check_passed: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TurnProduct {
    pub turn: MessageId,
    pub session: SessionId,
    pub start: Option<Timestamp>,
    pub start_evidence: crate::TimeEvidence,
    pub end: Option<Timestamp>,
    pub end_evidence: crate::TimeEvidence,
    pub prompts: Vec<MessageId>,
    pub answer: Option<MessageId>,
    pub slash_command: bool,
    pub tool_calls: u64,
    pub errors: u64,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cost_nusd: Option<i64>,
    pub abandoned: bool,
    pub input_unknown: u64,
    pub output_unknown: u64,
    pub cost_unknown: u64,
    pub unavailable: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WorkMessage {
    pub message: MessageId,
    pub role: Role,
    pub written_by: WrittenBy,
    pub prompt_kind: Option<PromptKind>,
    pub at: Option<Timestamp>,
    pub at_evidence: crate::TimeEvidence,
    pub source_event: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ReadingSource {
    pub event: String,
    pub input: String,
    pub chain: i64,
    /// The enclosing parser step; a compound record may cover the whole input.
    /// The causal fact retains its exact native record range and output identity.
    pub start: u64,
    pub end: u64,
    pub selected: bool,
    pub projection: Option<crate::bundle::MessageProjection>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeEvidence {
    pub event: String,
    pub material: String,
    pub anchor: u64,
    pub data: serde_json::Value,
}

/// One complete logical-session reduction at a confirmed position. Replacing
/// this record retires its preceding complete set, including omitted entries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WorkSnapshot {
    pub reducer_version: u32,
    pub position: i64,
    pub session_id: SessionId,
    pub session: Option<crate::Session>,
    pub retired: bool,
    pub unavailable: Vec<String>,
    pub messages: Vec<WorkMessage>,
    pub calls: Vec<ToolCallProduct>,
    pub turns: Vec<TurnProduct>,
    pub sources: Vec<ReadingSource>,
    pub native: Vec<NativeEvidence>,
}

impl WorkSnapshot {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.reducer_version != VERSION {
            return Err("unsupported work reducer version");
        }
        if self.position < 0 {
            return Err("snapshot position is negative");
        }
        if self
            .session
            .as_ref()
            .is_some_and(|session| session.id != self.session_id)
        {
            return Err("snapshot session metadata has another identity");
        }
        if self.retired
            && (self.session.is_some()
                || !self.messages.is_empty()
                || !self.calls.is_empty()
                || !self.turns.is_empty()
                || !self.native.is_empty()
                || !self.sources.is_empty())
        {
            return Err("retired snapshot contains source facts");
        }
        if self
            .calls
            .iter()
            .any(|call| call.session != self.session_id)
            || self
                .turns
                .iter()
                .any(|turn| turn.session != self.session_id)
        {
            return Err("snapshot member belongs to another session");
        }
        let mut entries = BTreeSet::new();
        for call in &self.calls {
            if call.entry_id.is_empty() || !entries.insert(&call.entry_id) {
                return Err("snapshot call entry identity is missing or repeated");
            }
            if call.outcome.is_none()
                && (call
                    .outcome_reason
                    .as_ref()
                    .is_none_or(|reason| reason.is_empty())
                    || call.unavailable.is_empty())
            {
                return Err("unavailable call outcome has no reason and evidence coverage");
            }
            if call.original_block.is_none()
                && !call
                    .unavailable
                    .iter()
                    .any(|reason| reason == "original_block_unavailable")
            {
                return Err("unknown call block origin has no unavailable reason");
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectedMessage {
    /// Raw physical-session facts remain unchanged.
    pub message: Message,
    pub logical_session: SessionId,
    /// None is a complete original row; a sparse row requires its source mapping.
    pub projection: Option<crate::bundle::MessageProjection>,
    pub event: String,
    pub cost_nusd: Option<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectedBackground {
    pub logical_session: SessionId,
    pub update: crate::BackgroundUpdate,
    pub event: String,
}

#[derive(Serialize)]
#[serde(tag = "origin", rename_all = "snake_case")]
enum BlockOrigin<'a> {
    Original {
        block: u64,
    },
    Unknown {
        event: &'a str,
        selected_slot: usize,
    },
}
#[derive(Serialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
enum CallKey<'a> {
    Native(&'a str),
    Local,
}
fn entry_id(
    selected: &SelectedMessage,
    original: Option<u64>,
    slot: usize,
    native: Option<&str>,
) -> String {
    let origin = match original {
        Some(block) => BlockOrigin::Original { block },
        None => BlockOrigin::Unknown {
            event: &selected.event,
            selected_slot: slot,
        },
    };
    let key = native.map_or(CallKey::Local, CallKey::Native);
    let identity = (
        "polylogue.work.tool_call",
        VERSION,
        &selected.logical_session,
        &selected.message.session,
        &selected.message.id,
        origin,
        key,
    );
    let bytes = serde_json::to_vec(&identity)
        .expect("structural call identity contains only serializable scalar fields");
    format!("work-call:{}", blake3::hash(&bytes).to_hex())
}

#[derive(Debug, Clone, PartialEq)]
pub struct Derived<T> {
    pub facts: T,
    pub parents: Vec<String>,
}

// Missing source fields do not conflict with explicit evidence. Different
// observed values remain unavailable rather than choosing by arrival or clock.
fn agreed<T: PartialEq>(values: impl IntoIterator<Item = T>) -> Result<Option<T>, ()> {
    let mut values = values.into_iter();
    let Some(first) = values.next() else {
        return Ok(None);
    };
    if values.all(|value| value == first) {
        Ok(Some(first))
    } else {
        Err(())
    }
}

fn add_u64(
    total: &mut Option<u64>,
    value: Option<u64>,
    field: &str,
    unavailable: &mut Vec<String>,
) {
    let overflow = format!("{field}_overflow");
    if unavailable.contains(&overflow) {
        return;
    }
    if let Some(value) = value {
        if let Some(previous) = *total {
            *total = previous.checked_add(value);
            if total.is_none() {
                unavailable.push(overflow);
            }
        } else {
            *total = Some(value);
        }
    }
}
fn add_i64(
    total: &mut Option<i64>,
    value: Option<i64>,
    field: &str,
    unavailable: &mut Vec<String>,
) {
    let overflow = format!("{field}_overflow");
    if unavailable.contains(&overflow) {
        return;
    }
    if let Some(value) = value {
        if let Some(previous) = *total {
            *total = previous.checked_add(value);
            if total.is_none() {
                unavailable.push(overflow);
            }
        } else {
            *total = Some(value);
        }
    }
}

struct Turn {
    product: TurnProduct,
    parents: Vec<String>,
}
impl Turn {
    fn new(message: &Message, logical_session: &SessionId) -> Self {
        Self {
            product: TurnProduct {
                turn: message.id.clone(),
                session: logical_session.clone(),
                start: None,
                start_evidence: crate::TimeEvidence::UNKNOWN,
                end: None,
                end_evidence: crate::TimeEvidence::UNKNOWN,
                prompts: Vec::new(),
                answer: None,
                slash_command: message.prompt_kind == Some(PromptKind::Command),
                tool_calls: 0,
                errors: 0,
                input_tokens: None,
                output_tokens: None,
                cost_nusd: None,
                abandoned: false,
                input_unknown: 0,
                output_unknown: 0,
                cost_unknown: 0,
                unavailable: Vec::new(),
            },
            parents: Vec::new(),
        }
    }
    fn push(&mut self, message: &SelectedMessage) {
        let m = &message.message;
        if let Some(at) = m.at {
            if self.product.start.is_none_or(|old| at < old) {
                self.product.start = Some(at);
                self.product.start_evidence = m.time_evidence();
            }
            if self.product.end.is_none_or(|old| at > old) {
                self.product.end = Some(at);
                self.product.end_evidence = m.time_evidence();
            }
        }
        match WrittenBy::of(m.role, m.prompt_kind) {
            WrittenBy::Human => self.product.prompts.push(m.id.clone()),
            WrittenBy::Model if m.role == Role::Assistant => {
                self.product.answer = Some(m.id.clone())
            }
            _ => {}
        }
        add_u64(
            &mut self.product.input_tokens,
            m.usage.input,
            "input_tokens",
            &mut self.product.unavailable,
        );
        add_u64(
            &mut self.product.output_tokens,
            m.usage.output,
            "output_tokens",
            &mut self.product.unavailable,
        );
        add_i64(
            &mut self.product.cost_nusd,
            message.cost_nusd,
            "cost_nusd",
            &mut self.product.unavailable,
        );
        if m.role == Role::Assistant && WrittenBy::of(m.role, m.prompt_kind) == WrittenBy::Model {
            self.product.input_unknown += u64::from(m.usage.input.is_none());
            self.product.output_unknown += u64::from(m.usage.output.is_none());
            self.product.cost_unknown += u64::from(message.cost_nusd.is_none());
        }
        self.parents.push(message.event.clone());
    }
}

/// Each logical message occurs once, already selected and ordered by the
/// product's source-reading rules. Harness notices do not open turns.
pub fn derive(
    messages: impl IntoIterator<Item = SelectedMessage>,
) -> (Vec<Derived<ToolCallProduct>>, Vec<Derived<TurnProduct>>) {
    derive_with_background(messages, std::iter::empty())
}

/// Native IDs associate evidence only with a unique selected definition.
/// Conflicting observations stay unavailable; clocks and arrival do not resolve them.
pub fn derive_with_background(
    messages: impl IntoIterator<Item = SelectedMessage>,
    backgrounds: impl IntoIterator<Item = SelectedBackground>,
) -> (Vec<Derived<ToolCallProduct>>, Vec<Derived<TurnProduct>>) {
    struct Observation {
        result: Option<MessageId>,
        at: Option<Timestamp>,
        evidence: crate::TimeEvidence,
        outcome: ToolOutcome,
        duration: Option<u64>,
        event: String,
    }
    let mut calls = Vec::<Derived<ToolCallProduct>>::new();
    let mut by_call = BTreeMap::<(SessionId, SessionId, String), Vec<usize>>::new();
    let mut observations = BTreeMap::<(SessionId, SessionId, String), Vec<Observation>>::new();
    let mut turns = Vec::<Turn>::new();
    let mut active = BTreeMap::<SessionId, (usize, bool)>::new();
    let mut call_turns = Vec::new();
    let mut call_boundaries = Vec::new();
    let mut last_response = BTreeMap::new();
    for (ordinal, selected) in messages.into_iter().enumerate() {
        let message = &selected.message;
        let session = &selected.logical_session;
        let written_by = WrittenBy::of(message.role, message.prompt_kind);
        if message.role == Role::Assistant
            && written_by == WrittenBy::Model
            && message.production.response_boundary_unknown != Some(true)
        {
            last_response.insert((session.clone(), message.session.clone()), ordinal);
        }
        if matches!(written_by, WrittenBy::Human | WrittenBy::Model) {
            let human = written_by == WrittenBy::Human;
            let current = active.get(session).copied();
            let begins = match current {
                None => true,
                Some((_, previous_human)) => {
                    human && (!previous_human || message.prompt_kind == Some(PromptKind::Command))
                }
            };
            let turn = if begins {
                let n = turns.len();
                turns.push(Turn::new(message, session));
                n
            } else {
                current.expect("active turn").0
            };
            turns[turn].push(&selected);
            active.insert(session.clone(), (turn, human));
        }
        let mapping_valid = selected
            .projection
            .as_ref()
            .is_none_or(|projection| projection.validate(message.blocks.len()).is_ok());
        for (index, block) in message.blocks.iter().enumerate() {
            let original = if mapping_valid {
                Some(
                    selected
                        .projection
                        .as_ref()
                        .map_or(index as u64, |projection| projection.block_ordinals[index]),
                )
            } else {
                None
            };
            match block {
                Block::ToolCall {
                    call_id,
                    tool,
                    action,
                    command,
                    path,
                    run_by_script,
                    ..
                } => {
                    let mut unavailable = Vec::new();
                    if original.is_none() {
                        unavailable.push("original_block_unavailable".into());
                    }
                    if call_id.is_none() {
                        unavailable.push("native_call_id_unavailable".into());
                    }
                    let display = call_id.clone().unwrap_or_else(|| match original {
                        Some(block) => format!("{}/{block}", message.id),
                        None => format!("{}/unknown-{index}", message.id),
                    });
                    let call = Derived {
                        facts: ToolCallProduct {
                            entry_id: entry_id(&selected, original, index, call_id.as_deref()),
                            call: format!("tool:{}#{display}", message.session),
                            native_call_id: call_id.clone(),
                            original_block: original,
                            unavailable,
                            session: session.clone(),
                            physical_session: message.session.clone(),
                            message: message.id.clone(),
                            tool: tool.clone(),
                            action: *action,
                            command: command.clone(),
                            path: path.clone(),
                            at: message.at,
                            at_evidence: message.time_evidence(),
                            result: None,
                            result_at: None,
                            result_at_evidence: crate::TimeEvidence::UNKNOWN,
                            outcome: call_id.as_ref().map(|_| ToolOutcome::NoResult),
                            outcome_reason: Some(
                                if call_id.is_some() {
                                    "no_result"
                                } else {
                                    "native_call_id_unavailable"
                                }
                                .into(),
                            ),
                            reported_duration_ms: None,
                            run_by_script: *run_by_script,
                            revision: message
                                .attrs
                                .get("revision")
                                .and_then(serde_json::Value::as_str)
                                .map(str::to_owned),
                            check_name: message
                                .attrs
                                .get("check_name")
                                .and_then(serde_json::Value::as_str)
                                .map(str::to_owned),
                            check_passed: message
                                .attrs
                                .get("check_passed")
                                .and_then(serde_json::Value::as_bool),
                        },
                        parents: vec![selected.event.clone()],
                    };
                    if let Some(id) = call_id {
                        by_call
                            .entry((session.clone(), message.session.clone(), id.clone()))
                            .or_default()
                            .push(calls.len());
                    }
                    call_turns.push(active.get(session).map(|(turn, _)| *turn));
                    call_boundaries.push((
                        ordinal,
                        message.production.response_boundary_unknown == Some(true),
                    ));
                    calls.push(call);
                }
                Block::ToolResult {
                    call_id: Some(id),
                    outcome,
                    reported_duration_ms,
                    ..
                } => {
                    observations
                        .entry((session.clone(), message.session.clone(), id.clone()))
                        .or_default()
                        .push(Observation {
                            result: Some(message.id.clone()),
                            at: message.at,
                            evidence: message.time_evidence(),
                            outcome: outcome.clone(),
                            duration: *reported_duration_ms,
                            event: selected.event.clone(),
                        });
                }
                _ => {}
            }
        }
    }
    for selected in backgrounds {
        let update = selected.update;
        if let Some(id) = update.call_id {
            observations
                .entry((selected.logical_session, update.session, id))
                .or_default()
                .push(Observation {
                    result: None,
                    at: update.result_at,
                    evidence: update.result_at_evidence,
                    outcome: update.outcome,
                    duration: update.reported_duration_ms,
                    event: selected.event,
                });
        }
    }
    for (key, definitions) in by_call {
        let results = observations.remove(&key).unwrap_or_default();
        let mut evidence: Vec<_> = definitions
            .iter()
            .flat_map(|&index| calls[index].parents.clone())
            .collect();
        evidence.extend(results.iter().map(|result| result.event.clone()));
        let ambiguous = definitions.len() > 1;
        for index in definitions {
            let call = &mut calls[index];
            call.parents.extend(evidence.iter().cloned());
            if ambiguous {
                call.facts.outcome = None;
                call.facts.outcome_reason = Some("ambiguous_native_call_id".into());
                call.facts
                    .unavailable
                    .push("ambiguous_native_call_id".into());
            } else if !results.is_empty() {
                match agreed(results.iter().map(|result| &result.outcome)) {
                    Ok(outcome) => {
                        call.facts.outcome = outcome.cloned();
                        call.facts.outcome_reason = None;
                    }
                    Err(()) => {
                        call.facts.outcome = None;
                        call.facts.outcome_reason = Some("ambiguous_result_evidence".into());
                        call.facts
                            .unavailable
                            .push("ambiguous_result_evidence".into());
                    }
                }
                match agreed(results.iter().filter_map(|result| result.result.as_ref())) {
                    Ok(result) => call.facts.result = result.cloned(),
                    Err(()) => call
                        .facts
                        .unavailable
                        .push("result_message_ambiguous".into()),
                }
                match agreed(
                    results
                        .iter()
                        .filter_map(|result| result.at.map(|at| (at, result.evidence))),
                ) {
                    Ok(Some((at, evidence))) => {
                        call.facts.result_at = Some(at);
                        call.facts.result_at_evidence = evidence;
                    }
                    Ok(None) => {}
                    Err(()) => call.facts.unavailable.push("result_time_ambiguous".into()),
                }
                match agreed(results.iter().filter_map(|result| result.duration)) {
                    Ok(duration) => call.facts.reported_duration_ms = duration,
                    Err(()) => call
                        .facts
                        .unavailable
                        .push("reported_duration_ambiguous".into()),
                }
            }
        }
    }
    for (index, call) in calls.iter_mut().enumerate() {
        if matches!(call.facts.outcome, Some(ToolOutcome::NoResult))
            && call.facts.outcome_reason.as_deref() == Some("no_result")
        {
            let (ordinal, unknown) = call_boundaries[index];
            if !unknown
                && last_response
                    .get(&(
                        call.facts.session.clone(),
                        call.facts.physical_session.clone(),
                    ))
                    .is_some_and(|later| *later > ordinal)
            {
                call.facts.outcome_reason = Some("passed_without_result".into());
            }
        }
        call.facts.unavailable.sort();
        call.facts.unavailable.dedup();
        if let Some(turn) = call_turns[index].map(|turn| &mut turns[turn]) {
            turn.product.tool_calls += 1;
            turn.product.errors += u64::from(matches!(
                call.facts.outcome,
                Some(ToolOutcome::Error { .. })
            ));
            turn.product
                .unavailable
                .extend(call.facts.unavailable.iter().cloned());
            turn.parents.extend(call.parents.iter().cloned());
        }
        call.parents.sort();
        call.parents.dedup();
    }
    let turns = turns
        .into_iter()
        .enumerate()
        .map(|(index, mut turn)| {
            turn.product.abandoned = turn.product.answer.is_none()
                && active
                    .get(&turn.product.session)
                    .is_some_and(|(last, _)| *last == index);
            turn.parents.sort();
            turn.parents.dedup();
            turn.product.unavailable.sort();
            turn.product.unavailable.dedup();
            Derived {
                facts: turn.product,
                parents: turn.parents,
            }
        })
        .collect();
    (calls, turns)
}

#[cfg(test)]
mod tests;
