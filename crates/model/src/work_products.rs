//! Canonical work products derived from selected confirmed conversation facts.
//! Inputs retain causal event IDs. The reducer has no clock, storage or transport.
use crate::{
    Block, Message, MessageId, PromptKind, Role, SessionId, Timestamp, ToolAction, ToolOutcome,
    WrittenBy,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ToolCallProduct {
    pub call: String,
    pub session: SessionId,
    pub message: MessageId,
    pub tool: Option<String>,
    pub action: ToolAction,
    pub command: Option<String>,
    pub path: Option<String>,
    pub at: Option<Timestamp>,
    pub result: Option<MessageId>,
    pub result_at: Option<Timestamp>,
    pub outcome: ToolOutcome,
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
    pub end: Option<Timestamp>,
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
    pub source_event: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ReadingSource {
    pub event: String,
    pub input: String,
    pub chain: i64,
    pub start: u64,
    pub end: u64,
    pub selected: bool,
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
    pub messages: Vec<WorkMessage>,
    pub calls: Vec<ToolCallProduct>,
    pub turns: Vec<TurnProduct>,
    pub sources: Vec<ReadingSource>,
    pub native: Vec<NativeEvidence>,
}

impl WorkSnapshot {
    pub fn validate(&self) -> Result<(), &'static str> {
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
                || !self.native.is_empty())
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
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectedMessage {
    pub message: Message,
    pub event: String,
    pub cost_nusd: Option<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Derived<T> {
    pub facts: T,
    pub parents: Vec<String>,
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
    fn new(message: &Message) -> Self {
        Self {
            product: TurnProduct {
                turn: message.id.clone(),
                session: message.session.clone(),
                start: None,
                end: None,
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
            self.product.start = Some(self.product.start.map_or(at, |old| old.min(at)));
            self.product.end = Some(self.product.end.map_or(at, |old| old.max(at)));
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

/// Background notices only affect the call the source explicitly identifies.
pub fn derive_with_background(
    messages: impl IntoIterator<Item = SelectedMessage>,
    backgrounds: impl IntoIterator<Item = (crate::BackgroundUpdate, String)>,
) -> (Vec<Derived<ToolCallProduct>>, Vec<Derived<TurnProduct>>) {
    let mut calls = Vec::<Derived<ToolCallProduct>>::new();
    let mut by_call = BTreeMap::<(SessionId, String), usize>::new();
    let mut pending_results = BTreeMap::<
        (SessionId, String),
        (
            MessageId,
            Option<Timestamp>,
            ToolOutcome,
            Option<u64>,
            String,
        ),
    >::new();
    let mut turns = Vec::<Turn>::new();
    let mut active = BTreeMap::<SessionId, (usize, bool)>::new();
    let mut call_turns = Vec::new();
    let mut call_boundaries = Vec::new();
    let mut last_response = BTreeMap::new();
    for (ordinal, selected) in messages.into_iter().enumerate() {
        let message = &selected.message;
        let written_by = WrittenBy::of(message.role, message.prompt_kind);
        if message.role == Role::Assistant
            && written_by == WrittenBy::Model
            && message.production.response_boundary_unknown != Some(true)
        {
            last_response.insert(message.session.clone(), ordinal);
        }
        if matches!(written_by, WrittenBy::Human | WrittenBy::Model) {
            let human = written_by == WrittenBy::Human;
            let current = active.get(&message.session).copied();
            let begins = match current {
                None => true,
                Some((_, previous_human)) => {
                    human && (!previous_human || message.prompt_kind == Some(PromptKind::Command))
                }
            };
            let turn = if begins {
                let n = turns.len();
                turns.push(Turn::new(message));
                n
            } else {
                current.expect("active turn").0
            };
            turns[turn].push(&selected);
            active.insert(message.session.clone(), (turn, human));
        }
        for (index, block) in message.blocks.iter().enumerate() {
            match block {
                Block::ToolCall {
                    call_id,
                    tool,
                    action,
                    input: _,
                    command,
                    path,
                    run_by_script,
                } => {
                    let call_id = call_id
                        .clone()
                        .unwrap_or_else(|| format!("{}/{index}", message.id));
                    let key = (message.session.clone(), call_id.clone());
                    let mut call = Derived {
                        facts: ToolCallProduct {
                            call: format!("tool:{}#{call_id}", message.session),
                            session: message.session.clone(),
                            message: message.id.clone(),
                            tool: tool.clone(),
                            action: *action,
                            command: command.clone(),
                            path: path.clone(),
                            at: message.at,
                            result: None,
                            result_at: None,
                            outcome: ToolOutcome::NoResult,
                            outcome_reason: Some("no_result".into()),
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
                    if let Some((id, at, outcome, duration, event)) = pending_results.remove(&key) {
                        call.facts.result = Some(id);
                        call.facts.result_at = at;
                        call.facts.outcome = outcome;
                        call.facts.outcome_reason = None;
                        call.facts.reported_duration_ms = duration;
                        call.parents.push(event);
                    }
                    by_call.insert(key, calls.len());
                    call_turns.push(active.get(&message.session).map(|(turn, _)| *turn));
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
                    let key = (message.session.clone(), id.clone());
                    if let Some(index) = by_call.get(&key) {
                        let call = &mut calls[*index];
                        call.facts.result = Some(message.id.clone());
                        call.facts.result_at = message.at;
                        call.facts.outcome = outcome.clone();
                        call.facts.outcome_reason = None;
                        call.facts.reported_duration_ms = *reported_duration_ms;
                        call.parents.push(selected.event.clone());
                    } else {
                        pending_results.insert(
                            key,
                            (
                                message.id.clone(),
                                message.at,
                                outcome.clone(),
                                *reported_duration_ms,
                                selected.event.clone(),
                            ),
                        );
                    }
                }
                _ => {}
            }
        }
    }
    for (update, event) in backgrounds {
        if let Some(index) = update
            .call_id
            .as_ref()
            .and_then(|id| by_call.get(&(update.session.clone(), id.clone())))
        {
            let call = &mut calls[*index];
            call.facts.outcome = update.outcome;
            call.facts.outcome_reason = None;
            call.facts.result_at = update.result_at.or(call.facts.result_at);
            call.facts.reported_duration_ms = update
                .reported_duration_ms
                .or(call.facts.reported_duration_ms);
            call.parents.push(event);
        }
    }
    for (index, call) in calls.iter_mut().enumerate() {
        if matches!(call.facts.outcome, ToolOutcome::NoResult) {
            let (ordinal, unknown) = call_boundaries[index];
            if !unknown
                && last_response
                    .get(&call.facts.session)
                    .is_some_and(|later| *later > ordinal)
            {
                call.facts.outcome_reason = Some("passed_without_result".into());
            }
        }
        if let Some(turn) = call_turns[index].map(|turn| &mut turns[turn]) {
            turn.product.tool_calls += 1;
            turn.product.errors +=
                u64::from(matches!(call.facts.outcome, ToolOutcome::Error { .. }));
        }
    }
    for call in &mut calls {
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
