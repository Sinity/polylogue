//! Native `AgentMessage`, `Reasoning` and `UserMessage` completion snapshots
//! that repeat a response item, in either arrival order. An unmatched
//! snapshot waits, keyed by thread, turn and item.
use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletedMessageProof {
    raw: Vec<u8>,
    origins: Origins,
    unresolved: bool,
    prompt: Option<usize>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
enum Body {
    Agent {
        text: Vec<String>,
        phase: Option<String>,
    },
    Reasoning {
        summary: Vec<String>,
    },
    User {
        text: String,
    },
}
struct Claim {
    thread: String,
    turn: String,
    id: String,
    body: Body,
}
fn only(v: &Value, keys: &[&str]) -> bool {
    v.as_object()
        .is_some_and(|o| o.keys().all(|k| keys.contains(&k.as_str())))
}
fn string(v: &Value, key: &str) -> Option<String> {
    v.get(key)?
        .as_str()
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}
fn text_items(value: &Value, kinds: &[&str], user: bool) -> Option<Vec<String>> {
    value
        .as_array()?
        .iter()
        .map(|v| {
            if !only(
                v,
                if user {
                    &["type", "text", "text_elements"]
                } else {
                    &["type", "text"]
                },
            ) || !kinds.contains(&v.get("type")?.as_str()?)
                || (user
                    && v.get("text_elements")
                        .is_some_and(|v| !v.as_array().is_some_and(Vec::is_empty)))
            {
                return None;
            }
            Some(v.get("text")?.as_str()?.to_owned())
        })
        .collect()
}
fn claim(payload: &Value) -> Option<Claim> {
    if !only(
        payload,
        &[
            "type",
            "thread_id",
            "turn_id",
            "item",
            "started_at_ms",
            "completed_at_ms",
        ],
    ) || payload.get("type")?.as_str()? != "item_completed"
        || ["started_at_ms", "completed_at_ms"].iter().any(|key| {
            payload
                .get(key)
                .is_some_and(|v| !v.is_null() && v.as_i64().is_none() && v.as_u64().is_none())
        })
    {
        return None;
    }
    let item = payload.get("item")?;
    let body = match item.get("type")?.as_str()? {
        "AgentMessage" if only(item, &["id", "type", "content", "phase"]) => {
            let phase = match item.get("phase") {
                None => None,
                Some(v) => Some(
                    v.as_str()
                        .filter(|s| matches!(*s, "commentary" | "final_answer"))?
                        .to_owned(),
                ),
            };
            Body::Agent {
                text: text_items(item.get("content")?, &["Text"], false)?,
                phase,
            }
        }
        "Reasoning" if only(item, &["id", "type", "summary_text", "raw_content"]) => {
            // Empty native raw_content replaces nothing.
            if !item.get("raw_content")?.as_array()?.is_empty() {
                return None;
            }
            let summary = item
                .get("summary_text")?
                .as_array()?
                .iter()
                .map(|v| v.as_str().map(str::to_owned))
                .collect::<Option<Vec<_>>>()?;
            Body::Reasoning { summary }
        }
        "UserMessage" if only(item, &["id", "type", "content"]) => Body::User {
            text: text_items(item.get("content")?, &["text"], true)?.join("\n"),
        },
        _ => return None,
    };
    Some(Claim {
        thread: string(payload, "thread_id")?,
        turn: string(payload, "turn_id")?,
        id: string(item, "id")?,
        body,
    })
}
impl CompletedMessageProof {
    fn claim(&self) -> Option<Claim> {
        let raw: Value = serde_json::from_slice(&self.raw).ok()?;
        claim(raw.get("payload")?)
    }
    pub(super) fn valid_for(&self, turn: &str, id: &str) -> bool {
        self.claim().is_some_and(|c| {
            c.turn == turn
                && c.id == id
                && (self.prompt.is_none() || matches!(c.body, Body::User { .. }))
        })
    }
}
/// The item's turn: its metadata turn if that agrees with `context`, else the
/// context turn. Malformed metadata gives `None`, never the current turn.
pub(super) fn response_turn(item: &Value, context: Option<&str>) -> Option<String> {
    let metadata = match item.get("internal_chat_message_metadata_passthrough") {
        None => return context.filter(|s| !s.is_empty()).map(str::to_owned),
        Some(value) => {
            value.as_object()?;
            value
        }
    };
    match metadata.get("turn_id") {
        Some(v) => {
            let turn = v.as_str().filter(|s| !s.is_empty())?;
            if context.is_some_and(|c| c != turn) {
                return None;
            }
            Some(turn.into())
        }
        None => context.filter(|s| !s.is_empty()).map(str::to_owned),
    }
}
fn corresponds(body: &Body, response: &Value) -> bool {
    match body {
        Body::Agent { text, phase } => {
            response.get("type").and_then(Value::as_str) == Some("message")
                && response.get("role").and_then(Value::as_str) == Some("assistant")
                && text_items(
                    response.get("content").unwrap_or(&Value::Null),
                    &["output_text", "text"],
                    false,
                )
                .as_ref()
                    == Some(text)
                && match phase {
                    Some(phase) => {
                        response.get("phase").and_then(Value::as_str) == Some(phase.as_str())
                    }
                    None => response.get("phase").is_none(),
                }
        }
        Body::Reasoning { summary } => {
            response.get("type").and_then(Value::as_str) == Some("reasoning")
                && text_items(
                    response.get("summary").unwrap_or(&Value::Null),
                    &["summary_text"],
                    false,
                )
                .as_ref()
                    == Some(summary)
        }
        Body::User { text } => {
            response.get("type").and_then(Value::as_str) == Some("message")
                && response.get("role").and_then(Value::as_str) == Some("user")
                && text_items(
                    response.get("content").unwrap_or(&Value::Null),
                    &["input_text", "text"],
                    false,
                )
                .is_some_and(|v| v.join("\n") == *text)
        }
    }
}
impl<V: ReadView> KeyedRecords<V> {
    fn completed_key(p: &Value) -> Option<Key> {
        Some(Key::CompletedMessage(
            string(p, "turn_id")?,
            string(p.get("item")?, "id")?,
        ))
    }
    fn store_completed(&mut self, key: Key, proof: Option<CompletedMessageProof>) {
        let previous = match self.get(key.clone()) {
            Some(Cell::CompletedMessage(p)) => Some(p),
            _ => None,
        };
        let old = previous.as_ref().is_some_and(|p| p.unresolved);
        if proof.is_none() {
            if let Some(previous) = &previous {
                self.retract_prompt_echo(previous);
                if let Some(message) = previous.prompt {
                    self.remove(Key::PromptEcho(message));
                }
            }
        }
        let new = proof.as_ref().is_some_and(|p| p.unresolved);
        let count = match (old, new) {
            (false, true) => self.s.pending_completed_messages.checked_add(1),
            (true, false) => self.s.pending_completed_messages.checked_sub(1),
            _ => Some(self.s.pending_completed_messages),
        };
        let Some(count) = count else {
            self.missing("pending completed Message count");
            return;
        };
        self.s.pending_completed_messages = count;
        match proof {
            Some(p) => self.put(key, Cell::CompletedMessage(p)),
            None => {
                self.remove(key);
            }
        }
    }
    fn invalidate_completed(&mut self, key: Key) {
        if let Some(Cell::CompletedMessage(mut proof)) = self.get(key.clone()) {
            proof.unresolved = true;
            self.store_completed(key, Some(proof));
        }
    }
    fn response_at(&mut self, message: usize, id: Option<&str>) -> Option<Value> {
        let slot = match id {
            Some(id) => match self.get(Key::EvidenceItem(message, id.into())) {
                Some(Cell::Position(slot, _)) => slot,
                _ => {
                    self.missing("response item evidence position");
                    return None;
                }
            },
            None => 0,
        };
        let Some(Cell::Evidence(unit)) = self.get(Key::Evidence(message, 0, slot)) else {
            self.missing("response item evidence");
            return None;
        };
        let raw: Value = match serde_json::from_slice(&unit.canonical) {
            Ok(v) => v,
            Err(_) => {
                self.missing("response item evidence JSON");
                return None;
            }
        };
        Some(raw.get("payload").unwrap_or(&raw).clone())
    }
    fn response_with_id(&mut self, id: &str) -> Option<(usize, Value, String)> {
        let (message, _) = self.position(Key::NativeItem("response_item".into(), id.into()))?;
        let Some(Cell::Text(turn)) = self.get(Key::ResponseScope(id.into())) else {
            return None;
        };
        Some((message, self.response_at(message, Some(id))?, turn))
    }
    pub(super) fn invalidate_completed_response(&mut self, item: &Value) {
        let Some(id) = item.get("id").and_then(Value::as_str) else {
            return;
        };
        if let Some(Cell::Text(turn)) = self.get(Key::ResponseScope(id.into())) {
            self.invalidate_completed(Key::CompletedMessage(turn, id.into()));
        }
        if let Some((message, _)) =
            self.position(Key::NativeItem("response_item".into(), id.into()))
        {
            if let Some(Cell::PromptEcho { turn, id }) = self.get(Key::PromptEcho(message)) {
                let key = Key::CompletedMessage(turn, id);
                if self.get(key.clone()).is_none() {
                    self.missing("prompt echo claim");
                    return;
                }
                self.invalidate_completed(key);
            }
        }
    }
    fn independent_prompt_echo(&mut self, message: usize) -> bool {
        let Some(Cell::Evidence(unit)) = self.get(Key::Evidence(message, 1, 3)) else {
            return false;
        };
        let Ok(raw) = serde_json::from_slice::<Value>(&unit.canonical) else {
            return false;
        };
        let Some(payload) = raw.get("payload") else {
            return false;
        };
        if payload.get("type").and_then(Value::as_str) != Some("user_message") {
            return false;
        }
        let Some(response) = self.response_at(message, None) else {
            return false;
        };
        response.get("role").and_then(Value::as_str) == Some("user")
            && payload.get("message").and_then(Value::as_str)
                == Some(texts(response.get("content")).join("\n").as_str())
    }
    fn apply_prompt_echo(&mut self, message: usize, proof: &CompletedMessageProof) {
        let changed = self
            .header(message)
            .is_some_and(|(m, _)| m.written_by != WrittenBy::Human);
        self.mark_human(message);
        if changed && !self.independent_prompt_echo(message) {
            self.evidence_unit(
                message,
                1,
                3,
                EvidenceUnit {
                    tag: "context:echo".into(),
                    canonical: proof.raw.clone(),
                    whole_record: true,
                    origins: proof.origins.clone(),
                },
            );
        }
        if self.s.pending_prompt == Some(message) {
            self.s.pending_prompt = None;
            self.s.pending_prompt_turn = None;
            self.remove(Key::Prompt);
        }
    }
    fn retract_prompt_echo(&mut self, proof: &CompletedMessageProof) {
        let Some(message) = proof.prompt else {
            return;
        };
        let Some(response) = self.response_at(message, None) else {
            return;
        };
        let independent = self.independent_prompt_echo(message);
        if response.get("role").and_then(Value::as_str) == Some("user") {
            let text = texts(response.get("content")).join("\n");
            let author = if independent {
                WrittenBy::Human
            } else if is_harness_wrapper(&text) {
                WrittenBy::Harness
            } else {
                match self.s.mode {
                    SessionMode::Interactive => WrittenBy::Human,
                    SessionMode::Subagent => WrittenBy::Model,
                    SessionMode::Exec => WrittenBy::Harness,
                    SessionMode::Unknown => WrittenBy::Unknown,
                }
            };
            self.edit(message, |m| {
                m.written_by = author;
                m.prompt_kind = (author == WrittenBy::Human).then_some(PromptKind::Prose);
            });
        }
        if !independent {
            self.remove_evidence(message, 1, 3);
        }
    }
    fn retract_linked_prompt(&mut self, message: usize) {
        if let Some(Cell::PromptEcho { turn, id }) = self.get(Key::PromptEcho(message)) {
            if let Some(Cell::CompletedMessage(proof)) = self.get(Key::CompletedMessage(turn, id)) {
                self.retract_prompt_echo(&proof);
            }
        }
    }
    pub(super) fn resolve_completed_response(&mut self, item: &Value) {
        if !matches!(
            item.get("type").and_then(Value::as_str),
            Some("message" | "reasoning")
        ) {
            return;
        }
        let Some(id) = item.get("id").and_then(Value::as_str) else {
            return;
        };
        let Some((message, _)) = self.position(Key::NativeItem("response_item".into(), id.into()))
        else {
            return;
        };
        let Some(response) = self.response_at(message, Some(id)) else {
            return;
        };
        let Some(turn) = response_turn(&response, self.s.turn.as_deref()) else {
            self.remove(Key::ResponseScope(id.into()));
            self.retract_linked_prompt(message);
            return;
        };
        self.put(Key::ResponseScope(id.into()), Cell::Text(turn.clone()));
        let key = Key::CompletedMessage(turn.clone(), id.into());
        if let Some(Cell::CompletedMessage(mut proof)) = self.get(key.clone()) {
            if let Some(c) = proof.claim() {
                // A typed user echo that arrives before its response never pairs,
                // even when the IDs match.
                proof.unresolved = matches!(c.body, Body::User { .. })
                    || c.turn != turn
                    || !corresponds(&c.body, &response);
                self.store_completed(key, Some(proof));
            }
        }
        if let Some(Cell::PromptEcho {
            turn: echo_turn,
            id: echo_id,
        }) = self.get(Key::PromptEcho(message))
        {
            let key = Key::CompletedMessage(echo_turn.clone(), echo_id);
            if let Some(Cell::CompletedMessage(mut proof)) = self.get(key.clone()) {
                if let Some(c) = proof.claim() {
                    proof.unresolved = echo_turn != turn || !corresponds(&c.body, &response);
                    if !proof.unresolved {
                        self.apply_prompt_echo(message, &proof);
                    } else {
                        self.retract_prompt_echo(&proof);
                    }
                    self.store_completed(key, Some(proof));
                }
            }
        }
    }
    pub(super) fn completed_message(
        &mut self,
        p: &Value,
        _at: Option<Timestamp>,
    ) -> Option<RecordOutcome<Ignored>> {
        let kind = p
            .get("item")
            .and_then(|v| v.get("type"))
            .and_then(Value::as_str);
        let supported = matches!(kind, Some("Reasoning" | "AgentMessage" | "UserMessage"));
        let key = Self::completed_key(p);
        if !supported {
            if let Some(key) = key {
                if self.get(key.clone()).is_some() {
                    self.store_completed(key, None);
                    self.s.normalization_incomplete = true;
                }
            }
            return None;
        }
        let parsed = claim(p).filter(|c| {
            self.s
                .session
                .as_ref()
                .is_some_and(|s| s.native_id == c.thread)
        });
        let Some(c) = parsed else {
            if let Some(key) = key {
                self.store_completed(key, None);
            }
            self.s.normalization_incomplete = true;
            return Some(RecordOutcome::Unrecognized(
                "unsupported completed Message shape or scope",
            ));
        };
        let key = Key::CompletedMessage(c.turn.clone(), c.id.clone());
        let old = match self.get(key.clone()) {
            Some(Cell::CompletedMessage(p)) => Some(p),
            _ => None,
        };
        if old
            .as_ref()
            .and_then(CompletedMessageProof::claim)
            .is_some_and(|old| std::mem::discriminant(&old.body) != std::mem::discriminant(&c.body))
        {
            self.store_completed(key, None);
            self.s.normalization_incomplete = true;
            return Some(RecordOutcome::Unrecognized(
                "completed Message constructor changed",
            ));
        }
        let mut proof = CompletedMessageProof {
            raw: self.current_raw.clone(),
            origins: self.current_origins.clone(),
            unresolved: true,
            prompt: old.as_ref().and_then(|p| p.prompt),
        };
        match &c.body {
            Body::User { text } => {
                if old.is_none()
                    && proof.prompt.is_none()
                    && self.s.pending_prompt_turn.as_deref() == Some(c.turn.as_str())
                    && matches!(self.get(Key::Prompt),Some(Cell::Text(pending)) if pending==*text)
                {
                    proof.prompt = self.s.pending_prompt;
                }
                if let Some(message) = proof.prompt {
                    if self
                        .response_at(message, None)
                        .is_some_and(|response| corresponds(&c.body, &response))
                    {
                        proof.unresolved = false;
                        self.put(
                            Key::PromptEcho(message),
                            Cell::PromptEcho {
                                turn: c.turn,
                                id: c.id,
                            },
                        );
                        self.apply_prompt_echo(message, &proof);
                    }
                }
            }
            _ => {
                if let Some((_, response, turn)) = self.response_with_id(&c.id) {
                    proof.unresolved = turn != c.turn || !corresponds(&c.body, &response);
                }
            }
        }
        if proof.unresolved {
            self.retract_prompt_echo(&proof);
        }
        self.store_completed(key, Some(proof));
        Some(RecordOutcome::Used)
    }
}
