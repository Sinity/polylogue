//! Finite path-preserving events, decoded through the owning immutable codec.
//! This proves membership only; it never settles Event products or outcomes.
use super::*;
use polylogue_model::{RetainedJson, SessionId};

pub(crate) struct FlatSessionStep {
    step: u64,
    events: u64,
}
impl FlatSessionStep {
    pub(crate) fn header_key(step: u64) -> [u8; EFFECT_KEY_BYTES] {
        effect_key(step, 0, 0)
    }
    pub(crate) fn decode(bytes: &[u8], step: u64) -> Result<Option<Self>> {
        let header = EffectsHeader::decode(bytes, step)?;
        // Links, native conflicts and reachback need their own reduction.
        Ok([1, 3, 5]
            .iter()
            .all(|i| header.counts[*i] == 0)
            .then_some(Self {
                step,
                events: header.counts[0],
            }))
    }
    pub(crate) fn events(&self) -> u64 {
        self.events
    }
    pub(crate) fn event_key(&self, ordinal: u64) -> Result<[u8; EFFECT_KEY_BYTES]> {
        if ordinal > self.events {
            return Err(StoreError::Range);
        }
        Ok(effect_key(self.step, 1, ordinal))
    }
    pub(crate) fn event(&self, ordinal: u64, bytes: Option<&[u8]>) -> Result<Option<Event>> {
        if ordinal > self.events {
            return Err(StoreError::Range);
        }
        if ordinal == self.events {
            return if bytes.is_none() {
                Ok(None)
            } else {
                Err(invalid("flat effects", "extra event beyond declared count"))
            };
        }
        let bytes = bytes.ok_or_else(|| invalid("flat effects", "missing declared event"))?;
        match serde_json::from_slice::<Effect>(bytes).map_err(|e| invalid("flat event", e))? {
            Effect::Event(event) => Ok(Some(event)),
            _ => Err(invalid(
                "flat effects",
                "event key contains another effect kind",
            )),
        }
    }
}

pub(crate) struct FlatSessionEvents<'a> {
    session: &'a SessionId,
    turn: Option<String>,
}
impl<'a> FlatSessionEvents<'a> {
    pub(crate) fn new(session: &'a SessionId) -> Self {
        Self {
            session,
            turn: None,
        }
    }
    pub(crate) fn accepts(&mut self, event: &Event) -> bool {
        if event
            .after_message
            .as_ref()
            .is_some_and(|m| m.session() != self.session)
        {
            return false;
        }
        let data = &event.data;
        match event.kind.as_str() {
            "base_instructions" => {
                event.model_saw == Some(true)
                    && event.after_message.is_none()
                    && (data.as_str().is_some()
                        || data.get("text").and_then(RetainedJson::as_str).is_some())
            }
            "world_state" => {
                event.model_saw == Some(true)
                    && data.get("full").and_then(RetainedJson::as_bool).is_some()
                    && data.get("state").is_some_and(RetainedJson::is_object)
            }
            "rate_limit" => {
                event.model_saw == Some(false) && event.after_message.is_none() && rate_limits(data)
            }
            "turn_start" | "turn_context" | "turn_end" => {
                if event.model_saw != Some(false) || !data.is_object() {
                    return false;
                }
                for key in ["thread_id", "session_id"] {
                    if let Some(value) = data.get(key) {
                        let Some(native) = value.as_str() else {
                            return false;
                        };
                        let origin = polylogue_model::Origin::new("codex").expect("product origin");
                        if SessionId::new(&origin, native).as_ref() != Ok(self.session) {
                            return false;
                        }
                    }
                }
                let expected = match event.kind.as_str() {
                    "turn_start" => Some("task_started"),
                    "turn_end" => Some("task_complete"),
                    _ => None,
                };
                if expected
                    .is_some_and(|ty| data.get("type").and_then(RetainedJson::as_str) != Some(ty))
                    || (expected.is_none() && data.get("type").is_some())
                {
                    return false;
                }
                let Some(turn) = data
                    .get("turn_id")
                    .and_then(RetainedJson::as_str)
                    .filter(|s| !s.is_empty())
                else {
                    return false;
                };
                if self.turn.as_deref().is_some_and(|old| old != turn) {
                    return false;
                }
                if event.kind == "turn_end" {
                    if self.turn.is_none() {
                        return false;
                    }
                    self.turn = None;
                } else {
                    if event.kind == "turn_context"
                        && data.get("model").is_some_and(|m| m.as_str().is_none())
                    {
                        return false;
                    }
                    self.turn = Some(turn.to_owned());
                }
                true
            }
            _ => false,
        }
    }
}
fn rate_limits(data: &RetainedJson) -> bool {
    let Some(object) = data.as_object() else {
        return false;
    };
    if !object.keys().all(|key| {
        matches!(
            key.as_str(),
            "primary"
                | "secondary"
                | "credits"
                | "individual_limit"
                | "limit_id"
                | "limit_name"
                | "plan_type"
                | "rate_limit_reached_type"
                | "spend_control_reached"
        )
    }) {
        return false;
    }
    for key in ["primary", "secondary", "credits"] {
        if data
            .get(key)
            .is_some_and(|v| !v.is_null() && !v.is_object())
        {
            return false;
        }
    }
    true
}

#[cfg(test)]
#[path = "flat_session/tests.rs"]
mod tests;
