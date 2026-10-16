//! The session model-read wire contract: decode once, execute typed options,
//! and preserve those options when constructing the next public call.
use super::{MAX_BUDGET, MCP_COMPANION, Position, View, key, open, refusal, seal, unavailable};
use serde::Serialize;
use serde_json::{Value, json};

#[derive(Clone, Copy)]
enum Presentation {
    Json,
    Mcp,
}

pub(super) struct Request<'a> {
    pub reference: &'a str,
    pub view: View,
    pub branches: Option<bool>,
    pub continuation: Option<([u8; 32], [u8; 32], Position)>,
    pub budget: u64,
    key: &'static [u8; 32],
    presentation: Presentation,
}
impl<'a> Request<'a> {
    pub fn decode(request: &'a Value) -> Result<Self, Value> {
        if request.as_object().is_none_or(|o| {
            o.keys().any(|k| {
                !matches!(
                    k.as_str(),
                    "command"
                        | "reference"
                        | "budget_tokens"
                        | "continuation"
                        | "presentation"
                        | "view"
                        | "tools"
                        | "thinking"
                        | "harness"
                        | "branches"
                )
            })
        }) {
            return Err(refusal("unsupported_option"));
        }
        if request.get("command").is_some_and(|v| v != "model-read") {
            return Err(refusal("invalid_command"));
        }
        let presentation = match request.get("presentation") {
            None => Presentation::Json,
            Some(Value::String(s)) if s == "mcp" => Presentation::Mcp,
            _ => return Err(refusal("unsupported_presentation")),
        };
        let view = match request.get("view") {
            None => View::Native,
            Some(Value::String(s)) if s == "dialogue" => View::Dialogue,
            Some(Value::String(s)) if s == "transcript" => View::TranscriptFull,
            _ => return Err(refusal("unsupported_view")),
        };
        // Only the explicit full transcript combination is implemented. Other
        // planned detail/collapse/range products never silently acquire defaults.
        for option in ["tools", "thinking", "harness"] {
            if matches!(view, View::TranscriptFull) {
                if request.get(option).and_then(Value::as_str) != Some("full") {
                    return Err(refusal("transcript_requires_full_tools_thinking_harness"));
                }
            } else if request.get(option).is_some() {
                return Err(refusal("unsupported_selection_option"));
            }
        }
        let branches = match request.get("branches") {
            None => None,
            Some(Value::Bool(value)) if matches!(view, View::TranscriptFull) => Some(*value),
            _ => return Err(refusal("unsupported_branches_selection")),
        };
        let reference = request["reference"]
            .as_str()
            .filter(|s| !s.is_empty() && s.len() <= 16384)
            .ok_or_else(|| refusal("invalid_reference"))?;
        let key = key().map_err(|_| unavailable("continuation_key_unavailable"))?;
        let old = match request.get("continuation") {
            None => None,
            Some(Value::String(s)) => Some(open(s, key)?),
            _ => return Err(refusal("invalid_continuation")),
        };
        if old
            .as_ref()
            .is_some_and(|(_, r, _)| r != blake3::hash(reference.as_bytes()).as_bytes())
        {
            return Err(refusal("cross_reference_continuation"));
        }
        let budget = match request.get("budget_tokens") {
            None => old.as_ref().map_or(4000, |(_, _, p)| p.budget),
            Some(v) => v
                .as_u64()
                .filter(|v| *v > 0 && *v <= MAX_BUDGET)
                .ok_or_else(|| refusal("invalid_budget"))?,
        };
        Ok(Self {
            reference,
            view,
            branches,
            continuation: old,
            budget,
            key,
            presentation,
        })
    }

    pub fn cost(&self, value: &Value) -> u64 {
        match self.presentation {
            Presentation::Json => super::cost(value),
            Presentation::Mcp => {
                #[derive(Serialize)]
                struct Text<'a> {
                    r#type: &'a str,
                    text: &'a str,
                }
                #[derive(Serialize)]
                struct Envelope<'a> {
                    #[serde(rename = "structuredContent")]
                    structured_content: &'a Value,
                    content: [Text<'a>; 1],
                }
                super::cost(&Envelope {
                    structured_content: value,
                    content: [Text {
                        r#type: "text",
                        text: MCP_COMPANION,
                    }],
                })
            }
        }
    }

    pub fn follow_up(&self, basis: &[u8; 32], position: &Position) -> Value {
        let token = seal(basis, self.reference, position, self.key);
        let mut args = json!({"ref":self.reference,"continuation":token});
        if matches!(self.presentation, Presentation::Mcp) {
            args["budget_tokens"] = json!(self.budget);
        }
        match self.view {
            View::Native => {}
            View::Dialogue => args["view"] = json!("dialogue"),
            View::TranscriptFull => {
                args["view"] = json!("transcript");
                for option in ["tools", "thinking", "harness"] {
                    args[option] = json!("full");
                }
                if let Some(branches) = self.branches {
                    args["branches"] = json!(branches);
                }
            }
        }
        match self.presentation {
            Presentation::Json => args,
            Presentation::Mcp => json!({"tool":"read","arguments":args}),
        }
    }
}

#[cfg(test)]
mod tests;
