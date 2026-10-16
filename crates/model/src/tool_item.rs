//! Session-owned provider operations. Native coordinates are never Message IDs.
use crate::{IdError, SessionId};
use serde::{Deserialize, Serialize};

pub const CODEX_REVISION: &str = "092d3acd6bec3e3a14bdc7e7a2810ab628ab759d";
pub const CODEX_CONSTRUCTOR: &str = "codex-rust-v0.162.1/CommandExecution";

/// A scoped invocation key. Length framing preserves arbitrary opaque IDs.
pub fn native_id(session: &SessionId, turn: &str, item: &str) -> Result<String, IdError> {
    if turn.is_empty() || item.is_empty() {
        return Err(IdError::Empty);
    }
    let mut h = blake3::Hasher::new();
    h.update(b"polylogue.native-tool-item.v1\0");
    for s in ["codex", session.as_str(), turn, item] {
        h.update(&(s.len() as u64).to_be_bytes());
        h.update(s.as_bytes());
    }
    Ok(format!("operation:{}", h.finalize().to_hex()))
}

/// SQLite coordinates name a row in an immutable physical snapshot. The
/// row's native key is independent of physical page offsets and SQLite rowid.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryRow {
    pub thread_id: String,
    pub turn_id: String,
    pub item_id: String,
    pub rollout_ordinal: i64,
    pub updated_at_ordinal: i64,
    pub created_at_ms: i64,
    pub started_at_ms: Option<i64>,
    pub completed_at_ms: Option<i64>,
    pub item_type: String,
    pub item_json: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "representation", rename_all = "snake_case")]
pub enum ToolOrigin {
    Rollout {
        tag: String,
        start: u64,
        end: u64,
        raw_hash: [u8; 32],
    },
    History {
        schema: String,
        table: String,
        thread: String,
        turn: String,
        item: String,
        raw_hash: [u8; 32],
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Operation {
    pub kind: String,
    pub tool: Option<String>,
    pub session: SessionId,
    pub native_session: String,
    pub direct_anchors: Vec<String>,
    pub turn: Option<String>,
    pub invocation: String,
    pub native: bool,
    pub provisional_anchor: Option<String>,
    pub agent_class: String,
    pub correspondence: String,
    pub call_present: bool,
    pub command: Option<String>,
    pub result_present: bool,
    /// None is a missing display field. Some("") is explicit empty output.
    pub output: Option<String>,
    pub outcome: Option<String>,
    pub exit_code: Option<i64>,
    pub duration_ms: Option<u64>,
    pub message: Option<String>,
    pub wrapper: Option<String>,
    pub wrapper_overlaps: Vec<String>,
    pub origins: Vec<ToolOrigin>,
}
impl Operation {
    pub fn id(&self) -> Result<String, IdError> {
        if self.native {
            return native_id(
                &self.session,
                self.turn.as_deref().ok_or(IdError::Empty)?,
                &self.invocation,
            );
        }
        let anchor = self
            .provisional_anchor
            .as_deref()
            .filter(|v| !v.is_empty())
            .ok_or(IdError::Empty)?;
        let mut hash = blake3::Hasher::new();
        hash.update(b"polylogue.provisional-operation.v1\0");
        for component in [self.session.as_str(), anchor] {
            hash.update(&(component.len() as u64).to_be_bytes());
            hash.update(component.as_bytes());
        }
        Ok(format!(
            "operation:provisional-v1:{}",
            hash.finalize().to_hex()
        ))
    }
    pub fn counted(&self) -> bool {
        self.agent_class == "agent" && self.correspondence == "established" && self.call_present
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_operation_scope_does_not_collapse_turn_or_session() {
        let a = SessionId::from_stored("codex:a:b").unwrap();
        let b = SessionId::from_stored("codex:a").unwrap();
        assert_ne!(
            native_id(&a, "t", "c").unwrap(),
            native_id(&a, "t2", "c").unwrap()
        );
        assert_ne!(
            native_id(&a, "t", "c").unwrap(),
            native_id(&b, "b:t", "c").unwrap()
        );
        assert!(native_id(&a, "", "c").is_err());
    }
}
