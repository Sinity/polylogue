use crate::{Result, StoreError};
use polylogue_model::SessionId;
use serde::Deserialize;
use std::collections::HashSet;

/// Omission is historical unknown; a present value must be an ID array.
pub(super) fn session_selection<'de, D: serde::Deserializer<'de>>(
    de: D,
) -> std::result::Result<Option<Vec<SessionId>>, D::Error> {
    Vec::<SessionId>::deserialize(de).map(Some)
}

pub(super) fn legacy_empty(value: &str) -> bool {
    value
        .bytes()
        .filter(|b| !b.is_ascii_whitespace())
        .eq(b"[]".iter().copied())
}

/// The writer's record of what must not come back.
#[derive(Debug, Clone, Default)]
pub struct Refused {
    pub sessions: HashSet<String>,
    pub messages: HashSet<String>,
}

impl Refused {
    /// Loaded from `archive.db` when the writer starts.
    pub fn load(archive: &rusqlite::Connection) -> Result<Self> {
        let mut out = Self::default();
        // Decode only this field: publication manifests can be large and are
        // not needed to decide which sessions the writer excludes.
        #[derive(Deserialize)]
        struct SessionPolicy {
            #[serde(default)]
            #[serde(deserialize_with = "session_selection")]
            sessions: Option<Vec<SessionId>>,
        }
        let mut stmt =
            archive.prepare("SELECT scope, message_ids_json, refused_json, id FROM excision")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
            ))
        })?;
        for row in rows {
            let (scope, ids, policy, excision) = row?;
            if let Some(s) = scope.strip_prefix("session:") {
                out.sessions.insert(s.to_owned());
            }
            // The archive schema's historical default was an empty array.
            let sessions = if legacy_empty(&policy) {
                None
            } else {
                if !policy.trim_start().starts_with('{') {
                    return Err(StoreError::Invalid(
                        "invalid excision refusal payload".into(),
                    ));
                }
                let policy: SessionPolicy = serde_json::from_str(&policy)?;
                policy.sessions
            };
            if scope.starts_with("input:") && sessions.is_none() {
                return Err(StoreError::Invalid(format!(
                    "input excision {excision} has no retained session selection; recover its exclusion policy before collecting new inputs"
                )));
            }
            out.sessions
                .extend(sessions.into_iter().flatten().map(|id| id.to_string()));
            let ids: Vec<String> = serde_json::from_str(&ids)?;
            out.messages.extend(ids);
        }
        Ok(out)
    }

    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty() && self.messages.is_empty()
    }
}

#[cfg(test)]
mod tests;
