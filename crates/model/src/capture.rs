//! One completed discovery walk, aggregated by declared source areas.
use crate::Timestamp;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Coverage {
    pub version: u8,
    pub machine: String,
    pub observed_at: Timestamp,
    pub roots: Vec<RootCoverage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RootCoverage {
    /// Native path bytes; displays are derived without replacing invalid bytes.
    pub path_bytes: Vec<u8>,
    pub status: Accessibility,
    pub entries: Vec<Entry>,
    pub unavailable_paths: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Accessibility {
    Accessible,
    Absent,
    Unavailable,
    Incomplete,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Classification {
    Collected,
    Ignored,
    Unrecognized,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub class: Classification,
    pub pattern: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub files: u64,
    pub bytes: u64,
    pub newest_mtime_ms: Option<u64>,
}

impl Coverage {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.version != 1 {
            return Err("unsupported coverage version");
        }
        if self.machine.is_empty() {
            return Err("coverage machine is empty");
        }
        let mut roots = std::collections::BTreeSet::new();
        for root in &self.roots {
            if root.path_bytes.is_empty() || !roots.insert(&root.path_bytes) {
                return Err("coverage roots must be nonempty and unique");
            }
            let mut entries = std::collections::BTreeSet::new();
            for entry in &root.entries {
                if !entries.insert((entry.class, &entry.pattern)) {
                    return Err("coverage entries must be unique within a root");
                }
                if entry.class == Classification::Ignored
                    && entry.reason.as_deref().is_none_or(str::is_empty)
                {
                    return Err("ignored coverage needs a reason");
                }
            }
        }
        Ok(())
    }
}
