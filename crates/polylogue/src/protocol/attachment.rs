//! Attachment descriptors and bytes share one checked range operation.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Arguments {
    pub message: String,
    pub block: u64,
    #[serde(default)]
    pub attachment: u64,
    pub input: Option<String>,
    #[serde(default = "range_bytes")]
    pub length: u64,
    pub cursor: Option<String>,
}
fn range_bytes() -> u64 {
    crate::daemon::attachment::PAGE_BYTES as u64
}
pub(crate) type Answer = crate::daemon::attachment::BytePage;
