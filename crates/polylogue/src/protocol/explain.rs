//! Explaining a query, and pages of the language's own description.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Arguments {
    /// `query`, `syntax`, `units`, `fields`, `operators` or `capabilities`.
    pub subject: String,
    /// The query, for subject `query`.
    pub expression: Option<String>,
    /// List fields for this unit.
    pub unit: Option<String>,
    /// Declarations per page.
    pub page_size: Option<u64>,
    /// The calling session, which `self` names.
    pub session: Option<String>,
}
