//! The timeline of recorded activity and the cost of selected sessions.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Timeline {
    /// Logical date YYYY-MM-DD (default today)
    pub date: Option<String>,
    /// This month and day in recorded years through the chosen date
    #[serde(default)]
    #[schemars(extend("unlisted" = true))]
    pub across_years: bool,
    /// Records with unknown or unrepresentable time
    #[serde(default)]
    #[schemars(extend("unlisted" = true))]
    pub undated: bool,
    /// Rows per page
    pub limit: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Cost {
    /// Session query (default sessions)
    pub query: Option<String>,
    /// Restrict source record times to this logical date
    pub date: Option<String>,
    /// Show response details for this model; totals include all selected models
    #[schemars(extend("unlisted" = true))]
    pub model: Option<String>,
    /// Only source records with unknown or unrepresentable times
    #[serde(default)]
    #[schemars(extend("unlisted" = true))]
    pub undated: bool,
    /// Response details with no recorded model name
    #[serde(default)]
    #[schemars(extend("unlisted" = true))]
    pub unknown_model: bool,
    /// Calendar month for daily response subtotals, as YYYY-MM-DD (default selected day or today)
    #[schemars(extend("unlisted" = true))]
    pub month: Option<String>,
    /// Rows per page
    pub limit: Option<u64>,
}
