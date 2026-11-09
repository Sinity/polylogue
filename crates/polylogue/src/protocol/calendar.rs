//! The timeline of recorded activity and the cost of selected sessions.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Timeline {
    /// A logical date, month or year.
    pub date: Option<String>,
    #[serde(default)]
    pub across_years: bool,
    /// Records with unknown or unrepresentable time.
    #[serde(default)]
    pub undated: bool,
    /// Rows per page.
    pub limit: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Cost {
    /// The sessions query (default `sessions`).
    pub query: Option<String>,
    /// Restrict source record times to this logical date.
    pub date: Option<String>,
    /// Response details for this model; totals include all selected models.
    pub model: Option<String>,
    /// Only source records with unknown or unrepresentable times.
    #[serde(default)]
    pub undated: bool,
    /// Response details with no recorded model name.
    #[serde(default)]
    pub unknown_model: bool,
    /// The calendar month of daily response subtotals, as YYYY-MM-DD.
    pub month: Option<String>,
    /// Rows per page.
    pub limit: Option<u64>,
}
