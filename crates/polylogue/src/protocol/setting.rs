//! Runtime setting catalog, values and authored receipts share one declared operation.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Arguments {
    pub key: Option<String>,
    pub value: Option<String>,
    pub machine: Option<String>,
}

pub fn required_access(arguments: &Arguments) -> command_surface::Access {
    if arguments.value.is_some() {
        command_surface::Access::Write
    } else {
        command_surface::Access::Read
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Answer {
    Catalog(Vec<polylogue_model::settings::Current>),
    Current(Box<polylogue_model::settings::Current>),
    Receipt(polylogue_model::user::UserOutcome),
}

#[cfg(test)]
mod tests;
