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
mod tests {
    use super::*;
    #[test]
    fn catalog_and_named_host_inputs_are_typed_and_closed() {
        let catalog: Arguments = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(catalog.key.is_none());
        let host: Arguments = serde_json::from_value(serde_json::json!({
            "key":"signal.quiet_hours", "machine":"named-host"
        }))
        .unwrap();
        assert_eq!(host.machine.as_deref(), Some("named-host"));
        assert!(serde_json::from_value::<Arguments>(serde_json::json!({"machine":true})).is_err());
        assert!(
            serde_json::from_value::<Arguments>(serde_json::json!({"author":"operator"})).is_err()
        );
    }
}
