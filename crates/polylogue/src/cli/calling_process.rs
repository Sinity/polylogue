//! Options that act on the calling machine (`--to FILE`, `--exec`): the
//! command line reads them and they never enter a request.
//! Each struct declares a row's options the way a request type declares its
//! arguments; [`options`] reads them back typed.
use command_surface::{JsonSchema, ParsedCommand};
use serde::{Deserialize, de::DeserializeOwned};

/// `schema infer`, `schema unread` and `schema diff`: where the result goes.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct SchemaOutput {
    /// Write the result to this explicit local path (default: standard output)
    pub to: Option<String>,
}

/// The schema of a row's calling-process options, for the command table.
pub(crate) fn schema<T: JsonSchema>() -> schemars::Schema {
    schemars::schema_for!(T)
}

/// The calling-process options of a parsed command line, as `T`.
pub(crate) fn options<T: DeserializeOwned>(parsed: &ParsedCommand) -> Result<T, String> {
    T::deserialize(serde_json::Value::Object(parsed.calling_process.clone()))
        .map_err(|error| error.to_string())
}
