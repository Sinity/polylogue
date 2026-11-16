//! Options that act on the calling machine (`--to DIR`, `--out FILE`,
//! `--exec`): the command line reads them and they never enter a request.
//! Each struct declares a row's options the way a request type declares its
//! arguments; [`options`] reads them back typed.
use command_surface::{JsonSchema, ParsedCommand};
use serde::{Deserialize, de::DeserializeOwned};

/// `read`: where a bundle is written.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct Read {
    /// Write a bundle directory
    pub to: Option<String>,
}

/// `attachment`: where the verified bytes go.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct Attachment {
    /// Write the verified bytes to FILE
    pub out: Option<String>,
}

/// `continue`: run the plan instead of printing it.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct Continue {
    /// Run the harness resume command instead of printing it
    #[serde(default)]
    pub exec: bool,
}

/// `schema infer`, `schema unread` and `schema diff`: where the result goes.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct SchemaOutput {
    /// Write the result to this explicit local path (default: standard output)
    pub output: Option<String>,
}

/// `chat send`: the files read here and sent as bytes.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct ChatSend {
    /// Files to attach
    #[serde(default)]
    pub attach: Vec<String>,
}

/// `follow`: what each line shows and when it stops.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct Follow {
    #[schemars(description = polylogue_model::read::ShowKind::HELP)]
    #[serde(default)]
    pub show: Vec<String>,
    /// Item kinds to hide, from the same table as show
    #[serde(default)]
    pub hide: Vec<String>,
    /// Stop when the session's live state is no longer working
    #[serde(default)]
    pub until_idle: bool,
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
