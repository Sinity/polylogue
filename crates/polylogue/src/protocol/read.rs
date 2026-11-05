//! Public read arguments. Folds are the `show` and `hide` words of one
//! table (`FoldKind`); the daemon reads them into one `ReadOptions`.
use polylogue_model::read::{Format, ReadDelivery, View};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::num::NonZeroU64;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Arguments {
    pub target: String,
    #[serde(default)]
    pub view: View,
    pub format: Option<Format>,
    #[serde(default)]
    pub delivery: ReadDelivery,
    pub page_tokens: Option<NonZeroU64>,
    /// Item kinds to show, each `KIND` or `KIND:LEVEL`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub show: Vec<String>,
    /// Item kinds to hide.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hide: Vec<String>,
    #[serde(default = "output_lines")]
    pub output_lines: u64,
    #[serde(default)]
    pub for_others: bool,
    #[serde(default)]
    pub variant: Vec<String>,
    #[serde(default)]
    pub with_blobs: bool,
    /// The message a context reading is taken at.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<String>,
    /// The other place of a context difference: a message or `compaction[:N]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff: Option<String>,
    /// Messages before and after a message read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub around: Option<u64>,
    /// Start at the last page.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub latest: bool,
    /// With `root:REF`: read only the named physical session.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub physical: bool,
}
fn output_lines() -> u64 {
    100
}
pub type Answer = polylogue_model::read::ReadPage;

impl Arguments {
    /// The read request these arguments name, built by the command's own
    /// builder, the one parser of read arguments.
    #[cfg(test)]
    pub fn into_request(self) -> Result<crate::daemon::Request, String> {
        let command = crate::commands::command("read").ok_or("read is a command")?;
        let caller = crate::commands::Caller {
            surface: crate::commands::Surface::Mcp,
            cwd: None,
            json: true,
        };
        let arguments = serde_json::to_value(self).map_err(|error| error.to_string())?;
        (command.build_typed.ok_or("read has a typed builder")?)(arguments, &caller, "test-read")
            .map_err(|error| error.to_string())
    }

    pub fn from_request(request: &crate::daemon::Request) -> Result<Self, String> {
        let crate::daemon::Request::Read {
            target,
            options,
            delivery,
            format,
            budget,
            with_blobs,
            cursor: _,
        } = request
        else {
            return Err("a read request is required".into());
        };
        let (show, hide) = options.fold_words();
        Ok(Self {
            target: target.clone(),
            view: options.view,
            format: Some(
                Format::parse(format).ok_or_else(|| format!("unknown format {format:?}"))?,
            ),
            delivery: *delivery,
            page_tokens: budget
                .map(|value| {
                    NonZeroU64::new(value as u64)
                        .ok_or_else(|| "budget must be positive".to_string())
                })
                .transpose()?,
            show,
            hide,
            output_lines: options.output_lines,
            for_others: options.for_others,
            variant: options.variants.clone(),
            with_blobs: *with_blobs,
            at: options.at.as_ref().map(ToString::to_string),
            diff: options.diff.clone(),
            around: options.around,
            latest: options.latest,
            physical: options.physical,
        })
    }
}

#[cfg(test)]
mod tests;
