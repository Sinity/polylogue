//! Public read arguments use the same typed options as layout and transfer.
use polylogue_model::read::{Expansion, Format, ReadDelivery, View};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::num::NonZeroU64;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Arguments {
    Full(FullArguments),
    Resume(ResumeArguments),
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResumeArguments {
    pub root_run: polylogue_model::RunId,
    pub cursor: query_language::cursor::SealedCursor,
    pub budget: Option<NonZeroU64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FullArguments {
    pub target: String,
    #[serde(default)]
    pub view: View,
    pub format: Option<Format>,
    #[serde(default)]
    pub delivery: ReadDelivery,
    pub budget: Option<NonZeroU64>,
    pub tools: Option<Expansion>,
    pub thinking: Option<Expansion>,
    pub harness: Option<Expansion>,
    #[serde(default = "output_lines")]
    pub output_lines: u64,
    #[serde(default)]
    pub inherited: bool,
    #[serde(default)]
    pub branches: bool,
    #[serde(default)]
    pub turns: bool,
    #[serde(default = "enabled")]
    pub assertions: bool,
    #[serde(default)]
    pub for_others: bool,
    #[serde(default)]
    pub variant: Vec<String>,
    #[serde(default)]
    pub with_blobs: bool,
}
fn output_lines() -> u64 {
    100
}
fn enabled() -> bool {
    true
}
pub type Answer = polylogue_model::read::ReadPage;

impl Arguments {
    pub fn from_request(request: &crate::daemon::Request) -> Result<Self, String> {
        if let crate::daemon::Request::ReadResume {
            root_run,
            cursor,
            budget,
        } = request
        {
            return Ok(Self::Resume(ResumeArguments {
                root_run: root_run.clone(),
                cursor: cursor.clone(),
                budget: *budget,
            }));
        }
        let crate::daemon::Request::Read {
            target,
            options,
            delivery,
            format,
            budget,
            with_blobs,
        } = request
        else {
            return Err("a read request is required".into());
        };
        Ok(Self::Full(FullArguments {
            target: target.clone(),
            view: options.view,
            format: Some(
                Format::parse(format).ok_or_else(|| format!("unknown format {format:?}"))?,
            ),
            delivery: *delivery,
            budget: budget
                .map(|value| {
                    NonZeroU64::new(value as u64)
                        .ok_or_else(|| "budget must be positive".to_string())
                })
                .transpose()?,
            tools: options.tools,
            thinking: options.thinking,
            harness: options.harness,
            output_lines: options.output_lines,
            inherited: options.inherited,
            branches: options.branches,
            turns: options.turns,
            assertions: options.assertions,
            for_others: options.for_others,
            variant: options.variants.clone(),
            with_blobs: *with_blobs,
        }))
    }
}

/// Only the supplied resume fields reach this decoder. Full options are never
/// filled from current defaults or recovered from untrusted caller input.
pub(crate) fn resume_from_values(
    values: &crate::commands::Values,
) -> Result<crate::daemon::Request, String> {
    let mut object = values.0.clone();
    object.remove("json");
    let arguments: ResumeArguments = serde_json::from_value(serde_json::Value::Object(object))
        .map_err(|error| error.to_string())?;
    Ok(crate::daemon::Request::ReadResume {
        root_run: arguments.root_run,
        cursor: arguments.cursor,
        budget: arguments.budget,
    })
}

#[cfg(test)]
mod tests;
