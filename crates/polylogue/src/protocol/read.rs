//! Public read arguments use the same typed options as layout and transfer.
use polylogue_model::read::{Expansion, ReadDelivery, View};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Arguments {
    pub target: String,
    #[serde(default)]
    pub view: View,
    pub format: Option<polylogue_render::Format>,
    #[serde(default)]
    pub delivery: ReadDelivery,
    pub budget: Option<u64>,
    pub cursor: Option<String>,
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
        let crate::daemon::Request::Read {
            target,
            options,
            delivery,
            format,
            budget,
            cursor,
            with_blobs,
        } = request
        else {
            return Err("a read request is required".into());
        };
        Ok(Self {
            target: target.clone(),
            view: options.view,
            format: Some(
                polylogue_render::Format::parse(format)
                    .ok_or_else(|| format!("unknown format {format:?}"))?,
            ),
            delivery: *delivery,
            budget: budget.map(|value| value as u64),
            cursor: cursor.clone(),
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
        })
    }
}
