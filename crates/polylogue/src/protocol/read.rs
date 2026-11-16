//! Public read arguments. Show and hide options are the `show` and `hide` words of one
//! table (`ShowKind`); the daemon reads them into one `ReadOptions`.
use polylogue_model::read::{Format, ReadDelivery, View};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::num::NonZeroU64;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Arguments {
    /// A query or a reference
    #[schemars(extend("argument_form" = "positional"))]
    pub target: String,
    /// The reading view
    #[serde(default)]
    pub view: View,
    /// The output format
    #[schemars(extend("unlisted" = true))]
    pub format: Option<Format>,
    /// Page or finite SSE transfer
    #[serde(default)]
    #[schemars(extend("unlisted" = true))]
    pub delivery: ReadDelivery,
    /// Tokens transferred in one read page
    pub page_tokens: Option<NonZeroU64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[schemars(description = polylogue_model::read::ShowKind::HELP)]
    #[schemars(extend("unlisted" = true))]
    pub show: Vec<String>,
    /// Item kinds to hide, from the same table as show
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[schemars(extend("unlisted" = true))]
    pub hide: Vec<String>,
    /// Tool output head and tail lines; zero keeps all lines
    #[serde(default = "output_lines")]
    #[schemars(extend("unlisted" = true))]
    pub output_lines: u64,
    /// Prepare a reading for another caller
    #[serde(default)]
    #[schemars(extend("unlisted" = true))]
    pub for_others: bool,
    /// Ordered variant layers
    #[serde(default)]
    #[schemars(extend("unlisted" = true))]
    pub variant: Vec<String>,
    /// Include stored blob bytes in a stream
    #[serde(default)]
    #[schemars(extend("unlisted" = true))]
    pub with_blobs: bool,
    /// The message a context reading is taken at
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("unlisted" = true))]
    pub at: Option<String>,
    /// With the context view: a message or compaction[:N] to compare with
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("unlisted" = true))]
    pub diff: Option<String>,
    /// With a message: this many messages before and after
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("unlisted" = true))]
    pub around: Option<u64>,
    /// Start at the last page
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    #[schemars(extend("unlisted" = true))]
    pub latest: bool,
    /// With root:REF: read only the named physical session
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    #[schemars(extend("unlisted" = true))]
    pub physical: bool,
}
fn output_lines() -> u64 {
    100
}
pub type Answer = polylogue_model::read::ReadPage;

impl Arguments {
    /// The public arguments of a resolved read, for its continuation call.
    pub(crate) fn from_request(
        request: &crate::daemon::service::read::ReadRequest,
    ) -> Result<Self, String> {
        let crate::daemon::service::read::ReadRequest {
            target,
            options,
            delivery,
            format,
            budget,
            with_blobs,
            cursor: _,
        } = request;
        let (show, hide) = options.show_hide_words();
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
