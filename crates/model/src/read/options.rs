use super::ReadError;
use crate::{Block, BlockKind, Message, PromptKind, Role};
use serde::{Deserialize, Serialize};

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum View {
    #[default]
    Transcript,
    Dialogue,
    Summary,
    Context,
    Links,
    Changes,
    Live,
    Raw,
}
impl View {
    pub const ALL: [Self; 8] = [
        Self::Transcript,
        Self::Dialogue,
        Self::Summary,
        Self::Context,
        Self::Links,
        Self::Changes,
        Self::Live,
        Self::Raw,
    ];
    pub const NAMES: &[&str] = &[
        "transcript",
        "dialogue",
        "summary",
        "context",
        "links",
        "changes",
        "live",
        "raw",
    ];
    pub const fn name(self) -> &'static str {
        match self {
            Self::Transcript => "transcript",
            Self::Dialogue => "dialogue",
            Self::Summary => "summary",
            Self::Context => "context",
            Self::Links => "links",
            Self::Changes => "changes",
            Self::Live => "live",
            Self::Raw => "raw",
        }
    }
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|view| view.name() == name)
    }
    /// The metadata filter and layout consult this same rule.
    pub const fn block_kinds(self) -> &'static [BlockKind] {
        use BlockKind::*;
        match self {
            Self::Transcript | Self::Context | Self::Raw => {
                &[Text, Thinking, ToolCall, ToolResult, Image, File, Other]
            }
            Self::Dialogue => &[Text],
            Self::Changes => &[ToolCall, ToolResult],
            Self::Summary | Self::Links | Self::Live => &[],
        }
    }
    pub fn keeps_message(self, message: &Message) -> bool {
        match self {
            Self::Dialogue => {
                message.role != Role::System && message.prompt_kind != Some(PromptKind::Harness)
            }
            Self::Transcript | Self::Context | Self::Changes | Self::Raw => true,
            Self::Summary | Self::Links | Self::Live => false,
        }
    }
    pub fn keeps(self, message: &Message, block: &Block) -> bool {
        self.keeps_message(message) && self.block_kinds().contains(&block.kind())
    }
}
impl std::fmt::Display for View {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
impl std::str::FromStr for View {
    type Err = ReadError;
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::parse(name).ok_or_else(|| ReadError::UnknownView { name: name.into() })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Expansion {
    None,
    #[serde(rename = "headers")]
    Collapsed,
    Full,
}

/// Absent expansions distinguish defaults from explicit unsupported options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadOptions {
    #[serde(default)]
    pub view: View,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Expansion>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Expansion>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub harness: Option<Expansion>,
    /// Head and tail lines of expanded tool output. Zero keeps the complete output.
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub variants: Vec<String>,
}
fn output_lines() -> u64 {
    100
}
fn enabled() -> bool {
    true
}
impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            view: View::Transcript,
            tools: None,
            thinking: None,
            harness: None,
            output_lines: 100,
            inherited: false,
            branches: false,
            turns: false,
            assertions: true,
            for_others: false,
            variants: vec![],
        }
    }
}
impl ReadOptions {
    pub fn tools(&self) -> Expansion {
        self.tools.unwrap_or(Expansion::Collapsed)
    }
    pub fn thinking(&self) -> Expansion {
        self.thinking.unwrap_or(Expansion::Collapsed)
    }
    pub fn harness(&self) -> Expansion {
        self.harness.unwrap_or(Expansion::Collapsed)
    }
    pub fn validate(&self) -> Result<(), ReadError> {
        let prose = matches!(self.view, View::Transcript | View::Context);
        for (supplied, option, accepted) in [
            (
                self.tools.is_some() && !prose,
                "tools",
                "transcript, context",
            ),
            (
                self.thinking.is_some() && !prose,
                "thinking",
                "transcript, context",
            ),
            (
                self.harness.is_some() && !prose,
                "harness",
                "transcript, context",
            ),
            (self.turns && !prose, "turns", "transcript, context"),
            (
                self.output_lines != 100 && !prose,
                "output_lines",
                "transcript, context",
            ),
            (
                self.inherited
                    && matches!(
                        self.view,
                        View::Summary | View::Links | View::Live | View::Raw
                    ),
                "inherited",
                "transcript, dialogue, context, changes",
            ),
            (
                self.branches && matches!(self.view, View::Summary | View::Links | View::Live),
                "branches",
                "transcript, dialogue, context, changes, raw",
            ),
        ] {
            if supplied {
                return Err(ReadError::UnsupportedOption {
                    view: self.view,
                    option: option.into(),
                    accepted_by: accepted.into(),
                });
            }
        }
        Ok(())
    }
}
