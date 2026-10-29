//! Content selection shared by every reading surface.

use crate::{Block, Message};

/// Which blocks a view keeps.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum View {
    /// Everything, tool calls and results included.
    Full,
    /// What the people and models said: prose and thinking, no tool output.
    Dialogue,
    /// The operator's own prompts only.
    Prompts,
    /// Tool calls and their outcomes.
    Actions,
}

impl View {
    pub const NAMES: &[&str] = &["full", "dialogue", "prompts", "actions"];

    pub fn parse(name: &str) -> Option<View> {
        Some(match name {
            "full" => View::Full,
            "dialogue" => View::Dialogue,
            "prompts" => View::Prompts,
            "actions" => View::Actions,
            _ => return None,
        })
    }

    /// Whether this view shows `block` of `message`.
    pub fn keeps(self, message: &Message, block: &Block) -> bool {
        self.keeps_block(crate::is_prompt(message.role, message.prompt_kind), block)
    }

    pub fn block_kinds(self) -> &'static [crate::BlockKind] {
        use crate::BlockKind::*;
        match self {
            Self::Full => &[Text, Thinking, ToolCall, ToolResult, Image, File, Other],
            Self::Dialogue => &[Text, Thinking],
            Self::Prompts => &[Text],
            Self::Actions => &[ToolCall, ToolResult],
        }
    }

    pub fn prompts_only(self) -> bool {
        self == Self::Prompts
    }

    pub fn keeps_block(self, prompt: bool, block: &Block) -> bool {
        (!self.prompts_only() || prompt) && self.block_kinds().contains(&block.kind())
    }
}
