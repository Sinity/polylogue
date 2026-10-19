use crate::{BlobHash, Json};
use serde::{Deserialize, Serialize};

/// One part of a message's content, in the order the source wrote it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Block {
    Text {
        text: String,
    },
    Thinking {
        text: Option<String>,
        /// Present when the provider sent the reasoning encrypted or
        /// redacted; the text is then absent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        redacted: bool,
    },
    ToolCall {
        call_id: Option<String>,
        tool: Option<String>,
        input: Json,
        /// The shell command, when the call runs one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        command: Option<String>,
        /// The file the call reads or edits, when it names one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
    },
    /// A tool's result. It lives in the message that carried it and points
    /// at its call by `call_id`; the call may be in an earlier message.
    ToolResult {
        call_id: Option<String>,
        outcome: ToolOutcome,
        output: Option<String>,
        /// Structured result data the source attached beside the text.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<Json>,
        /// Images or files the tool returned (screenshots, rendered pages).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        attachments: Vec<Attachment>,
    },
    Image(Attachment),
    File(Attachment),
    /// A block kind this origin's parser does not interpret yet. Kept whole.
    Other {
        type_name: String,
        data: Json,
    },
}

/// A block's variant without its contents: what the index stores as the
/// block's kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockKind {
    Text,
    Thinking,
    ToolCall,
    ToolResult,
    Image,
    File,
    Other,
}

stored_names!(BlockKind {
    Text => "text",
    Thinking => "thinking",
    ToolCall => "tool_call",
    ToolResult => "tool_result",
    Image => "image",
    File => "file",
    Other => "other",
});

impl Block {
    pub fn kind(&self) -> BlockKind {
        match self {
            Block::Text { .. } => BlockKind::Text,
            Block::Thinking { .. } => BlockKind::Thinking,
            Block::ToolCall { .. } => BlockKind::ToolCall,
            Block::ToolResult { .. } => BlockKind::ToolResult,
            Block::Image(_) => BlockKind::Image,
            Block::File(_) => BlockKind::File,
            Block::Other { .. } => BlockKind::Other,
        }
    }

    /// Prose a person or model wrote, for search and display.
    pub fn text(&self) -> Option<&str> {
        match self {
            Block::Text { text } => Some(text),
            Block::Thinking { text, .. } => text.as_deref(),
            _ => None,
        }
    }
}

/// How a tool call ended. Kinds never collapse into one success figure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolOutcome {
    Ok,
    Error {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<ErrorReason>,
    },
    /// The call was made and no result ever arrived.
    NoResult,
    /// A result exists but its status cannot be read.
    Unknown {
        reason: UnknownReason,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum ErrorReason {
    /// Refused by a permission rule, mode or the user; the source's own label.
    Denied(String),
    Interrupted,
    TimedOut,
    ExitCode(i64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnknownReason {
    StatusFieldUnreadable,
    /// Started in the background; completion has not been seen yet.
    RunningInBackground,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attachment {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    /// The bytes, when they were inline or found on disk and stored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob: Option<BlobHash>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// The source's own ID for the file (ChatGPT `file-…`), by which an
    /// export's separately stored bytes are found.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    pub direction: AttachmentDirection,
}

/// Whether the attachment went to the model or came from it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentDirection {
    #[default]
    Input,
    Output,
}

#[cfg(test)]
mod tests;
