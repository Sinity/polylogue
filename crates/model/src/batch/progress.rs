//! How far parsing has read a chain: what one step reached, and the chain's
//! standing that the writer records from it. Each is one value, so no reader
//! rebuilds it from flags or from text.

use serde::{Deserialize, Serialize};

/// How far one parse step read the chain's stored bytes, or why it stopped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StepProgress {
    /// The step stopped before the end of the stored bytes; more remain.
    Partway,
    /// The step parsed through the end of the stored bytes. An unconsumed
    /// suffix is a record still being written and waits for growth.
    AtEnd,
    /// The origin has no parser yet: the step's bytes are collected and kept,
    /// not parsed, through the end of the stored bytes or not.
    NoParserYet { at_end: bool },
    /// The parser declined the bytes as a whole, for the stated reason; the
    /// step carries no rows.
    Refused { reason: String },
    /// The parser failed at the step's start, with its error; the step
    /// carries no rows and the cursor stays where it was.
    Failed { error: String },
}

impl StepProgress {
    /// A parsing step that did or did not reach the end of the stored bytes.
    pub fn parsed(at_end: bool) -> Self {
        if at_end { Self::AtEnd } else { Self::Partway }
    }

    /// Whether the step read through the end of the stored bytes.
    pub fn reached_end(&self) -> bool {
        match self {
            Self::Partway | Self::Refused { .. } | Self::Failed { .. } => false,
            Self::AtEnd => true,
            Self::NoParserYet { at_end } => *at_end,
        }
    }

    /// The same step, as having read through the end of the stored bytes or not.
    pub fn with_end(self, at_end: bool) -> Self {
        match self {
            Self::Partway | Self::AtEnd => Self::parsed(at_end),
            Self::NoParserYet { .. } => Self::NoParserYet { at_end },
            Self::Refused { .. } | Self::Failed { .. } => self,
        }
    }
}

/// Where one chain's parse stands, recorded by the writer after each step.
/// A failure is a parse-state fact of its input, with its error or reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ParseProgress {
    /// Stored bytes past the parsed position still need parsing.
    Parsing,
    /// Parsed through the last complete record of `length` stored bytes; the
    /// rest is a record still being written, so the chain waits for growth
    /// past `length`.
    Waiting { length: u64 },
    /// Parsed through the stored length.
    Done,
    /// The origin has no parser yet; the bytes are kept unparsed.
    NoParserYet,
    /// Confirmed events omit part of the source this reading needs.
    Partial,
    /// The parser declined the `length` stored bytes, for the stated reason.
    /// The chain is read again once its stored length or parse version
    /// changes.
    Refused { reason: String, length: u64 },
    /// The parser failed over `length` stored bytes, with its error. The
    /// chain is read again once its stored length or parse version changes.
    Failed { error: String, length: u64 },
}

stored_names! {
    /// The stored status word of a [`ParseProgress`]: its variant without
    /// the length or reason.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ParseStatus {
        Parsing => "parsing",
        Waiting => "waiting",
        Done => "done",
        NoParserYet => "no_parser_yet",
        Partial => "partial",
        Refused => "refused",
        Failed => "failed",
    }
}

impl ParseProgress {
    /// The standing after a normal step over a chain of `length` stored bytes.
    pub fn after(step: StepProgress, parsed_upto: u64, length: u64) -> Self {
        // A collection that stopped inside a record still being written waits
        // like a parse does, whether or not the origin has a parser.
        match step {
            StepProgress::Refused { reason } => Self::Refused { reason, length },
            StepProgress::Failed { error } => Self::Failed { error, length },
            _ if step.reached_end() && parsed_upto < length => Self::Waiting { length },
            StepProgress::NoParserYet { .. } => Self::NoParserYet,
            StepProgress::AtEnd => Self::Done,
            StepProgress::Partway => Self::Parsing,
        }
    }

    /// The stored status word.
    pub fn status(&self) -> &'static str {
        self.kind().as_str()
    }

    pub fn kind(&self) -> ParseStatus {
        match self {
            Self::Parsing => ParseStatus::Parsing,
            Self::Waiting { .. } => ParseStatus::Waiting,
            Self::Done => ParseStatus::Done,
            Self::NoParserYet => ParseStatus::NoParserYet,
            Self::Partial => ParseStatus::Partial,
            Self::Refused { .. } => ParseStatus::Refused,
            Self::Failed { .. } => ParseStatus::Failed,
        }
    }

    /// The stored length the standing was decided at: what a waiting chain
    /// waits to grow past, or what a refusal or failure read.
    pub fn at_length(&self) -> Option<u64> {
        match self {
            Self::Waiting { length }
            | Self::Refused { length, .. }
            | Self::Failed { length, .. } => Some(*length),
            _ => None,
        }
    }

    /// A refusal or failure, which waits for different bytes or a new reading.
    pub fn stopped(&self) -> bool {
        matches!(self, Self::Refused { .. } | Self::Failed { .. })
    }

    /// The stored reason of a refusal or the error of a failure.
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Refused { reason, .. } => Some(reason),
            Self::Failed { error, .. } => Some(error),
            _ => None,
        }
    }

    /// The standing from its stored status word, length and reason.
    pub fn from_stored(
        status: &str,
        at_length: Option<u64>,
        reason: Option<String>,
    ) -> Result<Self, String> {
        Ok(match (ParseStatus::from_name(status), at_length, reason) {
            (Some(ParseStatus::Parsing), None, None) => Self::Parsing,
            (Some(ParseStatus::Waiting), Some(length), None) => Self::Waiting { length },
            (Some(ParseStatus::Done), None, None) => Self::Done,
            (Some(ParseStatus::NoParserYet), None, None) => Self::NoParserYet,
            (Some(ParseStatus::Partial), None, None) => Self::Partial,
            (Some(ParseStatus::Refused), Some(length), Some(reason)) => {
                Self::Refused { reason, length }
            }
            (Some(ParseStatus::Failed), Some(length), Some(error)) => {
                Self::Failed { error, length }
            }
            (_, at_length, reason) => {
                return Err(format!(
                    "parse status {status:?} at length {at_length:?} with reason {reason:?} is not a parse standing"
                ));
            }
        })
    }
}

#[cfg(test)]
mod tests;
