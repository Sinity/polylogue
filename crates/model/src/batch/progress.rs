//! How far parsing has read a chain: what one step reached, and the chain's
//! standing that the writer records from it. Each is one value, so no reader
//! rebuilds it from flags or from text.

use serde::{Deserialize, Serialize};

/// How far one parse step read the chain's stored bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
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
}

impl StepProgress {
    /// A parsing step that did or did not reach the end of the stored bytes.
    pub fn parsed(at_end: bool) -> Self {
        if at_end { Self::AtEnd } else { Self::Partway }
    }

    /// Whether the step read through the end of the stored bytes.
    pub fn reached_end(self) -> bool {
        match self {
            Self::Partway => false,
            Self::AtEnd => true,
            Self::NoParserYet { at_end } => at_end,
        }
    }

    /// The same step, as having read through the end of the stored bytes or not.
    pub fn with_end(self, at_end: bool) -> Self {
        match self {
            Self::Partway | Self::AtEnd => Self::parsed(at_end),
            Self::NoParserYet { .. } => Self::NoParserYet { at_end },
        }
    }
}

/// Where one chain's parse stands, recorded by the writer after each step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
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
    /// Confirmed facts omit part of the source this reading needs.
    Partial,
}

impl ParseProgress {
    /// The standing after an ordinary step over a chain of `length` stored bytes.
    pub fn after(step: StepProgress, parsed_upto: u64, length: u64) -> Self {
        // A collection that stopped inside a record still being written waits
        // like a parse does, whether or not the origin has a parser.
        match step {
            _ if step.reached_end() && parsed_upto < length => Self::Waiting { length },
            StepProgress::NoParserYet { .. } => Self::NoParserYet,
            StepProgress::AtEnd => Self::Done,
            StepProgress::Partway => Self::Parsing,
        }
    }

    /// The stored status word.
    pub fn status(self) -> &'static str {
        match self {
            Self::Parsing => "parsing",
            Self::Waiting { .. } => "waiting",
            Self::Done => "done",
            Self::NoParserYet => "no_parser_yet",
            Self::Partial => "partial",
        }
    }

    /// The stored length a waiting chain waits at.
    pub fn waiting_at(self) -> Option<u64> {
        match self {
            Self::Waiting { length } => Some(length),
            _ => None,
        }
    }

    /// The standing from its stored status word and waiting length.
    pub fn from_stored(status: &str, waiting_at: Option<u64>) -> Result<Self, String> {
        Ok(match (status, waiting_at) {
            ("parsing", None) => Self::Parsing,
            ("waiting", Some(length)) => Self::Waiting { length },
            ("done", None) => Self::Done,
            ("no_parser_yet", None) => Self::NoParserYet,
            ("partial", None) => Self::Partial,
            (status, waiting_at) => {
                return Err(format!(
                    "parse status {status:?} with waiting length {waiting_at:?} is not a parse standing"
                ));
            }
        })
    }
}

#[cfg(test)]
mod tests;
