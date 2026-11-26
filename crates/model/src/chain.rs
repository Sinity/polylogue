//! A chain's state in the index, what the archive says of it, and the one
//! answer to whether it is due and how it is shown. The index stores the
//! state (`parse_state`); the facts are read from the archive and never
//! copied into the index, so no reader judges a chain from a second copy.

use crate::{ChainFailure, ChainFailureKind, Json};
use serde::{Deserialize, Serialize};

/// One chain of one input, as the archive names it: the input's ID and the
/// number of the revision that started the chain.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ChainKey {
    pub input: String,
    pub chain: i64,
}

/// What a parser carries from its last complete record to the next one,
/// stored with the chain's state at the byte it stands after. `None` is a
/// parser that carries nothing.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(transparent)]
pub struct ParserState(pub Option<Json>);

/// How a chain's records ended so far: the three outcome counts, which sum
/// to the records read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Outcomes {
    pub used: u64,
    pub skipped: u64,
    pub unreadable: u64,
}

impl Outcomes {
    pub fn records(&self) -> u64 {
        self.used + self.skipped + self.unreadable
    }

    /// The counts of `counts` added to these.
    pub fn plus(self, counts: &crate::RecordCounts) -> Option<Self> {
        let sum = |map: &std::collections::BTreeMap<String, u64>| -> Option<u64> {
            map.values().try_fold(0u64, |a, b| a.checked_add(*b))
        };
        Some(Self {
            used: self.used.checked_add(sum(&counts.used)?)?,
            skipped: self.skipped.checked_add(sum(&counts.skipped)?)?,
            unreadable: self.unreadable.checked_add(sum(&counts.unreadable)?)?,
        })
    }
}

/// A chain's state in the index: one row of `parse_state`, or none.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ChainState {
    /// No row: never parsed, or its parse left because a newer chain of
    /// its input repeats it.
    #[default]
    Unparsed,
    /// The parse holds every complete record within `parsed_to` bytes
    /// under `version`. `seen_to` is the stored length the parser saw; when
    /// it is past `parsed_to`, a record is still being written there.
    Parsed {
        version: String,
        parsed_to: u64,
        seen_to: u64,
        resume: Option<ParserState>,
        counts: Outcomes,
    },
    /// The parser could not go on at `parsed_to` over the `seen_to` stored
    /// bytes; the rows of the records before it are kept. Only the chain's
    /// bytes under this version fail it.
    Failed {
        version: String,
        parsed_to: u64,
        seen_to: u64,
        failure: ChainFailure,
        counts: Outcomes,
    },
}

impl ChainState {
    /// The byte after the last complete record the index holds.
    pub fn parsed_to(&self) -> u64 {
        match self {
            Self::Unparsed => 0,
            Self::Parsed { parsed_to, .. } | Self::Failed { parsed_to, .. } => *parsed_to,
        }
    }

    /// The parser version the state is of.
    pub fn version(&self) -> Option<&str> {
        match self {
            Self::Unparsed => None,
            Self::Parsed { version, .. } | Self::Failed { version, .. } => Some(version),
        }
    }

    /// The stored length the parser saw.
    pub fn seen_to(&self) -> u64 {
        match self {
            Self::Unparsed => 0,
            Self::Parsed { seen_to, .. } | Self::Failed { seen_to, .. } => *seen_to,
        }
    }

    pub fn counts(&self) -> Outcomes {
        match self {
            Self::Unparsed => Outcomes::default(),
            Self::Parsed { counts, .. } | Self::Failed { counts, .. } => *counts,
        }
    }

    /// The parser's carried state at `parsed_to`, when the chain can go on
    /// from there.
    pub fn resume(&self) -> Option<&ParserState> {
        match self {
            Self::Parsed { resume, .. } => resume.as_ref(),
            _ => None,
        }
    }
}

/// What the archive says of a chain, and what the origin's declaration
/// says of its parser. Read from `archive.db` and the origin registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainFacts {
    /// The chain's stored length.
    pub stored: u64,
    /// A newer chain of the input known, from the agreement the archive
    /// records, to begin with all of this one.
    pub repeated_by: Option<i64>,
    /// Records ending at or before this byte are a newer chain's: the most
    /// any newer chain of the input is known to agree with this one.
    pub own_from: u64,
    /// The parser version of the chain's origin; `None`: it has no parser.
    pub parser: Option<String>,
    /// An import or export member: the end of its bytes ends its last
    /// record.
    pub finished: bool,
}

/// What a chain needs, from its state and the archive's facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Due {
    No,
    /// Parse on from the stored state at `from`.
    Extend {
        from: u64,
    },
    /// Parse from the first byte and replace the parse whole.
    Whole,
}

/// The one answer to whether a chain is due. A chain is read when its
/// stored length or its parser's version differs from what its state
/// names, and for no other cause. A repeated chain is never read: the
/// transaction that gives the newer chain its rows removes its parse.
pub fn due(state: &ChainState, facts: &ChainFacts) -> Due {
    if facts.repeated_by.is_some() {
        return Due::No;
    }
    let Some(parser) = facts.parser.as_deref() else {
        return Due::No;
    };
    match state {
        ChainState::Unparsed if facts.stored == 0 => Due::No,
        ChainState::Unparsed => Due::Whole,
        ChainState::Parsed { version, .. } | ChainState::Failed { version, .. }
            if version != parser =>
        {
            Due::Whole
        }
        ChainState::Parsed { seen_to, .. } | ChainState::Failed { seen_to, .. }
            if *seen_to == facts.stored =>
        {
            Due::No
        }
        // A chain newer ones agree with is read whole, so its rows are those
        // a whole parse holds less the newer chains' part.
        ChainState::Parsed {
            parsed_to,
            resume: Some(_),
            ..
        } if *parsed_to <= facts.stored && facts.own_from == 0 => Due::Extend { from: *parsed_to },
        _ => Due::Whole,
    }
}

/// How a chain is shown in `status` and in completeness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainStatus {
    /// Due a parse.
    Parsing,
    /// A newer chain of its input repeats it.
    Repeated,
    /// Its origin has no parser.
    NoParser,
    Done,
    /// A record is still being written past the last complete one.
    Waiting {
        length: u64,
    },
    /// The input failed its origin's shape check.
    Refused {
        reason: String,
    },
    Failed {
        error: String,
    },
}

stored_names! {
    /// The words of [`ChainStatus`], as status and the CLI show them.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum ChainStatusName {
        Parsing => "parsing",
        Repeated => "repeated",
        NoParser => "no_parser",
        Done => "done",
        Waiting => "waiting",
        Refused => "refused",
        Failed => "failed",
    }
}

impl ChainStatus {
    pub fn name(&self) -> ChainStatusName {
        match self {
            Self::Parsing => ChainStatusName::Parsing,
            Self::Repeated => ChainStatusName::Repeated,
            Self::NoParser => ChainStatusName::NoParser,
            Self::Done => ChainStatusName::Done,
            Self::Waiting { .. } => ChainStatusName::Waiting,
            Self::Refused { .. } => ChainStatusName::Refused,
            Self::Failed { .. } => ChainStatusName::Failed,
        }
    }
}

/// How a chain is shown: due is `parsing` whatever its state; otherwise
/// its state at the stored length.
pub fn status(state: &ChainState, facts: &ChainFacts) -> ChainStatus {
    if facts.repeated_by.is_some() {
        return ChainStatus::Repeated;
    }
    if facts.parser.is_none() {
        return ChainStatus::NoParser;
    }
    if due(state, facts) != Due::No {
        return ChainStatus::Parsing;
    }
    match state {
        ChainState::Unparsed => ChainStatus::Done,
        ChainState::Parsed {
            parsed_to, seen_to, ..
        } if parsed_to < seen_to => ChainStatus::Waiting { length: *seen_to },
        ChainState::Parsed { .. } => ChainStatus::Done,
        ChainState::Failed { failure, .. } if failure.kind == ChainFailureKind::Refused => {
            ChainStatus::Refused {
                reason: failure.detail.clone(),
            }
        }
        ChainState::Failed { failure, .. } => ChainStatus::Failed {
            error: failure.detail.clone(),
        },
    }
}

#[cfg(test)]
mod tests;
