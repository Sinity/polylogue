//! Completeness is read from the index snapshot of the read: the chain
//! counts the writer keeps, the derivation marks, search's pending entries
//! and, in primary mode, the applied stream position.
use super::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Supplied by the origin registry, never inferred from a stored cursor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReadingContract {
    Expected {
        origin: String,
        version: String,
    },
    Kept {
        origin: String,
    },
    /// Captured provider bytes whose declaration intentionally has no parser.
    RawOnly {
        origin: String,
    },
}
impl ReadingContract {
    pub fn origin(&self) -> &str {
        match self {
            Self::Expected { origin, .. } | Self::Kept { origin } | Self::RawOnly { origin } => {
                origin
            }
        }
    }
}

use polylogue_model::freshness::{CompletenessDimension, CompletenessReason, ReadCompleteness};

impl Reader<'_> {
    pub fn completeness(&self, contracts: &[ReadingContract]) -> Result<ReadCompleteness> {
        let token = self.scope_token();
        let conn = self.conn()?;
        let mut declared = BTreeMap::new();
        for contract in contracts {
            if contract.origin().is_empty()
                || declared.insert(contract.origin(), contract).is_some()
            {
                return Err(StoreError::Invalid(
                    "reading contracts need unique nonempty origins".into(),
                ));
            }
            if let ReadingContract::Expected { version, .. } = contract
                && version.is_empty()
            {
                return Err(StoreError::Invalid(
                    "an expected reading needs its declared version".into(),
                ));
            }
        }
        let mut missing = BTreeSet::new();
        let mut unknown = false;
        let mut kept_chains = 0_u64;
        let mut raw_only_chains = 0_u64;
        // Chains counted by origin, reading version and state, which the
        // writer keeps with each change (`crate::chain_progress`).
        for (origin, reading_version, state, n) in crate::chain_progress::counts(conn)? {
            use crate::chain_progress::State;
            match declared.get(origin.as_str()).copied() {
                None => {
                    unknown = true;
                    missing.insert(CompletenessReason::MissingDeclaration);
                }
                Some(ReadingContract::Kept { .. }) => kept_chains += n,
                Some(ReadingContract::RawOnly { .. }) => {
                    raw_only_chains += n;
                    missing.insert(CompletenessReason::KnownRawOnly);
                }
                Some(ReadingContract::Expected { version, .. }) => match state {
                    State::NoCheckpoint => {
                        missing.insert(CompletenessReason::MissingCheckpoint);
                    }
                    _ if &reading_version != version => {
                        missing.insert(CompletenessReason::ReadingVersionChanged);
                    }
                    State::Current => {}
                    State::NoParser => {
                        raw_only_chains += n;
                        missing.insert(CompletenessReason::KnownRawOnly);
                    }
                    State::Partial => {
                        missing.insert(CompletenessReason::MissingSourcePrefix);
                    }
                    State::Behind => {
                        missing.insert(CompletenessReason::ParserBehind);
                    }
                },
            }
        }
        let parsing = if missing.is_empty() {
            CompletenessDimension::Complete
        } else if unknown {
            CompletenessDimension::Unknown(missing.into_iter().collect())
        } else {
            CompletenessDimension::Incomplete(missing.into_iter().collect())
        };
        // The marks the derive step works from, and the versions it last
        // derived at; read in constant time.
        let derivation = if crate::derive::caught_up(conn)? {
            CompletenessDimension::Complete
        } else {
            CompletenessDimension::Incomplete(vec![CompletenessReason::DerivationBehind])
        };
        let mut search_missing = Vec::new();
        if self.pool.reads_build() {
            search_missing.push(CompletenessReason::FirstBuild);
        }
        if conn.exists("SELECT EXISTS(SELECT 1 FROM search_pending)", [])? {
            search_missing.push(CompletenessReason::SearchPending);
        }
        let search = if search_missing.is_empty() {
            CompletenessDimension::Complete
        } else {
            CompletenessDimension::Incomplete(search_missing)
        };
        let confirmed_stream = if self.pool.archive.is_primary() {
            let state = crate::primary::confirmed::replay_state(conn)?;
            match (state.applied_position, state.latest_position) {
                (None, _) => {
                    CompletenessDimension::Incomplete(vec![CompletenessReason::StreamNotApplied])
                }
                (Some(applied), Some(latest_position)) if applied >= latest_position => {
                    CompletenessDimension::Complete
                }
                (Some(_), Some(_)) => {
                    CompletenessDimension::Incomplete(vec![CompletenessReason::StreamBehind])
                }
                (Some(_), None) => {
                    CompletenessDimension::Unknown(vec![CompletenessReason::StreamBehind])
                }
            }
        } else {
            CompletenessDimension::NotApplicable
        };
        token.check()?;
        Ok(ReadCompleteness {
            parsing,
            derivation,
            search,
            confirmed_stream,
            kept_chains,
            raw_only_chains,
        })
    }
}

#[cfg(test)]
mod tests;
