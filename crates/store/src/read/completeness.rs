//! Completeness is proven from the same source and index snapshot as the read.
use super::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::ops::ControlFlow;

/// Supplied by the owning native-origin registry, never inferred from a stored cursor.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletenessReason {
    MissingDeclaration,
    MissingCheckpoint,
    ReadingVersionChanged,
    ParserBehind,
    MissingSourcePrefix,
    KnownRawOnly,
    DerivationBehind,
    SearchPending,
    FirstBuild,
    /// Primary mode: the index has not yet applied the confirmed stream.
    StreamNotApplied,
    /// Primary mode: the index is behind Sinex's commit position; the lag is
    /// in the read's position report.
    StreamBehind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "reasons", rename_all = "snake_case")]
pub enum CompletenessDimension {
    Complete,
    Incomplete(Vec<CompletenessReason>),
    Unknown(Vec<CompletenessReason>),
    NotApplicable,
}
impl CompletenessDimension {
    fn current(&self) -> bool {
        matches!(self, Self::Complete | Self::NotApplicable)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadCompleteness {
    pub parsing: CompletenessDimension,
    pub derivation: CompletenessDimension,
    pub search: CompletenessDimension,
    pub projection: CompletenessDimension,
    /// Deliberately retained without promising parsed products.
    pub kept_chains: u64,
    /// Known nonparsed chains, independently of retained raw-byte coverage.
    pub raw_only_chains: u64,
}
impl ReadCompleteness {
    pub fn caught_up(&self) -> bool {
        self.parsing.current()
            && self.derivation.current()
            && self.search.current()
            && self.projection.current()
    }
    pub fn degraded(&self) -> bool {
        !self.caught_up()
    }
}

impl Reader<'_> {
    pub fn completeness(&self, contracts: &[ReadingContract]) -> Result<ReadCompleteness> {
        let token = self.scope_token();
        let conn = self.conn()?;
        let mut declared = BTreeMap::new();
        let mut versions = Vec::new();
        for contract in contracts {
            if contract.origin().is_empty()
                || declared.insert(contract.origin(), contract).is_some()
            {
                return Err(StoreError::Invalid(
                    "reading contracts need unique nonempty origins".into(),
                ));
            }
            if let ReadingContract::Expected { origin, version } = contract {
                if version.is_empty() {
                    return Err(StoreError::Invalid(
                        "an expected reading needs its declared version".into(),
                    ));
                }
                versions.push((origin.clone(), version.clone()));
            }
        }
        let mut missing = BTreeSet::new();
        let mut unknown = false;
        let mut kept_chains = 0_u64;
        let mut raw_only_chains = 0_u64;
        self.visit_chain_states(
            &versions,
            true,
            super::behind_index::StoppedChains::Settled,
            &mut |state| {
                match declared.get(state.origin.as_str()).copied() {
                    None => {
                        unknown = true;
                        missing.insert(CompletenessReason::MissingDeclaration);
                    }
                    Some(ReadingContract::Kept { .. }) => {
                        kept_chains += 1;
                    }
                    Some(ReadingContract::RawOnly { .. }) => {
                        raw_only_chains += 1;
                        missing.insert(CompletenessReason::KnownRawOnly);
                    }
                    Some(ReadingContract::Expected { version, .. }) => {
                        let progress = &state.plan.progress;
                        if !progress.parsed {
                            missing.insert(CompletenessReason::MissingCheckpoint);
                        } else if state.version.as_ref() != Some(version) {
                            missing.insert(CompletenessReason::ReadingVersionChanged);
                        } else {
                            use polylogue_model::batch::ParseProgress;
                            match &state.progress {
                                Some(ParseProgress::NoParserYet) => {
                                    raw_only_chains += 1;
                                    missing.insert(CompletenessReason::KnownRawOnly);
                                }
                                Some(ParseProgress::Partial) => {
                                    missing.insert(CompletenessReason::MissingSourcePrefix);
                                }
                                Some(ParseProgress::Waiting { .. }) if progress.waiting => {}
                                Some(ParseProgress::Done)
                                    if progress.parsed_upto >= progress.length => {}
                                _ => {
                                    missing.insert(CompletenessReason::ParserBehind);
                                }
                            }
                        }
                    }
                }
                Ok(ControlFlow::Continue(()))
            },
        )?;
        let parsing = if missing.is_empty() {
            CompletenessDimension::Complete
        } else if unknown {
            CompletenessDimension::Unknown(missing.into_iter().collect())
        } else {
            CompletenessDimension::Incomplete(missing.into_iter().collect())
        };
        // The marks the derive step works from, and the versions it last
        // derived at; read in constant time.
        let derive_behind = !crate::derive::caught_up(conn)?;
        let derivation = if derive_behind {
            CompletenessDimension::Incomplete(vec![CompletenessReason::DerivationBehind])
        } else {
            CompletenessDimension::Complete
        };
        let mut search_missing = Vec::new();
        if crate::schema::building(conn)? {
            search_missing.push(CompletenessReason::FirstBuild);
        }
        if conn.query_row("SELECT EXISTS(SELECT 1 FROM search_pending)", [], |row| {
            row.get::<_, bool>(0)
        })? {
            search_missing.push(CompletenessReason::SearchPending);
        }
        let search = if search_missing.is_empty() {
            CompletenessDimension::Complete
        } else {
            CompletenessDimension::Incomplete(search_missing)
        };
        let projection = if self.pool.archive.is_primary() {
            let state = crate::primary::projection::replay_state(conn)?;
            match (state.applied_position, state.head) {
                (None, _) => {
                    CompletenessDimension::Incomplete(vec![CompletenessReason::StreamNotApplied])
                }
                (Some(applied), Some(head)) if applied >= head => CompletenessDimension::Complete,
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
            projection,
            kept_chains,
            raw_only_chains,
        })
    }
}

#[cfg(test)]
mod tests;
