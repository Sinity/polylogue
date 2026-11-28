//! Completeness is read from the index snapshot of the read: every chain's
//! status as the model judges it, the derivation marks, search's pending
//! entries (the applied stream position of primary mode is not applicable
//! here).
use super::*;
use serde::{Deserialize, Serialize};

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

use polylogue_model::freshness::{
    CompletenessDimension, CompletenessReason, CountedReason, ReadCompleteness,
};

impl Reader<'_> {
    /// [`Reader::chain_census`] of this read's position, computed once for
    /// every reader of the pool at that position: completeness is part of
    /// every reply.
    pub(crate) fn chain_census_at_position(
        &self,
        contracts: &[ReadingContract],
    ) -> Result<std::sync::Arc<ChainCensus>> {
        let position = self.change_position()?;
        if let Some(census) = self.cached_chain_census(&position, contracts) {
            return Ok(census);
        }
        let census = std::sync::Arc::new(self.chain_census(contracts)?);
        *self.pool.census.locked() = Some((position, contracts.to_vec(), census.clone()));
        Ok(census)
    }

    fn cached_chain_census(
        &self,
        position: &polylogue_model::ChangePosition,
        contracts: &[ReadingContract],
    ) -> Option<std::sync::Arc<ChainCensus>> {
        self.pool
            .census
            .locked()
            .as_ref()
            .and_then(|(at, of, census)| {
                (at == position && of.as_slice() == contracts).then(|| census.clone())
            })
    }

    pub fn completeness(&self, contracts: &[ReadingContract]) -> Result<ReadCompleteness> {
        self.read_completeness(contracts, true)
    }

    /// While inputs are catching up, report parsing counts only when the
    /// background reader has already measured this snapshot.
    pub fn completeness_during_catch_up(
        &self,
        contracts: &[ReadingContract],
    ) -> Result<ReadCompleteness> {
        self.read_completeness(contracts, false)
    }

    fn read_completeness(
        &self,
        contracts: &[ReadingContract],
        count: bool,
    ) -> Result<ReadCompleteness> {
        let token = self.scope_token();
        let conn = self.conn()?;
        let mut declared = BTreeMap::new();
        for contract in contracts {
            if contract.origin().is_empty()
                || declared.insert(contract.origin(), contract).is_some()
            {
                return Err(StoreError::internal(
                    "reading contracts need unique nonempty origins",
                ));
            }
            if let ReadingContract::Expected { version, .. } = contract
                && version.is_empty()
            {
                return Err(StoreError::internal(
                    "an expected reading needs its declared version",
                ));
            }
        }
        // Every chain judged by the model's `status`, from its state and the
        // archive's facts; read once per read position.
        let census = if count {
            Some(self.chain_census_at_position(contracts)?)
        } else {
            self.cached_chain_census(&self.change_position()?, contracts)
        };
        // Chains by why they are behind and by origin; and those whose
        // parse failed or was refused, which waiting does not end and which
        // therefore leave parsing complete.
        let mut missing: BTreeMap<(CompletenessReason, &str), u64> = BTreeMap::new();
        let mut stopped: BTreeMap<(CompletenessReason, &str), u64> = BTreeMap::new();
        let mut unknown = false;
        let mut kept_chains = 0_u64;
        let mut raw_only_chains = 0_u64;
        if let Some(census) = &census {
            for ((origin, status), n) in &census.chains {
                use polylogue_model::chain::ChainStatusName as Status;
                let n = *n;
                let add = |into: &mut BTreeMap<_, u64>, reason, n| {
                    *into.entry((reason, origin.as_str())).or_default() += n;
                };
                match declared.get(origin.as_str()).copied() {
                    None => {
                        unknown = true;
                        add(&mut missing, CompletenessReason::MissingDeclaration, n);
                    }
                    Some(ReadingContract::Kept { .. }) => kept_chains += n,
                    // Bytes kept with no parser are as far as they can be: they
                    // are counted, and leave the index complete.
                    Some(ReadingContract::RawOnly { .. }) => raw_only_chains += n,
                    Some(ReadingContract::Expected { .. }) => match status {
                        Status::Done | Status::Waiting | Status::Repeated => {}
                        Status::NoParser => raw_only_chains += n,
                        Status::Refused => add(&mut stopped, CompletenessReason::Refused, n),
                        Status::Failed => add(&mut stopped, CompletenessReason::ParseFailed, n),
                        Status::Parsing => {
                            for (reason, n) in census.behind.get(origin).into_iter().flatten() {
                                add(&mut missing, *reason, *n);
                            }
                        }
                    },
                }
            }
        }
        let counted = |reasons: BTreeMap<(CompletenessReason, &str), u64>| -> Vec<CountedReason> {
            reasons
                .into_iter()
                .map(|((reason, origin), n)| CountedReason::inputs(reason, origin, n))
                .collect()
        };
        let parsing = if census.is_none() {
            CompletenessDimension::Unknown(vec![CountedReason::new(
                CompletenessReason::ParsingCoveragePending,
            )])
        } else if missing.is_empty() {
            CompletenessDimension::Complete
        } else if unknown {
            CompletenessDimension::Unknown(counted(missing))
        } else {
            CompletenessDimension::Incomplete(counted(missing))
        };
        // The marks the derive step works from, and the versions it last
        // derived at; read in constant time.
        let derivation = if crate::summary::caught_up(conn)? {
            CompletenessDimension::Complete
        } else {
            CompletenessDimension::Incomplete(vec![CountedReason::new(
                CompletenessReason::DerivationBehind,
            )])
        };
        let mut search_missing = Vec::new();
        if self.pool.reads_build() {
            search_missing.push(CountedReason::new(CompletenessReason::IndexBuilding));
        }
        // Only whether texts wait: counting them is status's, which bounds
        // its scan; this runs for every reply.
        if conn.exists("SELECT EXISTS(SELECT 1 FROM search_pending)", [])? {
            search_missing.push(CountedReason::new(CompletenessReason::SearchPending));
        }
        let search = if search_missing.is_empty() {
            CompletenessDimension::Complete
        } else {
            CompletenessDimension::Incomplete(search_missing)
        };
        token.check()?;
        Ok(ReadCompleteness {
            parsing,
            derivation,
            search,
            confirmed_stream: CompletenessDimension::NotApplicable,
            kept_chains: census.as_ref().map(|_| kept_chains),
            raw_only_chains: census.as_ref().map(|_| raw_only_chains),
            failed_parses: counted(stopped),
        })
    }
}

#[cfg(test)]
mod tests;
