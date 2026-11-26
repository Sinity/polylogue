//! Completeness is read from the index snapshot of the read: every chain's
//! status as the model judges it, the derivation marks, search's pending
//! entries (the applied stream position of primary mode is not applicable
//! here).
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
    /// [`Reader::chain_census`] of this read's position, computed once for
    /// every reader of the pool at that position: completeness is part of
    /// every reply.
    pub(crate) fn chain_census_at_position(
        &self,
        contracts: &[ReadingContract],
    ) -> Result<std::sync::Arc<ChainCensus>> {
        let position = self.change_position()?;
        if let Some((at, of, census)) = self.pool.census.locked().as_ref()
            && *at == position
            && of.as_slice() == contracts
        {
            return Ok(census.clone());
        }
        let census = std::sync::Arc::new(self.chain_census(contracts)?);
        *self.pool.census.locked() = Some((position, contracts.to_vec(), census.clone()));
        Ok(census)
    }

    pub fn completeness(&self, contracts: &[ReadingContract]) -> Result<ReadCompleteness> {
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
        let mut missing = BTreeSet::new();
        let mut unknown = false;
        let mut kept_chains = 0_u64;
        let mut raw_only_chains = 0_u64;
        // Every chain judged by the model's `status`, from its state and the
        // archive's facts; read once per read position.
        let census = self.chain_census_at_position(contracts)?;
        for ((origin, status), n) in &census.chains {
            use polylogue_model::chain::ChainStatusName as Status;
            let n = *n;
            match declared.get(origin.as_str()).copied() {
                None => {
                    unknown = true;
                    missing.insert(CompletenessReason::MissingDeclaration);
                }
                Some(ReadingContract::Kept { .. }) => kept_chains += n,
                // Bytes kept with no parser are as far as they can be: they
                // are counted, and leave the index complete.
                Some(ReadingContract::RawOnly { .. }) => raw_only_chains += n,
                Some(ReadingContract::Expected { .. }) => match status {
                    Status::Done | Status::Waiting | Status::Repeated => {}
                    Status::NoParser => raw_only_chains += n,
                    Status::Refused => {
                        missing.insert(CompletenessReason::Refused);
                    }
                    Status::Failed => {
                        missing.insert(CompletenessReason::ParseFailed);
                    }
                    Status::Parsing => {
                        missing.extend(census.behind.get(origin).into_iter().flatten().copied());
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
        let derivation = if crate::summary::caught_up(conn)? {
            CompletenessDimension::Complete
        } else {
            CompletenessDimension::Incomplete(vec![CompletenessReason::DerivationBehind])
        };
        let mut search_missing = Vec::new();
        if self.pool.reads_build() {
            search_missing.push(CompletenessReason::IndexBuilding);
        }
        if conn.exists("SELECT EXISTS(SELECT 1 FROM search_pending)", [])? {
            search_missing.push(CompletenessReason::SearchPending);
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
            kept_chains,
            raw_only_chains,
        })
    }
}

#[cfg(test)]
mod tests;
