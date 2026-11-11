//! What a read cache knows about its owner. A recent observation describes a
//! completed position; it is never a synchronous claim about the owner's head.
use serde::{Deserialize, Serialize};

/// The owner status poll normally runs every second. Reads retain the age of
/// the observation and stop presenting its lag after five seconds.
pub const OWNER_OBSERVATION_SECONDS: f64 = 5.;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CurrencyState {
    Current,
    Behind,
    Unknown,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum IndexCurrency {
    Standalone,
    Primary {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scope: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        applied_position: Option<i64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        owner_position: Option<i64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        owner_position_seconds_ago: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reachable: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lag_positions: Option<i64>,
        state: CurrencyState,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
}
impl IndexCurrency {
    pub fn primary(
        applied_position: Option<i64>,
        owner_position: Option<i64>,
        age: Option<f64>,
        reachable: Option<bool>,
        error: Option<String>,
    ) -> Self {
        let age = age.filter(|age| age.is_finite() && *age >= 0.);
        let lag_positions =
            if reachable == Some(true) && age.is_some_and(|age| age <= OWNER_OBSERVATION_SECONDS) {
                applied_position
                    .zip(owner_position)
                    .filter(|(applied, owner)| *applied >= 0 && owner >= applied)
                    .map(|(applied, owner)| owner - applied)
            } else {
                None
            };
        let state = match (reachable, lag_positions) {
            (Some(false), _) => CurrencyState::Unavailable,
            (_, Some(0)) => CurrencyState::Current,
            (_, Some(_)) => CurrencyState::Behind,
            _ => CurrencyState::Unknown,
        };
        Self::Primary {
            scope: None,
            applied_position,
            owner_position,
            owner_position_seconds_ago: age,
            reachable,
            lag_positions,
            state,
            error,
        }
    }
    pub fn degraded(&self) -> bool {
        matches!(
            self,
            Self::Primary {
                state: CurrencyState::Behind | CurrencyState::Unknown | CurrencyState::Unavailable,
                ..
            }
        )
    }
    pub fn with_scope(mut self, applied_scope: Option<String>) -> Self {
        if let Self::Primary { scope, .. } = &mut self {
            *scope = applied_scope;
        }
        self
    }
    pub fn scope(&self) -> Option<&str> {
        match self {
            Self::Standalone => None,
            Self::Primary { scope, .. } => scope.as_deref(),
        }
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
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
    /// No outstanding local work, as in a newly created standalone archive.
    pub const fn empty() -> Self {
        Self {
            parsing: CompletenessDimension::Complete,
            derivation: CompletenessDimension::Complete,
            search: CompletenessDimension::Complete,
            projection: CompletenessDimension::NotApplicable,
            kept_chains: 0,
            raw_only_chains: 0,
        }
    }
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

/// Freshness of one answer, from that answer's index snapshot and the cached
/// owner observation. Local parser/search work also matters in standalone mode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadFreshness {
    pub local: ReadCompleteness,
    pub owner: IndexCurrency,
}
impl ReadFreshness {
    pub fn up_to_date(&self) -> bool {
        self.local.caught_up() && !self.owner.degraded()
    }
    pub fn notes(&self) -> Vec<&'static str> {
        let mut notes = std::collections::BTreeSet::new();
        for dimension in [
            &self.local.parsing,
            &self.local.derivation,
            &self.local.search,
            &self.local.projection,
        ] {
            if let CompletenessDimension::Incomplete(reasons)
            | CompletenessDimension::Unknown(reasons) = dimension
            {
                for reason in reasons {
                    notes.insert(match reason {
                        CompletenessReason::MissingDeclaration => {
                            "some stored inputs have no recorded reading declaration"
                        }
                        CompletenessReason::MissingCheckpoint => {
                            "some stored inputs have not been parsed"
                        }
                        CompletenessReason::ReadingVersionChanged => {
                            "some stored inputs await the current parser"
                        }
                        CompletenessReason::ParserBehind => "parsing is behind the stored bytes",
                        CompletenessReason::MissingSourcePrefix => {
                            "some stored inputs have missing earlier bytes"
                        }
                        CompletenessReason::KnownRawOnly => "some stored inputs have no parser",
                        CompletenessReason::DerivationBehind => "derived values are pending",
                        CompletenessReason::SearchPending => "search text is pending",
                        CompletenessReason::FirstBuild => "the search index is being built",
                        CompletenessReason::StreamNotApplied => {
                            "the confirmed stream has not been applied"
                        }
                        CompletenessReason::StreamBehind => {
                            "the index is behind the confirmed stream"
                        }
                    });
                }
            }
        }
        if self.owner.degraded() {
            notes.insert(match &self.owner {
                IndexCurrency::Primary {
                    state: CurrencyState::Behind,
                    ..
                } => "the index is behind the observed owner position",
                IndexCurrency::Primary {
                    state: CurrencyState::Unavailable,
                    ..
                } => "the owner is unavailable",
                _ => "the owner position is unknown",
            });
        }
        notes.into_iter().collect()
    }
}

#[cfg(test)]
mod tests;
