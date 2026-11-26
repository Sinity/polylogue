//! What a read cache knows about Sinex. A recent observation describes a
//! completed position; it never says where Sinex's latest position is now.
use serde::{Deserialize, Serialize};

/// The Sinex status poll normally runs every second. Reads keep the age of
/// the observation and stop presenting its lag after five seconds.
pub const SINEX_OBSERVATION_MAX_AGE_SECONDS: f64 = 5.;

stored_names! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum FreshnessState {
        Current => "current",
        Behind => "behind",
        Unknown => "unknown",
        Unavailable => "unavailable",
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum IndexFreshness {
    Standalone,
    Primary {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scope: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        applied_position: Option<i64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sinex_position: Option<i64>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "observation_age"
        )]
        sinex_position_seconds_ago: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reachable: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lag_positions: Option<i64>,
        state: FreshnessState,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
}

// Tagged enums buffer arbitrary-precision JSON decimals as Number maps.
// Decode that representation before converting the finite observation age.
fn observation_age<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<f64>, D::Error> {
    Option::<serde_json::Number>::deserialize(deserializer)?
        .map(|number| {
            number
                .as_f64()
                .ok_or_else(|| serde::de::Error::custom("observation age must be finite"))
        })
        .transpose()
}

impl IndexFreshness {
    pub fn primary(
        applied_position: Option<i64>,
        sinex_position: Option<i64>,
        age: Option<f64>,
        reachable: Option<bool>,
        error: Option<String>,
    ) -> Self {
        let age = age.filter(|age| age.is_finite() && *age >= 0.);
        let lag_positions = if reachable == Some(true)
            && age.is_some_and(|age| age <= SINEX_OBSERVATION_MAX_AGE_SECONDS)
        {
            applied_position
                .zip(sinex_position)
                .filter(|(applied, sinex)| *applied >= 0 && sinex >= applied)
                .map(|(applied, sinex)| sinex - applied)
        } else {
            None
        };
        let state = match (reachable, lag_positions) {
            (Some(false), _) => FreshnessState::Unavailable,
            (_, Some(0)) => FreshnessState::Current,
            (_, Some(_)) => FreshnessState::Behind,
            _ => FreshnessState::Unknown,
        };
        Self::Primary {
            scope: None,
            applied_position,
            sinex_position,
            sinex_position_seconds_ago: age,
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
                state: FreshnessState::Behind
                    | FreshnessState::Unknown
                    | FreshnessState::Unavailable,
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
    ParseVersionChanged,
    ParserBehind,
    /// Some stored inputs failed their origin's shape check.
    Refused,
    /// Some stored inputs' bytes could not be parsed under the current parser.
    ParseFailed,
    KnownRawOnly,
    DerivationBehind,
    SearchPending,
    IndexBuilding,
    /// Primary mode: the index has not yet applied the confirmed stream.
    StreamNotApplied,
    /// Primary mode: the index is behind Sinex's latest position; the lag is
    /// in the read's position report.
    StreamBehind,
}

impl CompletenessReason {
    /// The reason in words, as notes and the status view say it.
    pub fn text(self) -> &'static str {
        match self {
            CompletenessReason::MissingDeclaration => {
                "some stored inputs have no recorded reading declaration"
            }
            CompletenessReason::MissingCheckpoint => "some stored inputs have not been parsed",
            CompletenessReason::ParseVersionChanged => {
                "some stored inputs await the current parser"
            }
            CompletenessReason::ParserBehind => "parsing is behind the stored bytes",
            CompletenessReason::Refused => "some stored inputs were refused by their parser",
            CompletenessReason::ParseFailed => "some stored inputs failed to parse",
            CompletenessReason::KnownRawOnly => "some stored inputs have no parser",
            CompletenessReason::DerivationBehind => "session figures are still being computed",
            CompletenessReason::SearchPending => "search text is pending",
            CompletenessReason::IndexBuilding => "the index is being built",
            CompletenessReason::StreamNotApplied => "the confirmed stream has not been applied",
            CompletenessReason::StreamBehind => "the index is behind the confirmed stream",
        }
    }
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
    pub confirmed_stream: CompletenessDimension,
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
            confirmed_stream: CompletenessDimension::NotApplicable,
            kept_chains: 0,
            raw_only_chains: 0,
        }
    }
    pub fn caught_up(&self) -> bool {
        self.parsing.current()
            && self.derivation.current()
            && self.search.current()
            && self.confirmed_stream.current()
    }
    /// Whether work the daemon finishes by itself, without its loop, is
    /// still to come: derived figures, search text. Parse states the loop
    /// cannot finish (no parser, missing earlier bytes) are not such work.
    pub fn background_work_pending(&self) -> bool {
        self.derivation != CompletenessDimension::Complete
            || matches!(&self.search, CompletenessDimension::Incomplete(reasons)
                if reasons.contains(&CompletenessReason::SearchPending))
    }
}

/// Freshness of one answer, from that answer's index snapshot and the cached
/// Sinex observation. Local parser/search work also matters in standalone mode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadFreshness {
    pub local: ReadCompleteness,
    pub sinex: IndexFreshness,
}
impl ReadFreshness {
    pub fn up_to_date(&self) -> bool {
        self.local.caught_up() && !self.sinex.degraded()
    }
    /// The note for a degraded Sinex position, with the state it describes.
    pub fn sinex_note(&self) -> Option<(FreshnessState, &'static str)> {
        if !self.sinex.degraded() {
            return None;
        }
        Some(match &self.sinex {
            IndexFreshness::Primary {
                state: FreshnessState::Behind,
                ..
            } => (
                FreshnessState::Behind,
                "the index is behind the observed Sinex position",
            ),
            IndexFreshness::Primary {
                state: FreshnessState::Unavailable,
                ..
            } => (FreshnessState::Unavailable, "Sinex is unavailable"),
            _ => (FreshnessState::Unknown, "the Sinex position is unknown"),
        })
    }
    pub fn notes(&self) -> Vec<&'static str> {
        let mut notes = std::collections::BTreeSet::new();
        for dimension in [
            &self.local.parsing,
            &self.local.derivation,
            &self.local.search,
            &self.local.confirmed_stream,
        ] {
            if let CompletenessDimension::Incomplete(reasons)
            | CompletenessDimension::Unknown(reasons) = dimension
            {
                for reason in reasons {
                    notes.insert(reason.text());
                }
            }
        }
        if let Some((_, note)) = self.sinex_note() {
            notes.insert(note);
        }
        notes.into_iter().collect()
    }
}

#[cfg(test)]
mod tests;
