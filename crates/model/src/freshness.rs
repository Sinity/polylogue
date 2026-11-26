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
    /// Some stored inputs are of an origin no reading declaration names.
    MissingDeclaration,
    MissingCheckpoint,
    ParseVersionChanged,
    ParserBehind,
    /// Some stored inputs failed their origin's shape check.
    Refused,
    /// Some stored inputs' bytes could not be parsed under the current parser.
    ParseFailed,
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
    /// Whether the daemon ends it by itself, so that waiting for it ends.
    /// An undeclared origin and a failed or refused parse stand until the
    /// bytes, the parser or the declarations change.
    pub fn waiting_ends(self) -> bool {
        !matches!(
            self,
            Self::MissingDeclaration | Self::Refused | Self::ParseFailed
        )
    }

    /// What it says: of stored inputs, in the singular and the plural, or
    /// of the index as a whole.
    fn words(self) -> Words {
        Words::Inputs(match self {
            Self::MissingDeclaration => (
                "has no recorded reading declaration",
                "have no recorded reading declaration",
            ),
            Self::MissingCheckpoint => ("has not been parsed yet", "have not been parsed yet"),
            Self::ParseVersionChanged => ("awaits the current parser", "await the current parser"),
            Self::ParserBehind => ("has bytes not parsed yet", "have bytes not parsed yet"),
            Self::Refused => ("was refused by its parser", "were refused by their parser"),
            Self::ParseFailed => ("failed to parse", "failed to parse"),
            Self::DerivationBehind => {
                return Words::Index("session figures are still being computed");
            }
            Self::SearchPending => return Words::Index("search text is pending"),
            Self::IndexBuilding => return Words::Index("the index is being built"),
            Self::StreamNotApplied => {
                return Words::Index("the confirmed stream has not been applied");
            }
            Self::StreamBehind => return Words::Index("the index is behind the confirmed stream"),
        })
    }
}

enum Words {
    Inputs((&'static str, &'static str)),
    Index(&'static str),
}

/// One reason a reply is not up to date: how many things it covers (stored
/// inputs, or texts for search) and, for stored inputs, their origin, and
/// whether waiting ends it.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
pub struct CountedReason {
    pub reason: CompletenessReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u64>,
    /// Whether the daemon ends it by itself, so that waiting for it ends.
    pub waiting_ends: bool,
}

impl CountedReason {
    /// A reason with no count or origin: the index being built, figures
    /// being computed.
    pub fn new(reason: CompletenessReason) -> Self {
        Self {
            reason,
            origin: None,
            count: None,
            waiting_ends: reason.waiting_ends(),
        }
    }

    /// `count` stored inputs of `origin` held back for `reason`.
    pub fn inputs(reason: CompletenessReason, origin: &str, count: u64) -> Self {
        Self {
            origin: Some(origin.to_owned()),
            count: Some(count),
            ..Self::new(reason)
        }
    }

    /// `count` texts not yet in the search index.
    pub fn search_pending(count: Option<u64>) -> Self {
        Self {
            count,
            ..Self::new(CompletenessReason::SearchPending)
        }
    }

    /// The reason in words, as the status view says it: how many, and of
    /// which origin.
    pub fn text(&self) -> String {
        match (self.reason.words(), self.count) {
            (Words::Inputs((one, many)), count) => match (count, &self.origin) {
                (Some(1), Some(origin)) => format!("1 {origin} input {one}"),
                (Some(n), Some(origin)) => format!("{n} {origin} inputs {many}"),
                (Some(1), None) => format!("1 stored input {one}"),
                (Some(n), None) => format!("{n} stored inputs {many}"),
                (None, _) => format!("some stored inputs {many}"),
            },
            (Words::Index(_), Some(1)) if self.reason == CompletenessReason::SearchPending => {
                "1 text awaits search".to_owned()
            }
            (Words::Index(_), Some(n)) if self.reason == CompletenessReason::SearchPending => {
                format!("{n} texts await search")
            }
            (Words::Index(words), _) => words.to_owned(),
        }
    }

    /// The reason as a reply's note: its words, and where what it counts
    /// is shown.
    pub fn note(&self) -> String {
        match self.reason.words() {
            // Stored inputs are no query unit yet; status counts them.
            Words::Inputs(_) => format!(
                "{}; status counts them, and no query lists stored inputs yet",
                self.text()
            ),
            Words::Index(_) => self.text(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "state", content = "reasons", rename_all = "snake_case")]
pub enum CompletenessDimension {
    Complete,
    Incomplete(Vec<CountedReason>),
    Unknown(Vec<CountedReason>),
    NotApplicable,
}
impl CompletenessDimension {
    fn current(&self) -> bool {
        matches!(self, Self::Complete | Self::NotApplicable)
    }
    /// The reasons it is not current.
    pub fn reasons(&self) -> &[CountedReason] {
        match self {
            Self::Incomplete(reasons) | Self::Unknown(reasons) => reasons,
            Self::Complete | Self::NotApplicable => &[],
        }
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
    /// Chains whose parse failed or was refused, by origin. Waiting does not
    /// end them, so a read is up to date beside them; status reports them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failed_parses: Vec<CountedReason>,
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
            failed_parses: Vec::new(),
        }
    }
    fn dimensions(&self) -> [&CompletenessDimension; 4] {
        [
            &self.parsing,
            &self.derivation,
            &self.search,
            &self.confirmed_stream,
        ]
    }
    pub fn caught_up(&self) -> bool {
        self.dimensions()
            .into_iter()
            .all(CompletenessDimension::current)
    }
    /// Whether work the daemon finishes by itself, without its loop, is
    /// still to come: derived figures, search text. Parse states the loop
    /// cannot finish (no parser, a failed or refused parse) are not such
    /// work.
    pub fn background_work_pending(&self) -> bool {
        self.derivation != CompletenessDimension::Complete
            || self
                .search
                .reasons()
                .iter()
                .any(|reason| reason.reason == CompletenessReason::SearchPending)
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
    /// Whether the index is catching up, so that waiting brings it up to
    /// date: it is behind, and only for reasons the daemon ends by itself.
    pub fn catching_up(&self) -> bool {
        !self.up_to_date()
            && self
                .local
                .dimensions()
                .into_iter()
                .flat_map(CompletenessDimension::reasons)
                .all(|reason| reason.waiting_ends)
            && match &self.sinex {
                IndexFreshness::Standalone => true,
                IndexFreshness::Primary { state, .. } => {
                    matches!(state, FreshnessState::Current | FreshnessState::Behind)
                }
            }
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
    /// Why the reply is not up to date, one note for each reason.
    pub fn notes(&self) -> Vec<String> {
        self.words(CountedReason::note, false)
    }

    /// What status says of the index: each reason it is not up to date, and
    /// each standing failed or refused parse.
    pub fn status_notes(&self) -> Vec<String> {
        self.words(CountedReason::text, true)
    }

    fn words(&self, say: fn(&CountedReason) -> String, failed_parses: bool) -> Vec<String> {
        let reasons = self
            .local
            .dimensions()
            .into_iter()
            .flat_map(CompletenessDimension::reasons);
        let failed = failed_parses
            .then_some(self.local.failed_parses.as_slice())
            .unwrap_or_default();
        let mut notes: Vec<String> = reasons.chain(failed).map(say).collect();
        if let Some((_, note)) = self.sinex_note() {
            notes.push(note.to_owned());
        }
        notes.sort();
        notes.dedup();
        notes
    }
}

#[cfg(test)]
mod tests;
