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

#[cfg(test)]
mod tests;
