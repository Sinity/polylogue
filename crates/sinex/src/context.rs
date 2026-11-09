//! One owner-observed temporal and policy snapshot. Resolution performs no I/O
//! except the explicit named-zone callback supplied by the application boundary.
use crate::Sinex;
use serde::Serialize;
use sinex_protocol::PublicContext;
use std::time::{Duration, Instant};

#[derive(Clone, Serialize, schemars::JsonSchema)]
pub struct ResolvedContext {
    #[serde(skip_serializing)]
    #[schemars(skip)]
    pub context_token: String,
    pub settings_identity: String,
    pub owner_revision: u64,
    pub day: query_language::time::DaySetting,
    pub quiet_hours: Option<shared_signals::QuietHours>,
    pub interrupt_budget: sinex_protocol::PublicInterruptBudget,
    pub observed_at: sinex_model::Timestamp,
    pub received_seconds_ago: f64,
}

impl std::fmt::Debug for ResolvedContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedContext")
            .field("settings_identity", &self.settings_identity)
            .field("owner_revision", &self.owner_revision)
            .field("day", &self.day)
            .field("quiet_hours", &self.quiet_hours)
            .field("interrupt_budget", &self.interrupt_budget)
            .field("observed_at", &self.observed_at)
            .field("received_seconds_ago", &self.received_seconds_ago)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ContextFailureKind {
    Missing,
    Pending,
    Unavailable,
    Refused,
    Stale,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, thiserror::Error)]
#[error("{reason}")]
pub struct ContextFailure {
    pub kind: ContextFailureKind,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_known_identity: Option<String>,
}

#[derive(Default, Clone)]
pub(crate) struct Cache {
    reply: Option<PublicContext>,
    received: Option<Instant>,
    attempted: Option<Instant>,
    transport_error: Option<String>,
    last_known_identity: Option<String>,
}

impl Cache {
    pub(crate) fn refresh_due(&self) -> bool {
        self.attempted
            .is_none_or(|at| at.elapsed() >= Duration::from_secs(1))
    }

    pub(crate) fn record(&mut self, result: Result<PublicContext, String>) {
        self.attempted = Some(Instant::now());
        match result {
            Ok(reply) => {
                if let PublicContext::Available {
                    settings_identity,
                    context_token,
                    quiet_hours,
                    interrupt_budget,
                    ..
                } = &reply
                    && valid(
                        settings_identity,
                        context_token,
                        quiet_hours.as_ref(),
                        interrupt_budget,
                    )
                {
                    self.last_known_identity = Some(settings_identity.clone());
                }
                self.reply = Some(reply);
                self.received = Some(Instant::now());
                self.transport_error = None;
            }
            Err(error) => self.transport_error = Some(error),
        }
    }

    fn failure(&self, kind: ContextFailureKind, reason: impl Into<String>) -> ContextFailure {
        ContextFailure {
            kind,
            reason: reason.into(),
            last_known_identity: self.last_known_identity.clone(),
        }
    }

    fn resolve(
        &self,
        resolve_zone: impl FnOnce(&str) -> Result<jiff::tz::TimeZone, String>,
    ) -> Result<ResolvedContext, ContextFailure> {
        if let Some(reason) = &self.transport_error {
            return Err(self.failure(ContextFailureKind::Unavailable, reason.clone()));
        }
        let reply = self.reply.as_ref().ok_or_else(|| {
            self.failure(
                ContextFailureKind::Missing,
                "Sinex public context has not been observed",
            )
        })?;
        let (
            settings_identity,
            owner_revision,
            day,
            quiet_hours,
            interrupt_budget,
            observed_at,
            context_token,
        ) = match reply {
            PublicContext::Pending { reason } => {
                return Err(self.failure(ContextFailureKind::Pending, reason.clone()));
            }
            PublicContext::Unavailable { reason } => {
                return Err(self.failure(ContextFailureKind::Unavailable, reason.clone()));
            }
            PublicContext::Refused { reason } => {
                return Err(self.failure(ContextFailureKind::Refused, reason.clone()));
            }
            PublicContext::Available {
                settings_identity,
                owner_revision,
                day,
                quiet_hours,
                interrupt_budget,
                observed_at,
                context_token,
                ..
            } => (
                settings_identity,
                owner_revision,
                day,
                quiet_hours,
                interrupt_budget,
                observed_at,
                context_token,
            ),
        };
        let age = self
            .received
            .map(|at| at.elapsed().as_secs_f64())
            .ok_or_else(|| {
                self.failure(
                    ContextFailureKind::Missing,
                    "Sinex public context has no observation age",
                )
            })?;
        if age > polylogue_model::currency::OWNER_OBSERVATION_SECONDS {
            return Err(self.failure(
                ContextFailureKind::Stale,
                "Sinex public context observation is stale",
            ));
        }
        if !valid(
            settings_identity,
            context_token,
            quiet_hours.as_ref(),
            interrupt_budget,
        ) {
            return Err(self.failure(
                ContextFailureKind::Invalid,
                "Sinex public context contains invalid policy evidence",
            ));
        }
        let zone = resolve_zone(&day.zone)
            .map_err(|reason| self.failure(ContextFailureKind::Invalid, reason))?;
        let day = day
            .clone()
            .resolve(zone)
            .map_err(|error| self.failure(ContextFailureKind::Invalid, error.to_string()))?;
        Ok(ResolvedContext {
            context_token: context_token.clone(),
            settings_identity: settings_identity.clone(),
            owner_revision: *owner_revision,
            day,
            quiet_hours: *quiet_hours,
            interrupt_budget: interrupt_budget.clone(),
            observed_at: *observed_at,
            received_seconds_ago: age,
        })
    }
}

fn valid(
    identity: &str,
    token: &str,
    quiet: Option<&shared_signals::QuietHours>,
    budget: &sinex_protocol::PublicInterruptBudget,
) -> bool {
    !identity.is_empty()
        && !token.is_empty()
        && budget.window_ms > 0
        && budget.remaining == budget.per_hour.saturating_sub(budget.used)
        && quiet.is_none_or(|hours| hours.from < 1440 && hours.to < 1440)
}

impl Sinex {
    /// Reads the last observed owner context. No network request or local
    /// deployment setting participates in this operation.
    pub fn context(
        &self,
        resolve_zone: impl FnOnce(&str) -> Result<jiff::tz::TimeZone, String>,
    ) -> Result<ResolvedContext, ContextFailure> {
        let context = self
            .health
            .lock()
            .map_err(|_| ContextFailure {
                kind: ContextFailureKind::Unavailable,
                reason: "Sinex public context cache is unavailable".into(),
                last_known_identity: None,
            })?
            .context
            .clone();
        context.resolve(resolve_zone)
    }
}

#[cfg(test)]
mod tests;
