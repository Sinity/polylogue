//! One accepting invocation, extended with a canonical evaluation context when needed.
use crate::daemon::{Response, service::error};
use polylogue_model::{
    Timestamp,
    authorship::{Authorship, Channel},
    context::{Owner, Provenance},
    user::{UserRequest, UserWrite},
};
use query_language::time::{Clock, DaySetting};
use std::collections::BTreeMap;
#[cfg(feature = "sinex")]
use std::sync::Arc;

/// Clock-free writes capture this once without requiring an owner's logical day.
#[derive(Debug, Clone)]
pub struct Invocation {
    pub at: Timestamp,
    pub machine: String,
    pub authorship: Authorship,
}
impl Invocation {
    pub fn capture(
        metadata: &polylogue_model::user::RequestMetadata,
        authorship: &Authorship,
    ) -> Self {
        Self {
            at: metadata.now(),
            machine: metadata.machine().into(),
            authorship: authorship.clone(),
        }
    }
    pub fn request(&self, id: String, write: UserWrite) -> UserRequest {
        UserRequest {
            id,
            machine: self.machine.clone(),
            authorship: self.authorship.clone(),
            at: self.at,
            write,
        }
    }
}

/// The private policy is never serialized, rendered, or included in a schema.
#[derive(Clone)]
pub struct Context {
    pub ran_at: Timestamp,
    pub clock: Clock,
    pub machine: String,
    pub authorship: Authorship,
    owner: Owner,
    #[cfg(feature = "sinex")]
    policy: Option<Arc<polylogue_sinex::ResolvedContext>>,
}
impl std::fmt::Debug for Context {
    fn fmt(&self, format: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        format
            .debug_struct("Context")
            .field("provenance", &self.provenance())
            .finish()
    }
}
#[cfg(feature = "sinex")]
pub struct Attention<'a> {
    pub quiet_hours: Option<shared_signals::QuietHours>,
    pub budget: &'a polylogue_sinex::PublicInterruptBudget,
}
impl Context {
    pub fn standalone(invocation: Invocation, day: DaySetting) -> Self {
        Self {
            ran_at: invocation.at,
            clock: Clock::new(invocation.at.into(), day),
            machine: invocation.machine,
            authorship: invocation.authorship,
            owner: Owner::Standalone,
            #[cfg(feature = "sinex")]
            policy: None,
        }
    }
    #[cfg(feature = "sinex")]
    pub fn primary(
        invocation: Invocation,
        policy: polylogue_sinex::ResolvedContext,
    ) -> Result<Self, Response> {
        let observed_at =
            Timestamp::from_unix_ns(policy.observed_at.unix_ns()).ok_or_else(|| {
                error(
                    "primary_context_invalid",
                    "the owner observation time is outside the supported range",
                )
            })?;
        let owner = Owner::Primary {
            settings_identity: policy.settings_identity.clone(),
            owner_revision: policy.owner_revision,
            observed_at,
            received_seconds_ago: policy.received_seconds_ago,
        };
        Ok(Self {
            ran_at: invocation.at,
            clock: Clock::new(invocation.at.into(), policy.day.clone()),
            machine: invocation.machine,
            authorship: invocation.authorship,
            owner,
            policy: Some(Arc::new(policy)),
        })
    }
    pub fn invocation(&self) -> Invocation {
        Invocation {
            at: self.ran_at,
            machine: self.machine.clone(),
            authorship: self.authorship.clone(),
        }
    }
    pub fn request(&self, id: String, write: UserWrite) -> UserRequest {
        self.invocation().request(id, write)
    }
    pub fn provenance(&self) -> Provenance {
        Provenance {
            ran_at: self.ran_at,
            clock: self.clock.wire(),
            machine: self.machine.clone(),
            authorship: self.authorship.clone(),
            owner: self.owner.clone(),
        }
    }
    pub fn owner(&self) -> &Owner {
        &self.owner
    }
    /// Callers authenticate the cursor and compare these versions before replay.
    pub fn cursor_versions(&self) -> BTreeMap<String, String> {
        let owner = match &self.owner {
            Owner::Standalone => serde_json::json!({"mode":"standalone"}),
            Owner::Primary {
                settings_identity,
                owner_revision,
                ..
            } => {
                serde_json::json!({"mode":"primary","settings_identity":settings_identity,"owner_revision":owner_revision})
            }
        };
        let value = serde_json::json!({"owner":owner,"day":self.clock.day.wire()});
        [(
            "operation_context".into(),
            blake3::hash(value.to_string().as_bytes())
                .to_hex()
                .to_string(),
        )]
        .into()
    }
    /// Only an authenticated evaluation clock may be replayed; capture time stays current.
    pub fn replay_clock(&self, clock: Clock) -> Result<Self, Response> {
        if clock.day != self.clock.day {
            return Err(error(
                "stale_cursor",
                "the canonical logical day changed since this cursor was issued",
            ));
        }
        let mut replay = self.clone();
        replay.clock = clock;
        Ok(replay)
    }
    #[cfg(feature = "sinex")]
    pub fn attention(&self) -> Option<Attention<'_>> {
        self.policy.as_ref().map(|policy| Attention {
            quiet_hours: policy.quiet_hours,
            budget: &policy.interrupt_budget,
        })
    }
    #[cfg(feature = "sinex")]
    pub(crate) fn owner_token(&self) -> Option<&str> {
        self.policy
            .as_ref()
            .map(|policy| policy.context_token.as_str())
    }
    pub fn worker_authorship(program: &str) -> Authorship {
        Authorship {
            author: polylogue_model::Author::Script(program.into()),
            via: Channel::Daemon,
        }
    }
}

#[cfg(test)]
mod tests;
