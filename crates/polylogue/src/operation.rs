//! One accepted invocation, extended with its evaluation context when needed.
use crate::daemon::Failure;
use polylogue_model::{
    Timestamp,
    authorship::{Authorship, Channel},
    context::{Owner, Provenance},
    user::{UserRequest, UserWrite},
};
use query_language::time::{Clock, DaySetting};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Server run identity is entropy, independent of the captured invocation time.
pub(crate) fn new_run_id() -> polylogue_model::RunId {
    polylogue_model::RunId::parse(&uuid::Uuid::new_v4().hyphenated().to_string())
        .expect("a generated UUID is a valid run identity")
}

pub(crate) use polylogue_store::writer::ParentLifetime;

/// Clock-free writes capture this once without requiring an owner's logical day.
#[derive(Clone)]
pub struct Invocation {
    pub at: Timestamp,
    pub machine: String,
    pub authorship: Authorship,
    accepted_request: Option<Arc<command_surface::RunnableRequest>>,
    parent: Option<Arc<ParentLifetime>>,
}
impl std::fmt::Debug for Invocation {
    fn fmt(&self, format: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        format
            .debug_struct("Invocation")
            .field("at", &self.at)
            .field("machine", &self.machine)
            .field("authorship", &self.authorship)
            .finish()
    }
}
impl Invocation {
    #[cfg(test)]
    pub fn new(at: Timestamp, machine: String, authorship: Authorship) -> Self {
        Self {
            at,
            machine,
            authorship,
            accepted_request: None,
            parent: None,
        }
    }
    pub(crate) fn with_accepted_request(
        mut self,
        request: Option<Arc<command_surface::RunnableRequest>>,
    ) -> Self {
        self.accepted_request = request;
        self
    }
    pub(crate) fn with_parent(mut self, parent: Option<Arc<ParentLifetime>>) -> Self {
        self.parent = parent;
        self
    }
    pub(crate) fn parent_lifetime(&self) -> Option<Arc<ParentLifetime>> {
        self.parent.clone()
    }
    /// Check only before accepting new work, never after a write is handed over or for owned cleanup.
    pub(crate) fn check_parent_lifetime(&self) -> polylogue_store::Result<()> {
        self.parent.as_ref().map_or(Ok(()), |parent| parent.check())
    }
    pub fn capture(
        metadata: &polylogue_model::user::RequestMetadata,
        authorship: &Authorship,
    ) -> Self {
        Self {
            at: metadata.now(),
            machine: metadata.machine().into(),
            authorship: authorship.clone(),
            accepted_request: None,
            parent: None,
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
    accepted_request: Option<Arc<command_surface::RunnableRequest>>,
    parent: Option<Arc<ParentLifetime>>,
}
impl std::fmt::Debug for Context {
    fn fmt(&self, format: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        format
            .debug_struct("Context")
            .field("provenance", &self.provenance())
            .finish()
    }
}
impl Context {
    pub fn standalone(invocation: Invocation, day: DaySetting) -> Self {
        Self {
            ran_at: invocation.at,
            clock: Clock::new(invocation.at.into(), day),
            machine: invocation.machine,
            authorship: invocation.authorship,
            owner: Owner::Standalone,
            accepted_request: invocation.accepted_request,
            parent: invocation.parent,
        }
    }
    #[cfg(feature = "sinex")]
    pub fn primary(
        invocation: Invocation,
        policy: polylogue_sinex::ResolvedContext,
    ) -> Result<Self, Failure> {
        let observed_at =
            Timestamp::from_unix_ns(policy.observed_at.unix_ns()).ok_or_else(|| {
                Failure::new(
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
            accepted_request: invocation.accepted_request,
            parent: invocation.parent,
        })
    }
    pub fn invocation(&self) -> Invocation {
        Invocation {
            at: self.ran_at,
            machine: self.machine.clone(),
            authorship: self.authorship.clone(),
            accepted_request: self.accepted_request.clone(),
            parent: self.parent.clone(),
        }
    }
    pub(crate) fn parent_lifetime(&self) -> Option<Arc<ParentLifetime>> {
        self.parent.clone()
    }
    pub(crate) fn check_parent_lifetime(&self) -> polylogue_store::Result<()> {
        self.parent.as_ref().map_or(Ok(()), |parent| parent.check())
    }
    pub fn accepted_request(&self) -> Option<&command_surface::RunnableRequest> {
        self.accepted_request.as_deref()
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
    pub fn replay_clock(&self, clock: Clock) -> Result<Self, Failure> {
        if clock.day != self.clock.day {
            return Err(Failure::new(
                "stale_cursor",
                "the logical day setting changed since this cursor was issued",
            ));
        }
        let mut replay = self.clone();
        replay.clock = clock;
        Ok(replay)
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
