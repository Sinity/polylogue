//! One accepted invocation, extended with its evaluation context when needed.
use crate::Error;
use polylogue_model::{
    Timestamp,
    authorship::{Authorship, Channel},
    context::{RunContext, SinexContext},
    user::{UserRequest, UserWrite},
};
use query_language::time::{Clock, DaySetting};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Server run identity is entropy, independent of the captured invocation time.
pub fn new_run_id() -> polylogue_model::RunId {
    polylogue_model::RunId::parse(&uuid::Uuid::new_v4().hyphenated().to_string())
        .expect("a generated UUID is a valid run identity")
}

pub use polylogue_store::writer::ParentLifetime;

/// Clock-free writes capture this once without requiring Sinex's logical day.
#[derive(Clone)]
pub struct Invocation {
    pub at: Timestamp,
    pub machine: String,
    pub authorship: Authorship,
    calling_session: Option<String>,
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
    pub fn new(at: Timestamp, machine: String, authorship: Authorship) -> Self {
        Self {
            at,
            machine,
            authorship,
            calling_session: None,
            accepted_request: None,
            parent: None,
        }
    }
    /// The session whose request is being evaluated, independent of its author.
    pub fn with_calling_session(mut self, session: Option<String>) -> Self {
        self.calling_session = session;
        self
    }
    pub fn with_accepted_request(
        mut self,
        request: Option<Arc<command_surface::RunnableRequest>>,
    ) -> Self {
        self.accepted_request = request;
        self
    }
    pub fn with_parent(mut self, parent: Option<Arc<ParentLifetime>>) -> Self {
        self.parent = parent;
        self
    }
    pub fn parent_lifetime(&self) -> Option<Arc<ParentLifetime>> {
        self.parent.clone()
    }
    /// Check only before accepting new work, never after a write is handed over or for owned cleanup.
    pub fn check_parent_lifetime(&self) -> polylogue_store::Result<()> {
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
            calling_session: None,
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
    sinex: SinexContext,
    calling_session: Option<String>,
    accepted_request: Option<Arc<command_surface::RunnableRequest>>,
    parent: Option<Arc<ParentLifetime>>,
}
impl std::fmt::Debug for Context {
    fn fmt(&self, format: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        format
            .debug_struct("Context")
            .field("run_context", &self.run_context())
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
            sinex: SinexContext::Standalone,
            calling_session: invocation.calling_session,
            accepted_request: invocation.accepted_request,
            parent: invocation.parent,
        }
    }
    pub fn primary(invocation: Invocation, day: DaySetting, sinex: SinexContext) -> Self {
        Self {
            ran_at: invocation.at,
            clock: Clock::new(invocation.at.into(), day),
            machine: invocation.machine,
            authorship: invocation.authorship,
            sinex,
            calling_session: invocation.calling_session,
            accepted_request: invocation.accepted_request,
            parent: invocation.parent,
        }
    }
    pub fn invocation(&self) -> Invocation {
        Invocation {
            at: self.ran_at,
            machine: self.machine.clone(),
            authorship: self.authorship.clone(),
            calling_session: self.calling_session.clone(),
            accepted_request: self.accepted_request.clone(),
            parent: self.parent.clone(),
        }
    }
    pub fn parent_lifetime(&self) -> Option<Arc<ParentLifetime>> {
        self.parent.clone()
    }
    pub fn check_parent_lifetime(&self) -> polylogue_store::Result<()> {
        self.parent.as_ref().map_or(Ok(()), |parent| parent.check())
    }
    pub fn calling_session(&self) -> Option<&str> {
        self.calling_session.as_deref()
    }
    pub fn accepted_request(&self) -> Option<&command_surface::RunnableRequest> {
        self.accepted_request.as_deref()
    }
    pub fn request(&self, id: String, write: UserWrite) -> UserRequest {
        self.invocation().request(id, write)
    }
    pub fn run_context(&self) -> RunContext {
        RunContext {
            ran_at: self.ran_at,
            clock: self.clock.wire(),
            machine: self.machine.clone(),
            authorship: self.authorship.clone(),
            sinex: self.sinex.clone(),
        }
    }
    /// Callers authenticate the cursor and compare these versions before replay.
    pub fn cursor_versions(&self) -> BTreeMap<String, String> {
        let sinex = match &self.sinex {
            SinexContext::Standalone => serde_json::json!({"mode":"standalone"}),
            SinexContext::Primary {
                settings_identity,
                settings_revision,
                ..
            } => {
                serde_json::json!({"mode":"primary","settings_identity":settings_identity,"settings_revision":settings_revision})
            }
        };
        let value = serde_json::json!({"sinex":sinex,"day":self.clock.day.wire()});
        [(
            "operation_context".into(),
            canonical_json::canonical_hash("polylogue-operation-context", &value)
                .expect("a JSON value has a canonical form")
                .to_hex()
                .to_string(),
        )]
        .into()
    }
    /// Only an authenticated evaluation clock may be replayed; capture time stays current.
    pub fn replay_clock(&self, clock: Clock) -> Result<Self, Error> {
        if clock.day != self.clock.day {
            return Err(Error::StaleCursor {
                changed: "the logical day setting changed since this cursor was issued".into(),
            });
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
