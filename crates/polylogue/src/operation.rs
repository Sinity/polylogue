//! The request as its accepting transport saw it.
use polylogue_model::authorship::Authorship;
use polylogue_reading::operation::{Invocation, ParentLifetime};
use std::sync::Arc;

/// A request as its accepting transport saw it: who sent it, how, from
/// where, and what bounds its life.
#[derive(Clone)]
pub struct Accepted {
    pub authorship: Authorship,
    pub cancellation: polylogue_store::cancellation::Cancellation,
    /// Process-private outer request bound, captured once by its accepting transport.
    pub(crate) parent: Option<Arc<ParentLifetime>>,
    pub working_directory: Option<std::path::PathBuf>,
    pub invocation: Option<Invocation>,
    /// The read scope token captured after classifying the request.
    pub read_scope: Option<polylogue_store::ReadScopeToken>,
}

impl Accepted {
    /// The author of a request from the caller's facts, how it came, and the
    /// open tool call of the caller's harness, if one matches.
    pub fn new(
        open_calls: &crate::hook::open_calls::OpenCalls,
        caller: &command_surface::CallerEvidence,
        via: command_surface::Via,
        cancellation: polylogue_store::cancellation::Cancellation,
    ) -> Self {
        let hook = open_calls.identity(caller);
        Self {
            authorship: command_surface::author_of(caller, via, hook.as_ref()).into(),
            cancellation,
            parent: None,
            working_directory: None,
            invocation: None,
            read_scope: None,
        }
    }

    /// Coalescing must use the same actor and channel as the stored request identity.
    pub fn scope(&self) -> String {
        serde_json::to_string(&self.authorship).expect("computed authorship serializes")
    }
}

