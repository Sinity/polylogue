//! Who is asking: one record, made once by the transport that accepted a
//! request.
use polylogue_model::authorship::{Authorship, Channel};
use polylogue_reading::operation::{Invocation, ParentLifetime};
use polylogue_store::cancellation::Cancellation;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// Who is asking, how, from where, and what bounds the request's life.
#[derive(Clone, Debug)]
pub struct Caller {
    /// Computed once from the peer's evidence and the channel; its `via`
    /// is the channel that carried the request.
    pub authorship: Authorship,
    pub calling_session: Option<String>,
    /// The caller's working directory, against which relative paths
    /// resolve; HTTP callers have none and name absolute paths.
    pub cwd: Option<PathBuf>,
    /// Ends the request: a child of its connection's cancellation, itself a
    /// child of the daemon's stop.
    pub stop: Cancellation,
    /// The transport's bound on the whole request, when the caller gave one.
    pub outer_deadline: Option<Instant>,
    /// A test's moment of acceptance in place of the daemon's clock.
    #[cfg(test)]
    pub accepted_at: Option<polylogue_model::Timestamp>,
    /// A test's read limit, observed by the test, in place of the one the
    /// request would compute.
    #[cfg(test)]
    pub read_limit: Option<polylogue_store::read::limit::ReadLimit>,
}

impl Caller {
    /// A caller whose author is computed from the peer's facts, how it came,
    /// and the open tool call of the peer's harness, if one matches.
    pub fn from_peer(
        open_calls: &crate::hook::open_calls::OpenCalls,
        evidence: &command_surface::CallerEvidence,
        via: command_surface::Via,
        stop: Cancellation,
    ) -> Self {
        let hook = open_calls.identity(evidence);
        Self {
            authorship: command_surface::author_of(evidence, via, hook.as_ref()).into(),
            calling_session: evidence.session.as_ref().map(ToString::to_string),
            cwd: None,
            stop,
            outer_deadline: None,
            #[cfg(test)]
            accepted_at: None,
            #[cfg(test)]
            read_limit: None,
        }
    }

    /// A caller with no peer process: the web views, the daemon's own
    /// pages, tests.
    pub fn local(via: command_surface::Via, stop: Cancellation) -> Self {
        Self {
            authorship: command_surface::author_of(
                &command_surface::CallerEvidence::default(),
                via,
                None,
            )
            .into(),
            calling_session: None,
            cwd: None,
            stop,
            outer_deadline: None,
            #[cfg(test)]
            accepted_at: None,
            #[cfg(test)]
            read_limit: None,
        }
    }

    pub fn channel(&self) -> Channel {
        self.authorship.via
    }

    /// The sender a prepared context is recorded under: MCP hands it to a
    /// model and a hook to its harness; the command line and the web show
    /// it to a person.
    pub fn sender(&self) -> Option<polylogue_model::DeliverySender> {
        sender(self.channel())
    }

    /// The request as accepted: its clock, authorship, calling session and
    /// working directory, and the caller's bound and cancellation, which a
    /// write checks before the writer takes it.
    pub fn invocation(
        &self,
        daemon: &crate::daemon::service::Daemon,
        request: &command_surface::RunnableRequest,
    ) -> Invocation {
        #[cfg_attr(not(test), allow(unused_mut))]
        let mut invocation = daemon.invocation(&self.authorship);
        #[cfg(test)]
        if let Some(at) = self.accepted_at {
            invocation.at = at;
        }
        invocation
            .with_accepted_request(Some(Arc::new(request.clone())))
            .with_calling_session(self.calling_session.clone())
            .with_working_directory(self.cwd.clone())
            .with_parent(Some(ParentLifetime::capture(
                self.outer_deadline,
                self.stop.clone(),
            )))
    }
}

/// The sender a channel's prepared context is recorded under.
pub fn sender(channel: Channel) -> Option<polylogue_model::DeliverySender> {
    match channel {
        Channel::Mcp => Some(polylogue_model::DeliverySender::Mcp),
        Channel::Hook => Some(polylogue_model::DeliverySender::Hook),
        _ => None,
    }
}

/// Whether a channel is the command line or a process like it: anything
/// but MCP, HTTP and a hook. Only there are terminal options taken.
pub fn command_line(channel: Channel) -> bool {
    !matches!(channel, Channel::Mcp | Channel::Web | Channel::Hook)
}
