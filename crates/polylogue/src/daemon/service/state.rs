//! Shared daemon state and its archive mode.
use crate::daemon::Failure;
use polylogue_engine::{LoopMsg, SharedLoopStatus};
use polylogue_model::LockExt as _;
use polylogue_store::{Archive, ReadPool, StoreError, WriterSender};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// What request handlers share.
pub struct Daemon {
    pub(crate) clock: polylogue_model::WallClock,
    pub(crate) config: Arc<crate::config::Config>,
    pub archive: Archive,
    pub reads: ReadPool,
    /// The requests being answered, which a stop waits for.
    pub(crate) requests: Requests,
    pub(crate) callers: crate::hook::open_calls::OpenCalls,

    pub(crate) warm: crate::daemon::warm::Warm,
    /// Sessions by working directory, kept current by the refresh worker
    /// ([`crate::daemon::scope`]).
    pub(crate) scope: crate::daemon::scope::SharedScopeIndex,
    /// What `status` reads from the index, kept current by the same worker.
    pub(crate) status_snapshot: super::status::StatusSnapshot,
    pub(crate) live_spool: crate::live_events::SharedStatus,
    pub(crate) advisories: crate::advisory::Cooldowns,
    /// User writes (notes, deliveries, settings) through the one writer.
    pub(crate) writes: Option<WriterSender>,
    /// Live-session acts: the mailbox, resumed runs and hosted sessions.
    pub(crate) acts: crate::live_acts::Acts,
    pub(super) mode: ModeState,
    pub(super) loop_tx: Mutex<crossbeam_channel::Sender<LoopMsg>>,
    pub(crate) status: Arc<SharedLoopStatus>,
    pub(super) captures_sources: bool,
    pub(super) started: Instant,
    /// The signal rules in force and the ones refused, for `status`.
    pub(crate) signals: Mutex<serde_json::Value>,
    pub(in crate::daemon) idle_gate: crate::daemon::schema::IdleGate,
}

/// How many requests are being answered. Stopping refuses new ones and
/// waits a bounded time for these.
#[derive(Default)]
pub(crate) struct Requests {
    running: Mutex<usize>,
    stopping: AtomicBool,
    ended: std::sync::Condvar,
}

/// One request counted as running until dropped.
pub(crate) struct Running<'a>(&'a Requests);

impl Requests {
    /// Counts a request in, or refuses it while the daemon stops.
    pub(crate) fn enter(&self) -> Result<Running<'_>, Failure> {
        let mut running = self.running.locked();
        if self.stopping.load(Ordering::Acquire) {
            return Err(super::shutting_down());
        }
        *running += 1;
        Ok(Running(self))
    }

    /// Refuses requests from now on.
    pub(crate) fn refuse(&self) {
        let _running = self.running.locked();
        self.stopping.store(true, Ordering::Release);
    }

    /// How many requests are being answered now.
    #[cfg(test)]
    pub(crate) fn running(&self) -> usize {
        *self.running.locked()
    }

    /// Waits up to `bound` for running requests to end; how many did not.
    pub(crate) fn wait(&self, bound: std::time::Duration) -> usize {
        let until = Instant::now() + bound;
        let mut running = self.running.locked();
        while *running > 0 {
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            running = polylogue_model::unpoisoned(self.ended.wait_timeout(running, left)).0;
        }
        *running
    }
}

impl Drop for Running<'_> {
    fn drop(&mut self) {
        *self.0.running.locked() -= 1;
        self.0.ended.notify_all();
    }
}

impl Daemon {
    pub(crate) fn is_stopping(&self) -> bool {
        self.requests.stopping.load(Ordering::Acquire)
    }

    pub(crate) fn invocation(
        &self,
        authorship: &polylogue_model::authorship::Authorship,
    ) -> polylogue_reading::operation::Invocation {
        polylogue_reading::operation::Invocation::capture(&self.config.requests, authorship)
    }
    #[cfg(test)]
    pub(crate) fn operation_context(
        &self,
        authorship: &polylogue_model::authorship::Authorship,
    ) -> Result<Arc<polylogue_reading::operation::Context>, Failure> {
        self.resolve_operation_context(self.invocation(authorship))
    }
    pub(crate) fn resolve_operation_context(
        &self,
        invocation: polylogue_reading::operation::Invocation,
    ) -> Result<Arc<polylogue_reading::operation::Context>, Failure> {
        let context = match &self.mode {
            ModeState::Standalone => polylogue_reading::operation::Context::standalone(
                invocation,
                self.config
                    .standalone_day()
                    .map_err(|problem| Failure::not_yet_available(problem.to_string()))?
                    .clone(),
            ),
            ModeState::Blocked { error: reason, .. } => {
                return Err(Failure::sinex_unavailable(reason.clone()));
            }
        };
        Ok(Arc::new(context))
    }

    pub(crate) async fn operation_context_async(
        &self,
        authorship: &polylogue_model::authorship::Authorship,
    ) -> Result<Arc<polylogue_reading::operation::Context>, Failure> {
        let invocation = self.invocation(authorship);
        self.resolve_operation_context(invocation)
    }

    pub(crate) fn ready_call(
        &self,
        request: &command_surface::RunnableRequest,
    ) -> Result<polylogue_model::read::ReadyCall, command_surface::Failure> {
        use command_surface::{Failure, FailureKind};
        let row = crate::commands::command(&request.operation)
            .ok_or_else(|| Failure::new(FailureKind::BadRequest, "unknown operation"))?;
        crate::daemon::handler::policy(
            row,
            &request.arguments,
            polylogue_model::authorship::Channel::Web,
        )
        .map_err(|failure| failure.error)?;
        if !row.http {
            return Err(Failure::new(
                FailureKind::BadRequest,
                "operation has no HTTP route",
            ));
        }
        Ok(polylogue_model::read::ReadyCall {
            request: request.clone(),
        })
    }

    /// Freshness comes from this reader snapshot.
    pub(crate) fn freshness(
        &self,
        reader: &polylogue_store::Reader<'_>,
    ) -> polylogue_store::Result<polylogue_model::freshness::ReadFreshness> {
        // The filesystem can still hold not-yet-discovered inputs, which
        // a snapshot of the archive cannot count.
        let loop_status = self.status.get();
        let contracts = polylogue_engine::reading_contracts();
        let local = if loop_status.busy {
            reader.completeness_during_catch_up(&contracts)?
        } else {
            reader.completeness(&contracts)?
        };
        Ok(self.freshness_at(local, &loop_status))
    }

    pub(super) fn freshness_at(
        &self,
        mut local: polylogue_model::freshness::ReadCompleteness,
        status: &polylogue_engine::LoopStatus,
    ) -> polylogue_model::freshness::ReadFreshness {
        use polylogue_model::freshness::{
            CompletenessDimension, CompletenessReason, CountedReason,
        };
        if self.captures_sources
            && (status.full_scans == 0 || status.last_pass.capture_pending)
            && local.parsing == CompletenessDimension::Complete
        {
            local.parsing = CompletenessDimension::Incomplete(vec![CountedReason::new(
                CompletenessReason::CapturePending,
            )]);
        }
        freshness_of(local)
    }
}

/// Freshness from the index's completeness. A blocked archive's
/// configuration is reported by status, not as freshness.
pub(crate) fn freshness_of(
    local: polylogue_model::freshness::ReadCompleteness,
) -> polylogue_model::freshness::ReadFreshness {
    polylogue_model::freshness::ReadFreshness {
        local,
        sinex: polylogue_model::freshness::IndexFreshness::Standalone,
    }
}

impl Daemon {
    /// Writes are paused until the Sinex configuration is fixed or Sinex
    /// can be reached.
    pub(crate) fn blocked(&self) -> bool {
        self.mode.blocked()
    }
    pub(crate) fn write_refusal(&self) -> Failure {
        self.mode.write_refusal()
    }
    /// Holds idle-priority jobs back while an ordinary request runs.
    pub(in crate::daemon) fn foreground(&self) -> crate::daemon::schema::Foreground<'_> {
        self.idle_gate.foreground()
    }
}

#[derive(serde::Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub(super) enum ModeState {
    Standalone,
    Blocked {
        configured: Option<&'static str>,
        path: std::path::PathBuf,
        error: String,
    },
}

impl ModeState {
    /// Writes are paused until the Sinex configuration is fixed or Sinex
    /// can be reached.
    pub(super) fn blocked(&self) -> bool {
        matches!(self, Self::Blocked { .. })
    }

    /// The answer to a write the daemon cannot take.
    pub(super) fn write_refusal(&self) -> Failure {
        if let Self::Blocked {
            path,
            error: reason,
            ..
        } = self
        {
            let mut failure =
                Failure::sinex_unavailable(format!("archive writes are paused: {reason}"));
            failure.error.hint = Some(format!("fix the Sinex settings in {}", path.display()));
            return failure;
        }
        Failure::from(StoreError::WriterGone)
    }
}
