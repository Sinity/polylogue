//! Shared daemon state and its archive mode.
use crate::daemon::Failure;
use polylogue_engine::{LoopMsg, LoopStatus};
use polylogue_store::{Archive, ReadPool, StoreError, WriterSender};
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(test)]
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// What request handlers share.
pub struct Daemon {
    pub(crate) clock: polylogue_model::WallClock,
    pub(crate) config: Arc<crate::config::Config>,
    pub archive: Archive,
    pub reads: ReadPool,
    /// Runs every request's work within bounded slots.
    pub(crate) executor: crate::daemon::executor::Executor,
    pub(crate) completion_path: std::path::PathBuf,
    pub(crate) callers: crate::hook::open_calls::OpenCalls,
    pub(super) operations: std::sync::OnceLock<std::sync::Weak<command_surface::OperationTable>>,

    pub(crate) warm: crate::daemon::warm::Warm,
    /// Sessions by working directory, kept current by the refresh worker
    /// ([`crate::daemon::scope`]).
    pub(crate) scope: crate::daemon::scope::SharedScopeIndex,
    /// What `status` reads from the index, kept current by the same worker.
    pub(crate) status_snapshot: super::status::StatusSnapshot,
    /// Briefs answered whose delivery records are still queued.
    pub(crate) queued_deliveries: crate::daemon::queued_deliveries::QueuedDeliveries,
    pub(crate) live_spool: crate::live_events::SharedStatus,
    pub(crate) advisories: crate::advisory::Cooldowns,
    /// User writes (notes, deliveries, settings) through the one writer.
    pub(crate) writes: Option<WriterSender>,
    /// Live-session acts: the mailbox, resumed runs and hosted sessions.
    pub(crate) acts: crate::live_acts::Acts,
    pub(super) mode: ModeState,
    pub(super) loop_tx: Mutex<crossbeam_channel::Sender<LoopMsg>>,
    pub(crate) status: Arc<Mutex<LoopStatus>>,
    pub(super) started: Instant,
    /// The signal rules in force and the ones refused, for `status`.
    pub(crate) signals: Mutex<serde_json::Value>,
    #[cfg(test)]
    pub(crate) signal_trace: Mutex<Option<Sender<crate::signals::observations::Phase>>>,
    pub(super) stopping: AtomicBool,
    pub(in crate::daemon) idle_gate: crate::daemon::schema::IdleGate,
    #[cfg(test)]
    pub(crate) read_limits: Mutex<Option<Sender<crate::test_daemon::ReadObservation>>>,
}

impl Daemon {
    pub(crate) fn is_stopping(&self) -> bool {
        self.stopping.load(Ordering::Acquire)
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
        let table = self
            .operations
            .get()
            .and_then(std::sync::Weak::upgrade)
            .ok_or_else(|| {
                Failure::new(
                    FailureKind::NotRunning,
                    "the operation table is unavailable",
                )
            })?;
        let description = table
            .descriptions()
            .iter()
            .find(|description| description.spec.name == request.operation)
            .ok_or_else(|| Failure::new(FailureKind::BadRequest, "unknown operation"))?;
        let command =
            crate::commands::command(&request.operation).expect("registered command exists");
        if let Some(registration) = command.registration() {
            registration
                .admit(
                    &request.arguments,
                    command,
                    &crate::commands::Caller {
                        surface: crate::commands::Surface::Http,
                        cwd: None,
                    },
                )
                .map_err(|failure| failure.error)?;
        } else if let crate::commands::Run::Feed { registration, .. } = command.run {
            registration
                .validate(request)
                .map_err(|failure| failure.error)?;
        }
        if description.spec.http_route.is_none() {
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
        let local = reader.completeness(&polylogue_engine::reading_contracts())?;
        Ok(freshness_of(local))
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
