//! Test-only phase evidence. This channel never wakes the controller or writer.
use super::Daemon;
use serde_json::Value;

#[derive(Clone, Debug)]
pub(crate) enum Phase {
    Subscribed,
    SubscriptionReceived,
    CueForwarded { queued: bool },
    ControllerWake { refresh: bool },
    ControllerWaiting,
    TickStarted { at: i64 },
    RulesRead { ids: Vec<String> },
    Status(Value),
    PersistStarted { request: String },
    PersistAcknowledged { request: String, confirmed: bool },
    TickCompleted { persisted: bool },
    TickFailed(Value),
    CommittedSource { request: String },
    IndependentCommitCue,
}

impl std::fmt::Display for Phase {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Subscribed => output.write_str("subscribed"),
            Self::SubscriptionReceived => output.write_str("subscription received a cue"),
            Self::CueForwarded { queued } => write!(output, "cue forwarded queued={queued}"),
            Self::ControllerWake { refresh } => write!(output, "controller wake refresh={refresh}"),
            Self::ControllerWaiting => output.write_str("controller waiting"),
            Self::TickStarted { at } => write!(output, "tick started at={at}"),
            Self::RulesRead { ids } => write!(output, "rules read {ids:?}"),
            Self::Status(value) => write!(output, "status {value}"),
            Self::PersistStarted { request } => write!(output, "persist started {request}"),
            Self::PersistAcknowledged { request, confirmed } => write!(
                output,
                "persist acknowledged {request} confirmed={confirmed}"
            ),
            Self::TickCompleted { persisted } => {
                write!(output, "tick completed persisted={persisted}")
            }
            Self::TickFailed(value) => write!(output, "tick failed {value}"),
            Self::CommittedSource { request } => write!(output, "source row committed {request}"),
            Self::IndependentCommitCue => {
                output.write_str("independent subscription has a commit cue")
            }
        }
    }
}

pub(crate) fn emit(d: &Daemon, phase: Phase) {
    let sink = d.signal_trace.lock().unwrap().clone();
    if let Some(sink) = sink {
        let _ = sink.send(phase);
    }
}

pub(crate) fn status(d: &Daemon) {
    let value = d.signals.lock().unwrap().clone();
    emit(d, Phase::Status(value));
}
