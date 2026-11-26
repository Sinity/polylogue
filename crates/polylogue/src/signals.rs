//! The common evaluator owns firing, occurrence consumption and host policy.
//! This adapter owns archive transactions and deployment-selected effects.
mod desktop;
#[cfg(test)]
pub(crate) mod observations;
mod rules;

use crate::daemon::service::Daemon;
use polylogue_model::{
    LockExt as _,
    authored::RunsHere,
    signal::{Checkpoint, Commit, Event, PendingDelivery},
    user::UserWrite,
};
use serde_json::{Value, json};
use shared_signals::{DayCalendar, Evaluator, HostPolicy, HostSettings, InterruptBudget};
use std::sync::Arc;

pub fn duration(text: &str) -> Option<i64> {
    query_language::time::duration_ms(text).filter(|n| *n >= 0)
}

/// How many of the day's notices status shows.
const LATEST_NOTICES: usize = 3;

pub fn recent(reader: &polylogue_store::Reader<'_>, now: i64) -> Value {
    match reader.notices_since(now.saturating_sub(86_400_000)) {
        Ok(found) => {
            json!({"last_day":found.len(),"latest":found.iter().take(LATEST_NOTICES).map(|(row,text)|json!({"id":row.id,"at":row.at,"text":text,"events":row.events})).collect::<Vec<_>>()})
        }
        Err(error) => json!({"error":error.to_string()}),
    }
}

#[derive(Debug, serde::Serialize)]
struct Held {
    reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<Box<command_surface::Answer<()>>>,
}
impl From<String> for Held {
    fn from(reason: String) -> Self {
        Self {
            reason,
            detail: None,
        }
    }
}
impl From<&str> for Held {
    fn from(reason: &str) -> Self {
        reason.to_owned().into()
    }
}
impl From<crate::daemon::Failure> for Held {
    fn from(failure: crate::daemon::Failure) -> Self {
        Self {
            reason: "Sinex context is unavailable".into(),
            detail: Some(Box::new(failure.answer())),
        }
    }
}
fn report(d: &Daemon, hold: &Held) {
    *d.signals.locked() = json!({"held":[hold]});
    #[cfg(test)]
    observations::status(d);
}

#[cfg(test)]
fn operation(d: &Daemon) -> Result<Arc<polylogue_reading::operation::Context>, Held> {
    d.operation_context(&polylogue_reading::operation::Context::worker_authorship(
        "polylogue-signals",
    ))
    .map_err(Held::from)
}
struct LocalAttention {
    quiet: Option<shared_signals::QuietHours>,
    budget: usize,
}
fn local_attention(
    reader: &polylogue_store::Reader<'_>,
    context: &polylogue_reading::operation::Context,
) -> Result<LocalAttention, String> {
    let catalog = reader
        .settings_catalog(&context.machine)
        .map_err(|e| e.to_string())?;
    let (quiet, budget) = polylogue_model::settings::Snapshot::new(catalog)
        .standalone_attention()
        .map_err(|e| e.to_string())?;
    Ok(LocalAttention {
        quiet: quiet.map(|[from, to]| shared_signals::QuietHours { from, to }),
        budget,
    })
}
fn initial(
    d: &Daemon,
    context: &polylogue_reading::operation::Context,
    attention: Option<&LocalAttention>,
) -> Result<Checkpoint, String> {
    let calendar = DayCalendar::new(context.clock.day.clone());
    let host = attention.map(|a| {
        HostPolicy::new(
            HostSettings {
                product: "polylogue".into(),
                quiet_hours: a.quiet,
                away_command: d.config.signals.notifications.away_command.is_some(),
                ..HostSettings::default()
            },
            InterruptBudget {
                per_hour: a.budget,
                handoffs: vec![],
            },
        )
    });
    Ok(Checkpoint {
        owner: context.machine.clone(),
        evaluator: Evaluator::new(context.clock.now.as_millisecond(), vec![], &calendar)
            .map_err(|e| e.to_string())?,
        rules: vec![],
        host,
        notifications: None,
    })
}

/// A rule held back because nothing confirms its dependencies.
fn held_rule(id: &str) -> Value {
    json!({"id":id,"reason":"confirmed rule dependencies and whether the rule may run here are unknown"})
}

/// The stored checkpoint with its identity, or the initial checkpoint when
/// none is stored yet.
fn load_or_initial(
    d: &Daemon,
    reader: &polylogue_store::Reader<'_>,
    context: &polylogue_reading::operation::Context,
) -> Result<(Option<polylogue_model::BlobHash>, Checkpoint), String> {
    let previous = reader
        .signal_checkpoint(&context.machine)
        .map_err(|e| e.to_string())?;
    let prior = previous
        .as_ref()
        .map(Checkpoint::identity)
        .transpose()
        .map_err(|e| e.to_string())?;
    let checkpoint = match previous {
        Some(value) => value,
        None => {
            let attention = Some(local_attention(reader, context)?);
            initial(d, context, attention.as_ref())?
        }
    };
    Ok((prior, checkpoint))
}

fn load(d: &Daemon) -> polylogue_store::Result<Option<Checkpoint>> {
    d.reads
        .reader()
        .and_then(|r| r.signal_checkpoint(d.config.requests.machine()))
}

/// Whether the checkpoint write is confirmed, as opposed to queued. Effects and
/// worker acknowledgments are released only after the request is readable.
fn persist(
    d: &Daemon,
    context: &polylogue_reading::operation::Context,
    prior: Option<polylogue_model::BlobHash>,
    checkpoint: Checkpoint,
    pending_delivery: Vec<PendingDelivery>,
    events: Vec<Event>,
) -> Result<bool, String> {
    let writes = d.writes.as_ref().ok_or("writer is unavailable")?;
    let commit = Commit {
        prior,
        checkpoint,
        pending_delivery,
        events,
    };
    let id = canonical_json::canonical_hash("polylogue-signal-commit", &commit)
        .map_err(|e| e.to_string())?;
    let id = format!("signal:{}", id.to_hex());
    let request = context.request(id.clone(), UserWrite::SignalCommit(Box::new(commit)));
    #[cfg(test)]
    observations::emit(
        d,
        observations::Phase::PersistStarted {
            request: id.clone(),
        },
    );
    let outcome = writes.archive().user(request).map_err(|e| e.to_string())?;
    #[cfg(test)]
    observations::emit(
        d,
        observations::Phase::PersistAcknowledged {
            request: id.clone(),
            confirmed: !outcome.pending,
        },
    );
    Ok(!outcome.pending)
}

struct Tick {
    notices: Vec<shared_signals::notifications::Notice>,
    next: Option<i64>,
}

fn tick(d: &Daemon, context: Arc<polylogue_reading::operation::Context>) -> Result<Tick, Held> {
    let clock = &context.clock;
    let reader = d.reads.reader().map_err(|e| e.to_string())?;
    let (prior, checkpoint) = load_or_initial(d, &reader, &context)?;
    if clock.now.as_millisecond() < checkpoint.evaluator.now() {
        return Err("clock precedes the stored evaluator checkpoint".into());
    }
    let annotations = crate::daemon::query::annotations::list(&reader, "kind:signal", &context)
        .map_err(|e| e.to_string())?;
    #[cfg(test)]
    observations::emit(
        d,
        observations::Phase::RulesRead {
            ids: annotations
                .iter()
                .map(|annotation| annotation.fact.id.clone())
                .collect(),
        },
    );
    let firings = reader.last_firings().map_err(|e| e.to_string())?;
    let mut definitions = Vec::new();
    let mut unusable = Vec::new();
    let mut held_rules = Vec::new();
    for annotation in annotations {
        match rules::definition(&annotation, &d.config.signals.rule_configuration()) {
            Ok(definition) => {
                let id = format!("authored:{}", definition.family);
                held_rules.push(held_rule(&id));
                definitions.push(json!({
                    "id":id,"name":definition.rule.name,"query":definition.rule.query,
                    "author":annotation.fact.author.to_string(),
                    "last_fired":firings.get(&definition.rule.name).copied()
                        .and_then(polylogue_model::Timestamp::from_unix_ms),
                }));
            }
            Err(reason) => unusable.push(json!({"id":annotation.fact.id,"reason":reason})),
        }
    }
    // A current-looking row set does not show the complete revision graph.
    // In particular, an empty read cannot authorize withdrawal of kept work.
    for kept in &checkpoint.rules {
        if !held_rules.iter().any(|row| row["id"] == kept.version.id) {
            held_rules.push(held_rule(&kept.version.id));
        }
    }
    *d.signals.locked() = json!({"rules":definitions,"unusable":unusable,"held":held_rules});
    #[cfg(test)]
    observations::status(d);
    drop(reader);
    let bootstrap = prior.is_none();
    if bootstrap {
        // The empty bootstrap contains no asserted rule or inferred epoch.
        // Existing checkpoints remain byte-for-byte unchanged while held.
        let confirmed = persist(d, &context, None, checkpoint, vec![], vec![])?;
        if !confirmed {
            return Err("signal checkpoint is durably pending confirmation".into());
        }
    }
    #[cfg(test)]
    observations::emit(
        d,
        observations::Phase::TickCompleted {
            persisted: bootstrap,
        },
    );
    Ok(Tick {
        notices: vec![],
        next: None,
    })
}

/// Runs the signal worker until `stop`, a child of the archive's stop
/// request, is cancelled.
pub fn run(
    d: Arc<Daemon>,
    stop: polylogue_store::cancellation::Cancellation,
) -> std::io::Result<()> {
    let result = (|| -> Result<(), Held> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| Held::from(format!("signal runtime: {error}")))?;
        let token = tokio_util::sync::CancellationToken::new();
        let _wake = {
            let token = token.clone();
            stop.on_cancel(move || token.cancel())
        };
        runtime.block_on(desktop::run(Arc::clone(&d), token))
    })();
    if let Err(error) = &result {
        report(&d, error);
    }
    result.map_err(|error| {
        // Preserve structured Sinex context failures as well as their reason.
        std::io::Error::other(serde_json::to_string(&error).unwrap_or(error.reason))
    })
}

#[cfg(test)]
mod tests;
