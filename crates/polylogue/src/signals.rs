//! The common evaluator owns firing, occurrence consumption and host policy.
//! This adapter owns archive transactions and deployment-selected effects.
mod desktop;
#[cfg(test)]
pub(crate) mod observations;
mod rules;

use crate::daemon::service::Daemon;
use polylogue_model::{
    authored::ExecutionEligibility,
    signal::{Checkpoint, Commit, Event, Outbox},
    user::UserWrite,
};
use serde_json::{Value, json};
use shared_signals::{DayCalendar, Evaluator, HostPolicy, HostSettings, InterruptBudget};
use std::sync::Arc;

pub fn duration(text: &str) -> Option<i64> {
    query_language::time::duration_ms(text).filter(|n| *n >= 0)
}

pub fn recent(reader: &polylogue_store::Reader<'_>, now: i64) -> Value {
    match reader.notices_since(now.saturating_sub(86_400_000)) {
        Ok(found) => {
            json!({"last_day":found.len(),"latest":found.iter().take(5).map(|(row,text)|json!({"id":row.id,"at":row.at,"text":text,"events":row.events})).collect::<Vec<_>>()})
        }
        Err(error) => json!({"error":error.to_string()}),
    }
}

fn held(d: &Daemon, reason: &str) {
    if let Ok(mut status) = d.signals.lock() {
        *status = json!({"held":[{"reason":reason}]});
    }
    #[cfg(test)]
    observations::status(d);
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
            reason: "owner context is unavailable".into(),
            detail: Some(Box::new(failure.answer())),
        }
    }
}
fn report(d: &Daemon, hold: Held) {
    if let Ok(mut status) = d.signals.lock() {
        *status = json!({"held":[hold]});
    }
    #[cfg(test)]
    observations::status(d);
}

fn operation(d: &Daemon) -> Result<Arc<crate::operation::Context>, Held> {
    d.operation_context(&crate::operation::Context::worker_authorship(
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
    context: &crate::operation::Context,
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
    context: &crate::operation::Context,
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

fn load(d: &Daemon) -> Result<Option<Checkpoint>, String> {
    d.reads
        .reader()
        .and_then(|r| r.signal_checkpoint(d.config.requests.machine()))
        .map_err(|e| e.to_string())
}

/// The result distinguishes durable enqueue from confirmed state. Effects and
/// worker acknowledgements are released only after the request is readable.
fn persist(
    d: &Daemon,
    context: &crate::operation::Context,
    prior: Option<polylogue_model::BlobHash>,
    checkpoint: Checkpoint,
    outbox: Vec<Outbox>,
    events: Vec<Event>,
) -> Result<(String, bool), String> {
    let writes = d.writes.as_ref().ok_or("writer is unavailable")?;
    let commit = Commit {
        prior,
        checkpoint,
        outbox,
        events,
    };
    let body = serde_json::to_vec(&commit).map_err(|e| e.to_string())?;
    let id = format!("signal:{}", blake3::hash(&body).to_hex());
    let request = context.request(id.clone(), UserWrite::SignalCommit(Box::new(commit)));
    #[cfg(test)]
    observations::emit(
        d,
        observations::Phase::PersistStarted {
            request: id.clone(),
        },
    );
    let outcome = writes.user(request).map_err(|e| e.to_string())?;
    #[cfg(test)]
    observations::emit(
        d,
        observations::Phase::PersistAcknowledged {
            request: id.clone(),
            confirmed: !outcome.pending,
        },
    );
    Ok((id, !outcome.pending))
}

struct Tick {
    notices: Vec<shared_signals::notifications::Notice>,
    next: Option<i64>,
}

fn tick(d: &Daemon) -> Result<Tick, Held> {
    let context = operation(d)?;
    let clock = &context.clock;
    let reader = d.reads.reader().map_err(|e| e.to_string())?;
    let attention = if d.archive.is_primary() {
        None
    } else {
        Some(local_attention(&reader, &context)?)
    };
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
        None => initial(d, &context, attention.as_ref())?,
    };
    if clock.now.as_millisecond() < checkpoint.evaluator.now() {
        return Err("clock precedes the durable evaluator checkpoint".into());
    }
    let assertions = crate::daemon::query::assertions::list(&reader, "kind:signal", &context)
        .map_err(|e| e.to_string())?;
    #[cfg(test)]
    observations::emit(
        d,
        observations::Phase::RulesRead {
            ids: assertions
                .iter()
                .map(|assertion| assertion.fact.id.clone())
                .collect(),
        },
    );
    let mut definitions = Vec::new();
    let mut unusable = Vec::new();
    let mut held_rules = Vec::new();
    for assertion in assertions {
        match rules::definition(&assertion, &d.config.signals.rule_configuration()) {
            Ok(definition) => {
                let id = format!("authored:{}", definition.family);
                held_rules.push(json!({"id":id,"reason":"canonical rule closure and native invocation eligibility are unavailable"}));
                definitions.push(json!({
                    "id":id,"name":definition.rule.name,"query":definition.rule.query,
                    "author":"unavailable",
                }));
            }
            Err(reason) => unusable.push(json!({"id":assertion.fact.id,"reason":reason})),
        }
    }
    // A current-looking row set is not proof of the complete revision graph.
    // In particular, an empty read cannot authorize withdrawal of retained work.
    for retained in &checkpoint.rules {
        if !held_rules
            .iter()
            .any(|row| row["id"] == retained.version.id)
        {
            held_rules.push(json!({"id":retained.version.id,"reason":"canonical rule closure and native invocation eligibility are unavailable"}));
        }
    }
    if let Ok(mut status) = d.signals.lock() {
        *status = json!({"rules":definitions,"unusable":unusable,"held":held_rules});
    }
    #[cfg(test)]
    observations::status(d);
    drop(reader);
    let bootstrap = prior.is_none();
    if bootstrap {
        // The empty bootstrap contains no asserted rule or inferred epoch.
        // Existing checkpoints remain byte-for-byte unchanged while held.
        let (_, confirmed) = persist(d, &context, None, checkpoint, vec![], vec![])?;
        if !confirmed {
            return Err("signal checkpoint is durably pending canonical confirmation".into());
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

pub fn run(d: Arc<Daemon>, stop: std::sync::mpsc::Receiver<()>) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            held(&d, &format!("signal runtime: {error}"));
            return;
        }
    };
    let mut stop = match desktop::StopBridge::new(stop) {
        Ok(stop) => stop,
        Err(error) => {
            held(&d, &format!("signal stop bridge: {error}"));
            return;
        }
    };
    runtime.block_on(desktop::run(Arc::clone(&d), stop.token()));
    if let Err(error) = stop.finish() {
        held(&d, &error.to_string());
    }
}

#[cfg(test)]
mod tests;
