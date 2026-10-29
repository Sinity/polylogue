//! The common evaluator owns firing, occurrence consumption and host policy.
//! This adapter owns archive transactions and deployment-selected effects.
mod desktop;
mod rules;

use crate::daemon::service::Daemon;
use polylogue_model::{
    authored::ExecutionEligibility,
    signal::{Checkpoint, Commit, Event, Metadata, Outbox},
    user::{Delivery, UserWrite},
};
use serde_json::{Value, json};
use shared_signals::{
    DayCalendar, Evaluation, EvaluationMode, Evaluator, HostPolicy, InterruptBudget, Policy,
};
use std::{collections::BTreeMap, sync::Arc};

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
}

#[derive(serde::Serialize)]
struct Held {
    reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<command_surface::Answer<()>>,
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
impl From<crate::daemon::Response> for Held {
    fn from(response: crate::daemon::Response) -> Self {
        Self {
            reason: "owner context is unavailable".into(),
            detail: Some(response.answer().map(|_| ())),
        }
    }
}
fn report(d: &Daemon, hold: Held) {
    if let Ok(mut status) = d.signals.lock() {
        *status = json!({"held":[hold]});
    }
}

fn initial(d: &Daemon, clock: &query_language::time::Clock) -> Result<Checkpoint, String> {
    let calendar = DayCalendar::new(clock.day.clone());
    let host = if d.archive.is_primary() {
        None
    } else {
        let reader = d.reads.reader().map_err(|e| e.to_string())?;
        let settings = reader
            .settings_snapshot(d.config.requests.machine())
            .map_err(|e| e.to_string())?;
        let (quiet, budget) = settings.standalone_attention().map_err(|e| e.to_string())?;
        Some(HostPolicy::new(
            InterruptBudget {
                per_hour: budget,
                handoffs: vec![],
            },
            quiet.map(|[from, to]| shared_signals::QuietHours { from, to }),
        ))
    };
    Ok(Checkpoint {
        owner: d.config.requests.machine().into(),
        evaluator: Evaluator::new(
            clock.now.as_millisecond(),
            vec![],
            Policy::default(),
            &calendar,
        )
        .map_err(|e| e.to_string())?,
        rules: vec![],
        rule_sources: BTreeMap::new(),
        rule_revisions: BTreeMap::new(),
        host,
        notifications: None,
        outlet_attempts: BTreeMap::new(),
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
    let request = d
        .config
        .requests
        .request(id.clone(), UserWrite::SignalCommit(Box::new(commit)));
    let outcome = writes.user(request).map_err(|e| e.to_string())?;
    Ok((id, !outcome.pending))
}

fn tick(d: &Daemon) -> Result<Vec<shared_signals::notifications::Notice>, Held> {
    let clock = d.query_clock().map_err(Held::from)?;
    let now = clock.now.as_millisecond();
    let calendar = DayCalendar::new(clock.day.clone());
    let previous = load(d)?;
    let prior = previous
        .as_ref()
        .map(Checkpoint::identity)
        .transpose()
        .map_err(|e| e.to_string())?;
    let mut checkpoint = match previous {
        Some(value) => value,
        None => initial(d, &clock)?,
    };
    if now < checkpoint.evaluator.now() {
        return Err("clock precedes the durable evaluator checkpoint".into());
    }
    let reader = d.reads.reader().map_err(|e| e.to_string())?;
    let notes = crate::daemon::query::notes::list(&reader, "kind:signal", &clock)
        .map_err(|e| e.to_string())?;
    let mut active = Vec::new();
    let mut unusable = Vec::new();
    let mut held_rules = Vec::new();
    for note in notes {
        match rules::active(&note, &checkpoint, &d.config.signals.rule_configuration()) {
            Ok(rule) => {
                if !checkpoint.rules.iter().any(|old| old == &rule) {
                    checkpoint
                        .evaluator
                        .replace_rule(rule.clone(), &calendar)
                        .map_err(|e| e.to_string())?;
                }
                checkpoint
                    .rule_revisions
                    .insert(rule.version.id.clone(), rule.version.revision);
                checkpoint
                    .rule_sources
                    .insert(rule.version.id.clone(), note.id.clone());
                // The producer has not supplied exclusion-before-aggregation and canonical
                // episode evidence. An inexact read preserves the prior occurrence state.
                checkpoint
                    .evaluator
                    .observe(Evaluation {
                        rule: rule.version.clone(),
                        at: now,
                        complete: false,
                        rows: vec![],
                        value: None,
                    })
                    .map_err(|e| e.to_string())?;
                held_rules.push(json!({"id":rule.version.id,"reason":"signal input exclusion and causal evidence are unavailable"}));
                active.push(rule);
            }
            Err(reason) => unusable.push(json!({"id":note.first_note,"reason":reason})),
        }
    }
    for old in &checkpoint.rules {
        if !active.iter().any(|rule| rule.version.id == old.version.id) {
            checkpoint.evaluator.withdraw(&old.version);
        }
    }
    checkpoint.rules = active;
    if let Some(host) = &mut checkpoint.host {
        let settings = reader
            .settings_snapshot(d.config.requests.machine())
            .map_err(|e| e.to_string())?;
        let (quiet, budget) = settings.standalone_attention().map_err(|e| e.to_string())?;
        host.quiet_hours = quiet.map(|[from, to]| shared_signals::QuietHours { from, to });
        host.budget.per_hour = budget;
    }
    drop(reader);
    let output = if d.archive.is_primary() {
        if checkpoint.host.is_some() {
            return Err("primary checkpoint contains a local host policy".into());
        }
        checkpoint
            .evaluator
            .advance_in_mode(now, &calendar, EvaluationMode::Forward)
    } else {
        let host = checkpoint
            .host
            .as_mut()
            .ok_or("standalone host policy missing")?;
        checkpoint.evaluator.advance_in_mode(
            now,
            &calendar,
            EvaluationMode::Local {
                host,
                forwarded: &|_| shared_signals::Condition::Incomplete,
            },
        )
    }
    .map_err(|e| e.to_string())?;
    if !output.intents.is_empty() {
        return Err(
            "forwarded intent lacks a certified native input footprint and producer transport"
                .into(),
        );
    }
    let mut outbox = Vec::new();
    let mut notices = Vec::new();
    for delivery in output.deliveries {
        let rule = checkpoint
            .rules
            .iter()
            .find(|rule| rule.version == delivery.trigger.rule)
            .ok_or("delivery rule missing")?;
        let reference = delivery
            .basis
            .first()
            .map(|r| r.reference.clone())
            .unwrap_or_default();
        let notice = shared_signals::notifications::Notice {
            delivery_id: delivery.id.clone(),
            product: "polylogue".into(),
            icon: "polylogue".into(),
            rule: rule.rule.name.clone(),
            occurrence: delivery.trigger.occurrence.id.clone(),
            summary: rule.rule.name.clone(),
            body: delivery
                .basis
                .iter()
                .map(|r| r.reference.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
            reference: reference.clone(),
            command_hint: format!("polylogue read {reference}"),
            interrupt: delivery.channel == shared_signals::Channel::Interrupt,
            current: rule.current(),
            data_loss: false,
            budget_passive_count: None,
        };
        if matches!(
            delivery.channel,
            shared_signals::Channel::Action | shared_signals::Channel::Value
        ) {
            return Err("configured outlet admission is unavailable".into());
        }
        if notice.interrupt {
            notices.push(notice.clone());
        }
        outbox.push(Outbox {
            record: Delivery {
                recipient_kind: "operator".into(),
                recipient_id: None,
                sender: "signal".into(),
                event: "signal".into(),
                cwd: None,
                text: format!("{}: {}", notice.summary, notice.body),
                refs: delivery.basis.iter().map(|r| r.reference.clone()).collect(),
                left_out: vec![],
            },
            metadata: Metadata {
                delivery,
                notice,
                forward: None,
                outlet: None,
                execution: ExecutionEligibility::Live,
            },
        });
    }
    if let Ok(mut status) = d.signals.lock() {
        *status = json!({"rules":checkpoint.rules.iter().map(|r|json!({"id":r.version.id,"name":r.rule.name,"query":r.rule.query,"author":rules::author_name(&r.author)})).collect::<Vec<_>>(),"unusable":unusable,"held":held_rules});
    }
    let current = checkpoint.identity().map_err(|e| e.to_string())?;
    if prior == Some(current) {
        return Ok(vec![]);
    }
    let (_, confirmed) = persist(d, prior, checkpoint, outbox, vec![])?;
    if !confirmed {
        return Err("signal checkpoint is durably pending canonical confirmation".into());
    }
    Ok(notices)
}

pub fn run(d: Arc<Daemon>, stop: &std::sync::mpsc::Receiver<()>) {
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
    runtime.block_on(desktop::run(d, stop));
}

#[cfg(test)]
mod tests;
