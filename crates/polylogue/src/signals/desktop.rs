use super::*;
use shared_signals::notifications::{NotificationState, SendDecision};
use shared_signals::{Calendar, Condition};
use signal_delivery::{
    Authorization, Deliverer, NotificationRequest, NotificationUpdate, PolicyCheck, PolicyRequest,
};
use tokio::sync::mpsc;

/// The SQLite read snapshot ends inside this synchronous calculation. The
/// returned future owns immutable decisions and no database handle or guard.
fn admission(d: &Daemon, request: PolicyRequest) -> Authorization {
    let denied = || Authorization {
        decisions: request
            .notices
            .iter()
            .map(|n| (n.delivery_id.clone(), SendDecision::Cancel))
            .collect(),
    };
    if d.archive.is_primary() {
        return denied();
    }
    let Ok(clock) = d.query_clock() else {
        return denied();
    };
    let Ok(reader) = d.reads.reader() else {
        return denied();
    };
    let Ok(Some(checkpoint)) = reader.signal_checkpoint(d.config.requests.machine()) else {
        return denied();
    };
    let Ok(settings) = reader.settings_snapshot(d.config.requests.machine()) else {
        return denied();
    };
    let Ok((quiet, _)) = settings.standalone_attention() else {
        return denied();
    };
    let Ok(catalog) = reader.settings_catalog(d.config.requests.machine()) else {
        return denied();
    };
    let Ok(policy_bytes) = serde_json::to_vec(&catalog) else {
        return denied();
    };
    let policy = blake3::hash(&policy_bytes).to_hex().to_string();
    let calendar = DayCalendar::new(clock.day.clone());
    let now = clock.now.as_millisecond();
    let quiet_until = quiet
        .and_then(|[from, to]| calendar.quiet_until(now, shared_signals::QuietHours { from, to }));
    let mut outboxes = BTreeMap::new();
    if reader
        .visit_signals(|value| {
            outboxes.insert(value.id.clone(), value);
            Ok(())
        })
        .is_err()
    {
        return denied();
    }
    let decisions = request
        .notices
        .iter()
        .map(|notice| {
            let decision = match outboxes.get(&notice.delivery_id) {
                Some(outbox) if outbox.outbox.metadata.execution == ExecutionEligibility::Live => {
                    let metadata = &outbox.outbox.metadata;
                    if !checkpoint
                        .rules
                        .iter()
                        .any(|r| r.version == metadata.delivery.trigger.rule && r.rule.enabled)
                    {
                        SendDecision::Cancel
                    } else if let Some(at) = quiet_until {
                        SendDecision::HoldUntil {
                            at,
                            policy: policy.clone(),
                        }
                    } else {
                        // Recheck the owning evaluator's complete observation. A missing fresh
                        // producer proof never becomes permission when a hold deadline passes.
                        let intent = shared_signals::FiredIntent {
                            delivery: metadata.delivery.clone(),
                            author: shared_signals::Author::Agent("adapter".into()),
                            quiet_hours: None,
                            rate_limit: None,
                            current: notice.current,
                        };
                        match checkpoint.evaluator.condition(&intent, now) {
                            Condition::Ended => SendDecision::Cancel,
                            Condition::Incomplete => SendDecision::HoldUntil {
                                at: now.saturating_add(30_000),
                                policy: policy.clone(),
                            },
                            Condition::Matching => SendDecision::HoldUntil {
                                at: now.saturating_add(30_000),
                                policy: format!("{policy}:input-proof-unavailable"),
                            },
                        }
                    }
                }
                _ => SendDecision::Cancel,
            };
            (notice.delivery_id.clone(), decision)
        })
        .collect();
    Authorization { decisions }
}

pub(super) fn policy(d: Arc<Daemon>) -> PolicyCheck {
    Arc::new(move |request| {
        let d = Arc::clone(&d);
        Box::pin(async move { admission(&d, request) })
    })
}

async fn connect(d: &Daemon) -> Option<zbus::Connection> {
    let address = d.config.notification_bus_address.as_deref()?;
    match zbus::connection::Builder::address(address) {
        Ok(builder) => builder.build().await.ok(),
        Err(_) => None,
    }
}

fn persist_update(d: &Daemon, update: &NotificationUpdate) -> Result<bool, String> {
    if update.released.is_some() {
        return Err(
            "notification ownership transfer requires the confirmed producer outbox".into(),
        );
    }
    let clock = d.query_clock().map_err(|e| e.to_string())?;
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
    checkpoint.notifications = Some(update.state.clone());
    let events = update
        .receipts
        .iter()
        .cloned()
        .map(|receipt| Event::Desktop { receipt })
        .collect();
    persist(d, prior, checkpoint, vec![], events).map(|(_, confirmed)| confirmed)
}

pub(super) async fn run(d: Arc<Daemon>, stop: &std::sync::mpsc::Receiver<()>) {
    if d.writes.is_none() {
        return;
    }
    let (requests, receiver) = mpsc::unbounded_channel();
    let (updates, mut returned) = mpsc::unbounded_channel();
    let worker = if d.archive.is_primary() {
        None
    } else {
        let state = match load(&d) {
            Ok(value) => value
                .and_then(|c| c.notifications)
                .unwrap_or_else(NotificationState::default),
            Err(error) => {
                held(&d, &error);
                return;
            }
        };
        let deliverer = Deliverer::new(
            connect(&d).await,
            d.config.signals.notification_config(),
            state,
        );
        let wall = d.config.requests.clone();
        Some(tokio::spawn(signal_delivery::run(
            deliverer,
            receiver,
            updates,
            policy(Arc::clone(&d)),
            Arc::new(move || wall.now().unix_ms()),
        )))
    };
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
    loop {
        if !matches!(stop.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty)) {
            break;
        }
        tokio::select! {
         Some(update)=returned.recv(),if worker.is_some()=>{
          match persist_update(&d,&update) {
           Ok(true)=>{let _=update.persisted.send(());}
           Ok(false)=>held(&d,"notification receipt is pending canonical confirmation"),
           Err(error)=>held(&d,&error),
          }
         }
         _=interval.tick()=>{
          match tick(&d){
           Ok(notices)=>{for notice in notices {let _=requests.send(NotificationRequest::Deliver(Box::new(notice)));}}
           Err(error)=>held(&d,&error),
          }
         }
        }
    }
    drop(requests);
    if let Some(worker) = worker {
        worker.abort();
        let _ = worker.await;
    }
}
