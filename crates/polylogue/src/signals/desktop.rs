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
    let Ok(context) = operation(d) else {
        return denied();
    };
    let clock = &context.clock;
    let Ok(reader) = d.reads.reader() else {
        return denied();
    };
    let Ok(Some(checkpoint)) = reader.signal_checkpoint(&context.machine) else {
        return denied();
    };
    let Ok(attention) = local_attention(&reader, &context) else {
        return denied();
    };
    let policy = attention.identity;

    let calendar = DayCalendar::new(clock.day.clone());
    let now = clock.now.as_millisecond();
    let quiet_until = attention
        .quiet
        .and_then(|hours| calendar.quiet_until(now, hours));
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

fn persist_update(d: &Daemon, update: &NotificationUpdate) -> Result<bool, Held> {
    if update.released.is_some() {
        return Err(
            "notification ownership transfer requires the confirmed producer outbox".into(),
        );
    }
    let context = operation(d)?;
    let reader = d.reads.reader().map_err(|e| e.to_string())?;
    let previous = reader
        .signal_checkpoint(&context.machine)
        .map_err(|e| e.to_string())?;
    let prior = previous
        .as_ref()
        .map(Checkpoint::identity)
        .transpose()
        .map_err(|e| e.to_string())?;
    let mut checkpoint = match previous {
        Some(value) => value,
        None => {
            let attention = if d.archive.is_primary() {
                None
            } else {
                Some(local_attention(&reader, &context)?)
            };
            initial(&context, attention.as_ref())?
        }
    };
    drop(reader);
    checkpoint.notifications = Some(update.state.clone());
    let events = update
        .receipts
        .iter()
        .cloned()
        .map(|receipt| Event::Desktop { receipt })
        .collect();
    Ok(persist(d, &context, prior, checkpoint, vec![], events)?.1)
}

fn recover(
    d: &Daemon,
    state: &NotificationState,
    requests: &mpsc::UnboundedSender<NotificationRequest>,
) -> Result<(), String> {
    let reader = d.reads.reader().map_err(|e| e.to_string())?;
    reader
        .visit_signals(|stored| {
            let metadata = stored.outbox.metadata;
            let attempted = state.attempt(&stored.id).is_some()
                || stored
                    .events
                    .iter()
                    .any(|event| matches!(event.event, Event::Desktop { .. }));
            if metadata.execution == ExecutionEligibility::Live
                && metadata.delivery.channel == shared_signals::Channel::Interrupt
                && !attempted
            {
                requests
                    .send(NotificationRequest::Deliver(Box::new(metadata.notice)))
                    .map_err(|_| polylogue_store::StoreError::WriterGone)?;
            }
            Ok(())
        })
        .map_err(|e| e.to_string())
}

pub(super) async fn run(d: Arc<Daemon>, stop: &std::sync::mpsc::Receiver<()>) {
    let Some(writes) = &d.writes else {
        return;
    };
    // The subscription precedes both startup snapshots. It carries wakeups,
    // not work counters or authoritative observation timestamps.
    let commits = writes.completion_commits();
    let wake = commits.wake_handle();
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stopped = Arc::clone(&done);
    let (changed, mut changes) = mpsc::channel(1);
    let bridge = tokio::task::spawn_blocking(move || {
        while !stopped.load(std::sync::atomic::Ordering::Acquire) {
            if commits.recv().is_err() {
                break;
            }
            if stopped.load(std::sync::atomic::Ordering::Acquire) {
                break;
            }
            if matches!(
                changed.try_send(()),
                Err(mpsc::error::TrySendError::Closed(()))
            ) {
                break;
            }
        }
    });
    let result = drive(&d, stop, &mut changes).await;
    done.store(true, std::sync::atomic::Ordering::Release);
    let _ = wake.try_send(());
    let _ = bridge.await;
    if let Err(error) = result {
        report(&d, error);
    }
}

async fn drive(
    d: &Arc<Daemon>,
    stop: &std::sync::mpsc::Receiver<()>,
    changes: &mut mpsc::Receiver<()>,
) -> Result<(), Held> {
    let (requests, receiver) = mpsc::unbounded_channel();
    let (updates, mut returned) = mpsc::unbounded_channel();
    let worker = if d.archive.is_primary() {
        None
    } else {
        let state = load(d)?.and_then(|c| c.notifications).unwrap_or_default();
        recover(d, &state, &requests)?;
        let deliverer = Deliverer::new(
            connect(d).await,
            d.config.signals.notification_config(),
            state,
        );
        let wall = d.config.requests.clone();
        Some(tokio::spawn(signal_delivery::run(
            deliverer,
            receiver,
            updates,
            policy(Arc::clone(d)),
            Arc::new(move || wall.now().unix_ms()),
        )))
    };
    let mut stop_probe = tokio::time::interval(std::time::Duration::from_millis(250));
    let mut evaluate = true;
    let mut due = d.config.requests.now().unix_ms();
    let mut failure = None;
    loop {
        if !matches!(stop.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty)) {
            break;
        }
        if evaluate {
            let now = d.config.requests.now().unix_ms();
            due = now.saturating_add(30_000);
            match tick(d) {
                Ok(tick) => {
                    if let Some(next) = tick.next {
                        due = due.min(next);
                    }
                    for notice in tick.notices {
                        if requests
                            .send(NotificationRequest::Deliver(Box::new(notice)))
                            .is_err()
                        {
                            failure = Some(Held::from("desktop worker stopped"));
                            break;
                        }
                    }
                }
                Err(error) => report(d, error),
            }
            if failure.is_some() {
                break;
            }
            evaluate = false;
        }
        let wait = due.saturating_sub(d.config.requests.now().unix_ms()).max(1) as u64;
        tokio::select! {
            Some(update)=returned.recv(),if worker.is_some()=>{
                match persist_update(d,&update) {
                    Ok(true)=>{let _=update.persisted.send(());}
                    Ok(false)=>{failure=Some(Held::from("notification receipt is pending canonical confirmation"));break;}
                    Err(error)=>{failure=Some(error);break;}
                }
            }
            changed=changes.recv()=>{if changed.is_none(){break;}evaluate=true;}
            _=tokio::time::sleep(std::time::Duration::from_millis(wait))=>evaluate=true,
            _=stop_probe.tick()=>{}
        }
    }
    drop(requests);
    if let Some(worker) = worker {
        worker.abort();
        let _ = worker.await;
    }
    match failure {
        Some(error) => Err(error),
        None => Ok(()),
    }
}
