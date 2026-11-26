//! Desktop notification delivery: the bus connection, receipt persistence,
//! recovery of undelivered notices, and the loop that drives them beside the
//! signal evaluator.
use super::*;
use shared_signals::notifications::NotificationState;
use signal_delivery::{Deliverer, NotificationRequest, NotificationUpdate};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

async fn connect(d: &Daemon) -> Option<zbus::Connection> {
    let address = d.config.notification_bus_address.as_deref()?;
    let builder = zbus::connection::Builder::address(address).ok()?;
    builder.build().await.ok()
}

async fn persist_update(d: &Daemon, update: &NotificationUpdate) -> Result<bool, Held> {
    let context = d
        .operation_context_async(&polylogue_reading::operation::Context::worker_authorship(
            "polylogue-signals",
        ))
        .await
        .map_err(Held::from)?;
    let reader = d.reads.reader().map_err(|e| e.to_string())?;
    let (prior, mut checkpoint) = load_or_initial(d, &reader, &context)?;
    drop(reader);
    checkpoint.notifications = Some(update.state.clone());
    let events = update
        .receipts
        .iter()
        .cloned()
        .map(|receipt| Event::Desktop { receipt })
        .collect();
    persist(d, &context, prior, checkpoint, vec![], events).map_err(Held::from)
}

fn recover(
    d: &Daemon,
    state: &NotificationState,
    requests: &mpsc::UnboundedSender<NotificationRequest>,
) -> polylogue_store::Result<()> {
    let reader = d.reads.reader()?;
    reader.visit_signals(|stored| {
        let metadata = stored.pending_delivery.metadata;
        let attempted = state.attempt(&stored.id).is_some()
            || stored
                .events
                .iter()
                .any(|event| matches!(event.event, Event::Desktop { .. }));
        if metadata.execution == RunsHere::Live
            && metadata.delivery.decision.interrupts()
            && !attempted
        {
            requests
                .send(NotificationRequest::Deliver(Box::new(metadata.notice)))
                .map_err(|_| polylogue_store::StoreError::WriterGone)?;
        }
        Ok(())
    })
}

pub(super) async fn run(d: Arc<Daemon>, stop: CancellationToken) -> Result<(), Held> {
    let Some(writes) = &d.writes else {
        return Ok(());
    };
    // The subscription precedes both startup snapshots. It carries wakeups,
    // not work counters or observation timestamps to rely on.
    let commits = writes.completion_commits();
    #[cfg(test)]
    super::observations::emit(&d, super::observations::Phase::Subscribed);
    let wake = commits.wake_handle();
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stopped = Arc::clone(&done);
    let (changed, mut changes) = mpsc::channel(1);
    #[cfg(test)]
    let trace_daemon = Arc::clone(&d);
    let archive = d.archive.clone();
    let bridge = tokio::task::spawn_blocking(move || {
        let mut building = archive.index_build().is_some();
        while !stopped.load(std::sync::atomic::Ordering::Acquire) {
            let Ok(committed) = commits.recv() else {
                break;
            };
            #[cfg(test)]
            super::observations::emit(
                &trace_daemon,
                super::observations::Phase::SubscriptionReceived,
            );
            if stopped.load(std::sync::atomic::Ordering::Acquire) {
                break;
            }
            // A tick reads the rule set, the attention settings and its own
            // checkpoint, all authored records. A rule's query reads
            // content, but every rule is held until all its dependencies are confirmed
            // (`rules::definition`), so while none runs, a commit of content
            // or of a read's run record changes nothing a tick reads.
            // Installing a built index is a content commit, but it is the
            // first commit after which a tick can read anything.
            let installed = building && archive.index_build().is_none();
            building = archive.index_build().is_some();
            if !committed.authored && !installed {
                #[cfg(test)]
                super::observations::emit(&trace_daemon, super::observations::Phase::CueIgnored);
                continue;
            }
            let forwarded = changed.try_send(());
            #[cfg(test)]
            super::observations::emit(
                &trace_daemon,
                super::observations::Phase::CueForwarded {
                    queued: forwarded.is_ok(),
                },
            );
            if matches!(forwarded, Err(mpsc::error::TrySendError::Closed(()))) {
                break;
            }
        }
    });
    let result = drive(&d, &stop, &mut changes).await;
    done.store(true, std::sync::atomic::Ordering::Release);
    let _ = wake.try_send(());
    let joined = bridge
        .await
        .map_err(|error| Held::from(format!("signal commit bridge: {error}")));
    result.and(joined)
}

async fn drive(
    d: &Arc<Daemon>,
    stop: &CancellationToken,
    changes: &mut mpsc::Receiver<()>,
) -> Result<(), Held> {
    if stop.is_cancelled() {
        return Ok(());
    }
    let (requests, receiver) = mpsc::unbounded_channel();
    let (updates, mut returned) = mpsc::unbounded_channel();
    let worker = {
        let restored = loop {
            let restored = (|| -> polylogue_store::Result<NotificationState> {
                let state = load(d)?.and_then(|c| c.notifications).unwrap_or_default();
                recover(d, &state, &requests)?;
                Ok(state)
            })();
            // Nothing is readable until the index being built is installed,
            // which is a commit: restore once it is.
            if !matches!(restored, Err(polylogue_store::StoreError::IndexBuilding(_))) {
                break restored;
            }
            tokio::select! {
                _ = stop.cancelled() => return Ok(()),
                changed = changes.recv() => if changed.is_none() {
                    return Ok(());
                },
            }
        };
        let state = match restored {
            Ok(state) => state,
            // The stop request ends reads and closes the reader pool. A
            // worker starting in that same stop has no restoration left to
            // perform; corruption and other failures keep their diagnosis.
            Err(
                polylogue_store::StoreError::Stopped | polylogue_store::StoreError::ReadPoolClosed,
            ) if d.is_stopping() => return Ok(()),
            Err(error) => return Err(error.to_string().into()),
        };
        let connection = tokio::select! {
            _ = stop.cancelled() => return Ok(()),
            connection = connect(d) => connection,
        };
        let deliverer = Deliverer::new(connection, d.config.signals.notification_config(), state);
        // The worker ends when the request channel closes; stopping drops it.
        Some(tokio::spawn(signal_delivery::run(
            deliverer, receiver, updates,
        )))
    };
    let mut evaluate = true;
    let mut due = d.config.requests.now().unix_ms();
    let mut failure = None;
    let mut worker_ended = false;
    loop {
        if stop.is_cancelled() {
            break;
        }
        // Rules read the index; while it is being built with nothing to
        // serve, the next commit (its install among them) evaluates again.
        if evaluate && d.archive.index_files().is_ok_and(|files| !files.serves()) {
            evaluate = false;
            due = d.config.requests.now().unix_ms().saturating_add(30_000);
        }
        if evaluate {
            let now = d.config.requests.now().unix_ms();
            due = now.saturating_add(30_000);
            #[cfg(test)]
            super::observations::emit(d, super::observations::Phase::TickStarted { at: now });
            let authorship =
                polylogue_reading::operation::Context::worker_authorship("polylogue-signals");
            let context = tokio::select! {
                _ = stop.cancelled() => break,
                context = d.operation_context_async(&authorship) => context,
            };
            match context
                .map_err(Held::from)
                .and_then(|context| tick(d, context))
            {
                Ok(tick) => {
                    if let Some(next) = tick.next {
                        due = due.min(next);
                    }
                    for notice in tick.notices {
                        if requests
                            .send(NotificationRequest::Deliver(Box::new(notice)))
                            .is_err()
                        {
                            worker_ended = true;
                            break;
                        }
                    }
                }
                Err(error) => {
                    #[cfg(test)]
                    super::observations::emit(
                        d,
                        super::observations::Phase::TickFailed(json!(&error)),
                    );
                    report(d, &error);
                }
            }
            if worker_ended {
                break;
            }
            evaluate = false;
        }
        let wait = due.saturating_sub(d.config.requests.now().unix_ms()).max(1) as u64;
        #[cfg(test)]
        super::observations::emit(d, super::observations::Phase::ControllerWaiting);
        tokio::select! {
            update=returned.recv(),if worker.is_some()=>{
                let Some(update) = update else {
                    worker_ended = true;
                    break;
                };
                match persist_update(d,&update).await {
                    Ok(true)=>{let _=update.persisted.send(());}
                    Ok(false)=>{failure=Some(Held::from("notification receipt is pending confirmation"));break;}
                    Err(error)=>{failure=Some(error);break;}
                }
            }
            changed=changes.recv()=>{
                if changed.is_none(){
                    if !stop.is_cancelled() && !d.is_stopping() {
                        failure=Some(Held::from("signal commit subscription stopped"));
                    }
                    break;
                }
                #[cfg(test)]
                super::observations::emit(d, super::observations::Phase::ControllerWake { refresh: false });
                evaluate=true;
            }
            _=tokio::time::sleep(std::time::Duration::from_millis(wait))=>{
                #[cfg(test)]
                super::observations::emit(d, super::observations::Phase::ControllerWake { refresh: true });
                evaluate=true;
            }
            _=stop.cancelled()=>break,
        }
    }
    stop.cancel();
    drop(requests);
    if let Some(worker) = worker
        && let Err(error) = drain_worker(d, &mut returned, worker).await
    {
        failure.get_or_insert(error);
    }
    if worker_ended {
        // Joining above keeps a delivery error; channel closure alone is
        // only the fallback when that worker returned successfully.
        failure.get_or_insert_with(|| Held::from("desktop worker stopped"));
    }
    match failure {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Receipt persistence is cleanup after cancellation, never another effect.
async fn drain_worker(
    d: &Daemon,
    returned: &mut mpsc::UnboundedReceiver<NotificationUpdate>,
    worker: tokio::task::JoinHandle<zbus::Result<()>>,
) -> Result<(), Held> {
    let mut failure = None;
    while let Some(update) = returned.recv().await {
        match persist_update(d, &update).await {
            Ok(true) => {
                let _ = update.persisted.send(());
            }
            Ok(false) => {
                failure.get_or_insert_with(|| {
                    Held::from("notification receipt is pending confirmation")
                });
            }
            Err(error) => {
                failure.get_or_insert(error);
            }
        }
    }
    match worker.await {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            failure.get_or_insert_with(|| Held::from(format!("desktop worker: {error}")));
        }
        Err(error) => {
            failure.get_or_insert_with(|| Held::from(format!("desktop worker join: {error}")));
        }
    }
    failure.map_or(Ok(()), Err)
}

#[cfg(test)]
mod tests;
