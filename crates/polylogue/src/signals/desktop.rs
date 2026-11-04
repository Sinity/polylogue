use super::*;
use shared_signals::notifications::NotificationState;
use signal_delivery::{Deliverer, NotificationRequest, NotificationUpdate};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

async fn connect(d: &Daemon) -> Option<zbus::Connection> {
    let address = d.config.notification_bus_address.as_deref()?;
    match zbus::connection::Builder::address(address) {
        Ok(builder) => builder.build().await.ok(),
        Err(_) => None,
    }
}

fn persist_update(d: &Daemon, update: &NotificationUpdate) -> Result<bool, Held> {
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
            initial(d, &context, attention.as_ref())?
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
                && metadata.delivery.decision.interrupts()
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

pub(super) async fn run(d: Arc<Daemon>, stop: CancellationToken) {
    let Some(writes) = &d.writes else {
        return;
    };
    // The subscription precedes both startup snapshots. It carries wakeups,
    // not work counters or authoritative observation timestamps.
    let commits = writes.completion_commits();
    #[cfg(test)]
    super::observations::emit(&d, super::observations::Phase::Subscribed);
    let wake = commits.wake_handle();
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stopped = Arc::clone(&done);
    let (changed, mut changes) = mpsc::channel(1);
    #[cfg(test)]
    let trace_daemon = Arc::clone(&d);
    let bridge = tokio::task::spawn_blocking(move || {
        while !stopped.load(std::sync::atomic::Ordering::Acquire) {
            if commits.recv().is_err() {
                break;
            }
            #[cfg(test)]
            super::observations::emit(
                &trace_daemon,
                super::observations::Phase::SubscriptionReceived,
            );
            if stopped.load(std::sync::atomic::Ordering::Acquire) {
                break;
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
    let _ = bridge.await;
    if let Err(error) = result {
        report(&d, error);
    }
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
    let worker = if d.archive.is_primary() {
        None
    } else {
        let state = load(d)?.and_then(|c| c.notifications).unwrap_or_default();
        recover(d, &state, &requests)?;
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
    loop {
        if stop.is_cancelled() {
            break;
        }
        if evaluate {
            let now = d.config.requests.now().unix_ms();
            due = now.saturating_add(30_000);
            #[cfg(test)]
            super::observations::emit(d, super::observations::Phase::TickStarted { at: now });
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
                Err(error) => {
                    #[cfg(test)]
                    super::observations::emit(
                        d,
                        super::observations::Phase::TickFailed(json!(&error)),
                    );
                    report(d, error);
                }
            }
            if failure.is_some() {
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
                    failure = Some(Held::from("desktop worker stopped"));
                    break;
                };
                match persist_update(d,&update) {
                    Ok(true)=>{let _=update.persisted.send(());}
                    Ok(false)=>{failure=Some(Held::from("notification receipt is pending canonical confirmation"));break;}
                    Err(error)=>{failure=Some(error);break;}
                }
            }
            changed=changes.recv()=>{
                if changed.is_none(){break;}
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
        match persist_update(d, &update) {
            Ok(true) => {
                let _ = update.persisted.send(());
            }
            Ok(false) => {
                failure.get_or_insert_with(|| {
                    Held::from("notification receipt is pending canonical confirmation")
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

/// Converts the actual native worker stop channel into asynchronous owner
/// cancellation. The timeout only lets an already-cancelled owner join this
/// bridge; receiving a stop or disconnect cancels immediately.
pub(super) struct StopBridge {
    cancellation: CancellationToken,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl StopBridge {
    pub(super) fn new(stopped: std::sync::mpsc::Receiver<()>) -> std::io::Result<Self> {
        let cancellation = CancellationToken::new();
        if !matches!(
            stopped.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ) {
            cancellation.cancel();
            return Ok(Self {
                cancellation,
                thread: None,
            });
        }
        let owner = cancellation.clone();
        let thread = std::thread::Builder::new()
            .name("polylogue-signal-stop".into())
            .spawn(move || {
                loop {
                    match stopped.recv_timeout(std::time::Duration::from_millis(100)) {
                        Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                            owner.cancel();
                            break;
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) if owner.is_cancelled() => {
                            break;
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    }
                }
            })?;
        Ok(Self {
            cancellation,
            thread: Some(thread),
        })
    }
    pub(super) fn token(&self) -> CancellationToken {
        self.cancellation.clone()
    }
    pub(super) fn finish(&mut self) -> std::io::Result<()> {
        self.cancellation.cancel();
        if let Some(thread) = self.thread.take() {
            thread
                .join()
                .map_err(|_| std::io::Error::other("signal stop bridge panicked"))?;
        }
        Ok(())
    }
}
impl Drop for StopBridge {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

#[cfg(test)]
mod tests;
