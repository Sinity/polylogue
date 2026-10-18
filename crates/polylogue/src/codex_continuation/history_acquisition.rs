//! Import of Codex's paginated history database, stepped without blocking the
//! Codex home collector. The writer copies it in chunks between other work.
use super::*;
use polylogue_engine::owner::ConfiguredImport;
use std::sync::mpsc::TryRecvError;
struct Pending {
    request: ImportRequest,
    before: Option<polylogue_store::configured_codex::HistoryStamp>,
    reply: Option<Receiver<Result<ConfiguredImport, Error>>>,
}
pub(super) struct Acquisition {
    pending: Option<Pending>,
    poll: HistoryPoll,
    next: Instant,
    initial_finished: bool,
    attempt: u64,
}
impl Acquisition {
    pub fn new(config: &Config) -> Self {
        Self {
            pending: None,
            poll: HistoryPoll::new(
                Duration::from_secs(config.paginated_interval_seconds),
                Instant::now(),
            ),
            next: Instant::now(),
            initial_finished: config.paginated_history.is_none(),
            attempt: 0,
        }
    }
    pub fn initial_finished(&self) -> bool {
        self.initial_finished
    }
    fn finish(
        &mut self,
        config: &Config,
        status: &mut Status,
        result: Result<ConfiguredImport, Error>,
    ) {
        let pending = self.pending.take().expect("physical attempt");
        status.paginated_pending = false;
        match result {
            Ok(reply) => {
                status.paginated_snapshots = status
                    .paginated_snapshots
                    .saturating_add(u64::from(reply.acquired));
                status.paginated_input = Some(reply.input.id);
                status.paginated_length = Some(reply.input.length);
                status.paginated_state = "stored";
                self.poll.verified(
                    pending.before,
                    polylogue_store::configured_codex::history_stamp(config).ok(),
                    Instant::now(),
                );
                status.paginated_history = Some(diagnostic(if reply.acquired {
                    "stored a new snapshot"
                } else {
                    "unchanged; checked byte for byte"
                }));
            }
            Err(error) => {
                status.paginated_state = "unavailable";
                status.paginated_history = Some(diagnostic(error));
            }
        }
        self.initial_finished = true;
        self.attempt = self.attempt.saturating_add(1);
        self.next = Instant::now() + Duration::from_secs(config.paginated_interval_seconds);
    }
    pub fn step(&mut self, handle: &Handle, config: &Config, request: &str, status: &mut Status) {
        if config.paginated_history.is_none() {
            return;
        }
        if self.pending.is_none() && Instant::now() >= self.next {
            let before = polylogue_store::configured_codex::history_stamp(config).ok();
            let decision = self.poll.decision(before.as_ref(), Instant::now());
            if decision != HistoryDecision::Capture {
                status.paginated_state = "metadata_only";
                status.paginated_history = Some(diagnostic(
                    if decision == HistoryDecision::UnchangedMetadata {
                        "size and time unchanged; full check not due yet"
                    } else {
                        "changed; waiting for it to stay quiet, or for the hourly full check"
                    },
                ));
                self.next = Instant::now() + Duration::from_secs(config.paginated_interval_seconds);
                return;
            }
            let built = (|| -> Result<ImportRequest, String> {
                let path = polylogue_store::configured_codex::paginated_path(config)
                    .map_err(reason)?
                    .ok_or("paginated history is not configured")?;
                let seen_at = now()?;
                let candidate = Candidate::paginated(path.clone());
                let id = candidate
                    .observation_id(&format!("{request}:history:{}", self.attempt), seen_at)
                    .map_err(reason)?;
                Ok(ImportRequest {
                    format: candidate.format(),
                    context: vec![],
                    path: path.clone(),
                    observation: Observation {
                        id,
                        machine: config.machine.clone(),
                        host: config.host.clone(),
                        path: path.to_str().ok_or("history path must be UTF-8")?.into(),
                        seen_at,
                    },
                    metadata_allowance: 131072,
                })
            })();
            let request = match built {
                Ok(request) => request,
                Err(error) => {
                    status.paginated_pending = false;
                    status.paginated_state = "unavailable";
                    status.paginated_history = Some(diagnostic(error));
                    self.initial_finished = true;
                    self.next =
                        Instant::now() + Duration::from_secs(config.paginated_interval_seconds);
                    return;
                }
            };
            self.pending = Some(Pending {
                request,
                before,
                reply: None,
            });
            status.paginated_pending = true;
            status.paginated_state = "pending";
            status.paginated_history = Some(diagnostic("copying the paginated history"));
        }
        let Some(pending) = &mut self.pending else {
            return;
        };
        if pending.reply.is_none() {
            match handle.try_history_import(pending.request.clone()) {
                Ok(reply) => pending.reply = Some(reply),
                Err(Error::Busy) => return,
                Err(error) => {
                    self.finish(config, status, Err(error));
                    return;
                }
            }
        }
        match pending.reply.as_ref().expect("physical reply").try_recv() {
            Ok(result) => self.finish(config, status, result),
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => self.finish(config, status, Err(Error::Stopped)),
        }
    }
}
