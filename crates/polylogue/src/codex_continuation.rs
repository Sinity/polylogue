//! Collect from a configured Codex home: poll it in short turns, each limited
//! in files and bytes, and import changed files through the writer. The limits
//! are per turn; every file is reached eventually.
use polylogue_engine::owner::{Error, Handle, ImportRequest, Observation};
use polylogue_model::BlobHash;
use polylogue_store::configured_codex::{Availability, Candidate, Config, RolloutCursor};
use serde::Serialize;
use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, RecvTimeoutError},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
mod history_acquisition;
const INTERVAL: Duration = Duration::from_secs(1);
const REQUESTS: usize = 512;
const BYTES: u64 = 32 * 1024 * 1024;
#[derive(Clone, Debug, Serialize)]
pub(crate) struct Diagnostic {
    pub message: String,
    pub truncated: bool,
}
fn diagnostic(error: impl ToString) -> Diagnostic {
    let text = error.to_string();
    Diagnostic {
        truncated: text.chars().count() > 256,
        message: text.chars().take(256).collect(),
    }
}
fn reason(error: impl ToString) -> String {
    error.to_string()
}
#[derive(Clone, Debug, Serialize)]
pub(crate) struct Status {
    pub mode: &'static str,
    pub state: &'static str,
    pub pause_between_turns_ms: u64,
    pub per_turn_entry_limit: usize,
    pub per_turn_candidate_limit: usize,
    pub per_turn_charged_source_byte_limit: u64,
    pub pass_active: bool,
    pub completed_passes: u64,
    pub enumeration_cycles_completed: u64,
    pub discovery_failures: u64,
    pub unchanged_passes: u64,
    pub successful_snapshots: u64,
    pub acquired_source_bytes: u64,
    pub last_attempt_finished_at: Option<i64>,
    pub last_pass: Option<Pass>,
    pub last_error: Option<Diagnostic>,
    pub conversation_coverage: Option<bool>,
    pub paginated_history: Option<Diagnostic>,
    pub paginated_snapshots: u64,
    pub paginated_pending: bool,
    pub paginated_state: &'static str,
    /// The last stored snapshot of the paginated history, not necessarily the
    /// provider's current state.
    pub paginated_input: Option<String>,
    pub paginated_length: Option<u64>,
}
#[derive(Clone, Debug, Serialize)]
pub(crate) struct Pass {
    pub catalog: Availability,
    pub rollouts: Availability,
    pub enumeration_advanced: bool,
    pub enumeration_cycle: u64,
    pub enumeration_cycle_complete: bool,
    pub visited_entries: usize,
    pub admitted_rollout_paths: usize,
    pub rollout_count_unit: &'static str,
    pub rollout_count_scope: &'static str,
    pub evaluated_candidates: usize,
    pub examined_files: usize,
    pub unchanged_files: usize,
    pub acquired_files: usize,
    pub failed_files: usize,
    pub charged_source_bytes: u64,
    pub deferred_candidates: usize,
    pub evaluation_complete: bool,
    pub budget_exhausted: bool,
    pub history_state: &'static str,
    pub history_bytes: Option<u64>,
    pub failure_sample: Vec<Diagnostic>,
}
impl Status {
    pub fn initial() -> Self {
        Self {
            mode: "codex_home",
            state: "polling",
            pause_between_turns_ms: 1000,
            per_turn_entry_limit: REQUESTS,
            per_turn_candidate_limit: REQUESTS,
            per_turn_charged_source_byte_limit: BYTES,
            pass_active: false,
            completed_passes: 0,
            enumeration_cycles_completed: 0,
            discovery_failures: 0,
            unchanged_passes: 0,
            successful_snapshots: 0,
            acquired_source_bytes: 0,
            last_attempt_finished_at: None,
            last_pass: None,
            last_error: None,
            conversation_coverage: None,
            paginated_history: None,
            paginated_snapshots: 0,
            paginated_pending: false,
            paginated_state: "disabled",
            paginated_input: None,
            paginated_length: None,
        }
    }
}
fn now() -> Result<i64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(reason)
        .and_then(|d| i64::try_from(d.as_micros()).map_err(reason))
}
fn pause(stop: &AtomicBool, duration: Duration) {
    let until = Instant::now() + duration;
    while !stop.load(Ordering::Acquire) && Instant::now() < until {
        thread::sleep(Duration::from_millis(10));
    }
}
fn receive<T>(rx: Receiver<Result<T, Error>>, stop: &AtomicBool) -> Result<T, Error> {
    loop {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(result) => return result,
            Err(RecvTimeoutError::Disconnected) => return Err(Error::Stopped),
            Err(RecvTimeoutError::Timeout) if stop.load(Ordering::Acquire) => {
                return Err(Error::Stopped);
            }
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
}
fn permission(handle: &Handle, stop: &AtomicBool) -> Result<(), String> {
    let until = Instant::now() + Duration::from_secs(30);
    loop {
        let result = match handle.try_check_import_acquisition() {
            Ok(rx) => receive(rx, stop),
            Err(e) => Err(e),
        };
        match result {
            Ok(()) => return Ok(()),
            Err(Error::Busy) if Instant::now() < until && !stop.load(Ordering::Acquire) => {
                pause(stop, Duration::from_millis(10))
            }
            Err(e) => return Err(reason(e)),
        }
    }
}
fn import(
    handle: &Handle,
    config: &Config,
    candidate: Candidate,
    hash: BlobHash,
    length: u64,
    request: &str,
    seen_at: i64,
    stop: &AtomicBool,
) -> Result<bool, String> {
    let path = candidate.path().to_path_buf();
    let id = candidate.observation_id(request, seen_at).map_err(reason)?;
    let request = ImportRequest {
        format: candidate.format(),
        context: vec![],
        path: path.clone(),
        observation: Observation {
            id,
            machine: config.machine.clone(),
            host: config.host.clone(),
            path: path.to_str().ok_or("configured path must be UTF-8")?.into(),
            seen_at,
        },
        metadata_allowance: 131072,
    };
    let until = Instant::now() + Duration::from_secs(30);
    loop {
        if stop.load(Ordering::Acquire) {
            return Err("continuation stopped".into());
        }
        let result =
            match handle.try_configured_import(request.clone(), candidate.clone(), hash, length) {
                Ok(rx) => receive(rx, stop),
                Err(e) => Err(e),
            };
        match result {
            Ok(reply) => return Ok(reply.acquired),
            Err(Error::Busy) if Instant::now() < until => {}
            Err(e) => return Err(reason(e)),
        }
        pause(stop, Duration::from_millis(10));
    }
}
#[derive(Debug, PartialEq, Eq)]
enum HistoryDecision {
    Capture,
    UnchangedMetadata,
    WaitingForQuiet,
}
struct HistoryPoll {
    interval: Duration,
    verified_stamp: Option<polylogue_store::configured_codex::HistoryStamp>,
    observed_stamp: Option<polylogue_store::configured_codex::HistoryStamp>,
    quiet_since: Instant,
    verified_at: Option<Instant>,
}
impl HistoryPoll {
    fn new(interval: Duration, now: Instant) -> Self {
        Self {
            interval,
            verified_stamp: None,
            observed_stamp: None,
            quiet_since: now,
            verified_at: None,
        }
    }
    fn decision(
        &mut self,
        stamp: Option<&polylogue_store::configured_codex::HistoryStamp>,
        now: Instant,
    ) -> HistoryDecision {
        let Some(stamp) = stamp else {
            return HistoryDecision::Capture;
        };
        if self.observed_stamp.as_ref() != Some(stamp) {
            self.observed_stamp = Some(stamp.clone());
            self.quiet_since = now;
        }
        let Some(verified_at) = self.verified_at else {
            return HistoryDecision::Capture;
        };
        if now.duration_since(verified_at) >= Duration::from_secs(3600) {
            return HistoryDecision::Capture;
        }
        if self.verified_stamp.as_ref() == Some(stamp) {
            return HistoryDecision::UnchangedMetadata;
        }
        if now.duration_since(self.quiet_since) >= self.interval {
            HistoryDecision::Capture
        } else {
            HistoryDecision::WaitingForQuiet
        }
    }
    fn verified(
        &mut self,
        before: Option<polylogue_store::configured_codex::HistoryStamp>,
        after: Option<polylogue_store::configured_codex::HistoryStamp>,
        now: Instant,
    ) {
        self.verified_stamp = before.filter(|stamp| after.as_ref() == Some(stamp));
        self.verified_at = Some(now);
    }
}
#[cfg(test)]
#[path = "codex_continuation/history_poll_tests.rs"]
mod history_poll_tests;

pub(crate) fn run(
    handle: Handle,
    config: Config,
    request: String,
    stop: Arc<AtomicBool>,
    shared: Arc<Mutex<Option<Status>>>,
    continue_collecting: bool,
) {
    let mut cursor = RolloutCursor::default();
    let mut pending = VecDeque::new();
    let mut previous: Option<Pass> = None;
    let mut status = Status::initial();
    // Report the paginated history as pending even if the first attempt fails
    // before it starts.
    status.paginated_pending = config.paginated_history.is_some();
    status.paginated_state = if status.paginated_pending {
        "pending"
    } else {
        "disabled"
    };
    let mut physical = history_acquisition::Acquisition::new(&config);
    if !continue_collecting {
        while !stop.load(Ordering::Acquire) && !physical.initial_finished() {
            physical.step(&handle, &config, &request, &mut status);
            *shared.lock().expect("continuation status") = Some(status.clone());
            if !physical.initial_finished() {
                pause(&stop, Duration::from_millis(10));
            }
        }
        status.state = if stop.load(Ordering::Acquire) {
            "stopped"
        } else {
            "startup_attempt_finished"
        };
        *shared.lock().expect("continuation status") = Some(status);
        return;
    }
    while !stop.load(Ordering::Acquire) {
        pause(&stop, INTERVAL);
        if stop.load(Ordering::Acquire) {
            break;
        }
        status.pass_active = true;
        *shared.lock().expect("continuation status") = Some(status.clone());
        if let Err(error) = permission(&handle, &stop) {
            status.pass_active = false;
            status.state = "failed";
            status.last_pass = None;
            status.last_error = Some(diagnostic(error));
            status.last_attempt_finished_at = now().ok();
            *shared.lock().expect("continuation status") = Some(status.clone());
            continue;
        }
        physical.step(&handle, &config, &request, &mut status);
        let mut pass = if pending.is_empty() {
            let turn = match cursor.next_turn(&config, REQUESTS, &stop) {
                Ok(turn) => turn,
                Err(error) => {
                    status.pass_active = false;
                    status.state = "source_unavailable";
                    status.discovery_failures = status.discovery_failures.saturating_add(1);
                    status.last_pass = None;
                    status.last_error = Some(diagnostic(error));
                    status.last_attempt_finished_at = now().ok();
                    *shared.lock().expect("continuation status") = Some(status.clone());
                    continue;
                }
            };
            if turn.cycle_complete {
                status.enumeration_cycles_completed =
                    status.enumeration_cycles_completed.saturating_add(1);
            }
            let count = turn.discovery.candidates.len();
            pending.extend(turn.discovery.candidates);
            pending.push_back(Candidate::prompt(turn.discovery.history_path));
            Pass {
                catalog: turn.discovery.catalog,
                rollouts: turn.discovery.rollouts,
                enumeration_advanced: true,
                enumeration_cycle: turn.cycle,
                enumeration_cycle_complete: turn.cycle_complete,
                visited_entries: turn.visited_entries,
                admitted_rollout_paths: count,
                rollout_count_unit: "admitted_union_paths",
                rollout_count_scope: "observed_turn_prefix",
                evaluated_candidates: 0,
                examined_files: 0,
                unchanged_files: 0,
                acquired_files: 0,
                failed_files: 0,
                charged_source_bytes: 0,
                deferred_candidates: 0,
                evaluation_complete: false,
                budget_exhausted: false,
                history_state: "not_examined",
                history_bytes: None,
                failure_sample: vec![],
            }
        } else {
            let mut pass = previous.as_ref().expect("pending discovery").clone();
            pass.enumeration_advanced = false;
            pass.visited_entries = 0;
            pass.evaluated_candidates = 0;
            pass.examined_files = 0;
            pass.unchanged_files = 0;
            pass.acquired_files = 0;
            pass.failed_files = 0;
            pass.charged_source_bytes = 0;
            pass.deferred_candidates = 0;
            pass.evaluation_complete = false;
            pass.budget_exhausted = false;
            pass.history_state = "not_examined";
            pass.history_bytes = None;
            pass.failure_sample.clear();
            pass
        };
        status.state = "polling";
        status.last_error = None;
        while let Some(candidate) = pending.front() {
            if stop.load(Ordering::Acquire) {
                break;
            }
            if pass.evaluated_candidates >= REQUESTS {
                pass.budget_exhausted = true;
                break;
            }
            let declared = candidate.declared_length();
            if declared.as_ref().is_ok_and(|length| {
                pass.charged_source_bytes
                    .saturating_add(*length)
                    .saturating_add(1)
                    > BYTES
            }) {
                pass.budget_exhausted = true;
                break;
            }
            let candidate = pending.pop_front().expect("candidate front");
            pass.evaluated_candidates += 1;
            let prompt = candidate.format() == polylogue_store::InputFormat::CodexPromptHistory;
            if prompt {
                pass.history_state = "unavailable";
            }
            let result = (|| -> Result<(), String> {
                let declared = declared.map_err(reason)?;
                pass.charged_source_bytes += declared + 1;
                let (hash, length) = candidate.fingerprint(&stop, declared).map_err(reason)?;
                pass.examined_files += 1;
                if prompt {
                    pass.history_state = if length == 0 {
                        "accessible_empty"
                    } else {
                        "accessible"
                    };
                    pass.history_bytes = Some(length);
                }
                let acquired = import(
                    &handle,
                    &config,
                    candidate,
                    hash,
                    length,
                    &format!("{request}:{}", status.completed_passes),
                    now()?,
                    &stop,
                )?;
                if acquired {
                    pass.acquired_files += 1;
                    status.successful_snapshots = status.successful_snapshots.saturating_add(1);
                    status.acquired_source_bytes =
                        status.acquired_source_bytes.saturating_add(length);
                } else {
                    pass.unchanged_files += 1;
                }
                Ok(())
            })();
            if let Err(error) = result {
                pass.failed_files += 1;
                if pass.failure_sample.len() < 4 {
                    pass.failure_sample.push(diagnostic(error));
                }
            }
        }
        pass.deferred_candidates = pending.len();
        pass.evaluation_complete = pending.is_empty() && !stop.load(Ordering::Acquire);
        if !stop.load(Ordering::Acquire) {
            status.completed_passes = status.completed_passes.saturating_add(1);
        }
        if pass.evaluation_complete
            && pass.acquired_files == 0
            && pass.failed_files == 0
            && pass.rollouts.state != "unavailable"
            && pass.catalog.state != "unavailable"
        {
            status.unchanged_passes = status.unchanged_passes.saturating_add(1);
        }
        status.pass_active = false;
        status.last_attempt_finished_at = now().ok();
        previous = Some(pass.clone());
        status.last_pass = Some(pass);
        *shared.lock().expect("continuation status") = Some(status.clone());
    }
    status.pass_active = false;
    status.state = "stopped";
    *shared.lock().expect("continuation status") = Some(status);
}
