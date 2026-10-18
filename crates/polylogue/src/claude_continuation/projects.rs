//! Collect from a configured Claude Code projects directory. While there is
//! work to do it runs continuously; once caught up it waits before rescanning.
use super::*;
use polylogue_store::configured_claude::{Config, Cursor, Scan};
use std::collections::VecDeque;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProjectStatus {
    pub scan: Option<Scan>,
    pub availability: &'static str,
    pub queued_candidates: usize,
    pub active_path: Option<String>,
    pub acquired_source_attempts: u64,
    pub selected_source_bytes: u64,
    pub failed_attempts: u64,
    pub known_inputs: Vec<String>,
    pub known_inputs_truncated: bool,
    pub initial_cycle_finished: bool,
}
impl ProjectStatus {
    pub fn initial() -> Self {
        Self {
            scan: None,
            availability: "pending",
            queued_candidates: 0,
            active_path: None,
            acquired_source_attempts: 0,
            selected_source_bytes: 0,
            failed_attempts: 0,
            known_inputs: vec![],
            known_inputs_truncated: false,
            initial_cycle_finished: false,
        }
    }
}
fn publish(shared: &Mutex<Option<Status>>, status: &Status) {
    *shared.lock().expect("Claude status") = Some(status.clone());
}
pub(crate) fn run(
    handle: Handle,
    config: Config,
    request: String,
    stop: Arc<AtomicBool>,
    shared: Arc<Mutex<Option<Status>>>,
) {
    let mut status = Status::initial(config.acquisition_quantum_bytes);
    status.mode = "claude_projects";
    status.state = "discovering";
    status.projects = Some(ProjectStatus::initial());
    let mut cursor = Cursor::default();
    let mut queue = VecDeque::new();
    let mut cycle_complete = false;
    publish(&shared, &status);
    while !stop.load(Ordering::Acquire) {
        if queue.is_empty() {
            if cycle_complete {
                status.projects.as_mut().unwrap().initial_cycle_finished = true;
                publish(&shared, &status);
                for _ in 0..100 {
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    thread::sleep(Duration::from_millis(10));
                }
            }
            match cursor.next_turn(&config, 128, &stop) {
                Ok(turn) => {
                    cycle_complete = turn.scan.cycle_complete;
                    status.projects.as_mut().unwrap().availability = turn.scan.state;
                    status.projects.as_mut().unwrap().scan = Some(turn.scan);
                    queue.extend(turn.locations);
                    status.state = "discovering";
                }
                Err(error) => {
                    status.state = "source_unavailable";
                    status.last_error = Some(error.to_string().chars().take(256).collect());
                    let projects = status.projects.as_mut().unwrap();
                    projects.scan = None;
                    projects.availability = match &error {
                        polylogue_store::Error::Io(e)
                            if e.kind() == std::io::ErrorKind::NotFound =>
                        {
                            "missing"
                        }
                        polylogue_store::Error::Io(e)
                            if e.kind() == std::io::ErrorKind::PermissionDenied =>
                        {
                            "denied"
                        }
                        _ => "unavailable",
                    };
                    projects.failed_attempts += 1;
                    // A failed walk is neither an empty directory nor a full scan.
                    publish(&shared, &status);
                    for _ in 0..100 {
                        if stop.load(Ordering::Acquire) {
                            break;
                        }
                        thread::sleep(Duration::from_millis(10));
                    }
                    continue;
                }
            }
        }
        status.projects.as_mut().unwrap().queued_candidates = queue.len();
        let Some(location) = queue.pop_front() else {
            publish(&shared, &status);
            continue;
        };
        let seen_at = match SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .and_then(|v| i64::try_from(v.as_micros()).ok())
        {
            Some(v) => v,
            None => {
                status.state = "unavailable";
                status.last_error = Some("observation time unavailable".into());
                publish(&shared, &status);
                return;
            }
        };
        let attempt = format!("{request}:project:{}", status.attempts);
        let observation = Observation {
            id: match location.observation_id(&attempt, seen_at) {
                Ok(id) => id,
                Err(e) => {
                    status.last_error = Some(e.to_string());
                    publish(&shared, &status);
                    return;
                }
            },
            machine: config.machine.clone(),
            host: config.host.clone(),
            path: location.path().to_string_lossy().into_owned(),
            seen_at,
        };
        status.state = "acquiring";
        status.projects.as_mut().unwrap().active_path =
            Some(location.path().to_string_lossy().into_owned());
        publish(&shared, &status);
        let result = (|| {
            loop {
                if stop.load(Ordering::Acquire) {
                    return Err(Error::Stopped);
                }
                let receiver = match handle.try_claude_location(
                    location.clone(),
                    observation.clone(),
                    config.acquisition_quantum_bytes,
                ) {
                    Ok(r) => r,
                    Err(Error::Busy) => {
                        thread::sleep(Duration::from_millis(10));
                        continue;
                    }
                    Err(e) => return Err(e),
                };
                loop {
                    match receiver.recv_timeout(Duration::from_millis(50)) {
                        Ok(Err(Error::Busy)) => break,
                        Ok(result) => return result,
                        Err(RecvTimeoutError::Disconnected) => return Err(Error::Stopped),
                        Err(RecvTimeoutError::Timeout) if stop.load(Ordering::Acquire) => {
                            return Err(Error::Stopped);
                        }
                        Err(RecvTimeoutError::Timeout) => {}
                    }
                }
                thread::sleep(Duration::from_millis(10));
            }
        })();
        status.attempts += 1;
        let projects = status.projects.as_mut().unwrap();
        projects.queued_candidates = queue.len();
        projects.active_path = None;
        match result {
            Ok(reply) => {
                status.state = "discovering";
                status.last_error = None;
                status.stored_length = Some(reply.input.length);
                status.input = Some(reply.input.id.clone());
                if reply.acquired {
                    projects.acquired_source_attempts += 1;
                    projects.selected_source_bytes = projects
                        .selected_source_bytes
                        .saturating_add(reply.input.length);
                } else {
                    status.unchanged_passes += 1;
                }
                // This is a short sample for status, not the work queue.
                if !projects.known_inputs.contains(&reply.input.id) {
                    if projects.known_inputs.len() < 32 {
                        projects.known_inputs.push(reply.input.id);
                    } else {
                        projects.known_inputs_truncated = true;
                    }
                }
            }
            Err(error) => {
                status.state = "failed";
                status.last_error = Some(error.to_string().chars().take(256).collect());
                projects.failed_attempts += 1;
            }
        }
        publish(&shared, &status);
    }
    status.state = "stopped";
    status.projects.as_mut().unwrap().active_path = None;
    publish(&shared, &status);
}
