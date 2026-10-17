//! One configured main, with each acquisition delegated to the existing owner.
use polylogue_engine::owner::{Error, Handle, Observation};
use serde::Serialize;
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::RecvTimeoutError,
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
#[derive(Clone, Debug, Serialize)]
pub(crate) struct Status {
    pub mode: &'static str,
    pub state: &'static str,
    pub attempts: u64,
    pub last_error: Option<String>,
    pub input: Option<String>,
    pub retained_length: Option<u64>,
    pub unchanged_passes: u64,
    pub quantum_bytes: u64,
}
impl Status {
    pub fn initial(quantum_bytes: u64) -> Self {
        Self {
            mode: "bounded_configured_claude_polling_v1",
            state: "polling",
            attempts: 0,
            last_error: None,
            input: None,
            retained_length: None,
            unchanged_passes: 0,
            quantum_bytes,
        }
    }
}
pub(crate) fn run(
    handle: Handle,
    path: PathBuf,
    machine: String,
    host: String,
    request: String,
    quantum_bytes: u64,
    stop: Arc<AtomicBool>,
    shared: Arc<Mutex<Option<Status>>>,
) {
    let mut status = Status::initial(quantum_bytes);
    while !stop.load(Ordering::Acquire) {
        for _ in 0..100 {
            if stop.load(Ordering::Acquire) {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        if stop.load(Ordering::Acquire) {
            break;
        }
        let seen_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .and_then(|d| i64::try_from(d.as_micros()).ok())
            .unwrap_or(0);
        let observation = Observation {
            id: format!("{request}:poll:{}", status.attempts),
            machine: machine.clone(),
            host: host.clone(),
            path: path.to_string_lossy().into_owned(),
            seen_at,
        };
        status.state = "acquiring";
        *shared.lock().expect("Claude status") = Some(status.clone());
        let result = (|| {
            let receiver =
                handle.try_claude_acquisition(path.clone(), observation, quantum_bytes)?;
            loop {
                match receiver.recv_timeout(Duration::from_millis(50)) {
                    Ok(result) => return result,
                    Err(RecvTimeoutError::Disconnected) => return Err(Error::Stopped),
                    Err(RecvTimeoutError::Timeout) if stop.load(Ordering::Acquire) => {
                        return Err(Error::Stopped);
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                }
            }
        })();
        status.attempts += 1;
        match result {
            Ok(reply) => {
                status.state = if reply.input.length == 0 {
                    "accessible_empty"
                } else {
                    "polling"
                };
                status.last_error = None;
                status.input = Some(reply.input.id);
                status.retained_length = Some(reply.input.length);
                if !reply.acquired {
                    status.unchanged_passes += 1;
                }
            }
            Err(error) => {
                status.state = match &error {
                    Error::Store(polylogue_store::Error::Io(source))
                        if source.kind() == std::io::ErrorKind::NotFound =>
                    {
                        "missing"
                    }
                    Error::Store(polylogue_store::Error::Io(source))
                        if source.kind() == std::io::ErrorKind::PermissionDenied =>
                    {
                        "denied"
                    }
                    Error::Busy => "busy",
                    Error::Stopped => "stopped",
                    _ => "unavailable",
                };
                status.last_error = Some(error.to_string().chars().take(256).collect());
            }
        }
        *shared.lock().expect("Claude status") = Some(status.clone());
    }
    status.state = "stopped";
    *shared.lock().expect("Claude status") = Some(status);
}
