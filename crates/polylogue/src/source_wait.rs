//! A caller owns only its fixed-basis wait. It never stops or gates the owner.
use polylogue_model::{
    settlement::{SourceRowState, SourceRows},
    types::{ErrorKind, Failure},
};
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};
fn failure(mut value: Value, kind: ErrorKind, message: impl Into<String>) -> Value {
    value["outcome"] = json!("error");
    value["error"] = json!(Failure::new(kind, message));
    value
}
pub(crate) fn wait(socket: &Path, mut retained: Value, duration: Duration) -> Value {
    let captured: SourceRows = match serde_json::from_value(retained["source_rows"].clone()) {
        Ok(value) => value,
        Err(e) => {
            return failure(
                retained,
                ErrorKind::Io,
                format!("invalid admitted source settlement basis: {e}"),
            );
        }
    };
    let deadline = Instant::now() + duration;
    loop {
        let rows: SourceRows = match serde_json::from_value(retained["source_rows"].clone()) {
            Ok(value) => value,
            Err(e) => {
                return failure(
                    retained,
                    ErrorKind::Io,
                    format!("invalid source settlement result: {e}"),
                );
            }
        };
        if rows.basis != captured.basis || rows.scope != captured.scope {
            return failure(
                retained,
                ErrorKind::Conflict,
                "source interpretation basis changed during fixed-basis wait; retry the retained request explicitly",
            );
        }
        match rows.state {
            SourceRowState::Settled => {
                retained["outcome"] = json!("ok");
                return retained;
            }
            SourceRowState::Unavailable {
                usable_result,
                notes,
                record_errors,
            } => {
                if !usable_result {
                    return failure(retained, ErrorKind::NotYetAvailable, notes.join("; "));
                }
                retained["outcome"] = json!("degraded");
                retained["missing"] = json!(notes);
                retained["notes"] = retained["missing"].clone();
                retained["unknown"] = json!({"record_errors":record_errors});
                return retained;
            }
            SourceRowState::Pending { .. } => {}
        }
        if Instant::now() >= deadline {
            return failure(
                retained,
                ErrorKind::Timeout,
                "source row settlement is pending; retained owner work continues; retry the original import receipt",
            );
        }
        std::thread::sleep(
            Duration::from_millis(10).min(deadline.saturating_duration_since(Instant::now())),
        );
        let reply = match polylogue::request_value(
            socket,
            &json!({"command":"progress","input":captured.basis.input}),
        ) {
            Ok(reply) => reply,
            Err(e) => return failure(retained, ErrorKind::Io, e.to_string()),
        };
        if reply["outcome"] == "error" {
            return failure(
                retained,
                ErrorKind::Io,
                format!("source wait progress refused: {}", reply["error"]),
            );
        }
        retained["source_rows"] = reply["source_rows"].clone();
    }
}
