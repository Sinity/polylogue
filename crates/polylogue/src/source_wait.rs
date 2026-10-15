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
    if retained["retained"]["id"].as_str() != Some(captured.basis.input.as_str())
        || retained["retained"]["chain"].as_i64() != Some(captured.basis.chain)
        || retained["retained"]["length"].as_u64() != Some(captured.basis.length)
        || captured.basis.revision != 1
    {
        return failure(
            retained,
            ErrorKind::Io,
            "source settlement basis differs from the admitted retained input",
        );
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    fn pending() -> Value {
        json!({"outcome":"ok","retained":{"id":"synthetic","chain":1,"length":7},"source_rows":{"scope":"imported_source_rows","basis":{"input":"synthetic","original_hash":polylogue_model::BlobHash::of(b"raw"),"chain":1,"revision":1,"length":7,"parser_version":1,"interpretation":polylogue_model::BlobHash::of(b"basis")},"state":"pending","obligations":["canonical rows"]}})
    }
    #[test]
    fn zero_wait_preserves_receipt_and_never_converts_pending_into_empty() {
        let output = wait(Path::new("/synthetic/unused"), pending(), Duration::ZERO);
        assert_eq!(output["error"]["kind"], "timeout");
        assert_eq!(output["retained"]["id"], "synthetic");
        assert_eq!(output["source_rows"]["state"], "pending");
    }
    #[test]
    fn wrong_admission_identity_or_unknown_basis_refuses_before_settled_success() {
        for field in ["input", "chain", "length", "revision"] {
            let mut reply = pending();
            reply["source_rows"]["state"] = json!("settled");
            reply["source_rows"]
                .as_object_mut()
                .unwrap()
                .remove("obligations");
            reply["source_rows"]["basis"][field] = if field == "input" {
                json!("foreign")
            } else {
                json!(9)
            };
            let output = wait(Path::new("/synthetic/unused"), reply, Duration::ZERO);
            assert_eq!(output["outcome"], "error", "{field}: {output}");
        }
    }
}
