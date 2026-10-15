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
    let deadline = Instant::now() + duration;
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
    let mut checked_progress = false;
    loop {
        // An initially settled receipt may pass --wait-timeout 0 (check once).
        // A later response must arrive AND be accepted before the same deadline.
        if checked_progress && Instant::now() >= deadline {
            return timed_out(retained);
        }
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
            retained["requested_source_basis"] = json!(captured.basis);
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
            return timed_out(retained);
        }
        std::thread::sleep(
            Duration::from_millis(10).min(deadline.saturating_duration_since(Instant::now())),
        );
        let reply = match polylogue::request_value_until(
            socket,
            &json!({"command":"progress","input":captured.basis.input}),
            deadline,
        ) {
            Ok(reply) => reply,
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                ) || Instant::now() >= deadline =>
            {
                return timed_out(retained);
            }
            Err(e) => return failure(retained, ErrorKind::Io, e.to_string()),
        };
        if Instant::now() >= deadline {
            return timed_out(retained);
        }
        checked_progress = true;
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

fn timed_out(retained: Value) -> Value {
    failure(
        retained,
        ErrorKind::Timeout,
        "source row settlement is pending; retained owner work continues; retry the original import receipt",
    )
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
    #[test]
    fn later_interpretation_cannot_certify_the_captured_wait_basis() {
        use std::io::{BufRead, BufReader, Write};
        use std::os::unix::net::UnixListener;
        let dir = tempfile::tempdir_in("/tmp").unwrap();
        let socket = dir.path().join("peer");
        let listener = UnixListener::bind(&socket).unwrap();
        let mut reply = pending();
        reply["source_rows"]["basis"]["interpretation"] =
            json!(polylogue_model::BlobHash::of(b"later recipe"));
        reply["source_rows"]["state"] = json!("settled");
        reply["source_rows"]
            .as_object_mut()
            .unwrap()
            .remove("obligations");
        let peer = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = String::new();
            BufReader::new(stream.try_clone().unwrap())
                .read_line(&mut request)
                .unwrap();
            assert_eq!(
                serde_json::from_str::<Value>(&request).unwrap(),
                json!({"command":"progress","input":"synthetic"})
            );
            writeln!(stream, "{reply}").unwrap();
        });
        let expected = pending()["source_rows"]["basis"].clone();
        let out = wait(&socket, pending(), Duration::from_secs(5));
        peer.join().unwrap();
        assert_eq!(out["error"]["kind"], "conflict");
        assert_eq!(out["requested_source_basis"], expected);
    }
}
