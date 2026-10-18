//! `import --wait`: poll the daemon until the imported file is fully parsed.
//! Giving up only stops this client; the daemon keeps parsing.
use polylogue_model::{
    source_rows::{SourceRowState, SourceRows},
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
pub(crate) fn wait(socket: &Path, mut reply: Value, duration: Duration) -> Value {
    let deadline = Instant::now() + duration;
    let captured: SourceRows = match serde_json::from_value(reply["source_rows"].clone()) {
        Ok(value) => value,
        Err(e) => {
            return failure(
                reply,
                ErrorKind::Io,
                format!("invalid source_rows in import reply: {e}"),
            );
        }
    };
    let from = &captured.parsed_from;
    if reply["input"]["id"].as_str() != Some(from.input.as_str())
        || reply["input"]["chain"].as_i64() != Some(from.chain)
        || reply["input"]["length"].as_u64() != Some(from.length)
        || from.revision != 1
    {
        return failure(
            reply,
            ErrorKind::Io,
            "source_rows describe a different file than the import reply",
        );
    }
    let mut checked_progress = false;
    loop {
        // A reply that is already complete passes even with --wait-timeout 0.
        // A later progress reply must arrive and be checked before the deadline.
        if checked_progress && Instant::now() >= deadline {
            return timed_out(reply);
        }
        let rows: SourceRows = match serde_json::from_value(reply["source_rows"].clone()) {
            Ok(value) => value,
            Err(e) => {
                return failure(
                    reply,
                    ErrorKind::Io,
                    format!("invalid source_rows in progress reply: {e}"),
                );
            }
        };
        if rows.parsed_from != captured.parsed_from || rows.scope != captured.scope {
            reply["expected_parsed_from"] = json!(captured.parsed_from);
            return failure(
                reply,
                ErrorKind::Conflict,
                "the file was reparsed with a different parser while waiting; retry the import",
            );
        }
        match rows.state {
            SourceRowState::Complete => {
                reply["outcome"] = json!("ok");
                if checked_progress && Instant::now() >= deadline {
                    return timed_out(reply);
                }
                return reply;
            }
            SourceRowState::Unavailable {
                usable_result,
                notes,
                record_errors,
            } => {
                if !usable_result {
                    return failure(reply, ErrorKind::NotYetAvailable, notes.join("; "));
                }
                reply["outcome"] = json!("degraded");
                reply["missing"] = json!(notes);
                reply["notes"] = reply["missing"].clone();
                reply["unknown"] = json!({"record_errors":record_errors});
                return reply;
            }
            SourceRowState::Pending { .. } => {}
        }
        if Instant::now() >= deadline {
            return timed_out(reply);
        }
        std::thread::sleep(
            Duration::from_millis(10).min(deadline.saturating_duration_since(Instant::now())),
        );
        let progress = match polylogue::request_value_until(
            socket,
            &json!({"command":"progress","input":from.input}),
            deadline,
        ) {
            Ok(progress) => progress,
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                ) || Instant::now() >= deadline =>
            {
                return timed_out(reply);
            }
            Err(e) => return failure(reply, ErrorKind::Io, e.to_string()),
        };
        if Instant::now() >= deadline {
            return timed_out(reply);
        }
        checked_progress = true;
        if progress["outcome"] == "error" {
            return failure(
                reply,
                ErrorKind::Io,
                format!("progress request failed: {}", progress["error"]),
            );
        }
        reply["source_rows"] = progress["source_rows"].clone();
    }
}

fn timed_out(reply: Value) -> Value {
    failure(
        reply,
        ErrorKind::Timeout,
        "timed out waiting for parsing; the daemon keeps working; retry the same import to wait again",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pending() -> Value {
        json!({"outcome":"ok","input":{"id":"synthetic","chain":1,"length":7},"source_rows":{"scope":"imported_source_rows","parsed_from":{"input":"synthetic","original_hash":polylogue_model::BlobHash::of(b"raw"),"chain":1,"revision":1,"length":7,"parser_version":1,"interpretation":polylogue_model::BlobHash::of(b"basis")},"state":"pending","pending_work":["canonical rows"]}})
    }
    #[test]
    fn zero_timeout_on_pending_rows_is_a_timeout_not_an_empty_success() {
        let output = wait(Path::new("/synthetic/unused"), pending(), Duration::ZERO);
        assert_eq!(output["error"]["kind"], "timeout");
        assert_eq!(output["input"]["id"], "synthetic");
        assert_eq!(output["source_rows"]["state"], "pending");
    }
    #[test]
    fn rows_for_a_different_file_are_an_error_even_when_complete() {
        for field in ["input", "chain", "length", "revision"] {
            let mut reply = pending();
            reply["source_rows"]["state"] = json!("complete");
            reply["source_rows"]
                .as_object_mut()
                .unwrap()
                .remove("pending_work");
            reply["source_rows"]["parsed_from"][field] = if field == "input" {
                json!("foreign")
            } else {
                json!(9)
            };
            let output = wait(Path::new("/synthetic/unused"), reply, Duration::ZERO);
            assert_eq!(output["outcome"], "error", "{field}: {output}");
        }
    }
    #[test]
    fn rows_from_a_different_parser_are_a_conflict() {
        use std::io::{BufRead, BufReader, Write};
        use std::os::unix::net::UnixListener;
        let dir = tempfile::tempdir_in("/tmp").unwrap();
        let socket = dir.path().join("peer");
        let listener = UnixListener::bind(&socket).unwrap();
        let mut reply = pending();
        reply["source_rows"]["parsed_from"]["interpretation"] =
            json!(polylogue_model::BlobHash::of(b"later recipe"));
        reply["source_rows"]["state"] = json!("complete");
        reply["source_rows"]
            .as_object_mut()
            .unwrap()
            .remove("pending_work");
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
        let expected = pending()["source_rows"]["parsed_from"].clone();
        let out = wait(&socket, pending(), Duration::from_secs(5));
        peer.join().unwrap();
        assert_eq!(out["error"]["kind"], "conflict");
        assert_eq!(out["expected_parsed_from"], expected);
    }
}
