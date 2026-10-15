//! Public socket peer operations during unfinished source work. Canonical
//! publication/search are deliberately not certified by these retention tests.
use polylogue::{Running, request, request_value};
use serde_json::{Value, json};
use std::{
    fs,
    path::Path,
    thread,
    time::{Duration, Instant},
};
fn import(socket: &Path, path: &Path, id: &str) -> Value {
    request_value(socket,&json!({"command":"import","context":[],"id":id,"format":"codex-rollout","path":path,"machine":"synthetic-machine","host":"synthetic-host","seen_at":10})).unwrap()
}
fn progress(socket: &Path, id: &str) -> Value {
    request_value(socket, &json!({"command":"progress","input":id})).unwrap()
}
fn wait(socket: &Path, id: &str, condition: impl Fn(&Value) -> bool) -> Value {
    // This debug-build recovery control checks continued progress and exact EOF,
    // not throughput. Keep the full 10,000-message fixture under shared IO load.
    let deadline = Instant::now() + Duration::from_secs(300);
    let mut last_parser = Value::Null;
    let mut last_change = Instant::now();
    loop {
        let value = progress(socket, id);
        assert_eq!(value["outcome"], "ok", "{value}");
        assert!(value["failure"].is_null(), "{value}");
        assert!(value["recovery_error"].is_null(), "{value}");
        if condition(&value) {
            return value;
        }
        if value["parser"] != last_parser {
            last_parser = value["parser"].clone();
            last_change = Instant::now();
        }
        assert!(
            last_change.elapsed() < Duration::from_secs(30),
            "source progress stalled: {value}"
        );
        assert!(
            Instant::now() < deadline,
            "source progress exceeded recovery bound: {value}"
        );
        thread::sleep(Duration::from_millis(20));
    }
}
#[test]
fn actual_socket_peers_write_during_large_input_then_restart_keeps_eof_incomplete() {
    let temp = tempfile::tempdir().unwrap();
    let archive = temp.path().join("archive");
    // Unix socket names have a kernel length limit independent of TMPDIR.
    // Only socket dentries live here; database/blob work stays in managed TMPDIR.
    let runtime_owner = tempfile::tempdir_in("/tmp").unwrap();
    let runtime = runtime_owner.path().join("runtime");
    let large = temp.path().join("large.jsonl");
    let small = temp.path().join("small.jsonl");
    let mut bytes = json!({"type":"session_meta","payload":{"id":"synthetic-large","source":"cli"}})
        .to_string() + "\n";
    for n in 0..10000 {
        bytes.push_str(&json!({"type":"response_item","payload":{"type":"message","id":format!("m-{n}"),"role":"assistant","content":[{"type":"output_text","text":format!("synthetic text {n}")}]}}).to_string());
        bytes.push('\n');
    }
    fs::write(&large, &bytes).unwrap();
    fs::write(
        &small,
        json!({"type":"session_meta","payload":{"id":"synthetic-small","source":"cli"}})
            .to_string()
            + "\n",
    )
    .unwrap();
    let mut running = Running::start(&archive, &runtime).unwrap();
    let retained = import(
        running.socket(),
        &large,
        "01900000-0000-7000-8000-000000000001",
    );
    assert_eq!(retained["outcome"], "ok", "{retained}");
    let id = retained["retained"]["id"].as_str().unwrap().to_owned();
    let unfinished = wait(running.socket(), &id, |v| {
        v["parser"]["parsed_upto"]
            .as_u64()
            .is_some_and(|n| n > 0 && n < bytes.len() as u64)
    });
    assert_eq!(unfinished["source_rows"]["scope"], "imported_source_rows");
    assert_eq!(request(running.socket()).unwrap()["outcome"], "ok");
    let peer = import(
        running.socket(),
        &small,
        "01900000-0000-7000-8000-000000000002",
    );
    assert_eq!(peer["outcome"], "ok", "{peer}");
    let during = progress(running.socket(), &id);
    assert!(
        during["parser"]["parsed_upto"].as_u64().unwrap() < bytes.len() as u64,
        "{during}"
    );
    assert_eq!(during["source_rows"]["scope"], "imported_source_rows");
    running.stop();
    drop(running);
    // Neither the provider pathname nor a capture worker is recovery authority.
    fs::remove_file(&large).unwrap();
    let running = Running::start(&archive, &runtime).unwrap();
    let retry = import(
        running.socket(),
        &large,
        "01900000-0000-7000-8000-000000000001",
    );
    assert_eq!(retry["outcome"], "ok", "{retry}");
    assert_eq!(retry["retained"], retained["retained"]);
    let eof = wait(running.socket(), &id, |v| {
        v["parser"]["status"] == "finalizing"
    });
    assert_eq!(eof["parser"]["parsed_upto"], bytes.len() as u64);
    assert_eq!(eof["parser"]["records"]["framed"], 10001);
    assert_eq!(
        eof["parser"]["finalization_pending"],
        eof["source_rows"]["state"] != "settled"
    );
    assert_eq!(eof["source_rows"]["scope"], "imported_source_rows");
    let search = request_value(
        running.socket(),
        &json!({"command":"search","query":"synthetic"}),
    )
    .unwrap();
    assert_eq!(
        search["outcome"], "error",
        "EOF must not enable incomplete empty search: {search}"
    );
    assert_eq!(search["error"]["kind"], "not_yet_available");
}
#[test]
fn public_partial_tail_restart_waits_without_advancing_or_claiming_done() {
    let temp = tempfile::tempdir().unwrap();
    let archive = temp.path().join("archive");
    // Unix socket names have a kernel length limit independent of TMPDIR.
    // Only socket dentries live here; database/blob work stays in managed TMPDIR.
    let runtime_owner = tempfile::tempdir_in("/tmp").unwrap();
    let runtime = runtime_owner.path().join("runtime");
    let path = temp.path().join("tail.jsonl");
    let prefix = json!({"type":"session_meta","payload":{"id":"synthetic-tail","source":"cli"}})
        .to_string()
        + "\n";
    let bytes = prefix.clone() + "{\"type\":\"response_item\"";
    fs::write(&path, &bytes).unwrap();
    let mut running = Running::start(&archive, &runtime).unwrap();
    let retained = import(
        running.socket(),
        &path,
        "01900000-0000-7000-8000-000000000003",
    );
    assert_eq!(retained["outcome"], "ok", "{retained}");
    let id = retained["retained"]["id"].as_str().unwrap().to_owned();
    let before = wait(running.socket(), &id, |v| {
        v["parser"]["status"] == "waiting_tail"
    });
    assert_eq!(before["parser"]["parsed_upto"], prefix.len() as u64);
    assert_eq!(before["parser"]["inspected_upto"], bytes.len() as u64);
    running.stop();
    drop(running);
    fs::remove_file(&path).unwrap();
    let running = Running::start(&archive, &runtime).unwrap();
    let after = wait(running.socket(), &id, |v| {
        v["parser"]["status"] == "waiting_tail"
    });
    assert_eq!(before["parser"], after["parser"]);
    thread::sleep(Duration::from_millis(30));
    assert_eq!(progress(running.socket(), &id)["parser"], after["parser"]);
    assert_eq!(after["source_rows"]["scope"], "imported_source_rows");
}

#[test]
fn actual_request_conflict_and_invalid_identity_are_typed_before_source_open() {
    let temp = tempfile::tempdir().unwrap();
    let runtime = tempfile::tempdir_in("/tmp").unwrap();
    let running = Running::start(&temp.path().join("archive"), runtime.path()).unwrap();
    let first = temp.path().join("first.jsonl");
    fs::write(&first, b"").unwrap();
    let id = "01900000-0000-7000-8000-000000000004";
    assert_eq!(import(running.socket(), &first, id)["outcome"], "ok");
    let different = temp.path().join("does-not-exist.jsonl");
    let conflict = import(running.socket(), &different, id);
    assert_eq!(conflict["error"]["kind"], "conflict", "{conflict}");
    assert_eq!(conflict["error"]["retryable"], false);
    let invalid = import(running.socket(), &different, "invalid-request-id");
    assert_eq!(invalid["error"]["kind"], "invalid_query", "{invalid}");
    assert_eq!(import(running.socket(), &first, id)["outcome"], "ok");
}
