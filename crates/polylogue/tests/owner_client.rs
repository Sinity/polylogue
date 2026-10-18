//! Socket clients while the daemon is still parsing a large or unfinished file.
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
    // Checks that parsing keeps moving and stops at the right end, not speed.
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
fn clients_can_write_while_a_large_file_parses_and_restart_resumes_it() {
    let temp = tempfile::tempdir().unwrap();
    let archive = temp.path().join("archive");
    // Unix socket paths are limited to 108 bytes, so only the socket goes in /tmp.
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
    let id = retained["input"]["id"].as_str().unwrap().to_owned();
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
    // After restart, parsing resumes from the stored copy, not the original path.
    fs::remove_file(&large).unwrap();
    let running = Running::start(&archive, &runtime).unwrap();
    let retry = import(
        running.socket(),
        &large,
        "01900000-0000-7000-8000-000000000001",
    );
    assert_eq!(retry["outcome"], "ok", "{retry}");
    assert_eq!(retry["input"], retained["input"]);
    let eof = wait(running.socket(), &id, |v| {
        v["parser"]["status"] == "finalizing"
    });
    assert_eq!(eof["parser"]["parsed_upto"], bytes.len() as u64);
    assert_eq!(eof["parser"]["records"]["framed"], 10001);
    assert_eq!(
        eof["parser"]["finalization_pending"],
        eof["source_rows"]["state"] != "complete"
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
fn unfinished_last_line_waits_after_restart_without_being_parsed() {
    let temp = tempfile::tempdir().unwrap();
    let archive = temp.path().join("archive");
    // Unix socket paths are limited to 108 bytes, so only the socket goes in /tmp.
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
    let id = retained["input"]["id"].as_str().unwrap().to_owned();
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
fn conflicting_request_id_and_bad_ids_fail_before_the_file_is_opened() {
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

#[test]
fn many_ignored_records_parse_in_turns_while_clients_are_served() {
    use std::io::Write;
    let temp = tempfile::tempdir().unwrap();
    let archive = temp.path().join("archive");
    let runtime_owner = tempfile::tempdir_in("/tmp").unwrap();
    let runtime = runtime_owner.path().join("runtime");
    let path = temp.path().join("states.jsonl");
    let peer_path = temp.path().join("peer.jsonl");
    let states = 65_536_u64;
    let mut stream = fs::File::create(&path).unwrap();
    writeln!(
        stream,
        "{{\"type\":\"session_meta\",\"payload\":{{\"id\":\"state-turns\"}}}}"
    )
    .unwrap();
    for sequence in 0..states {
        writeln!(
            stream,
            "{{\"record_type\":\"state\",\"sequence\":{sequence}}}"
        )
        .unwrap();
    }
    stream.sync_all().unwrap();
    let length = stream.metadata().unwrap().len();
    drop(stream);
    fs::write(
        &peer_path,
        b"{\"type\":\"session_meta\",\"payload\":{\"id\":\"state-peer\"}}\n",
    )
    .unwrap();
    let mut running = Running::start(&archive, &runtime).unwrap();
    let retained = import(
        running.socket(),
        &path,
        "01900000-0000-7000-8000-000000000021",
    );
    assert_eq!(retained["outcome"], "ok", "{retained}");
    let id = retained["input"]["id"].as_str().unwrap().to_owned();
    let prefix = wait(running.socket(), &id, |v| {
        v["parser"]["records"]["ignored"]
            .as_u64()
            .is_some_and(|n| n > 0 && n < states)
    });
    // Reads and a second import are served while parsing is still running.
    assert_eq!(request(running.socket()).unwrap()["outcome"], "ok");
    let peer = import(
        running.socket(),
        &peer_path,
        "01900000-0000-7000-8000-000000000022",
    );
    assert_eq!(peer["outcome"], "ok", "{peer}");
    let during = progress(running.socket(), &id);
    assert!(
        during["parser"]["parsed_upto"].as_u64().unwrap() < length,
        "{during}"
    );
    running.stop();
    drop(running);
    fs::remove_file(&path).unwrap();
    let running = Running::start(&archive, &runtime).unwrap();
    let eof = wait(running.socket(), &id, |v| {
        v["parser"]["parsed_upto"] == length
    });
    assert!(
        eof["parser"]["parsed_upto"].as_u64().unwrap()
            > prefix["parser"]["parsed_upto"].as_u64().unwrap()
    );
    assert_eq!(
        eof["parser"]["records"],
        json!({"framed":states+1,"used":1,"ignored":states,"unrecognized":0,"undecodable":0})
    );
    // Records are parsed in large batches, not a few records per step.
    assert!(
        eof["parser"]["step"].as_u64().unwrap() < states / 32,
        "{eof}"
    );
}
