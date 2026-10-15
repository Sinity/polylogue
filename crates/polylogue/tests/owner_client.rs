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
    let deadline = Instant::now() + Duration::from_secs(90);
    loop {
        let value = progress(socket, id);
        assert_eq!(value["outcome"], "ok", "{value}");
        assert!(value["failure"].is_null(), "{value}");
        assert!(value["recovery_error"].is_null(), "{value}");
        if condition(&value) {
            return value;
        }
        assert!(
            Instant::now() < deadline,
            "source progress timed out: {value}"
        );
        thread::sleep(Duration::from_millis(2));
    }
}
#[test]
fn actual_socket_peers_write_during_large_input_then_restart_keeps_eof_incomplete() {
    let temp = tempfile::tempdir().unwrap();
    let archive = temp.path().join("archive");
    let runtime = temp.path().join("runtime");
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
    assert_eq!(unfinished["canonical_complete"], false);
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
    assert_eq!(during["canonical_complete"], false);
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
    assert_eq!(eof["parser"]["finalization_pending"], true);
    assert_eq!(eof["canonical_complete"], false);
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
    let runtime = temp.path().join("runtime");
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
    assert_eq!(after["canonical_complete"], false);
}
