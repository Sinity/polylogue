//! Real subprocess daemon/CLI Session reads; catalog coverage remains degraded.
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, Read, Seek, SeekFrom, Write},
    os::unix::net::UnixStream,
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};
const BINARY: &str = env!("CARGO_BIN_EXE_polylogue");
const WAIT: Duration = Duration::from_secs(120);
const SID: &str = "7f3a2c10-0001-4000-8000-000000000001";
const MORROW: &[u8] = include_bytes!("../../sources/tests/fixtures/morrow-main.jsonl");
struct OwnedChild(Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
struct World {
    child: Option<OwnedChild>,
    work: tempfile::TempDir,
    _socket_owner: tempfile::TempDir,
    archive: PathBuf,
    runtime: PathBuf,
    scratch: PathBuf,
    log: PathBuf,
}
impl World {
    fn new() -> Self {
        let work = tempfile::tempdir().unwrap();
        let archive = work.path().join("archive");
        let scratch = work.path().join("tmp");
        fs::create_dir(&scratch).unwrap();
        // Only short Unix-socket dentries use /tmp; retained bytes and command
        // captures stay in the test's isolated, harness-owned work directory.
        let socket_owner = tempfile::tempdir_in("/tmp").unwrap();
        let runtime = socket_owner.path().join("run");
        fs::create_dir(&runtime).unwrap();
        let log = work.path().join("daemon.log");
        let mut world = Self {
            child: None,
            work,
            _socket_owner: socket_owner,
            archive,
            runtime,
            scratch,
            log,
        };
        world.start();
        world
    }
    fn command(&self) -> Command {
        let mut command = Command::new(BINARY);
        command
            .arg("--archive")
            .arg(&self.archive)
            .env("XDG_RUNTIME_DIR", &self.runtime)
            .env("TMPDIR", &self.scratch);
        command
    }
    fn cli(&self, args: &[&str]) -> Output {
        // Files avoid pipe-buffer deadlock for large terminal output and let us
        // enforce a child deadline without adding a separate client protocol.
        let mut stdout = tempfile::tempfile_in(&self.scratch).unwrap();
        let mut stderr = tempfile::tempfile_in(&self.scratch).unwrap();
        let mut child = OwnedChild(
            self.command()
                .args(args)
                .stdout(Stdio::from(stdout.try_clone().unwrap()))
                .stderr(Stdio::from(stderr.try_clone().unwrap()))
                .spawn()
                .unwrap(),
        );
        let deadline = Instant::now() + Duration::from_secs(40);
        let status = loop {
            assert!(
                stdout.metadata().unwrap().len() <= 5_000_000,
                "CLI stdout exceeds test capture allowance"
            );
            assert!(
                stderr.metadata().unwrap().len() <= 1_000_000,
                "CLI stderr exceeds test capture allowance"
            );
            if let Some(status) = child.0.try_wait().unwrap() {
                break status;
            }
            assert!(
                Instant::now() < deadline,
                "CLI deadline {args:?}; daemon: {}",
                self.log()
            );
            thread::sleep(Duration::from_millis(2));
        };
        stdout.seek(SeekFrom::Start(0)).unwrap();
        stderr.seek(SeekFrom::Start(0)).unwrap();
        let mut out = vec![];
        let mut err = vec![];
        stdout.read_to_end(&mut out).unwrap();
        stderr.read_to_end(&mut err).unwrap();
        Output {
            status,
            stdout: out,
            stderr: err,
        }
    }
    fn log(&self) -> String {
        fs::read_to_string(&self.log).unwrap_or_default()
    }
    fn start(&mut self) {
        assert!(self.child.is_none());
        self.child = Some(OwnedChild(
            self.command()
                .arg("daemon")
                .stdout(Stdio::null())
                .stderr(Stdio::from(File::create(&self.log).unwrap()))
                .spawn()
                .unwrap(),
        ));
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            let out = self.cli(&["status"]);
            if out.status.success() {
                let value: Value = serde_json::from_slice(&out.stdout).unwrap();
                assert_eq!(value["archive"], self.archive.to_str().unwrap());
                return;
            }
            assert!(
                self.child.as_mut().unwrap().0.try_wait().unwrap().is_none(),
                "daemon exited: {}; status {out:?}",
                self.log()
            );
            assert!(
                Instant::now() < deadline,
                "daemon startup: {}; status {out:?}",
                self.log()
            );
            thread::sleep(Duration::from_millis(5));
        }
    }
    fn kill(&mut self) {
        drop(self.child.take());
    }
    fn restart(&mut self) {
        self.kill();
        self.start();
    }
    fn input(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.work.path().join(name);
        fs::write(&path, bytes).unwrap();
        path
    }
    fn import(&self, path: &Path, request_id: &str) -> Value {
        let context = self.work.path().join("context.json");
        fs::write(
            &context,
            json!({"version":1,"context":["claude-main",SID]}).to_string(),
        )
        .unwrap();
        let out = self.cli(&[
            "import",
            path.to_str().unwrap(),
            "--format",
            "claude-code-jsonl",
            "--context",
            context.to_str().unwrap(),
            "--request-id",
            request_id,
            "--seen-at",
            "17",
            "--machine",
            "00112233445566778899aabbccddeeff",
            "--host",
            "synthetic-collector",
        ]);
        assert_eq!(
            out.status.code(),
            Some(0),
            "import {out:?}; daemon {}",
            self.log()
        );
        let receipt: Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(receipt["request_id"], request_id);
        assert_eq!(receipt["seen_at"], 17);
        assert_eq!(receipt["source_rows"]["scope"], "imported_source_rows");
        receipt
    }
    fn raw_progress(&self, id: &str) -> Value {
        let out = self.cli(&["progress", id]);
        assert_eq!(out.status.code(), Some(0), "progress {out:?}");
        serde_json::from_slice(&out.stdout).unwrap()
    }
    fn progress(&self, id: &str) -> Value {
        let progress = self.raw_progress(id);
        assert_eq!(progress["source_rows"]["scope"], "imported_source_rows");
        assert!(progress["failure"].is_null(), "{progress}; {}", self.log());
        assert!(progress["recovery_error"].is_null(), "{progress}");
        progress
    }
    fn read(&self, reference: &str, id: &str, messages: usize) -> Value {
        let deadline = Instant::now() + WAIT;
        loop {
            let p = self.progress(id);
            let out = self.cli(&[reference, "read", "--format", "json"]);
            if out.status.code() == Some(5) {
                let document: Value = serde_json::from_slice(&out.stdout).unwrap();
                assert!(
                    String::from_utf8_lossy(&out.stderr).contains("coverage is incomplete"),
                    "{out:?}"
                );
                if document["messages"].as_array().unwrap().len() == messages {
                    assert_eq!(
                        p["parser"]["finalization_pending"],
                        p["source_rows"]["state"] != "settled"
                    );
                    return document;
                }
            } else {
                assert!(
                    matches!(out.status.code(), Some(2 | 4)),
                    "{out:?}; {}",
                    self.log()
                );
                assert!(
                    out.stdout.is_empty(),
                    "failed read emitted partial document: {out:?}"
                );
            }
            assert!(
                Instant::now() < deadline,
                "Session read deadline; {p}; {out:?}; {}",
                self.log()
            );
            thread::sleep(Duration::from_millis(5));
        }
    }
    fn socket_read(&self, reference: &str) -> (Value, Vec<u8>, Value) {
        let socket =
            polylogue::socket_path(&fs::canonicalize(&self.archive).unwrap(), &self.runtime);
        let mut stream = UnixStream::connect(socket).unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        writeln!(
            stream,
            "{}",
            json!({"command":"session-read","reference":reference,"format":"json"})
        )
        .unwrap();
        let mut reader = BufReader::new(stream);
        let mut raw = Vec::new();
        let mut text = String::new();
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            assert!(Instant::now() < deadline, "socket deadline");
            let mut line = Vec::new();
            reader
                .by_ref()
                .take(65537)
                .read_until(b'\n', &mut line)
                .unwrap();
            assert!(
                line.len() <= 65536 && line.ends_with(b"\n"),
                "invalid frame size/completion"
            );
            raw.extend_from_slice(&line);
            assert!(
                raw.len() < 5_000_000,
                "socket capture exceeds test allowance"
            );
            let frame: Value = serde_json::from_slice(&line).unwrap();
            match frame["type"].as_str().unwrap() {
                "session" => assert_eq!(frame["id"], reference),
                "text" => text.push_str(frame["data"].as_str().unwrap()),
                "end" => return (serde_json::from_str(&text).unwrap(), raw, frame),
                other => panic!("unexpected frame {other}"),
            }
        }
    }
}
impl Drop for World {
    fn drop(&mut self) {
        self.kill();
    }
}
fn reference() -> String {
    polylogue_model::SessionId::new(&polylogue_model::Origin::new("claude-code").unwrap(), SID)
        .unwrap()
        .as_str()
        .to_owned()
}
fn fixture_records() -> Vec<Value> {
    MORROW
        .split(|b| *b == b'\n')
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_slice(l).unwrap())
        .filter(|r: &Value| r.get("uuid").is_some())
        .collect()
}
fn assert_morrow(document: &Value) {
    assert_eq!(document["format"], "polylogue-session");
    assert_eq!(document["version"], 1);
    assert_eq!(document["session"]["id"], reference());
    assert_eq!(document["session"]["native_id"], SID);
    let messages = document["messages"].as_array().unwrap();
    let records = fixture_records();
    assert_eq!(messages.len(), 12);
    for (m, r) in messages.iter().zip(&records) {
        let id = polylogue_model::MessageId::new(
            polylogue_model::SessionId::from_stored(reference()).unwrap(),
            r["uuid"].as_str().unwrap(),
        )
        .unwrap()
        .as_str()
        .to_owned();
        assert_eq!(m["id"], id);
        assert_eq!(m["role"], r["type"]);
        assert_eq!(m["at"], r["timestamp"]);
        let content = &r["message"]["content"];
        let expected: Vec<Value> = if let Some(text) = content.as_str() {
            vec![json!({"type":"text","text":text})]
        } else {
            content.as_array().unwrap().clone()
        };
        let blocks = m["blocks"].as_array().unwrap();
        assert_eq!(blocks.len(), expected.len());
        for (index, (block, source)) in blocks.iter().zip(&expected).enumerate() {
            assert_eq!(block["id"], format!("{id}:{index}"));
            match source["type"].as_str().unwrap() {
                "text" => assert_eq!(block["text"], source["text"]),
                "thinking" => {
                    assert_eq!(block["text"], source["thinking"]);
                    assert_eq!(block["signature"], source["signature"]);
                }
                "tool_use" => {
                    assert_eq!(block["tool"], source["name"]);
                    assert_eq!(block["input"], source["input"]);
                    assert_eq!(block["call_id"], source["id"]);
                }
                "tool_result" => {
                    assert_eq!(block["text"], source["content"]);
                    assert_eq!(block["outcome"], "ok");
                    assert_eq!(block["call_id"], source["tool_use_id"]);
                }
                other => panic!("fixture block {other}"),
            }
        }
        if r["type"] == "assistant" {
            assert_eq!(m["usage"]["input"], r["message"]["usage"]["input_tokens"]);
            assert_eq!(m["usage"]["output"], r["message"]["usage"]["output_tokens"]);
            assert_eq!(m["usage"]["service_tier"], "standard");
            assert!(
                m["attrs"]
                    .as_object()
                    .unwrap()
                    .contains_key("stop_sequence")
            );
            assert!(m["attrs"]["stop_sequence"].is_null());
        }
        assert!(!m["field_origins"].as_array().unwrap().is_empty());
        assert!(!m["sources"].as_array().unwrap().is_empty());
    }
    assert_eq!(
        messages[4]["blocks"][0]["data"]["stdout"],
        records[4]["toolUseResult"]["stdout"]
    );
    assert_eq!(messages[4]["blocks"][0]["data"]["stderr"], "");
    assert_eq!(messages[4]["blocks"][0]["data"]["interrupted"], false);
}
#[test]
fn actual_morrow_json_read_survives_provider_loss_sigkill_and_observation_replay() {
    let mut world = World::new();
    let path = world.input("morrow.jsonl", MORROW);
    let request = "01900000-0000-7000-8000-000000000201";
    let receipt = world.import(&path, request);
    let id = receipt["retained"]["id"].as_str().unwrap();
    assert_eq!(receipt["retained"]["length"], MORROW.len() as u64);
    fs::remove_file(&path).unwrap();
    let document = world.read(&reference(), id, 12);
    assert_morrow(&document);
    world.restart(); // Child::kill sends SIGKILL; Drop waits before starting.
    let replay = world.import(&path, request);
    assert_eq!(replay["retained"], receipt["retained"]);
    assert_eq!(replay["request_id"], receipt["request_id"]);
    assert_eq!(replay["seen_at"], receipt["seen_at"]);
    let after = world.read(&reference(), id, 12);
    assert_eq!(after, document);
    let unknown = world.cli(&["claude-code:unknown:opaque", "read", "--format", "json"]);
    assert_eq!(unknown.status.code(), Some(4));
    assert!(unknown.stdout.is_empty());
    assert!(!unknown.stderr.is_empty());
    let search = world.cli(&["search", "morrow"]);
    assert_eq!(search.status.code(), Some(2));
    let wait = world.cli(&[
        "import",
        path.to_str().unwrap(),
        "--format",
        "claude-code-jsonl",
        "--wait",
    ]);
    assert_eq!(wait.status.code(), Some(2));
}
#[test]
fn daemon_raw_frames_and_cli_both_mask_cross_piece_and_dense_credentials() {
    let world = World::new();
    let token = format!("ghp_{}", "A".repeat(36));
    let text = format!("{} {} suffix λ", "x".repeat(4089), token);
    let dense = (0..80).map(|_| json!({"short":token})).collect::<Vec<_>>();
    let raw=json!({"type":"assistant","uuid":"masked-message","sessionId":SID,"requestId":"masked-request","message":{"id":"masked-response","content":[{"type":"text","text":text},{"type":"tool_use","id":"masked-call","name":"Bash","input":{"command":"true","nested":dense}}],"stop_reason":"tool_use"}}).to_string()+"\n";
    let path = world.input("masked.jsonl", raw.as_bytes());
    let receipt = world.import(&path, "01900000-0000-7000-8000-000000000202");
    let document = world.read(&reference(), receipt["retained"]["id"].as_str().unwrap(), 1);
    let (socket, frames, end) = world.socket_read(&reference());
    assert_eq!(socket, document);
    assert_eq!(end["outcome"], "degraded");
    assert_eq!(end["canonical_complete"], false);
    assert!(!end["missing"].as_array().unwrap().is_empty());
    assert!(!String::from_utf8(frames).unwrap().contains(&token));
    assert!(!serde_json::to_string(&document).unwrap().contains(&token));
    let actual = document["messages"][0]["blocks"][0]["text"]
        .as_str()
        .unwrap();
    assert!(actual.starts_with(&format!("{} ", "x".repeat(4089))));
    assert!(actual.ends_with(" suffix λ"));
    assert!(actual.contains("[credential:"));
    let nested = document["messages"][0]["blocks"][1]["input"]["nested"]
        .as_array()
        .unwrap();
    assert_eq!(nested.len(), 80);
    assert!(
        nested
            .iter()
            .all(|item| item["short"].as_str().unwrap().contains("[credential:"))
    );
}
fn simple(uuid: &str, text: Value) -> Vec<u8> {
    (json!({"type":"user","uuid":uuid,"sessionId":SID,"timestamp":"2026-10-01T08:00:00.000Z","message":{"role":"user","content":text}}).to_string()+"\n").into_bytes()
}
fn wait_message(world: &World, id: &str, native: &str) {
    let sid = polylogue_model::SessionId::from_stored(reference()).unwrap();
    let mid = polylogue_model::MessageId::new(sid, native).unwrap();
    let reference = format!("message:{}", mid.as_str());
    let deadline = Instant::now() + WAIT;
    loop {
        world.progress(id);
        let out = world.cli(&[&reference, "read", "--format", "terminal"]);
        if out.status.code() == Some(5) {
            return;
        }
        assert!(matches!(out.status.code(), Some(2 | 4)), "{out:?}");
        assert!(Instant::now() < deadline, "Message never selected: {out:?}");
        thread::sleep(Duration::from_millis(5));
    }
}
#[test]
fn actual_cli_refuses_pending_members_and_ambiguous_order_without_json_subset() {
    let world = World::new();
    let mut raw = simple("ready", json!("ready text"));
    let path = world.input("ready.jsonl", &raw);
    let first = world.import(&path, "01900000-0000-7000-8000-000000000203");
    let first_id = first["retained"]["id"].as_str().unwrap();
    wait_message(&world, first_id, "ready");
    raw.extend(simple(
        "pending",
        json!([{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}]),
    ));
    let path = world.input("pending.jsonl", &raw);
    let pending = world.import(&path, "01900000-0000-7000-8000-000000000204");
    let deadline = Instant::now() + WAIT;
    loop {
        let p = world.raw_progress(pending["retained"]["id"].as_str().unwrap());
        assert_eq!(p["source_rows"]["scope"], "imported_source_rows");
        if !p["failure"].is_null() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "unsupported image never became explicit: {p}"
        );
        thread::sleep(Duration::from_millis(5));
    }
    let out = world.cli(&[&reference(), "read", "--format", "json"]);
    assert_eq!(out.status.code(), Some(2), "{out:?}");
    assert!(out.stdout.is_empty());
    assert!(String::from_utf8_lossy(&out.stderr).contains("pending Message"));
    drop(world);

    let world = World::new();
    for (index, native) in ["first", "independent"].into_iter().enumerate() {
        let path = world.input(&format!("{native}.jsonl"), &simple(native, json!(native)));
        let receipt = world.import(
            &path,
            &format!("01900000-0000-7000-8000-{:012}", 205 + index),
        );
        wait_message(&world, receipt["retained"]["id"].as_str().unwrap(), native);
    }
    let out = world.cli(&[&reference(), "read", "--format", "json"]);
    assert_eq!(out.status.code(), Some(2), "{out:?}");
    assert!(out.stdout.is_empty());
    assert!(String::from_utf8_lossy(&out.stderr).contains("order is unresolved"));
}

fn wait_import(world: &World, path: &Path, request_id: &str, timeout: &str) -> Output {
    let context = world.work.path().join("context.json");
    fs::write(
        &context,
        json!({"version":1,"context":["claude-main",SID]}).to_string(),
    )
    .unwrap();
    world.cli(&[
        "import",
        path.to_str().unwrap(),
        "--format",
        "claude-code-jsonl",
        "--context",
        context.to_str().unwrap(),
        "--request-id",
        request_id,
        "--seen-at",
        "17",
        "--machine",
        "00112233445566778899aabbccddeeff",
        "--host",
        "synthetic-collector",
        "--wait",
        "--wait-timeout",
        timeout,
    ])
}
#[test]
fn source_row_wait_settles_fixed_basis_then_provider_loss_restart_replays_without_claiming_session_complete()
 {
    let mut world = World::new();
    let bytes = simple("wait-row", json!("settled retained prompt"));
    let path = world.input("wait.jsonl", &bytes);
    let id = "01900000-0000-7000-8000-000000000401";
    let output = wait_import(&world, &path, id, "20");
    assert_eq!(output.status.code(), Some(0), "{output:?}; {}", world.log());
    let receipt: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(receipt.get("canonical_complete").is_none());
    assert_eq!(receipt["source_rows"]["scope"], "imported_source_rows");
    assert_eq!(receipt["source_rows"]["state"], "settled");
    assert_eq!(receipt["source_rows"]["basis"]["length"], bytes.len());
    let read = world.cli(&[&reference(), "read", "--format", "json"]);
    assert_eq!(read.status.code(), Some(5));
    let first: Value = serde_json::from_slice(&read.stdout).unwrap();
    assert_eq!(first["result"]["outcome"], "degraded");
    assert!(!first["result"]["notes"].as_array().unwrap().is_empty());
    fs::remove_file(&path).unwrap();
    world.restart();
    let replay = wait_import(&world, &path, id, "20");
    assert_eq!(replay.status.code(), Some(0), "{replay:?}");
    let again: Value = serde_json::from_slice(&replay.stdout).unwrap();
    assert_eq!(again["retained"], receipt["retained"]);
    assert_eq!(again["source_rows"], receipt["source_rows"]);
    let read = world.cli(&[&reference(), "read", "--format", "json"]);
    assert_eq!(read.status.code(), Some(5));
    let second: Value = serde_json::from_slice(&read.stdout).unwrap();
    assert_eq!(first["messages"], second["messages"]);
}
#[test]
fn malformed_source_wait_keeps_usable_rows_degraded_and_names_retained_failure() {
    let mut world = World::new();
    let mut bytes = simple("malformed-row", json!("known retained row"));
    bytes.extend_from_slice(b"{not json}\n");
    let path = world.input("malformed-wait.jsonl", &bytes);
    let id = "01900000-0000-7000-8000-000000000402";
    let output = wait_import(&world, &path, id, "20");
    assert_eq!(output.status.code(), Some(5), "{output:?}; {}", world.log());
    let receipt: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(receipt["source_rows"]["state"], "unavailable");
    assert_eq!(receipt["source_rows"]["usable_result"], true);
    assert_eq!(receipt["unknown"]["record_errors"], 1);
    assert!(!receipt["notes"].as_array().unwrap().is_empty());
    let read = world.cli(&[&reference(), "read", "--format", "json"]);
    assert_eq!(read.status.code(), Some(5));
    fs::remove_file(&path).unwrap();
    world.restart();
    let replay = wait_import(&world, &path, id, "20");
    assert_eq!(replay.status.code(), Some(5), "{replay:?}");
}
#[test]
fn incomplete_source_wait_times_out_without_certifying_from_a_later_peer() {
    let world = World::new();
    let mut bytes = simple("partial-row", json!("known prefix"));
    bytes.extend_from_slice(b"{\"unfinished\":");
    let path = world.input("partial-wait.jsonl", &bytes);
    let id = "01900000-0000-7000-8000-000000000403";
    let output = wait_import(&world, &path, id, "0");
    assert_eq!(output.status.code(), Some(7), "{output:?}");
    let receipt: Value = serde_json::from_slice(&output.stdout).unwrap();
    let peer = world.input("peer-wait.jsonl", b"\n");
    let complete = wait_import(&world, &peer, "01900000-0000-7000-8000-000000000404", "20");
    assert_eq!(complete.status.code(), Some(0), "{complete:?}");
    let output = wait_import(&world, &path, id, "1");
    assert_eq!(output.status.code(), Some(7), "{output:?}");
    let later: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        later["source_rows"]["basis"],
        receipt["source_rows"]["basis"]
    );
    assert_eq!(later["source_rows"]["state"], "pending");
    assert!(
        later["retained"]["id"]
            != serde_json::from_slice::<Value>(&complete.stdout).unwrap()["retained"]["id"]
    );
}

#[test]
fn disconnecting_a_source_waiter_keeps_the_retained_owner_and_peer_available() {
    let world = World::new();
    let mut bytes = simple("disconnect-row", json!("prefix"));
    bytes.extend_from_slice(b"{\"unfinished\":");
    let path = world.input("disconnect-wait.jsonl", &bytes);
    let id = "01900000-0000-7000-8000-000000000405";
    let receipt = world.import(&path, id);
    let context = world.work.path().join("context.json");
    let mut client = OwnedChild(
        world
            .command()
            .args([
                "import",
                path.to_str().unwrap(),
                "--format",
                "claude-code-jsonl",
                "--context",
                context.to_str().unwrap(),
                "--request-id",
                id,
                "--seen-at",
                "17",
                "--machine",
                "00112233445566778899aabbccddeeff",
                "--host",
                "synthetic-collector",
                "--wait",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let p = world.raw_progress(receipt["retained"]["id"].as_str().unwrap());
        if p["parser"]["status"] == "waiting_tail" {
            break;
        }
        assert!(Instant::now() < deadline, "{p}");
        thread::sleep(Duration::from_millis(5));
    }
    assert!(client.0.try_wait().unwrap().is_none());
    drop(client);
    let peer = world.input("after-disconnect.jsonl", b"\n");
    let output = wait_import(&world, &peer, "01900000-0000-7000-8000-000000000406", "20");
    assert_eq!(output.status.code(), Some(0), "{output:?}; {}", world.log());
    assert_eq!(
        world.raw_progress(receipt["retained"]["id"].as_str().unwrap())["source_rows"]["state"],
        "pending"
    );
}

#[test]
fn source_wait_never_turns_no_usable_interpretation_or_unsupported_body_into_empty_success() {
    for (name, bytes, id) in [
        (
            "only-malformed",
            b"{invalid json}\n".to_vec(),
            "01900000-0000-7000-8000-000000000407",
        ),
        (
            "unsupported-image",
            simple(
                "unsupported",
                json!([{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}]),
            ),
            "01900000-0000-7000-8000-000000000408",
        ),
    ] {
        let mut world = World::new();
        let path = world.input(name, &bytes);
        let out = wait_import(&world, &path, id, "20");
        assert_eq!(
            out.status.code(),
            Some(2),
            "{name}: {out:?}; {}",
            world.log()
        );
        let reply: Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(reply["outcome"], "error");
        assert_eq!(reply["source_rows"]["state"], "unavailable");
        assert_eq!(reply["source_rows"]["usable_result"], false);
        assert!(reply["error"]["message"].as_str().unwrap().len() > 0);
        fs::remove_file(&path).unwrap();
        world.restart();
        let retry = wait_import(&world, &path, id, "20");
        assert_eq!(retry.status.code(), Some(2), "{retry:?}");
    }
}
#[test]
fn retained_byte_eof_stays_pending_until_real_message_effects_are_published() {
    let world = World::new();
    let text = "source effect ".repeat(15000);
    let bytes = simple("large-canonical-effect", json!(text));
    let path = world.input("eof-before-effects.jsonl", &bytes);
    let id = "01900000-0000-7000-8000-000000000409";
    let receipt = world.import(&path, id);
    let input = receipt["retained"]["id"].as_str().unwrap();
    let deadline = Instant::now() + WAIT;
    loop {
        let p = world.raw_progress(input);
        assert!(p["failure"].is_null(), "{p}");
        if p["parser"]["parsed_upto"] == receipt["retained"]["length"] {
            assert_eq!(
                p["source_rows"]["state"], "pending",
                "did not observe the actual EOF-before-canonical boundary: {p}"
            );
            assert_eq!(p["parser"]["finalization_pending"], true);
            break;
        }
        assert!(Instant::now() < deadline, "{p}");
        thread::sleep(Duration::from_millis(2));
    }
    let settled = wait_import(&world, &path, id, "30");
    assert_eq!(
        settled.status.code(),
        Some(0),
        "{settled:?}; {}",
        world.log()
    );
    let read = world.cli(&[&reference(), "read", "--format", "json"]);
    assert_eq!(read.status.code(), Some(5));
    let doc: Value = serde_json::from_slice(&read.stdout).unwrap();
    assert_eq!(doc["messages"][0]["blocks"][0]["text"], text);
}

#[test]
fn admission_refusal_and_provider_diagnostics_are_named_without_credential_disclosure() {
    let world = World::new();
    let bytes = simple("too-large-member", json!("capacity ".repeat(70000)));
    let path = world.input("admission-wait.jsonl", &bytes);
    let out = wait_import(&world, &path, "01900000-0000-7000-8000-000000000410", "20");
    assert_eq!(out.status.code(), Some(2), "{out:?}; {}", world.log());
    let failure: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(failure["source_rows"]["state"], "unavailable");
    assert!(
        failure["error"]["message"]
            .as_str()
            .unwrap()
            .contains("admission")
    );
    let secret = ["gh", "p_"].concat() + &"A".repeat(36);
    let mut record: Value =
        serde_json::from_slice(&simple("secret-diagnostic", json!("supported content"))).unwrap();
    record["message"][&secret] = json!("unsupported metadata");
    let bytes = (record.to_string() + "\n").into_bytes();
    let path = world.input("diagnostic-wait.jsonl", &bytes);
    let out = wait_import(&world, &path, "01900000-0000-7000-8000-000000000411", "20");
    assert_eq!(out.status.code(), Some(2), "{out:?}; {}", world.log());
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(!text.contains(&secret), "{text}");
    assert!(text.contains("credential:"), "{text}");
    let failure: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        failure["source_rows"]["basis"]["original_hash"],
        json!(polylogue_model::BlobHash::of(&bytes))
    );
    let progress = world.raw_progress(failure["retained"]["id"].as_str().unwrap());
    assert!(!progress.to_string().contains(&secret));
}
