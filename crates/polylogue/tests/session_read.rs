//! Real subprocess daemon/CLI reads with source-scoped completion qualifications.
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
            if matches!(out.status.code(), Some(0 | 5)) {
                let document: Value = serde_json::from_slice(&out.stdout).unwrap();
                if out.status.code() == Some(5) {
                    assert!(
                        String::from_utf8_lossy(&out.stderr).contains("coverage is incomplete"),
                        "{out:?}"
                    );
                } else {
                    assert_eq!(document["result"]["outcome"], "ok");
                    assert_eq!(document["result"]["settlement"]["state"], "settled");
                }
                if document["messages"].as_array().unwrap().len() == messages
                    && p["source_rows"]["state"] != "pending"
                {
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
    assert!(
        !document.as_object().unwrap().contains_key("recovery"),
        "an unrecovered standalone document omits recovery"
    );
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
    assert_eq!(document["result"]["outcome"], "ok");
    let (framed, _, end) = world.socket_read(&reference());
    assert_eq!(framed, document);
    assert_eq!(end["outcome"], "ok");
    assert_eq!(end["canonical_complete"], true);
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
    wait_import_context(world, path, request_id, timeout, &["claude-main", SID])
}
fn wait_import_context(
    world: &World,
    path: &Path,
    request_id: &str,
    timeout: &str,
    parts: &[&str],
) -> Output {
    let context = world.work.path().join("context.json");
    fs::write(&context, json!({"version":1,"context":parts}).to_string()).unwrap();
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

#[test]
fn session_document_includes_malformed_only_and_possible_foreign_link_sources() {
    let mut world = World::new();
    let clean = world.input(
        "qualified-main.jsonl",
        &simple("qualified-row", json!("retained direct Session row")),
    );
    let good = wait_import(&world, &clean, "01900000-0000-7000-8000-000000000420", "20");
    assert_eq!(good.status.code(), Some(0), "{good:?}");
    let malformed = world.input("only-malformed.jsonl", b"{not json}\n");
    let bad = wait_import(
        &world,
        &malformed,
        "01900000-0000-7000-8000-000000000421",
        "20",
    );
    assert_eq!(bad.status.code(), Some(2), "{bad:?}");
    let bad_receipt: Value = serde_json::from_slice(&bad.stdout).unwrap();
    assert_eq!(bad_receipt["source_rows"]["usable_result"], false);
    let peer = world.input("unrelated-pending.jsonl", b"{unfinished");
    let peer_out = wait_import_context(
        &world,
        &peer,
        "01900000-0000-7000-8000-000000000422",
        "0",
        &["claude-main", "unrelated-main"],
    );
    assert_eq!(peer_out.status.code(), Some(7));
    let read = world.cli(&[&reference(), "read", "--format", "json"]);
    assert_eq!(read.status.code(), Some(5), "{read:?}");
    let first: Value = serde_json::from_slice(&read.stdout).unwrap();
    assert_eq!(first["messages"].as_array().unwrap().len(), 1);
    assert_eq!(first["result"]["unknown"]["record_errors"], 1);
    assert!(
        first["result"]["notes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|note| note
                .as_str()
                .unwrap()
                .contains("undecodable complete records"))
    );
    assert!(
        first["result"]["notes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|note| note.as_str().unwrap().contains("pending interpretation")),
        "{first}"
    );
    let (socket, _, end) = world.socket_read(&reference());
    assert_eq!(socket, first);
    assert_eq!(end["unknown"], first["result"]["unknown"]);
    assert_eq!(end["notes"], first["result"]["notes"]);
    // A subagent's parent is not proved by its folder or by any matching row.
    // Its malformed record cannot be attributed to A; the dependency is named.
    let subagent = world.input("unresolved-subagent.jsonl", b"{unreadable}\n");
    let unresolved = wait_import_context(
        &world,
        &subagent,
        "01900000-0000-7000-8000-000000000424",
        "20",
        &["claude-subagent", "synthetic-agent"],
    );
    assert_eq!(unresolved.status.code(), Some(2), "{unresolved:?}");
    let qualified = world.cli(&[&reference(), "read", "--format", "json"]);
    assert_eq!(qualified.status.code(), Some(5));
    let first: Value = serde_json::from_slice(&qualified.stdout).unwrap();
    assert_eq!(first["result"]["unknown"]["record_errors"], 1);
    assert!(
        first["result"]["notes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|note| note
                .as_str()
                .unwrap()
                .contains("dependency shapes remain unresolved"))
    );
    // Another observation/path into the identical context-qualified input is
    // the same failure identity, not a second Session/candidate dependency.
    let copied = world.input("copied-malformed.jsonl", b"{not json}\n");
    let repeated = wait_import(
        &world,
        &copied,
        "01900000-0000-7000-8000-000000000423",
        "20",
    );
    assert_eq!(repeated.status.code(), Some(2));
    assert_eq!(
        serde_json::from_slice::<Value>(&repeated.stdout).unwrap()["retained"],
        bad_receipt["retained"]
    );
    let reread = world.cli(&[&reference(), "read", "--format", "json"]);
    assert_eq!(reread.status.code(), Some(5));
    assert_eq!(
        without_settlement(serde_json::from_slice::<Value>(&reread.stdout).unwrap()),
        without_settlement(first.clone())
    );
    fs::remove_file(&clean).unwrap();
    fs::remove_file(&malformed).unwrap();
    fs::remove_file(&copied).unwrap();
    fs::remove_file(&peer).unwrap();
    fs::remove_file(&subagent).unwrap();
    world.restart();
    let replay = wait_import(
        &world,
        &malformed,
        "01900000-0000-7000-8000-000000000421",
        "20",
    );
    assert_eq!(replay.status.code(), Some(2));
    assert_eq!(
        serde_json::from_slice::<Value>(&replay.stdout).unwrap()["retained"],
        bad_receipt["retained"]
    );
    let read = world.cli(&[&reference(), "read", "--format", "json"]);
    assert_eq!(read.status.code(), Some(5));
    assert_eq!(
        without_settlement(serde_json::from_slice::<Value>(&read.stdout).unwrap()),
        without_settlement(first.clone())
    );
}

#[test]
fn ordinary_listing_and_prose_search_share_retained_ids_and_population_qualification() {
    let mut world = World::new();
    let empty = world.cli(&["sessions", "--json"]);
    assert_eq!(empty.status.code(), Some(1), "{empty:?}");
    let value: Value = serde_json::from_slice(&empty.stdout).unwrap();
    assert_eq!(value["outcome"], "empty");
    assert_eq!(value["count"], 0);
    let empty_search = world.cli(&["absenttoken", "--json"]);
    assert_eq!(empty_search.status.code(), Some(1));
    let mut prose: Value = serde_json::from_slice(&simple(
        "older",
        json!("Łódź weekday_offset refactor carefully"),
    ))
    .unwrap();
    prose["promptSource"] = json!("typed");
    let source = world.input(
        "prose-search.jsonl",
        &(prose.to_string() + "\n").into_bytes(),
    );
    let ready = wait_import(
        &world,
        &source,
        "01900000-0000-7000-8000-000000000430",
        "20",
    );
    assert_eq!(ready.status.code(), Some(0), "{ready:?}");
    let listed = world.cli(&["origin:claude-code", "list", "--json"]);
    assert_eq!(listed.status.code(), Some(5), "{listed:?}");
    let listed: Value = serde_json::from_slice(&listed.stdout).unwrap();
    assert_eq!(listed["rows"][0]["id"], reference());
    assert_eq!(listed["count"], 1);
    for text in [
        "lodz",
        "łódź",
        "weekday offset",
        "\"weekday offset\"",
        "refact*",
        "sessions where origin:claude-code text:lodz",
    ] {
        let hit = world.cli(&[text, "--json"]);
        assert_eq!(hit.status.code(), Some(5), "{text}: {hit:?}");
        let hit: Value = serde_json::from_slice(&hit.stdout).unwrap();
        assert_eq!(hit["count"], 1, "{text}: {hit}");
        assert_eq!(hit["rows"][0]["id"], reference());
        assert!(
            hit["rows"][0]["hits"][0]["message"]
                .as_str()
                .unwrap()
                .ends_with(":older")
        );
    }
    for word in ["list", "status", "messages"] {
        let output = world.cli(&["find", word, "--json"]);
        assert_eq!(output.status.code(), Some(5), "{output:?}");
        assert_eq!(
            serde_json::from_slice::<Value>(&output.stdout).unwrap()["count"],
            0
        );
    }
    assert_eq!(world.cli(&["find", "--json"]).status.code(), Some(2));
    let nohit = world.cli(&["absenttoken", "--json"]);
    assert_eq!(nohit.status.code(), Some(5));
    let nohit: Value = serde_json::from_slice(&nohit.stdout).unwrap();
    assert_eq!(nohit["count"], 0);
    assert_eq!(nohit["outcome"], "degraded");
    fs::remove_file(&source).unwrap();
    world.restart();
    let again = world.cli(&["lodz", "--json"]);
    assert_eq!(again.status.code(), Some(5));
    let again: Value = serde_json::from_slice(&again.stdout).unwrap();
    assert_eq!(again["rows"][0]["id"], reference());
    let read = world.cli(&[
        again["rows"][0]["id"].as_str().unwrap(),
        "read",
        "--format",
        "json",
    ]);
    assert_eq!(read.status.code(), Some(5));
    assert_eq!(
        serde_json::from_slice::<Value>(&read.stdout).unwrap()["messages"][0]["blocks"][0]["text"],
        "Łódź weekday_offset refactor carefully"
    );
}
#[test]
fn prose_conjunction_crosses_text_blocks_but_phrases_do_not() {
    let world = World::new();
    let mut record: Value = serde_json::from_slice(&simple(
        "split",
        json!([{"type":"text","text":"boundaryfirst"},{"type":"text","text":"boundarysecond"}]),
    ))
    .unwrap();
    record["promptSource"] = json!("typed");
    let source = world.input("split-prose.jsonl", (record.to_string() + "\n").as_bytes());
    let ready = wait_import(
        &world,
        &source,
        "01900000-0000-7000-8000-000000000434",
        "20",
    );
    assert_eq!(ready.status.code(), Some(0), "{ready:?}");
    for (query, count) in [
        ("boundaryfirst boundarysecond", 1),
        ("\"boundaryfirst boundarysecond\"", 0),
    ] {
        let output = world.cli(&[query, "--json"]);
        assert_eq!(output.status.code(), Some(5), "{output:?}");
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["count"], count, "{result}");
        assert_eq!(result["ranking"]["state"], "unavailable");
    }
}
#[test]
fn query_unknown_population_is_not_completed_empty_and_request_shapes_refuse() {
    let world = World::new();
    let empty = world.input("query-empty.jsonl", b"\n");
    assert_eq!(
        wait_import(&world, &empty, "01900000-0000-7000-8000-000000000431", "20")
            .status
            .code(),
        Some(0)
    );
    assert_eq!(world.cli(&["sessions", "--json"]).status.code(), Some(1));
    let malformed = world.input("query-malformed-only.jsonl", b"{not json}\n");
    assert_eq!(
        wait_import(
            &world,
            &malformed,
            "01900000-0000-7000-8000-000000000432",
            "20"
        )
        .status
        .code(),
        Some(2)
    );
    let query = world.cli(&["sessions", "--json"]);
    assert_eq!(query.status.code(), Some(5), "{query:?}");
    let query: Value = serde_json::from_slice(&query.stdout).unwrap();
    assert_eq!(query["count"], 0);
    assert_eq!(query["unknown"]["record_errors"], 1);
    assert_eq!(query["unknown"]["unavailable_inputs"], 1);
    let sub = world.input("query-unresolved-subagent.jsonl", b"{bad}\n");
    assert_eq!(
        wait_import_context(
            &world,
            &sub,
            "01900000-0000-7000-8000-000000000433",
            "20",
            &["claude-subagent", "synthetic-agent"]
        )
        .status
        .code(),
        Some(2)
    );
    let query = world.cli(&["find", "missingneedle", "--json"]);
    assert_eq!(query.status.code(), Some(5));
    let query: Value = serde_json::from_slice(&query.stdout).unwrap();
    assert_eq!(query["unknown"]["record_errors"], 1);
    assert_eq!(query["unknown"]["unresolved_inputs"], 1);
    for query in [
        "near:semantic",
        "messages needle",
        "sessions | group by origin",
        "needle OR other",
        "--reveal",
    ] {
        let out = world.cli(&[query, "--json"]);
        assert!(matches!(out.status.code(), Some(2 | 3)), "{query}: {out:?}");
        assert!(out.stdout.is_empty());
    }
}

fn without_settlement(mut document: Value) -> Value {
    // New observations bind a new read basis even when source payloads and
    // retained failure identities are unchanged.
    document["result"]
        .as_object_mut()
        .unwrap()
        .remove("settlement");
    document
}

#[test]
fn metadata_only_source_is_not_a_complete_empty_selection() {
    let world = World::new();
    let source = world.input(
        "metadata-only.jsonl",
        b"{\"type\":\"ai-title\",\"aiTitle\":\"Synthetic title\"}\n",
    );
    let imported = wait_import(
        &world,
        &source,
        "01900000-0000-7000-8000-000000000440",
        "20",
    );
    assert_eq!(imported.status.code(), Some(0));
    let selected = world.cli(&[&reference(), "--json"]);
    assert_eq!(selected.status.code(), Some(5));
    let selected: Value = serde_json::from_slice(&selected.stdout).unwrap();
    assert_eq!(selected["count"], 0);
    assert!(selected["notes"].as_array().unwrap().iter().any(|note| {
        note.as_str()
            .unwrap()
            .contains("no published catalog owner")
    }));
}

#[test]
fn compatible_full_snapshot_append_settles_once_and_preserves_every_input_citation_after_restart() {
    // This positive, multi-snapshot fixture shares the suite's bounded WAIT.
    // A short CLI wait can expire under parallel debug-test load while the
    // retained owner continues; retry only that same receipt and fixed basis.
    fn settled_import(world: &World, path: &Path, request_id: &str) -> Output {
        let started = Instant::now();
        let deadline = started + WAIT;
        let mut original = None;
        let mut progress = Value::Null;
        let mut attempts = 0;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "compatible import {request_id} exceeded {WAIT:?}; progress {progress}; daemon {}",
                world.log()
            );
            let timeout = remaining.as_secs().clamp(1, 30).to_string();
            let out = wait_import(world, path, request_id, &timeout);
            attempts += 1;
            let receipt: Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
                panic!(
                    "compatible import {request_id}: {e}; {out:?}; daemon {}",
                    world.log()
                )
            });
            assert_eq!(receipt["request_id"], request_id, "{receipt}");
            assert_eq!(receipt["seen_at"], 17, "{receipt}");
            let retained_basis =
                json!({"retained":receipt["retained"],"basis":receipt["source_rows"]["basis"]});
            if let Some(expected) = &original {
                assert_eq!(
                    &retained_basis, expected,
                    "retry changed immutable retained basis"
                );
            } else {
                original = Some(retained_basis);
            }
            if out.status.success() {
                assert_eq!(receipt["source_rows"]["state"], "settled", "{receipt}");
                eprintln!(
                    "compatible import {request_id} settled after {attempts} attempt(s), {:?}",
                    started.elapsed()
                );
                return out;
            }
            assert_eq!(
                out.status.code(),
                Some(7),
                "{out:?}; progress {progress}; daemon {}",
                world.log()
            );
            assert_eq!(receipt["error"]["kind"], "timeout", "{receipt}");
            assert_eq!(receipt["source_rows"]["state"], "pending", "{receipt}");
            assert!(
                Instant::now() < deadline,
                "compatible import {request_id} exceeded {WAIT:?}; receipt {receipt}; prior progress {progress}; daemon {}",
                world.log()
            );
            progress = world.progress(receipt["retained"]["id"].as_str().unwrap());
            assert_eq!(
                progress["source_rows"]["basis"], receipt["source_rows"]["basis"],
                "retry target changed"
            );
            eprintln!(
                "compatible import {request_id} pending after {attempts} attempt(s), {:?}; progress {progress}",
                started.elapsed()
            );
        }
    }
    let mut world = World::new();
    let path = world.input("compatible-snapshot.jsonl", MORROW);
    let base = settled_import(&world, &path, "01900000-0000-7000-8000-000000000700");
    assert_eq!(base.status.code(), Some(0), "{base:?}; {}", world.log());
    let base: Value = serde_json::from_slice(&base.stdout).unwrap();
    let base_input = base["retained"]["id"].as_str().unwrap();
    let before = world.read(&reference(), base_input, 12);
    let last = fixture_records().last().unwrap()["uuid"]
        .as_str()
        .unwrap()
        .to_owned();
    let records = [
        json!({"type":"user","uuid":"compatible-append-user","sessionId":SID,"timestamp":"2026-10-16T00:00:00.000Z","parentUuid":last,"isSidechain":false,"promptSource":"typed","message":{"role":"user","content":"Neutral appended request"}}),
        json!({"type":"assistant","uuid":"compatible-append-model","sessionId":SID,"timestamp":"2026-10-16T00:01:00.000Z","parentUuid":"compatible-append-user","isSidechain":false,"requestId":"compatible-request","message":{"id":"compatible-response","role":"assistant","model":"claude-neutral-control","content":[{"type":"text","text":"Neutral appended response"}],"stop_reason":"end_turn"}}),
    ];
    let mut stream = fs::OpenOptions::new().append(true).open(&path).unwrap();
    for record in records {
        writeln!(stream, "{record}").unwrap()
    }
    stream.sync_all().unwrap();
    drop(stream);
    let appended = settled_import(&world, &path, "01900000-0000-7000-8000-000000000701");
    assert_eq!(
        appended.status.code(),
        Some(0),
        "{appended:?}; {}",
        world.log()
    );
    let appended: Value = serde_json::from_slice(&appended.stdout).unwrap();
    assert_eq!(appended["source_rows"]["state"], "settled");
    let input = appended["retained"]["id"].as_str().unwrap();
    let after = world.read(&reference(), input, 14);
    let messages = after["messages"].as_array().unwrap();
    for (prior, current) in before["messages"].as_array().unwrap().iter().zip(messages) {
        for field in ["id", "role", "at", "blocks"] {
            assert_eq!(current[field], prior[field], "{field}")
        }
        let sources = current["sources"].as_array().unwrap();
        assert_eq!(sources.len(), 2, "{sources:?}");
        let mut inputs = sources
            .iter()
            .map(|s| s["input"].as_str().unwrap())
            .collect::<Vec<_>>();
        inputs.sort();
        let mut expected = vec![base_input, input];
        expected.sort();
        assert_eq!(inputs, expected);
    }
    assert!(
        messages[12]["id"]
            .as_str()
            .unwrap()
            .ends_with(":compatible-append-user")
    );
    assert!(
        messages[13]["id"]
            .as_str()
            .unwrap()
            .ends_with(":compatible-append-model")
    );
    fs::remove_file(&path).unwrap();
    world.restart(); // Genuine SIGKILL; ordinary current-index WAL reuse remains supported.
    let replay = settled_import(&world, &path, "01900000-0000-7000-8000-000000000701");
    assert_eq!(replay.status.code(), Some(0), "{replay:?}; {}", world.log());
    let replay: Value = serde_json::from_slice(&replay.stdout).unwrap();
    assert_eq!(replay["retained"], appended["retained"]);
    assert_eq!(world.read(&reference(), input, 14), after);
}

fn model_fixture(huge: &str, extra: bool) -> Vec<u8> {
    let mut rows =
        vec![json!({"type":"ai-title","aiTitle":"Bounded neutral control","sessionId":SID})];
    let mut parent = Value::Null;
    let entries = if extra {
        vec![
            ("bounded-first", "A repeated neutral passage."),
            ("bounded-huge", huge),
            ("bounded-repeat", "A repeated neutral passage."),
            ("bounded-appended", "Later unrelated neutral fact."),
        ]
    } else {
        vec![
            ("bounded-first", "A repeated neutral passage."),
            ("bounded-huge", huge),
            ("bounded-repeat", "A repeated neutral passage."),
        ]
    };
    for (i, (id, text)) in entries.into_iter().enumerate() {
        rows.push(json!({"type":"user","uuid":id,"parentUuid":parent,"sessionId":SID,"isSidechain":false,"timestamp":format!("2026-10-01T08:{i:02}:00.000Z"),"promptSource":"typed","message":{"role":"user","content":text}}));
        parent = json!(id);
    }
    rows.into_iter()
        .flat_map(|v| (v.to_string() + "\n").into_bytes())
        .collect()
}
fn model_ref(native: &str) -> String {
    let sid = polylogue_model::SessionId::from_stored(reference()).unwrap();
    format!(
        "block:{}",
        polylogue_model::BlockId::new(polylogue_model::MessageId::new(sid, native).unwrap(), 0)
            .as_str()
    )
}
fn model_call(world: &World, call: &Value, budget: Option<u64>) -> Value {
    let mut args = vec![call["ref"].as_str().unwrap().to_owned(), "read".to_owned()];
    if let Some(b) = budget {
        args.extend(["--budget".into(), b.to_string()]);
    }
    if let Some(v) = call["view"].as_str() {
        args.extend(["--view".into(), v.into()]);
    }
    if let Some(c) = call["continuation"].as_str() {
        args.extend(["--continuation".into(), c.into()]);
    }
    let out = world.cli(&args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(
        matches!(out.status.code(), Some(0 | 2 | 3 | 4 | 5 | 7)),
        "{out:?}; daemon {}",
        world.log()
    );
    serde_json::from_slice(&out.stdout).unwrap_or_else(|e| panic!("{e}: {out:?}"))
}
#[test]
fn model_cli_follows_only_emitted_calls_through_unicode_middle_and_repeated_identities() {
    let world = World::new();
    let huge = format!(
        "BEGIN BLUE\n{}MIDDLE AMBER\n{}END GREEN\n",
        "🙂e\u{301} \"\\\u{1b} [neutral] ".repeat(1700),
        "λ東京🙂 ".repeat(1900)
    );
    let secret = format!("ghp_{}", "A".repeat(36));
    let huge = format!("{} {secret} {}", "n".repeat(4090), huge);
    let input = world.input("bounded.jsonl", &model_fixture(&huge, false));
    let receipt = world.import(&input, "01900000-0000-7000-8000-000000000800");
    world.read(&reference(), receipt["retained"]["id"].as_str().unwrap(), 3);
    let expected = [
        ("bounded-first", "A repeated neutral passage.".to_owned()),
        (
            "bounded-huge",
            huge.replace('\u{1b}', "\\x1b")
                .replace(&secret, "[credential: GitHub token, masked]"),
        ),
        ("bounded-repeat", "A repeated neutral passage.".to_owned()),
    ];
    let mut call = json!({"ref":reference()});
    let mut budget = 1700;
    let mut expected_index = 0;
    let mut scalar = 0u64;
    let mut reconstructed = String::new();
    let mut pages = 0;
    loop {
        let page = model_call(&world, &call, Some(budget));
        assert_ne!(page["outcome"], "error", "{page}");
        assert!((serde_json::to_vec(&page).unwrap().len() as u64).div_ceil(4) <= budget);
        assert!(!page.to_string().contains(&secret));
        assert_eq!(page["rendered_as"], "native-model-read-v1");
        assert_eq!(page["session"]["display_title"], "Bounded neutral control");
        assert!(page["settlement"].is_object());
        assert!(page["up_to_date"].is_boolean());
        let spans = page["shown"].as_array().unwrap();
        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        let (native, body) = &expected[expected_index];
        assert_eq!(span["block"], model_ref(native));
        assert_eq!(span["written_by"], "human");
        assert_eq!(span["display_start"], scalar);
        let text = span["text"].as_str().unwrap();
        assert!(!text.is_empty());
        assert_eq!(
            span["display_end"].as_u64().unwrap() - scalar,
            text.chars().count() as u64
        );
        assert!(!span["sources"].as_array().unwrap().is_empty());
        for source in span["sources"].as_array().unwrap() {
            assert!(source["start"].as_u64().unwrap() < source["end"].as_u64().unwrap());
        }
        reconstructed.push_str(text);
        scalar += text.chars().count() as u64;
        if reconstructed == *body {
            expected_index += 1;
            scalar = 0;
            reconstructed.clear();
        }
        pages += 1;
        assert!(pages < 250);
        if page["next"].is_null() {
            break;
        }
        let next = page["next"].clone();
        assert_ne!(next, call);
        if pages == 2 {
            let grow = model_call(&world, &next, Some(2200));
            let shrink = model_call(&world, &next, Some(1300));
            assert_eq!(
                grow["shown"][0]["display_start"],
                shrink["shown"][0]["display_start"]
            );
            assert!(
                grow["shown"][0]["display_end"].as_u64().unwrap()
                    > shrink["shown"][0]["display_end"].as_u64().unwrap()
            );
            let tiny = model_call(&world, &next, Some(128));
            assert_eq!(tiny["error"]["reason"], "insufficient_budget");
            assert!(serde_json::to_vec(&tiny).unwrap().len().div_ceil(4) <= 128);
            let sufficient = tiny["error"]["sufficient_budget"].as_u64().unwrap();
            let retry = model_call(&world, &next, Some(sufficient));
            assert_ne!(retry["outcome"], "error", "{retry}");
            assert_eq!(
                retry["shown"][0]["display_start"],
                grow["shown"][0]["display_start"]
            );
        }
        call = next;
        budget = if pages % 2 == 0 { 1700 } else { 1500 };
    }
    assert_eq!(expected_index, 3);
    assert!(pages > 10);
    fs::remove_file(&input).unwrap();
    // This world remains running here; restart qualification is in the second control.
}
#[test]
fn model_exact_block_append_restart_wrong_ref_options_and_selected_edit_refuse_correctly() {
    let mut world = World::new();
    let huge = "Neutral🙂 ".repeat(6000);
    let input = world.input("bounded.jsonl", &model_fixture(&huge, false));
    let receipt = world.import(&input, "01900000-0000-7000-8000-000000000801");
    world.read(&reference(), receipt["retained"]["id"].as_str().unwrap(), 3);
    let first = model_call(
        &world,
        &json!({"ref":model_ref("bounded-huge")}),
        Some(1600),
    );
    assert_ne!(first["outcome"], "error", "{first}");
    let next = first["next"].clone();
    assert!(next.is_object());
    let session = model_call(&world, &json!({"ref":reference()}), Some(1600));
    let session_next = session["next"].clone();
    let mut wrong = next.clone();
    wrong["ref"] = json!(model_ref("bounded-repeat"));
    assert_eq!(
        model_call(&world, &wrong, None)["error"]["reason"],
        "cross_reference_continuation"
    );
    let out = world.cli(&[
        &model_ref("bounded-huge"),
        "read",
        "--budget",
        "1600",
        "--view",
        "raw",
    ]);
    assert_eq!(out.status.code(), Some(2));
    assert!(out.stdout.is_empty());
    assert_eq!(
        model_call(&world, &json!({"ref":"note:unknown"}), Some(1600))["error"]["kind"],
        "not_yet_available"
    );
    let tiny = model_call(&world, &json!({"ref":model_ref("bounded-huge")}), Some(1));
    assert_eq!(tiny["error"]["minimum_budget"], 128);
    assert!(serde_json::to_vec(&tiny).unwrap().len().div_ceil(4) <= 128);
    // This supported unrelated main source cannot replace the selected donor.
    // A complete competing snapshot below can change its preferred origin.
    let other = "7f3a2c10-0002-4000-8000-000000000002";
    let extra = json!({"type":"user","uuid":"unrelated-message","parentUuid":null,"sessionId":other,"isSidechain":false,"timestamp":"2026-10-01T08:03:00.000Z","promptSource":"typed","message":{"role":"user","content":"Later unrelated neutral fact."}});
    let append = world.input("unrelated.jsonl", (extra.to_string() + "\n").as_bytes());
    let out = wait_import_context(
        &world,
        &append,
        "01900000-0000-7000-8000-000000000802",
        "20",
        &["claude-main", other],
    );
    assert!(matches!(out.status.code(), Some(0 | 5)), "{out:?}");
    let resumed = model_call(&world, &next, None);
    assert_ne!(resumed["outcome"], "error", "{resumed}");
    assert_eq!(
        resumed["shown"][0]["display_start"],
        first["shown"][0]["display_end"]
    );
    let competing = world.input("same-text-new-origin.jsonl", &model_fixture(&huge, true));
    let receipt = world.import(&competing, "01900000-0000-7000-8000-000000000804");
    world.read(&reference(), receipt["retained"]["id"].as_str().unwrap(), 4);
    assert_eq!(
        model_call(&world, &session_next, None)["error"]["reason"],
        "stale_read"
    );
    assert_eq!(
        model_call(&world, &next, None)["error"]["reason"],
        "stale_read"
    );
    fs::remove_file(&competing).unwrap();
    fs::remove_file(&input).unwrap();
    fs::remove_file(&append).unwrap();
    world.restart();
    assert_eq!(
        model_call(&world, &next, None)["error"]["reason"],
        "invalid_continuation"
    );
    let fresh = model_call(
        &world,
        &json!({"ref":model_ref("bounded-huge")}),
        Some(1600),
    );
    assert_ne!(fresh["outcome"], "error", "{fresh}");
    assert_eq!(fresh["shown"][0]["display_start"], 0);
    // Content replacement is checked through the native importer, never SQL.
    let edit = world.input(
        "edited.jsonl",
        &model_fixture(&(huge + " selected edit"), true),
    );
    let receipt = world.import(&edit, "01900000-0000-7000-8000-000000000803");
    let until = Instant::now() + WAIT;
    loop {
        let progress = world.raw_progress(receipt["retained"]["id"].as_str().unwrap());
        if !progress["failure"].is_null() {
            break;
        }
        assert!(Instant::now() < until);
        thread::sleep(Duration::from_millis(5));
    }
    let after = model_call(&world, &fresh["next"], Some(1600));
    assert_eq!(
        after["error"]["reason"],
        "current_selected_basis_unavailable"
    );
    assert!(after["outcome"] == "error", "{after}");
}

#[test]
fn model_mixed_session_refuses_unimplemented_included_tool_blocks_before_content() {
    let world = World::new();
    let input = world.input("mixed.jsonl", MORROW);
    let receipt = world.import(&input, "01900000-0000-7000-8000-000000000805");
    world.read(
        &reference(),
        receipt["retained"]["id"].as_str().unwrap(),
        12,
    );
    let page = model_call(&world, &json!({"ref":reference()}), Some(4000));
    assert_eq!(page["error"]["reason"], "unsupported_model_block_kind");
    assert!(page.get("shown").is_none());
    assert!(page.get("next").is_none());
}

#[test]
fn model_dialogue_keeps_the_unchanged_morrow_selection_and_truthful_omissions() {
    let world = World::new();
    let input = world.input("dialogue-morrow.jsonl", MORROW);
    let receipt = world.import(&input, "01900000-0000-7000-8000-000000000811");
    world.read(
        &reference(),
        receipt["retained"]["id"].as_str().unwrap(),
        12,
    );
    let rows: Vec<Value> = MORROW
        .split(|b| *b == b'\n')
        .filter(|r| !r.is_empty())
        .map(|r| serde_json::from_slice(r).unwrap())
        .filter(|r: &Value| matches!(r["type"].as_str(), Some("user" | "assistant")))
        .collect();
    let wanted = [1, 6, 7, 10, 11, 12];
    let mut call = json!({"ref":reference(),"view":"dialogue"});
    let mut seen = 0;
    loop {
        let page = model_call(&world, &call, Some(2000));
        assert_ne!(page["outcome"], "error", "{page}");
        assert!(serde_json::to_vec(&page).unwrap().len().div_ceil(4) <= 2000);
        assert_eq!(page["rendered_as"], "dialogue-v1");
        assert_eq!(page["omitted"]["blocks"], 13);
        assert_eq!(
            page["omitted"]["by_kind"],
            json!({"thinking":1,"tool_call":6,"tool_result":6})
        );
        assert_eq!(page["omitted"]["by_author"], json!({"model":7,"harness":6}));
        let row = &rows[wanted[seen] - 1];
        let expected = row["message"]["content"]
            .as_str()
            .unwrap_or_else(|| row["message"]["content"][0]["text"].as_str().unwrap());
        let span = &page["shown"][0];
        assert_eq!(span["text"], expected);
        let uuid = row["uuid"].as_str().unwrap();
        assert_eq!(span["message"], format!("message:{}:{uuid}", reference()));
        assert_eq!(span["block"], format!("block:{}:{uuid}:0", reference()));
        assert_eq!(
            span["written_by"],
            if row["type"] == "user" {
                "human"
            } else {
                "model"
            }
        );
        assert!(!span["sources"].as_array().unwrap().is_empty());
        seen += 1;
        if page["next"].is_null() {
            break;
        }
        call = page["next"].clone();
        assert_eq!(call["view"], "dialogue");
    }
    assert_eq!(seen, 6);
}

fn mixed_dialogue_fixture() -> Vec<u8> {
    let begin = "-----BEGIN PRIVATE KEY-----";
    let end = "-----END PRIVATE KEY-----";
    let mut rows =
        vec![json!({"type":"ai-title","aiTitle":"Mixed dialogue boundaries","sessionId":SID})];
    let contents = [
        (
            "mixed-model",
            "assistant",
            json!([
                {"type":"thinking","thinking":format!("{begin} hidden"),"signature":"synthetic"},
                {"type":"text","text":"PEM BODY MUST STAY HIDDEN"},
                {"type":"tool_use","id":"independent-close","name":"Bash","input":{"command":end}},
                {"type":"text","text":format!("STILL HIDDEN{end} visible α")},
                {"type":"text","text":"short ghp_"},
                {"type":"tool_use","id":"independent-open","name":"Write","input":{"content":begin}},
                {"type":"text","text":format!("{} open🙂", "A".repeat(36))}
            ]),
        ),
        (
            "mixed-result",
            "user",
            json!([
                {"type":"tool_result","tool_use_id":"result","content":format!("{begin} hidden"),"is_error":false},
                {"type":"text","text":format!("HIDDEN{end} visible result λ")}
            ]),
        ),
        (
            "mixed-large",
            "user",
            json!([
                {"type":"tool_result","tool_use_id":"large","content":"neutral".repeat(10000),"is_error":false},
                {"type":"text","text":"after oversized result"}
            ]),
        ),
        ("mixed-last", "user", json!("Next Message is independent.")),
    ];
    let mut parent = Value::Null;
    for (index, (id, role, content)) in contents.into_iter().enumerate() {
        let mut row = json!({"type":role,"uuid":id,"sessionId":SID,"parentUuid":parent,"isSidechain":false,"timestamp":format!("2026-10-01T08:{index:02}:00.000Z"),"message":{"role":role,"content":content}});
        if role == "assistant" {
            row["message"]["model"] = json!("synthetic-model");
            row["message"]["id"] = json!("response-mixed");
        } else if id == "mixed-last" {
            row["promptSource"] = json!("typed");
        }
        if id == "mixed-result" {
            row["toolUseResult"] = json!({"stdout":begin,"stderr":"","interrupted":false});
        }
        rows.push(row);
        parent = json!(id);
    }
    rows.into_iter()
        .flat_map(|row| (row.to_string() + "\n").into_bytes())
        .collect()
}
fn mixed_block(native: &str, slot: u32) -> String {
    let sid = polylogue_model::SessionId::from_stored(reference()).unwrap();
    format!(
        "block:{}",
        polylogue_model::BlockId::new(polylogue_model::MessageId::new(sid, native).unwrap(), slot)
            .as_str()
    )
}
#[test]
fn model_dialogue_and_exact_blocks_preserve_hidden_pem_and_independent_structured_values() {
    let world = World::new();
    let input = world.input("mixed-boundaries.jsonl", &mixed_dialogue_fixture());
    let receipt = world.import(&input, "01900000-0000-7000-8000-000000000812");
    let until = Instant::now() + WAIT;
    loop {
        let p = world.progress(receipt["retained"]["id"].as_str().unwrap());
        if p["source_rows"]["state"] != "pending" {
            break;
        }
        assert!(Instant::now() < until, "{p}");
        thread::sleep(Duration::from_millis(5));
    }
    let mut call = json!({"ref":reference(),"view":"dialogue"});
    let expected = [
        "".to_owned(),
        " visible α".to_owned(),
        "short ghp_".to_owned(),
        format!("{} open🙂", "A".repeat(36)),
        "Next Message is independent.".to_owned(),
    ];
    for text in expected {
        let page = model_call(&world, &call, Some(4000));
        assert_ne!(page["outcome"], "error", "{page}");
        assert_eq!(page["shown"][0]["text"], text);
        assert!(!page.to_string().contains("STILL HIDDEN"));
        call = page["next"].clone();
    }
    assert!(call.is_null());
    let page = model_call(
        &world,
        &json!({"ref":mixed_block("mixed-result",1)}),
        Some(4000),
    );
    assert_eq!(page["shown"][0]["text"], " visible result λ", "{page}");
    let page = model_call(
        &world,
        &json!({"ref":mixed_block("mixed-large",1)}),
        Some(4000),
    );
    assert_eq!(
        page["error"]["reason"],
        "prior_tool_output_unavailable_or_exceeds_admission"
    );
    assert!(page.get("shown").is_none());
    let page = model_call(
        &world,
        &json!({"ref":mixed_block("mixed-model",0),"view":"dialogue"}),
        Some(4000),
    );
    assert!(
        page["shown"][0]["text"]
            .as_str()
            .unwrap()
            .contains("masked"),
        "{page}"
    );
}

#[test]
fn model_disconnect_and_expired_deadline_release_the_reader_without_source_owner_cancellation() {
    let mut world = World::new();
    let body = "Neutral🙂 ".repeat(9000);
    let input = world.input("disconnect.jsonl", &model_fixture(&body, false));
    let receipt = world.import(&input, "01900000-0000-7000-8000-000000000806");
    world.read(&reference(), receipt["retained"]["id"].as_str().unwrap(), 3);
    let socket = polylogue::socket_path(&fs::canonicalize(&world.archive).unwrap(), &world.runtime);
    let mut peer = UnixStream::connect(socket).unwrap();
    writeln!(peer,"{}",json!({"command":"model-read","reference":model_ref("bounded-huge"),"budget_tokens":1000000})).unwrap();
    drop(peer);
    let before = Instant::now();
    assert_eq!(world.cli(&["status"]).status.code(), Some(0));
    assert!(before.elapsed() < Duration::from_secs(3));
    assert_ne!(
        model_call(
            &world,
            &json!({"ref":model_ref("bounded-huge")}),
            Some(1800)
        )["outcome"],
        "error"
    );
    // Reuse the real retained reader with an already expired shared deadline.
    world.kill();
    let reader = std::sync::Mutex::new(polylogue_store::ReadHandle::open(&world.archive).unwrap());
    let request =
        json!({"command":"model-read","reference":model_ref("bounded-huge"),"budget_tokens":1800});
    let expired = polylogue::model_read::execute(
        &reader,
        &request,
        Instant::now() - Duration::from_secs(1),
        &mut || false,
    );
    assert_eq!(expired["error"]["reason"], "reader_cancelled_or_deadline");
    assert_eq!(expired["error"]["kind"], "timeout");
    assert_eq!(
        serde_json::from_value::<polylogue_model::types::Failure>(expired["error"].clone())
            .unwrap()
            .outcome()
            .exit_code(),
        7
    );
    assert!(expired.get("shown").is_none());
    let fresh = polylogue::model_read::execute(
        &reader,
        &request,
        Instant::now() + Duration::from_secs(30),
        &mut || false,
    );
    assert_ne!(fresh["outcome"], "error", "{fresh}");
}

#[test]
fn model_late_native_metadata_growth_retries_same_cursor_without_rewinding() {
    let world = World::new();
    let huge = "Neutral🙂 ".repeat(9000);
    let bytes = model_fixture(&huge, false);
    let no_title = bytes
        .split(|b| *b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<Value>(line).unwrap())
        .filter(|r| r["type"] != "ai-title")
        .flat_map(|r| (r.to_string() + "\n").into_bytes())
        .collect::<Vec<_>>();
    let input = world.input("late-no-title.jsonl", &no_title);
    let receipt = world.import(&input, "01900000-0000-7000-8000-000000000807");
    world.read(&reference(), receipt["retained"]["id"].as_str().unwrap(), 3);
    let mut call = json!({"ref":model_ref("bounded-huge")});
    let start = loop {
        let page = model_call(&world, &call, Some(1800));
        assert_ne!(page["outcome"], "error", "{page}");
        let end = page["shown"][0]["display_end"].as_u64().unwrap();
        call = page["next"].clone();
        assert!(call.is_object());
        if end > 10000 {
            break end;
        }
    };
    let title = "Neutral title ".to_owned() + &"q".repeat(131073);
    let metadata = world.input(
        "late-title.jsonl",
        (json!({"type":"ai-title","aiTitle":title,"sessionId":SID}).to_string() + "\n").as_bytes(),
    );
    let receipt = world.import(&metadata, "01900000-0000-7000-8000-000000000808");
    world.read(&reference(), receipt["retained"]["id"].as_str().unwrap(), 3);
    let refusal = model_call(&world, &call, Some(1800));
    assert_eq!(
        refusal["error"]["reason"], "insufficient_budget",
        "{refusal}"
    );
    assert!(refusal.get("shown").is_none());
    assert!(serde_json::to_vec(&refusal).unwrap().len().div_ceil(4) <= 128);
    let sufficient = refusal["error"]["sufficient_budget"].as_u64().unwrap();
    assert!(sufficient > 32768);
    let retry = model_call(&world, &call, Some(sufficient));
    assert_ne!(retry["outcome"], "error", "{retry}");
    assert_eq!(retry["shown"][0]["display_start"], start);
    assert_eq!(retry["session"]["display_title"], title);
    assert!((serde_json::to_vec(&retry).unwrap().len() as u64).div_ceil(4) <= sufficient);
}
