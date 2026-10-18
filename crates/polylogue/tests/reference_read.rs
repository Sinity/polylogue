//! Actual daemon processes and normative terminal reference reads. These tests
//! qualify retained source reads, not complete canonical history/search/export.
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};
const BINARY: &str = env!("CARGO_BIN_EXE_polylogue");
const WAIT: Duration = Duration::from_secs(120);
fn assert_content(bytes: &[u8], expected: &str) {
    assert!(
        std::str::from_utf8(bytes).is_ok(),
        "terminal output is not UTF-8"
    );
    let difference = bytes
        .iter()
        .zip(expected.as_bytes())
        .position(|(a, b)| a != b);
    assert!(
        difference.is_none() && bytes.len() == expected.len(),
        "terminal content mismatch: first differing byte {difference:?}, actual bytes {}, expected {}",
        bytes.len(),
        expected.len()
    );
}
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
        let deadline = Instant::now() + Duration::from_secs(60);
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
        let out = self.cli(&[
            "import",
            path.to_str().unwrap(),
            "--format",
            "codex-rollout",
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
            "import: {out:?}; daemon {}",
            self.log()
        );
        let receipt: Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(receipt["outcome"], "ok");
        assert_eq!(receipt["request_id"], request_id);
        assert_eq!(receipt["seen_at"], 17);
        assert_eq!(receipt["source_rows"]["scope"], "imported_source_rows");
        receipt
    }
    fn progress(&self, id: &str) -> Value {
        let out = self.cli(&["progress", id]);
        assert!(out.status.success(), "progress {out:?}");
        let value: Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(value["outcome"], "ok");
        value
    }
    fn healthy(&self, id: &str) -> Value {
        let value = self.progress(id);
        assert!(value["failure"].is_null(), "{value}; daemon {}", self.log());
        assert!(value["recovery_error"].is_null(), "{value}");
        assert_eq!(value["source_rows"]["scope"], "imported_source_rows");
        value
    }
    fn wait_read(&self, reference: &str, expected: &str, id: &str) -> Value {
        let deadline = Instant::now() + WAIT;
        loop {
            let progress = self.healthy(id);
            let out = self.cli(&[reference, "read", "--format", "terminal"]);
            if out.status.code() == Some(5) {
                assert_content(&out.stdout, &expected);
                assert!(
                    String::from_utf8(out.stderr)
                        .unwrap()
                        .contains("Read is incomplete")
                );
                assert_eq!(progress["source_rows"]["scope"], "imported_source_rows");
                assert_eq!(
                    progress["parser"]["finalization_pending"],
                    progress["source_rows"]["state"] != "settled"
                );
                return progress;
            }
            assert!(
                matches!(out.status.code(), Some(2 | 4)),
                "read returned an unexpected outcome: {out:?}"
            );
            assert!(
                out.stdout.is_empty(),
                "unready reference emitted partial terminal success: {out:?}"
            );
            assert!(!out.stderr.is_empty(), "read failure has no diagnostic");
            assert!(
                Instant::now() < deadline,
                "reference never became selected: {progress}; read {out:?}; daemon {}",
                self.log()
            );
            thread::sleep(Duration::from_millis(5));
        }
    }
}
impl Drop for World {
    fn drop(&mut self) {
        self.kill();
    }
}
fn line(payload: Value) -> String {
    payload.to_string() + "\n"
}
fn input(session: &str, native: &str, text: &str) -> Vec<u8> {
    [line(json!({"type":"session_meta","payload":{"id":session,"source":"cli","cwd":"/synthetic"}})),line(json!({"type":"response_item","timestamp":"2026-10-01T00:00:00Z","payload":{"type":"message","id":native,"role":"assistant","content":[{"type":"output_text","text":text}]}}))].concat().into_bytes()
}
fn reference(session: &str, native: &str) -> String {
    let origin = polylogue_model::Origin::new("codex").unwrap();
    let session = polylogue_model::SessionId::new(&origin, session).unwrap();
    format!(
        "message:{}",
        polylogue_model::MessageId::new(session, native)
            .unwrap()
            .as_str()
    )
}
fn expected(session: &str, native: &str, text: &str) -> String {
    let id = reference(session, native);
    format!(
        "{}\nassistant · model\n\n{text}\n",
        id.strip_prefix("message:").unwrap()
    )
}
fn negatives(world: &World, reference: &str) {
    // Session JSON is now registered, but this exact opaque identity is absent.
    // A recognized absent reference is distinct from an unsupported operation.
    let absent = world.cli(&[
        "session:codex:reference-session",
        "read",
        "--format",
        "json",
    ]);
    assert_eq!(absent.status.code(), Some(4));
    assert!(absent.stdout.is_empty());
    assert!(!absent.stderr.is_empty());

    for args in [
        vec!["search", "retained phrase"],
        vec![reference, "read", "--format", "json"],
        vec!["export", reference],
    ] {
        let out = world.cli(&args);
        assert_eq!(
            out.status.code(),
            Some(2),
            "unsupported operation falsely completed: {out:?}"
        );
        assert!(out.stdout.is_empty());
        assert!(!out.stderr.is_empty());
    }
    let out = world.cli(&["message:codex:absent-session:absent-message", "read"]);
    assert_eq!(out.status.code(), Some(4), "{out:?}");
    assert!(out.stdout.is_empty());
    assert!(!out.stderr.is_empty());
}
#[test]
fn retained_unicode_reference_reads_exactly_but_coverage_and_search_stay_incomplete() {
    let world = World::new();
    let text = "ż🙂漢字é retained phrase\n".repeat(7000);
    assert!(text.len() > 65536);
    let path = world.input(
        "unicode.jsonl",
        &input("reference-session:opaque", "assistant-native:opaque", &text),
    );
    let receipt = world.import(&path, "01900000-0000-7000-8000-000000000101");
    let id = receipt["retained"]["id"].as_str().unwrap();
    fs::remove_file(&path).unwrap();
    let reference = reference("reference-session:opaque", "assistant-native:opaque");
    let expected = expected("reference-session:opaque", "assistant-native:opaque", &text);
    let progress = world.wait_read(&reference, &expected, id);
    assert_eq!(
        progress["parser"]["parsed_upto"],
        receipt["retained"]["length"]
    );
    assert_eq!(
        progress["parser"]["finalization_pending"],
        progress["source_rows"]["state"] != "settled"
    );
    let default = world.cli(&[&reference, "read"]);
    assert_eq!(default.status.code(), Some(5));
    assert_content(&default.stdout, &expected);
    negatives(&world, &reference);
}
#[test]
fn selected_read_and_receipt_survive_peer_work_kill_restart_and_provider_removal() {
    let mut world = World::new();
    let text = "first selected retained phrase 🙂";
    let first = world.input(
        "first.jsonl",
        &input("stable-session", "stable-message", text),
    );
    let request = "01900000-0000-7000-8000-000000000102";
    let receipt = world.import(&first, request);
    let id = receipt["retained"]["id"].as_str().unwrap().to_owned();
    fs::remove_file(&first).unwrap();
    let reference = reference("stable-session", "stable-message");
    let expected = expected("stable-session", "stable-message", text);
    world.wait_read(&reference, &expected, &id);
    let mut large = line(
        json!({"type":"session_meta","payload":{"id":"peer-many-messages","source":"cli","cwd":"/synthetic"}}),
    );
    let body = "peer content 漢🙂".repeat(500);
    for n in 0..700 {
        large.push_str(&line(json!({"type":"response_item","payload":{"type":"message","id":format!("prompt-{n}"),"role":"user","content":[{"type":"input_text","text":format!("synthetic prompt {n}")}]}})));
        large.push_str(&line(json!({"type":"response_item","payload":{"type":"message","id":format!("answer-{n}"),"role":"assistant","content":[{"type":"output_text","text":body}]}})));
    }
    assert!(large.len() > 4 * 1024 * 1024);
    assert!(large.lines().all(|line| line.len() < 65536));
    let second = world.input("peer.jsonl", large.as_bytes());
    let peer = world.import(&second, "01900000-0000-7000-8000-000000000103");
    let peer_id = peer["retained"]["id"].as_str().unwrap().to_owned();
    fs::remove_file(&second).unwrap();
    let deadline = Instant::now() + WAIT;
    let unfinished = loop {
        let p = world.healthy(&peer_id);
        if p["active"] == true
            && p["parser"]["parsed_upto"]
                .as_u64()
                .is_some_and(|n| n > 0 && n < large.len() as u64)
        {
            break p;
        }
        assert!(
            Instant::now() < deadline,
            "no observed unfinished retained peer work: {p}"
        );
        thread::sleep(Duration::from_millis(5));
    };
    assert_eq!(unfinished["source_rows"]["scope"], "imported_source_rows");
    let out = world.cli(&[&reference, "read", "--format", "terminal"]);
    assert_eq!(out.status.code(), Some(5));
    assert_content(&out.stdout, &expected);
    let observed = world.healthy(&peer_id);
    assert_eq!(
        observed["active"], true,
        "kill boundary must have actual unfinished work: {observed}"
    );
    assert_eq!(observed["parser"]["finalization_pending"], true);
    world.restart();
    world.wait_read(&reference, &expected, &id);
    let replay = world.import(&first, request);
    assert_eq!(replay["retained"], receipt["retained"]);
    assert_eq!(replay["request_id"], receipt["request_id"]);
    assert_eq!(replay["seen_at"], receipt["seen_at"]);
    let resumed = world.healthy(&peer_id);
    assert_eq!(resumed["length"], large.len() as u64);
    assert_eq!(resumed["source_rows"]["scope"], "imported_source_rows");
    negatives(&world, &reference);
}
#[test]
fn missing_call_id_message_reference_is_an_explicit_failure() {
    let world = World::new();
    let bytes=[line(json!({"type":"session_meta","payload":{"id":"unsupported-session","source":"cli","cwd":"/synthetic"}})),line(json!({"type":"response_item","payload":{"type":"function_call","id":"unsupported-native","name":"synthetic-tool","arguments":"{}"}}))].concat();
    let path = world.input("unsupported.jsonl", bytes.as_bytes());
    let receipt = world.import(&path, "01900000-0000-7000-8000-000000000104");
    let id = receipt["retained"]["id"].as_str().unwrap();
    fs::remove_file(path).unwrap();
    let deadline = Instant::now() + WAIT;
    loop {
        let progress = world.progress(id);
        if progress["failure"].is_string() {
            assert_eq!(progress["source_rows"]["scope"], "imported_source_rows");
            assert_eq!(
                progress["parser"]["finalization_pending"],
                progress["source_rows"]["state"] != "settled"
            );
            break;
        }
        assert!(
            Instant::now() < deadline,
            "unsupported mapping was not accounted: {progress}"
        );
        thread::sleep(Duration::from_millis(5));
    }
    let out = world.cli(&[
        &reference("unsupported-session", "unsupported-native"),
        "read",
    ]);
    assert_eq!(out.status.code(), Some(2), "{out:?}");
    assert!(out.stdout.is_empty());
    assert!(!out.stderr.is_empty());
}

#[test]
fn daemon_masks_credentials_before_both_cli_and_socket_text_frames() {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;
    let token = format!("ghp_{}", "A".repeat(36));
    for (session, prefix, secret, lead, tail, matched) in [
        (
            "mask-within",
            "before ".to_owned(),
            token.clone(),
            "",
            "",
            true,
        ),
        // The catalog requires a non-word boundary; keep the token at byte4090.
        (
            "mask-cross",
            "x".repeat(4089) + " ",
            token.clone(),
            "",
            "",
            true,
        ),
        (
            "mask-word-boundary",
            "identifier".to_owned(),
            token.clone(),
            "",
            "",
            false,
        ),
        (
            "mask-dense",
            "before ".to_owned(),
            "-u=a:b ".repeat(585),
            "",
            "",
            true,
        ),
        (
            "mask-overlap",
            "before ".to_owned(),
            format!("https://u:{token}!private@host"),
            "https://",
            "@host",
            true,
        ),
    ] {
        let world = World::new();
        let text = format!("{prefix}{secret} after");
        let original = input(session, "masked", &text);
        let original_hash = polylogue_model::BlobHash::of(&original);
        let path = world.work.path().join("provider.jsonl");
        fs::write(&path, &original).unwrap();
        let receipt = world.import(&path, "019d59d0-0000-7000-8000-000000000003");
        let id = receipt["retained"]["id"].as_str().unwrap();
        let reference = reference(session, "masked");
        let deadline = Instant::now() + WAIT;
        let output = loop {
            world.healthy(id);
            let output = world.cli(&[&reference, "read"]);
            if output.status.code() == Some(5) {
                break output;
            }
            assert!(matches!(output.status.code(), Some(2 | 4)), "{output:?}");
            assert!(Instant::now() < deadline, "masked reference never ready");
            thread::sleep(Duration::from_millis(5));
        };
        let endpoint = polylogue::socket_path(&world.archive, &world.runtime);
        let mut socket = UnixStream::connect(endpoint).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(15)))
            .unwrap();
        writeln!(
            socket,
            "{}",
            json!({"command":"read","reference":reference,"format":"terminal"})
        )
        .unwrap();
        let mut stream = BufReader::new(socket);
        let mut body = String::new();
        loop {
            let mut line = String::new();
            stream.by_ref().take(65537).read_line(&mut line).unwrap();
            assert!(line.ends_with('\n') && line.len() <= 65536);
            let frame: Value = serde_json::from_str(&line).unwrap();
            if frame["type"] == "text" {
                body.push_str(frame["data"].as_str().unwrap());
            }
            if frame["type"] == "end" {
                assert_eq!(frame["outcome"], "degraded");
                assert_eq!(frame["canonical_complete"], false);
                break;
            }
        }
        if session == "mask-dense" {
            let expected_body = format!(
                "before {} after",
                "-u=[credential: CLI credentials, masked] ".repeat(585)
            );
            assert!(expected_body.len() > 16384);
            assert_eq!(body, expected_body);
        } else if matched {
            assert!(
                !body.contains(&token),
                "raw catalog credential reached daemon response: {session}"
            );
            assert!(
                !body.contains("!private"),
                "overlapping outer credential leaked"
            );
            let marker = body
                .strip_prefix(&(prefix.clone() + lead))
                .unwrap()
                .strip_suffix(" after")
                .unwrap()
                .strip_suffix(tail)
                .unwrap();
            assert!(
                marker.starts_with("[credential:") && marker.ends_with(", masked]"),
                "{marker}"
            );
        } else {
            // A token-shaped substring inside a word is outside the v1 family;
            // this is an intentional catalog boundary, independent of framing.
            assert_eq!(body, text);
        }
        assert_eq!(output.stdout, expected(session, "masked", &body).as_bytes());
        assert_eq!(fs::read(&path).unwrap(), original);
        assert_eq!(
            polylogue_model::BlobHash::of(&fs::read(&path).unwrap()),
            original_hash
        );
    }
}
