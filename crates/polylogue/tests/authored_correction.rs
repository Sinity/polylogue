//! Actual CLI adaptation over the ordinary daemon and archive, with synthetic targets.
use serde_json::{Value, json};
use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
    thread,
    time::{Duration, Instant},
};
const BINARY: &str = env!("CARGO_BIN_EXE_polylogue");
struct World {
    archive: PathBuf,
    runtime: tempfile::TempDir,
    dir: tempfile::TempDir,
    running: Option<polylogue::Running>,
}
impl World {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("archive");
        fs::write(
            dir.path().join("machine-id"),
            "00112233445566778899aabbccddeeff\n",
        )
        .unwrap();
        let runtime = tempfile::tempdir_in("/tmp").unwrap();
        let running = Some(polylogue::Running::start(&archive, runtime.path()).unwrap());
        Self {
            archive,
            runtime,
            dir,
            running,
        }
    }
    fn raw(&self, args: &[&str]) -> Output {
        Command::new(BINARY)
            .arg("--archive")
            .arg(&self.archive)
            .args(args)
            .env_clear()
            .env("XDG_RUNTIME_DIR", self.runtime.path())
            .env("TMPDIR", self.dir.path())
            .env("CODEX_THREAD_ID", "synthetic-client")
            .env(
                "POLYLOGUE_MACHINE_ID_FILE",
                self.dir.path().join("machine-id"),
            )
            .output()
            .unwrap()
    }
    fn cli(&self, args: &[&str], expected: i32) -> Value {
        let o = self.raw(args);
        assert_eq!(o.status.code(), Some(expected), "{args:?}: {o:?}");
        serde_json::from_slice(&o.stdout).unwrap_or_else(|e| panic!("{args:?}: {e}: {o:?}"))
    }
    fn read(&self, reference: &str) -> Value {
        self.cli(&[reference, "read", "--budget", "12000"], 0)
    }
    fn restart_without_index(&mut self) {
        self.running.take().unwrap().stop();
        for name in ["index.db", "index.db-wal", "index.db-shm"] {
            let p = self.archive.join(name);
            if p.exists() {
                fs::remove_file(p).unwrap();
            }
        }
        self.running = Some(polylogue::Running::start(&self.archive, self.runtime.path()).unwrap());
    }
}
#[test]
fn authored_cli_retries_inherits_basis_and_reconstructs_backlinks_without_provider() {
    let mut w = World::new();
    let target = "block:claude-code:authored-cli:answer:0";
    let request = "01900000-0000-7000-8000-000000000301";
    let basis = w.dir.path().join("basis.json");
    fs::write(&basis,json!([{"ref":target,"seen_hash":"af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"}]).to_string()).unwrap();
    let args = [
        target,
        "note",
        "Correction: compare the retained answer with the prepared empty basis.",
        "--kind",
        "correction",
        "--target-basis",
        basis.to_str().unwrap(),
        "--request-id",
        request,
    ];
    let first = w.cli(&args, 0);
    assert_eq!(first["revision"]["author"], "codex:synthetic-client");
    assert_eq!(
        first["revision"]["machine"],
        "00112233445566778899aabbccddeeff"
    );
    assert_eq!(
        w.cli(&args, 0),
        first,
        "a new CLI process retry returns the original effect"
    );
    let mut changed = args;
    changed[2] = "Different intent";
    assert_eq!(w.cli(&changed, 4)["error"]["kind"], "conflict");
    let reference = first["ref"].as_str().unwrap();
    assert!(w.read(reference)["family"]["shown"][0]["stale"].is_null());
    let path = w.dir.path().join("provider.jsonl");
    let context = w.dir.path().join("context.json");
    fs::write(&path,format!("{}\n{}\n",json!({"type":"ai-title","aiTitle":"Synthetic authored CLI","sessionId":"authored-cli"}),json!({"type":"user","uuid":"answer","sessionId":"authored-cli","promptSource":"typed","message":{"role":"user","content":"abc"}}))).unwrap();
    fs::write(
        &context,
        json!({"version":1,"context":["claude-main","authored-cli"]}).to_string(),
    )
    .unwrap();
    let imported = w.cli(
        &[
            "import",
            path.to_str().unwrap(),
            "--format",
            "claude-code-jsonl",
            "--context",
            context.to_str().unwrap(),
            "--request-id",
            "01900000-0000-7000-8000-000000000302",
            "--seen-at",
            "17",
            "--machine",
            "00112233445566778899aabbccddeeff",
            "--host",
            "synthetic-host",
            "--wait",
            "--wait-timeout",
            "30",
        ],
        0,
    );
    let before = w.read(reference);
    assert_eq!(before["family"]["shown"][0]["stale"], true);
    assert_eq!(before["family"]["shown"][0]["revision"], first["revision"]);
    let target_read = w.cli(&[target, "read", "--budget", "12000"], 5);
    assert_eq!(
        target_read["authored_basis"]["seen_hash"],
        "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"
    );
    assert_eq!(
        target_read["authored_notes"]["families"][0]["root"],
        request
    );
    let revised = w.cli(
        &[
            reference,
            "revise",
            "Correction: the prepared basis remains the original empty text.",
            "--request-id",
            "01900000-0000-7000-8000-000000000303",
        ],
        0,
    );
    assert_eq!(revised["revision"]["targets"], first["revision"]["targets"]);
    let stale = w.cli(
        &[
            reference,
            "revise",
            "Stale edit",
            "--request-id",
            "01900000-0000-7000-8000-000000000304",
        ],
        4,
    );
    assert_eq!(stale["error"]["kind"], "conflict");
    let revision_ref = revised["ref"].as_str().unwrap();
    w.cli(
        &[
            revision_ref,
            "withdraw",
            "--reason",
            "Restore the earlier wording",
            "--request-id",
            "01900000-0000-7000-8000-000000000305",
        ],
        0,
    );
    let root = format!("note:{request}");
    let restored = w.read(&root);
    assert_eq!(restored["family"]["current"], json!([reference]));
    let unsupported = w.raw(&[
        target,
        "note",
        "Unsupported return",
        "--intent",
        "authored_return",
    ]);
    assert_eq!(unsupported.status.code(), Some(2));
    assert!(
        String::from_utf8(unsupported.stderr)
            .unwrap()
            .contains("unsupported authored option --intent")
    );
    fs::remove_file(path).unwrap();
    w.restart_without_index();
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let p = w.cli(
            &["progress", imported["retained"]["id"].as_str().unwrap()],
            0,
        );
        if p["source_rows"]["state"] == "settled" {
            break;
        }
        assert!(Instant::now() < deadline, "{p}");
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(w.read(&root)["family"], restored["family"]);
    assert_eq!(
        w.cli(&[target, "read", "--budget", "12000"], 5)["shown"],
        target_read["shown"]
    );
}

#[test]
fn authored_cli_lost_response_reports_unknown_commit_and_same_action_identity() {
    use std::{
        io::{BufRead, BufReader},
        os::unix::net::UnixListener,
    };
    let mut w = World::new();
    let socket = w.running.as_ref().unwrap().socket().to_path_buf();
    w.running.take().unwrap().stop();
    let listener = UnixListener::bind(socket).unwrap();
    let peer = thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut line = String::new();
        BufReader::new(stream).read_line(&mut line).unwrap();
        serde_json::from_str::<Value>(&line).unwrap()
    });
    let request = "01900000-0000-7000-8000-000000000401";
    let reply = w.cli(
        &[
            "message:future",
            "note",
            "Final attributed text",
            "--request-id",
            request,
        ],
        4,
    );
    assert_eq!(reply["request_id"], request);
    assert_eq!(reply["commit_state"], "unknown");
    assert!(
        reply["error"]["hint"]
            .as_str()
            .unwrap()
            .contains("same request_id")
    );
    let submitted = peer.join().unwrap();
    assert_eq!(submitted["request_id"], request);
    assert_eq!(submitted["command"], "note");
}

#[test]
fn authored_cli_rejects_a_success_for_another_action_identity() {
    use std::{
        io::{BufRead, BufReader, Write},
        os::unix::net::UnixListener,
    };
    let mut w = World::new();
    let socket = w.running.as_ref().unwrap().socket().to_path_buf();
    w.running.take().unwrap().stop();
    let listener = UnixListener::bind(socket).unwrap();
    let peer = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut line = String::new();
        BufReader::new(stream.try_clone().unwrap())
            .read_line(&mut line)
            .unwrap();
        let id = "01900000-0000-7000-8000-000000000999";
        writeln!(stream,"{}",json!({"outcome":"ok","request_id":id,"note":id,"root":id,"ref":format!("note:{id}@{id}"),"commit_state":"committed","revision":{"id":id,"root":id}})).unwrap();
    });
    let id = "01900000-0000-7000-8000-000000000402";
    let reply = w.cli(
        &[
            "message:future",
            "note",
            "Final attributed text",
            "--request-id",
            id,
        ],
        4,
    );
    assert_eq!(reply["request_id"], id);
    assert_eq!(reply["commit_state"], "unknown");
    assert!(reply["error"]["message"].as_str().unwrap().contains("bind"));
    peer.join().unwrap();
}
