//! Public daemon/CLI export, using retained synthetic Morrow bytes.
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};
const BINARY: &str = env!("CARGO_BIN_EXE_polylogue");
const SID: &str = "7f3a2c10-0001-4000-8000-000000000001";
const MORROW: &[u8] = include_bytes!("../../sources/tests/fixtures/morrow-main.jsonl");
struct World {
    child: Child,
    work: tempfile::TempDir,
    runtime: tempfile::TempDir,
    archive: PathBuf,
}
impl World {
    fn new() -> Self {
        let work = tempfile::tempdir().unwrap();
        // Only socket dentries use the short tmpfs path.
        let runtime = tempfile::tempdir_in("/tmp").unwrap();
        let archive = work.path().join("archive");
        let log = File::create(work.path().join("daemon.log")).unwrap();
        let child = Command::new(BINARY)
            .args(["--archive", archive.to_str().unwrap(), "daemon"])
            .env("XDG_RUNTIME_DIR", runtime.path())
            .stdout(Stdio::null())
            .stderr(log)
            .spawn()
            .unwrap();
        let mut world = Self {
            child,
            work,
            runtime,
            archive,
        };
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            let out = world.cli(&["status"]);
            if out.status.success() {
                break;
            }
            assert!(world.child.try_wait().unwrap().is_none(), "{out:?}");
            assert!(Instant::now() < deadline, "{out:?}");
            thread::sleep(Duration::from_millis(10));
        }
        world
    }
    fn cli(&self, args: &[&str]) -> Output {
        Command::new(BINARY)
            .args(["--archive", self.archive.to_str().unwrap()])
            .args(args)
            .env("XDG_RUNTIME_DIR", self.runtime.path())
            .output()
            .unwrap()
    }
    fn import(&self, native: &str, bytes: &[u8], request: &str) {
        let source = self.work.path().join("provider.jsonl");
        let context = self.work.path().join("context.json");
        fs::write(&source, bytes).unwrap();
        fs::write(
            &context,
            json!({"version":1,"context":["claude-main",native]}).to_string(),
        )
        .unwrap();
        let out = self.cli(&[
            "import",
            source.to_str().unwrap(),
            "--format",
            "claude-code-jsonl",
            "--context",
            context.to_str().unwrap(),
            "--machine",
            "00112233445566778899aabbccddeeff",
            "--host",
            "synthetic-export",
            "--request-id",
            request,
            "--seen-at",
            "17",
            "--wait",
            "--wait-timeout",
            "120",
        ]);
        assert!(out.status.success(), "{out:?}");
        fs::remove_file(source).unwrap();
    }
    fn export(&self, reference: &str, to: &Path) -> Output {
        self.cli(&[
            reference,
            "export",
            "--format",
            "json",
            "--to",
            to.to_str().unwrap(),
        ])
    }
}
impl Drop for World {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
#[test]
fn morrow_read_and_export_are_identical_and_atomic_replacement_removes_previous_tree() {
    let world = World::new();
    world.import(SID, MORROW, "01900000-0000-7000-8000-000000009501");
    let reference = format!("claude-code:{SID}");
    let out = world.cli(&[&reference, "read", "--format", "json"]);
    assert!(out.status.success(), "{out:?}");
    let document: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(document["messages"].as_array().unwrap().len(), 12);
    // Independent facts from the source fixture, rather than read/export equality alone.
    let records: Vec<Value> = MORROW
        .split(|b| *b == b'\n')
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_slice(l).unwrap())
        .filter(|v: &Value| v.get("uuid").is_some())
        .collect();
    for (message, record) in document["messages"].as_array().unwrap().iter().zip(records) {
        assert_eq!(
            message["id"],
            format!("{reference}:{}", record["uuid"].as_str().unwrap())
        );
        if let Some(text) = record["message"]["content"].as_str() {
            assert_eq!(message["blocks"][0]["text"], text);
        }
    }
    let destination = world.work.path().join("export");
    let out = world.export(&reference, &destination);
    assert!(out.status.success(), "{out:?}");
    assert!(out.stdout.is_empty());
    assert_eq!(
        read(&destination.join(format!("claude-code/{SID}.json"))),
        document
    );
    let manifest = read(&destination.join("manifest.json"));
    assert_eq!(manifest["result"], document["result"]);
    assert_eq!(manifest["query"], reference);
    assert_eq!(manifest["up_to_date"], true);
    assert_eq!(manifest["sessions"], 1);
    assert_eq!(
        manifest["view"],
        format!(
            "{}:{}",
            document["result"]["completion"]["serving_view"]["index_generation"]
                .as_str()
                .unwrap(),
            document["result"]["completion"]["serving_view"]["index_counter"]
        )
    );
    assert!(!manifest.as_object().unwrap().contains_key("recovery"));
    assert!(manifest["ran_at"].as_str().unwrap().ends_with('Z'));
    fs::write(destination.join("previous-only"), b"old complete tree").unwrap();
    let out = world.export(&reference, &destination);
    assert!(out.status.success(), "{out:?}");
    assert!(!destination.join("previous-only").exists());
    assert!(!fs::read_dir(world.work.path()).unwrap().any(|e| {
        e.unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".polylogue-export-")
    }));
    let before = fs::read(destination.join("manifest.json")).unwrap();
    let out = world.export("claude-code:unknown", &destination);
    assert_eq!(out.status.code(), Some(4));
    assert!(out.stdout.is_empty());
    assert_eq!(fs::read(destination.join("manifest.json")).unwrap(), before);
    let missing = world.work.path().join("missing-export");
    let out = world.export("claude-code:unknown", &missing);
    assert_eq!(out.status.code(), Some(4));
    assert!(!missing.exists());
    for extra in ["--with-blobs", "--no-inherited", "--no-notes", "--profile"] {
        let out = world.cli(&[
            &reference,
            "export",
            "--format",
            "json",
            "--to",
            destination.to_str().unwrap(),
            extra,
        ]);
        assert_eq!(out.status.code(), Some(2));
        assert!(out.stdout.is_empty());
        assert_eq!(fs::read(destination.join("manifest.json")).unwrap(), before);
    }
    let out = world.cli(&[
        &reference,
        "export",
        "--format",
        "jsonl",
        "--to",
        destination.to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(2));
    let ordinary = world.work.path().join("ordinary");
    fs::create_dir(&ordinary).unwrap();
    fs::write(ordinary.join("keep"), b"untouched").unwrap();
    assert!(!world.export(&reference, &ordinary).status.success());
    assert_eq!(fs::read(ordinary.join("keep")).unwrap(), b"untouched");
    let symlink = world.work.path().join("symlink");
    std::os::unix::fs::symlink(&destination, &symlink).unwrap();
    assert!(!world.export(&reference, &symlink).status.success());
    assert!(
        fs::symlink_metadata(symlink)
            .unwrap()
            .file_type()
            .is_symlink()
    );
}
#[test]
fn opaque_native_identity_is_not_split_or_interpreted_as_a_path() {
    let world = World::new();
    let native = "opaque:..λ%";
    let bytes = String::from_utf8(MORROW.to_vec())
        .unwrap()
        .replace(SID, native);
    world.import(
        native,
        bytes.as_bytes(),
        "01900000-0000-7000-8000-000000009502",
    );
    let destination = world.work.path().join("opaque");
    let reference = polylogue_model::SessionId::new(
        &polylogue_model::Origin::new("claude-code").unwrap(),
        native,
    )
    .unwrap()
    .as_str()
    .to_owned();
    let out = world.export(&reference, &destination);
    assert!(out.status.success(), "{out:?}");
    let document = read(&destination.join("claude-code/opaque%3A..%CE%BB%25.json"));
    assert_eq!(document["session"]["id"], reference);
    assert_eq!(document["session"]["native_id"], native);
}

#[test]
fn sigint_terminates_partial_frame_export_and_disconnects_without_publication() {
    use std::{
        io::{BufRead, BufReader, Read, Write},
        os::unix::{net::UnixListener, process::ExitStatusExt},
        sync::mpsc,
    };
    struct Owned(Child);
    impl Drop for Owned {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let work = tempfile::tempdir().unwrap();
    let runtime = tempfile::tempdir_in("/tmp").unwrap();
    let archive = work.path().join("archive");
    fs::create_dir(&archive).unwrap();
    let destination = work.path().join("export");
    fs::create_dir(&destination).unwrap();
    let manifest = json!({"format":"polylogue-export-manifest","version":1}).to_string();
    fs::write(destination.join("manifest.json"), &manifest).unwrap();
    fs::write(destination.join("prior"), b"previous complete export").unwrap();
    let socket = polylogue::socket_path(&fs::canonicalize(&archive).unwrap(), runtime.path());
    fs::create_dir_all(socket.parent().unwrap()).unwrap();
    let listener = UnixListener::bind(socket).unwrap();
    listener.set_nonblocking(true).unwrap();
    let (ready, started) = mpsc::channel();
    let producer = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(Instant::now() < deadline, "CLI never connected");
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("{error}"),
            }
        };
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut request = String::new();
        BufReader::new(stream.try_clone().unwrap())
            .read_line(&mut request)
            .unwrap();
        let request: Value = serde_json::from_str(&request).unwrap();
        assert_eq!(
            request,
            json!({"command":"session-read","reference":"test:cancel","format":"json"})
        );
        writeln!(stream,"{}",json!({"type":"session","id":"test:cancel","origin":"test","native_id":"cancel","format":"polylogue-session","version":1})).unwrap();
        writeln!(
            stream,
            "{}",
            json!({"type":"text","data":"{\"format\":\"polylogue-session\","})
        )
        .unwrap();
        // No newline or completion follows this frame. The old handler merely
        // set a flag, leaving the client stuck inside read_line until TIME.
        stream
            .write_all(b"{\"type\":\"text\",\"data\":\"unfinished")
            .unwrap();
        stream.flush().unwrap();
        ready.send(()).unwrap();
        let mut byte = [0];
        stream.read(&mut byte)
    });
    let mut client = Owned(
        Command::new(BINARY)
            .args([
                "--archive",
                archive.to_str().unwrap(),
                "test:cancel",
                "export",
                "--format",
                "json",
                "--to",
                destination.to_str().unwrap(),
            ])
            .env("XDG_RUNTIME_DIR", runtime.path())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    started.recv_timeout(Duration::from_secs(5)).unwrap();
    // The signal targets this exact owned child; the positional argument avoids
    // interpreting its numeric PID as shell text.
    assert!(
        Command::new("sh")
            .args([
                "-c",
                "kill -INT \"$1\"",
                "signal-export-client",
                &client.0.id().to_string()
            ])
            .status()
            .unwrap()
            .success()
    );
    let deadline = Instant::now() + Duration::from_secs(2);
    let status = loop {
        if let Some(status) = client.0.try_wait().unwrap() {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "export client ignored SIGINT while waiting for a partial frame"
        );
        thread::sleep(Duration::from_millis(5));
    };
    assert_eq!(status.signal(), Some(2));
    let disconnected = producer.join().unwrap();
    // A full close can reset the stream if the client had unread bytes. Both
    // observations release the producer; a timeout or open stream fails.
    assert!(
        matches!(disconnected, Ok(0))
            || disconnected.is_err_and(|error| error.kind() == std::io::ErrorKind::ConnectionReset),
        "producer must see full client disconnection"
    );
    assert_eq!(
        fs::read_to_string(destination.join("manifest.json")).unwrap(),
        manifest
    );
    assert_eq!(
        fs::read(destination.join("prior")).unwrap(),
        b"previous complete export"
    );
    let stages = fs::read_dir(work.path())
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with(".polylogue-export-")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        stages.len(),
        1,
        "abrupt SIGINT retains only private unfinished staging"
    );
    assert!(stages[0].path().join("session.json").is_file());
    assert!(!stages[0].path().join("manifest.json").exists());
}
