//! `export` through the CLI and a real daemon, using the synthetic Morrow fixture.
use serde_json::{Value, json};
use std::{
    fs,
    path::Path,
    process::{Command, Output, Stdio},
    thread,
};
const SID: &str = "7f3a2c10-0001-4000-8000-000000000001";
const MORROW: &[u8] = include_bytes!("../../sources/tests/fixtures/morrow-main.jsonl");
mod support;
use support::{BINARY, OwnedChild, World};
impl World {
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
fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
#[test]
fn export_matches_read_and_replaces_the_previous_export() {
    let world = World::new();
    world.import(SID, MORROW, "01900000-0000-7000-8000-000000009501");
    let reference = format!("claude-code:{SID}");
    let out = world.cli(&[&reference, "read", "--format", "json"]);
    assert!(out.status.success(), "{out:?}");
    let document: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(document["messages"].as_array().unwrap().len(), 12);
    // Check against the fixture itself, not only read/export equality.
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
fn native_id_with_colons_and_dots_is_one_file_name() {
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
fn ctrl_c_during_a_partial_frame_exits_without_writing_the_export() {
    use std::{
        io::{BufRead, BufReader, Read, Write},
        os::unix::{net::UnixListener, process::ExitStatusExt},
        sync::mpsc,
    };
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
    let (ready, started) = mpsc::channel();
    let producer = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
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
        // This frame never ends, so the client blocks in read_line until the
        // signal arrives.
        stream
            .write_all(b"{\"type\":\"text\",\"data\":\"unfinished")
            .unwrap();
        stream.flush().unwrap();
        ready.send(()).unwrap();
        let mut byte = [0];
        stream.read(&mut byte)
    });
    let mut client = OwnedChild(
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
    started.recv().unwrap();

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
    // The producer never finishes the frame, so the client exits only on the
    // signal.
    let status = client.0.wait().unwrap();
    assert_eq!(status.signal(), Some(2));
    let disconnected = producer.join().unwrap();
    // A close with unread bytes can show up as a reset instead of EOF.
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
        "Ctrl-C leaves only the hidden staging directory"
    );
    assert!(stages[0].path().join("session.json").is_file());
    assert!(!stages[0].path().join("manifest.json").exists());
}
