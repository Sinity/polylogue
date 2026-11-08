use super::*;
use crate::import::tests::engine;
use polylogue_testkit::ClaudeHome;
#[test]
fn recorded_directory_captures_original_assets_and_reports_missing_and_unresolved_references() {
    let w = ClaudeHome::new();
    let cwd = w.dir.path().join("recorded");
    std::fs::create_dir_all(&cwd).unwrap();
    let image = cwd.join("image.png");
    std::fs::write(&image, b"original image").unwrap();
    let dir = w.home.join(".codex/sessions");
    std::fs::create_dir_all(&dir).unwrap();
    let row = serde_json::json!({"type":"event_msg","payload":{"type":"user_message","message":"attachment","local_images":["image.png","missing.png"]}});
    let header =
        serde_json::json!({"type":"session_meta","payload":{"id":"local-native","cwd":cwd}});
    std::fs::write(
        dir.join("rollout-images.jsonl"),
        format!("{header}\n{row}\n"),
    )
    .unwrap();
    let unresolved = serde_json::json!({"type":"event_msg","payload":{"local_images":["relative-without-cwd.png"]}});
    std::fs::write(
        dir.join("rollout-unknown.jsonl"),
        format!("{{\"type\":\"session_meta\",\"payload\":{{\"id\":\"no-cwd\"}}}}\n{unresolved}\n"),
    )
    .unwrap();
    let (mut e, _lock) = engine(&w, 64);
    let p = e.run_pass().unwrap();
    assert!(p.errors.is_empty(), "{p:?}");
    let id = blake3::hash(b"original image").to_hex().to_string();
    let r = e.reads.reader().unwrap();
    let progress = r.input_progress(&id).unwrap().unwrap();
    assert_eq!(
        r.stored_range(&id, progress.chain, 0, progress.length, &e.blobs)
            .unwrap(),
        b"original image"
    );
    let rows=r.query("SELECT DISTINCT o.input FROM a.observation o JOIN a.input i ON i.id=o.input WHERE i.format='tool-context' AND o.path LIKE 'codex-local-references/%'",&[]).unwrap();
    let mut statuses = Vec::new();
    for row in rows {
        let id = row["input"].as_str().unwrap();
        let p = r.input_progress(id).unwrap().unwrap();
        let bytes = r.stored_range(id, p.chain, 0, p.length, &e.blobs).unwrap();
        for line in bytes.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
            let value: Json = serde_json::from_slice(line).unwrap();
            statuses.push(value["status"].as_str().unwrap().to_owned());
        }
    }
    statuses.sort();
    assert_eq!(statuses, ["collected", "missing", "unresolved"]);
}

#[test]
fn every_reference_after_four_thousand_is_still_reported() {
    let w = ClaudeHome::new();
    let dir = w.home.join(".codex/sessions");
    std::fs::create_dir_all(&dir).unwrap();
    let images = (0..4097)
        .map(|i| format!("missing-{i}.png"))
        .collect::<Vec<_>>();
    let row = serde_json::json!({"type":"event_msg","payload":{"type":"user_message","message":"attachment list","local_images":images}});
    std::fs::write(
        dir.join("rollout-many.jsonl"),
        format!("{{\"type\":\"session_meta\",\"payload\":{{\"id\":\"many-native\"}}}}\n{row}\n"),
    )
    .unwrap();
    let (mut e, _lock) = engine(&w, 4096);
    let p = e.run_pass().unwrap();
    assert!(p.errors.is_empty(), "{p:?}");
    let reader = e.reads.reader().unwrap();
    let rows=reader.query("SELECT o.input FROM a.observation o JOIN a.input i ON i.id=o.input WHERE i.format='tool-context' AND o.path='codex-local-references/codex:rollout:many-native.jsonl'",&[]).unwrap();
    assert_eq!(rows.len(), 1);
    let input = rows[0]["input"].as_str().unwrap();
    let progress = reader.input_progress(input).unwrap().unwrap();
    let bytes = reader
        .stored_range(input, progress.chain, 0, progress.length, &e.blobs)
        .unwrap();
    assert_eq!(bytes.iter().filter(|b| **b == b'\n').count(), 4097);
}

#[test]
fn ordinary_reopen_observes_local_references_after_raw_commit_before_parse() {
    for remove_source in [false, true] {
        let w = ClaudeHome::new();
        let dir = w.home.join(".codex/sessions");
        std::fs::create_dir_all(&dir).unwrap();
        let image = w.dir.path().join("retained.png");
        std::fs::write(&image, b"image after commit boundary").unwrap();
        let row = serde_json::json!({"type":"event_msg","payload":{"type":"user_message","message":"capture pending attachment","local_images":[image]}});
        let path = dir.join("rollout-crash.jsonl");
        std::fs::write(&path,format!("{{\"type\":\"session_meta\",\"payload\":{{\"id\":\"interrupted-native\"}}}}\n{row}\n")).unwrap();
        {
            let (mut e, _lock) = engine(&w, 64);
            let found = crate::collect::Found {
                path: path.clone(),
                origin: polylogue_sources::origin::by_name("codex").unwrap(),
            };
            let plan = e
                .capture_rollout(
                    &found,
                    FileFacts::of(&std::fs::metadata(&path).unwrap()),
                    &mut Pass::default(),
                )
                .unwrap()
                .unwrap();
            assert_eq!(plan.input.id, "codex:rollout:interrupted-native");
            assert_eq!(
                w.count("SELECT count(*) AS n FROM a.input WHERE format='codex-local-file'"),
                0
            );
        }
        if remove_source {
            std::fs::remove_file(&path).unwrap();
        }
        let (mut e, _lock) = engine(&w, 64);
        let p = e.run_pass().unwrap();
        assert!(p.errors.is_empty(), "{p:?}");
        let id = blake3::hash(b"image after commit boundary")
            .to_hex()
            .to_string();
        let r = e.reads.reader().unwrap();
        let progress = r.input_progress(&id).unwrap().unwrap();
        assert_eq!(
            r.stored_range(&id, progress.chain, 0, progress.length, &e.blobs)
                .unwrap(),
            b"image after commit boundary"
        );
    }
}

#[test]
fn regular_local_files_keep_exact_binary_bytes_empty_files_and_descriptor_facts() {
    let home = ClaudeHome::new();
    let (mut engine, _lock) = engine(&home, 64);
    for (name, bytes) in [
        ("binary", b"original\0bytes\xff".as_slice()),
        ("empty", b"".as_slice()),
    ] {
        let path = home.dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        let facts = FileFacts::of(&std::fs::metadata(&path).unwrap());
        let link = home.dir.path().join(format!("{name}-link"));
        std::os::unix::fs::symlink(&path, &link).unwrap();
        for reference in [&path, &link] {
            let mut pass = Pass::default();
            let id = engine
                .capture_local_file(reference, &mut pass)
                .unwrap()
                .unwrap();
            assert_eq!(id, polylogue_model::BlobHash::of(bytes).to_string());
            assert_eq!(pass.bytes_read, bytes.len() as u64);
            let reader = engine.reads.reader().unwrap();
            let progress = reader.input_progress(&id).unwrap().unwrap();
            assert_eq!(
                reader
                    .stored_range(&id, progress.chain, 0, progress.length, &engine.blobs)
                    .unwrap(),
                bytes
            );
            let observation = reader
                .known_input_at("test-machine", reference.as_os_str().as_encoded_bytes())
                .unwrap()
                .unwrap();
            assert_eq!(observation.facts, Some(facts));
            assert_eq!(FileFacts::of(&std::fs::metadata(&path).unwrap()), facts);
        }
    }
}

#[test]
fn fifo_local_reference_child() {
    let Some(path) = std::env::var_os("POLYLOGUE_LOCAL_FIFO") else {
        return;
    };
    let home = ClaudeHome::new();
    let (mut engine, _lock) = engine(&home, 64);
    let row = serde_json::json!({"local_images":[PathBuf::from(path)]});
    let mut pass = Pass::default();
    engine
        .capture_local_files(
            std::io::Cursor::new(format!("{row}\n")),
            "fifo-source",
            &mut pass,
        )
        .unwrap();
    assert_eq!(pass.bytes_read, 0);
    let reader = engine.reads.reader().unwrap();
    assert_eq!(
        reader
            .query(
                "SELECT count(*) AS n FROM a.input WHERE format='codex-local-file'",
                &[]
            )
            .unwrap()[0]["n"],
        0
    );
    let manifests = reader
        .query("SELECT id FROM a.input WHERE format='tool-context'", &[])
        .unwrap();
    assert_eq!(manifests.len(), 1);
    let id = manifests[0]["id"].as_str().unwrap();
    let progress = reader.input_progress(id).unwrap().unwrap();
    let bytes = reader
        .stored_range(id, progress.chain, 0, progress.length, &engine.blobs)
        .unwrap();
    let observation: Json = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(observation["status"], "unreadable");
    assert!(
        observation["reason"]
            .as_str()
            .unwrap()
            .contains("not a regular file")
    );
}

#[test]
#[allow(unsafe_code)]
fn a_local_fifo_without_a_writer_is_reported_unreadable() {
    use std::os::unix::ffi::OsStrExt;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};
    let home = ClaudeHome::new();
    let fifo = home.dir.path().join("unwritten-image.pipe");
    let name = std::ffi::CString::new(fifo.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let scratch = home.dir.path().join("child");
    std::fs::create_dir(&scratch).unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "local_files::tests::fifo_local_reference_child",
            "--nocapture",
        ])
        .env("POLYLOGUE_LOCAL_FIFO", &fifo)
        .env("TMPDIR", &scratch)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // This watchdog bounds the regression itself, not capture throughput. The
    // predecessor blocks in open forever because this fixture has no writer.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if child.try_wait().unwrap().is_some() {
            break;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            let output = child.wait_with_output().unwrap();
            panic!(
                "FIFO capture did not return without a writer: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "FIFO child failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
