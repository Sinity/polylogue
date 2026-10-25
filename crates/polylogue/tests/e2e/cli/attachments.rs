//! Export attachments: what the descriptor says about stored files, and the
//! byte path that only the operator's terminal may use.
use super::*;
use serde_json::{Value, json};

mod terminal;

const MESSAGE: &str = "chatgpt:asset-message";
const SESSION: &str = "chatgpt:asset-session";
const SECRET: &[u8] =
    b"-----BEGIN PRIVATE KEY-----\nsynthetic-private-file-value\n-----END PRIVATE KEY-----\n\0\xff";

struct World {
    _daemon: Daemon,
    _dir: tempfile::TempDir,
    home: std::path::PathBuf,
    archive: std::path::PathBuf,
    config: std::path::PathBuf,
}
fn conversation() -> Vec<u8> {
    serde_json::to_vec(&json!([{"id":"asset-session","title":"Synthetic export attachment","mapping":{"node":{"id":"node","parent":null,"children":[],"message":{"id":"asset-message","author":{"role":"user"},"create_time":1700000000,"content":{"content_type":"multimodal_text","parts":[{"content_type":"image_asset_pointer","asset_pointer":"file-service://file-Native"}]},"metadata":{}}}},"current_node":"node"}])).unwrap()
}

impl World {
    fn env(&self) -> [(&str, &Path); 3] {
        [
            ("POLYLOGUE_HOME", &self.home),
            ("POLYLOGUE_ARCHIVE", &self.archive),
            ("XDG_CONFIG_HOME", &self.config),
        ]
    }
    fn new(zip: bool) -> Self {
        Self::with_bytes(zip, SECRET)
    }
    fn with_bytes(zip: bool, bytes: &[u8]) -> Self {
        Self::with_label(zip, bytes, "supplied λ🙂.png")
    }
    fn with_label(zip: bool, bytes: &[u8], label: &str) -> Self {
        Self::with_source_name(zip, bytes, label, if zip { "export.zip" } else { "export" })
    }
    fn with_source_name(zip: bool, bytes: &[u8], label: &str, source_name: &str) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        let archive = dir.path().join("archive");
        let config = dir.path().join("config");
        std::fs::create_dir_all(&home).unwrap();
        let env = [
            ("POLYLOGUE_HOME", home.as_path()),
            ("POLYLOGUE_ARCHIVE", archive.as_path()),
            ("XDG_CONFIG_HOME", config.as_path()),
        ];
        let daemon = start_daemon(&env);
        let conversation = conversation();
        let naming = serde_json::to_vec(&json!({"file-Native.dat":label})).unwrap();
        let members = [
            ("conversations.json", conversation.as_slice()),
            ("file-Native.dat", bytes),
            ("conversation_asset_file_names.json", naming.as_slice()),
            ("notes-file-Native.dat", b"basename decoy".as_slice()),
        ];
        let source = dir.path().join(source_name);
        if zip {
            let mut output = zip::ZipWriter::new(std::fs::File::create(&source).unwrap());
            for (name, bytes) in members {
                output
                    .start_file(name, zip::write::SimpleFileOptions::default())
                    .unwrap();
                output.write_all(bytes).unwrap();
            }
            output.finish().unwrap();
        } else {
            std::fs::create_dir_all(&source).unwrap();
            for (name, bytes) in members {
                std::fs::write(source.join(name), bytes).unwrap();
            }
        }
        let imported = cli(&env, &["--json", "import", source.to_str().unwrap()]);
        assert!(imported.status.success(), "{imported:?}");
        assert_eq!(json(&imported)["pass"]["errors"], json!([]));
        if zip {
            std::fs::remove_file(source).unwrap();
        } else {
            std::fs::remove_dir_all(source).unwrap();
        }
        Self {
            _daemon: daemon,
            _dir: dir,
            home,
            archive,
            config,
        }
    }
    fn descriptor(&self) -> Value {
        let output = cli(&self.env(), &["--json", "attachment", MESSAGE, "0"]);
        assert!(output.status.success(), "{output:?}");
        json(&output)
    }
}

#[test]
fn directory_and_zip_imports_describe_stored_bytes_without_exposing_them() {
    for zip in [false, true] {
        let world = World::new(zip);
        let descriptor = world.descriptor();
        assert_eq!(descriptor["state"], "available");
        assert_eq!(descriptor["candidates"][0]["size"], SECRET.len());
        assert_eq!(
            descriptor["candidates"][0]["names"],
            json!(["supplied λ🙂.png"])
        );
        assert!(descriptor.get("content_base64").is_none());
        assert!(
            !descriptor
                .to_string()
                .contains("synthetic-private-file-value")
        );
        let read = cli(&world.env(), &["--json", "read", SESSION]);
        assert!(read.status.success(), "{read:?}");
        assert!(json(&read)["messages"][0]["blocks"][0]["blob"].is_null());
        let shown = cli(
            &world.env(),
            &["--json", "read", SESSION, "--format", "markdown"],
        );
        assert!(shown.status.success(), "{shown:?}");
        let shown = json(&shown);
        assert!(
            shown["content"].as_str().unwrap().contains("stored bytes"),
            "{shown}"
        );
        assert!(!shown.to_string().contains("synthetic-private-file-value"));
    }
}

#[test]
fn exact_attachment_output_requires_observed_terminal_and_no_harness() {
    let world = World::new(true);
    let target = world._dir.path().join("exact.bin");
    std::fs::write(&target, b"existing output stays until complete").unwrap();
    let args = [
        "--json",
        "attachment",
        MESSAGE,
        "0",
        "--reveal",
        "--out",
        target.to_str().unwrap(),
        "--length",
        "7",
    ];
    let no_terminal = cli(&world.env(), &args);
    assert!(!no_terminal.status.success());
    assert_eq!(
        std::fs::read(&target).unwrap(),
        b"existing output stays until complete"
    );
    let harness = terminal::cli(&world.env(), &args, true);
    assert!(!harness.0.success(), "{harness:?}");
    assert_eq!(
        std::fs::read(&target).unwrap(),
        b"existing output stays until complete"
    );
    let operator = terminal::cli(&world.env(), &args, false);
    assert!(operator.0.success(), "{operator:?}");
    assert_eq!(std::fs::read(&target).unwrap(), SECRET);
    assert!(!String::from_utf8_lossy(&operator.1).contains("synthetic-private-file-value"));
}

#[test]
fn changed_attachment_between_exact_ranges_does_not_replace_existing_output() {
    let bytes = vec![b'Q'; 64 * 1024];
    let world = World::with_bytes(false, &bytes);
    let directory = world._dir.path().join("output");
    std::fs::create_dir(&directory).unwrap();
    let destination = directory.join("exact.bin");
    std::fs::write(&destination, b"existing complete output").unwrap();
    let args = [
        "--json",
        "attachment",
        MESSAGE,
        "0",
        "--reveal",
        "--out",
        destination.to_str().unwrap(),
        "--length",
        "1",
    ];
    let operator = std::thread::scope(|scope| {
        let operator = scope.spawn(|| terminal::cli(&world.env(), &args, false));
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let partial = std::fs::read_dir(&directory)
                .unwrap()
                .filter_map(Result::ok)
                .any(|entry| {
                    entry.path() != destination
                        && entry
                            .metadata()
                            .is_ok_and(|m| m.len() > 0 && m.len() < bytes.len() as u64)
                });
            if partial {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "exact CLI did not write its private partial output"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let conflicting = world._dir.path().join("conflicting-export");
        std::fs::create_dir(&conflicting).unwrap();
        std::fs::write(conflicting.join("conversations.json"), conversation()).unwrap();
        std::fs::write(
            conflicting.join("file-Native.dat"),
            b"a different stored file",
        )
        .unwrap();
        let imported = cli(
            &world.env(),
            &["--json", "import", conflicting.to_str().unwrap()],
        );
        assert!(imported.status.success(), "{imported:?}");
        assert_eq!(json(&imported)["pass"]["errors"], json!([]));
        operator.join().unwrap()
    });
    assert!(!operator.0.success(), "{operator:?}");
    assert!(
        String::from_utf8_lossy(&operator.1).contains("stale_cursor"),
        "{operator:?}"
    );
    assert_eq!(
        std::fs::read(&destination).unwrap(),
        b"existing complete output"
    );
    assert_eq!(
        std::fs::read_dir(directory).unwrap().count(),
        1,
        "failed output must discard its private temporary file"
    );
}

#[test]
fn an_arbitrary_socket_reveal_flag_does_not_grant_exact_byte_output() {
    let world = World::new(false);
    let mut socket =
        std::os::unix::net::UnixStream::connect(world.archive.join("daemon.sock")).unwrap();
    serde_json::to_writer(
        &mut socket,
        &json!({"op":"attachment","message":MESSAGE,"block":0,"reveal":true,"length":1}),
    )
    .unwrap();
    socket.write_all(b"\n").unwrap();
    let mut response = String::new();
    BufReader::new(socket).read_line(&mut response).unwrap();
    let response: Value = serde_json::from_str(&response).unwrap();
    assert_eq!(response["kind"], "operator_required", "{response}");
    assert!(!response.to_string().contains("content_base64"));
}

#[test]
fn unicode_attachment_pages_check_the_descriptor_and_refuse_a_changed_file() {
    let label = "label λ🙂字 ".repeat(120);
    let world = World::with_label(false, SECRET, &label);
    let whole = cli(
        &world.env(),
        &["--json", "read", SESSION, "--budget", "100000"],
    );
    assert!(whole.status.success(), "{whole:?}");
    let whole = json(&whole);
    let mut text = String::new();
    let mut next = None::<String>;
    let mut original_cursor = None;
    let mut described = false;
    loop {
        let mut args = vec!["--json", "read", SESSION, "--budget", "3"];
        if let Some(cursor) = &next {
            args.extend(["--cursor", cursor]);
        }
        let output = cli(&world.env(), &args);
        assert!(output.status.success(), "{output:?}");
        let page = json(&output);
        let shown = page["text"].as_str().unwrap();
        assert!(shown.chars().count() <= 12, "{page}");
        if text.is_empty() {
            assert!(
                page.get("attachments").is_none(),
                "header-only page revealed attachment metadata"
            );
        }
        if let Some(descriptors) = page["attachments"].as_array() {
            described = true;
            assert_eq!(descriptors.len(), 1);
            assert_eq!(descriptors[0]["attachment_index"], 0);
            assert_eq!(descriptors[0]["message"], MESSAGE);
        }
        text.push_str(shown);
        next = page["next"].as_str().map(str::to_owned);
        if original_cursor.is_none() {
            original_cursor = next.clone();
        }
        if next.is_none() {
            break;
        }
    }
    assert!(described);
    assert_eq!(text, whole["text"].as_str().unwrap());
    let conflict = world._dir.path().join("later-export");
    std::fs::create_dir(&conflict).unwrap();
    std::fs::write(conflict.join("conversations.json"), conversation()).unwrap();
    std::fs::write(conflict.join("file-Native.dat"), b"a changed stored file").unwrap();
    assert!(
        cli(
            &world.env(),
            &["--json", "import", conflict.to_str().unwrap()]
        )
        .status
        .success()
    );
    let refused = cli(
        &world.env(),
        &[
            "--json",
            "read",
            SESSION,
            "--budget",
            "3",
            "--cursor",
            original_cursor.as_deref().unwrap(),
        ],
    );
    assert!(!refused.status.success());
    assert_eq!(json(&refused)["error"]["kind"], "stale_cursor");
}

#[test]
fn an_available_empty_attachment_installs_an_empty_verified_file() {
    let world = World::with_bytes(false, b"");
    assert_eq!(world.descriptor()["state"], "available");
    let destination = world._dir.path().join("empty.bin");
    let result = terminal::cli(
        &world.env(),
        &[
            "--json",
            "attachment",
            MESSAGE,
            "0",
            "--reveal",
            "--out",
            destination.to_str().unwrap(),
        ],
        false,
    );
    assert!(result.0.success(), "{result:?}");
    assert!(std::fs::read(destination).unwrap().is_empty());
}

mod promoted;

#[test]
fn removing_a_zip_keeps_its_members_then_removing_a_member_succeeds() {
    let world = World::new(true);
    let before = cli(&world.env(), &["--json", "read", SESSION]);
    assert!(before.status.success(), "{before:?}");
    let pool =
        polylogue_store::ReadPool::open(&polylogue_store::Archive::at(&world.archive)).unwrap();
    let origins = pool
        .reader()
        .unwrap()
        .query(
            "SELECT DISTINCT container_input FROM a.observation_container",
            &[],
        )
        .unwrap();
    assert_eq!(origins.len(), 1);
    let original = origins[0]["container_input"].as_str().unwrap();
    let removed = cli(
        &world.env(),
        &["--json", "excise", &format!("input:{original}"), "--yes"],
    );
    assert!(removed.status.success(), "{removed:?}");
    assert_eq!(
        json(&removed)["zips_removed"],
        serde_json::json!([original])
    );
    let after = cli(&world.env(), &["--json", "read", SESSION]);
    assert!(after.status.success(), "{after:?}");
    assert_eq!(json(&after), json(&before));
    for target in [
        format!("input:{}", polylogue_model::BlobHash::of(SECRET)),
        format!("session:{SESSION}"),
    ] {
        let result = cli(&world.env(), &["--json", "excise", &target, "--yes"]);
        assert!(result.status.success(), "{result:?}");
    }
    assert_eq!(
        pool.reader()
            .unwrap()
            .query("SELECT count(*) AS n FROM a.observation_container", &[])
            .unwrap()[0]["n"],
        0
    );
}

#[test]
fn original_container_name_is_not_a_member_attachment_identity() {
    let world = World::with_source_name(true, SECRET, "actual member.png", "file-Native.dat");
    let descriptor = world.descriptor();
    assert_eq!(descriptor["candidates"].as_array().unwrap().len(), 1);
    assert_eq!(
        descriptor["candidates"][0]["blob"],
        json!(polylogue_model::BlobHash::of(SECRET))
    );
}

#[test]
fn removing_a_zip_member_removes_the_original_zip_and_marks_the_attachment_unavailable() {
    let world = World::new(true);
    let original = {
        let pool =
            polylogue_store::ReadPool::open(&polylogue_store::Archive::at(&world.archive)).unwrap();
        pool.reader()
            .unwrap()
            .query(
                "SELECT DISTINCT container_input FROM a.observation_container",
                &[],
            )
            .unwrap()[0]["container_input"]
            .as_str()
            .unwrap()
            .to_owned()
    };
    let result = cli(
        &world.env(),
        &[
            "--json",
            "excise",
            &format!("input:{}", polylogue_model::BlobHash::of(SECRET)),
            "--yes",
        ],
    );
    assert!(result.status.success(), "{result:?}");
    assert_eq!(json(&result)["zips_removed"], serde_json::json!([original]));
    let descriptor = world.descriptor();
    assert_eq!(descriptor["state"], "unavailable", "{descriptor}");
    assert!(descriptor["candidates"].as_array().unwrap().is_empty());
    let read = cli(&world.env(), &["--json", "read", SESSION]);
    assert!(read.status.success(), "{read:?}");
}
