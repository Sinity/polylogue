//! Stored attachment metadata and checked byte pages through the real CLI.
use super::*;
use serde_json::{Value, json};

const MESSAGE: &str = "chatgpt:asset-message";
const SESSION: &str = "chatgpt:asset-session";
const SECRET: &[u8] =
    b"-----BEGIN PRIVATE KEY-----\nsynthetic-private-file-value\n-----END PRIVATE KEY-----\n\0\xff";

struct TestArchive {
    _daemon: Daemon,
    _dir: tempfile::TempDir,
    home: std::path::PathBuf,
    archive: std::path::PathBuf,
    config: std::path::PathBuf,
}
fn conversation() -> Vec<u8> {
    serde_json::to_vec(&json!([{"id":"asset-session","title":"Synthetic export attachment","mapping":{"node":{"id":"node","parent":null,"children":[],"message":{"id":"asset-message","author":{"role":"user"},"create_time":1700000000,"content":{"content_type":"multimodal_text","parts":[{"content_type":"image_asset_pointer","asset_pointer":"file-service://file-Native"}]},"metadata":{}}}},"current_node":"node"}])).unwrap()
}

impl TestArchive {
    fn env(&self) -> crate::Fixture {
        crate::Fixture::new(&self.archive, &self.home, &self.config)
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
        let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
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
        json(&output)["descriptor"].clone()
    }
}

#[test]
fn directory_and_zip_imports_describe_exact_stored_attachment_bytes() {
    for zip in [false, true] {
        let fixture = TestArchive::new(zip);
        let descriptor = fixture.descriptor();
        assert_eq!(descriptor["state"], "available");
        assert_eq!(descriptor["matches"][0]["size"], SECRET.len());
        assert_eq!(
            descriptor["matches"][0]["names"],
            json!(["supplied λ🙂.png"])
        );
        assert!(
            !descriptor
                .to_string()
                .contains("synthetic-private-file-value")
        );
        let read = cli(&fixture.env(), &["--json", "read", SESSION]);
        assert!(read.status.success(), "{read:?}");
        assert!(json(&read)["messages"][0]["blocks"][0]["blob"].is_null());
        let shown = cli(
            &fixture.env(),
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
fn exact_attachment_output_is_available_to_agents_without_terminal_stdout() {
    let fixture = TestArchive::new(true);
    let target = fixture._dir.path().join("exact.bin");
    let args = [
        "--json",
        "attachment",
        MESSAGE,
        "0",
        "--out",
        target.to_str().unwrap(),
        "--length",
        "7",
    ];
    for agent in [false, true] {
        std::fs::write(&target, b"previous complete output").unwrap();
        let mut child = command(&fixture.env());
        for (key, value) in fixture.env() {
            child.env(key, value);
        }
        if agent {
            child.env("CODEX_SESSION_ID", "synthetic-harness");
        }
        let output = child.args(args).output().unwrap();
        assert!(output.status.success(), "{output:?}");
        assert_eq!(std::fs::read(&target).unwrap(), SECRET);
    }
}

#[test]
fn changed_attachment_between_exact_ranges_does_not_replace_existing_output() {
    let bytes = vec![b'Q'; 64 * 1024];
    let fixture = TestArchive::with_bytes(false, &bytes);
    let directory = fixture._dir.path().join("output");
    std::fs::create_dir(&directory).unwrap();
    let destination = directory.join("exact.bin");
    std::fs::write(&destination, b"existing complete output").unwrap();
    let args = [
        "--json",
        "attachment",
        MESSAGE,
        "0",
        "--out",
        destination.to_str().unwrap(),
        "--length",
        "1",
    ];
    let operator = std::thread::scope(|scope| {
        let operator = scope.spawn(|| cli(&fixture.env(), &args));
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
            std::thread::sleep(Duration::from_millis(1));
        }
        let conflicting = fixture._dir.path().join("conflicting-export");
        std::fs::create_dir(&conflicting).unwrap();
        std::fs::write(conflicting.join("conversations.json"), conversation()).unwrap();
        std::fs::write(
            conflicting.join("file-Native.dat"),
            b"a different stored file",
        )
        .unwrap();
        let imported = cli(
            &fixture.env(),
            &["--json", "import", conflicting.to_str().unwrap()],
        );
        assert!(imported.status.success(), "{imported:?}");
        assert_eq!(json(&imported)["pass"]["errors"], json!([]));
        operator.join().unwrap()
    });
    assert!(!operator.status.success(), "{operator:?}");
    assert!(
        String::from_utf8_lossy(&[operator.stdout.clone(), operator.stderr.clone()].concat())
            .contains("stale_cursor"),
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
fn ordinary_attachment_pages_reconstruct_credential_shaped_binary_bytes() {
    use base64::Engine as _;
    let fixture = TestArchive::new(false);
    let mut bytes = Vec::new();
    let mut cursor = None::<String>;
    loop {
        let mut args = vec!["--json", "attachment", MESSAGE, "0", "--length", "7"];
        if let Some(cursor) = &cursor {
            args.extend(["--cursor", cursor]);
        }
        let output = cli(&fixture.env(), &args);
        assert!(output.status.success(), "{output:?}");
        let page = json(&output);
        assert_eq!(page["offset"], bytes.len());
        let part = base64::engine::general_purpose::STANDARD
            .decode(page["content_base64"].as_str().unwrap())
            .unwrap();
        bytes.extend(part);
        cursor = page["next"].as_str().map(str::to_owned);
        if cursor.is_none() {
            break;
        }
    }
    assert_eq!(bytes, SECRET);
}

#[test]
fn unicode_attachment_pages_check_the_descriptor_and_refuse_a_changed_file() {
    let label = "label λ🙂字 ".repeat(120);
    let fixture = TestArchive::with_label(false, SECRET, &label);
    let whole = cli(
        &fixture.env(),
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
        let output = cli(&fixture.env(), &args);
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
    let conflict = fixture._dir.path().join("later-export");
    std::fs::create_dir(&conflict).unwrap();
    std::fs::write(conflict.join("conversations.json"), conversation()).unwrap();
    std::fs::write(conflict.join("file-Native.dat"), b"a changed stored file").unwrap();
    assert!(
        cli(
            &fixture.env(),
            &["--json", "import", conflict.to_str().unwrap()]
        )
        .status
        .success()
    );
    let refused = cli(
        &fixture.env(),
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
    let fixture = TestArchive::with_bytes(false, b"");
    assert_eq!(fixture.descriptor()["state"], "available");
    let destination = fixture._dir.path().join("empty.bin");
    let result = cli(
        &fixture.env(),
        &[
            "--json",
            "attachment",
            MESSAGE,
            "0",
            "--out",
            destination.to_str().unwrap(),
        ],
    );
    assert!(result.status.success(), "{result:?}");
    assert!(std::fs::read(destination).unwrap().is_empty());
}

mod promoted;

#[test]
fn original_container_name_is_not_a_member_attachment_identity() {
    let fixture =
        TestArchive::with_source_name(true, SECRET, "actual member.png", "file-Native.dat");
    let descriptor = fixture.descriptor();
    assert_eq!(descriptor["matches"].as_array().unwrap().len(), 1);
    assert_eq!(
        descriptor["matches"][0]["blob"],
        json!(polylogue_model::BlobHash::of(SECRET))
    );
}
