//! Stored attachment metadata and checked byte pages through the real CLI.
use super::read_current::Items;
use super::*;
use polylogue_model::read::{ItemKind, ReadItem, ReadPage};
use serde_json::{Value, json};

const MESSAGE: &str = "chatgpt:asset-message";
const SESSION: &str = "session:chatgpt:asset-session";
const CAPTION: &str = "caption λ🙂字";
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
    serde_json::to_vec(&json!([{"id":"asset-session","title":"Synthetic export attachment","mapping":{"node":{"id":"node","parent":null,"children":[],"message":{"id":"asset-message","author":{"role":"user"},"create_time":1700000000,"content":{"content_type":"multimodal_text","parts":[{"content_type":"image_asset_pointer","asset_pointer":"file-service://file-Native"},CAPTION]},"metadata":{}}}},"current_node":"node"}])).unwrap()
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

fn typed_page(output: &Output) -> ReadPage {
    assert!(output.status.success(), "{output:?}");
    serde_json::from_value(json(output)).unwrap()
}
fn one_attachment(items: &[ReadItem]) -> &ReadItem {
    let attachments: Vec<_> = items
        .iter()
        .filter(|item| matches!(item.kind, ItemKind::Attachment { .. }))
        .collect();
    assert_eq!(attachments.len(), 1);
    attachments[0]
}
fn assert_available(env: &crate::Fixture, item: &ReadItem, descriptor: &Value) {
    assert_eq!(item.source.reference, format!("block:{MESSAGE}#0"));
    assert_eq!(item.source.block, Some(0));
    let ItemKind::Attachment {
        attachment,
        availability,
        open,
        ..
    } = &item.kind
    else {
        panic!("attachment expected");
    };
    assert!(
        attachment.blob.is_none(),
        "presentation must preserve the original canonical descriptor"
    );
    assert_eq!(attachment.source_id.as_deref(), Some("file-Native"));
    assert_eq!(availability.value().map(String::as_str), Some("present"));
    let args = continuation_arguments(
        open.as_deref()
            .expect("available stored attachment needs its typed open action"),
    );
    let output = cli(env, &args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(output.status.success(), "{output:?}");
    assert_eq!(json(&output)["descriptor"], *descriptor);
    use base64::Engine as _;
    let page = json(&output);
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(page["content_base64"].as_str().unwrap())
        .unwrap();
    assert_eq!(page["offset"], 0);
    assert!(
        page["next"].is_null(),
        "the small fixture's exact bytes must fit"
    );
    assert_eq!(page["size"], bytes.len());
    assert_eq!(page["blob"], json!(polylogue_model::BlobHash::of(&bytes)));
    assert_eq!(descriptor["matches"][0]["size"], page["size"]);
    assert_eq!(descriptor["matches"][0]["blob"], page["blob"]);
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
        let page = typed_page(&read);
        let item = one_attachment(&page.document.items);
        assert_available(&fixture.env(), item, &descriptor);
        let destination = fixture._dir.path().join("canonical");
        std::fs::create_dir(&destination).unwrap();
        let delivered = cli(
            &fixture.env(),
            &[
                "--json",
                "read",
                SESSION,
                "--format",
                "jsonl",
                "--to",
                destination.to_str().unwrap(),
            ],
        );
        assert!(
            matches!(delivered.status.code(), Some(0 | 5)),
            "{delivered:?}"
        );
        let records: Vec<polylogue_model::bundle::Record> =
            std::fs::read_to_string(destination.join("records.jsonl"))
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
        assert!(matches!(
            records.first(),
            Some(polylogue_model::bundle::Record::Manifest(_))
        ));
        let messages: Vec<_> = records
            .iter()
            .filter_map(|record| match record {
                polylogue_model::bundle::Record::Message(message) => Some(message),
                _ => None,
            })
            .collect();
        assert_eq!(messages.len(), 1);
        let message = messages[0];
        assert_eq!(message.reference, format!("message:{MESSAGE}"));
        assert!(
            matches!(&message.message.blocks[0], polylogue_model::Block::Image(attachment) if attachment.blob.is_none() && attachment.source_id.as_deref()==Some("file-Native"))
        );
        assert!(
            !message.bindings.is_empty(),
            "canonical attachment membership must survive export"
        );
        for binding in &message.bindings {
            assert_eq!(binding.source_version, message.source_version);
            assert_eq!((binding.block, binding.attachment), (0, 0));
            assert_eq!(binding.source_id, "file-Native");
            assert_eq!(binding.candidates.len(), 1);
            assert_eq!(
                binding.candidates[0].hash,
                polylogue_model::BlobHash::of(SECRET)
            );
            assert_eq!(binding.candidates[0].size, SECRET.len() as u64);
            binding.validate().unwrap();
        }
        let mut shown = Vec::new();
        polylogue_render::encode_document(
            &page.document,
            polylogue_render::Format::Markdown,
            Default::default(),
            &mut shown,
        )
        .unwrap();
        let shown = String::from_utf8(shown).unwrap();
        assert!(!shown.contains("synthetic-private-file-value"));
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
        &["--json", "read", SESSION, "--page-tokens", "100000"],
    );
    assert!(whole.status.success(), "{whole:?}");
    let whole = typed_page(&whole);
    let expected = one_attachment(&whole.document.items);
    let descriptor = fixture.descriptor();
    assert_available(&fixture.env(), expected, &descriptor);
    assert_eq!(descriptor["matches"][0]["names"], json!([label]));
    let mut reconstructed = Items::default();
    let mut args = vec![
        "--json".to_owned(),
        "read".into(),
        SESSION.into(),
        "--page-tokens".into(),
        "3".into(),
    ];
    let mut original_next = None;
    let mut seen = std::collections::BTreeSet::new();
    let mut complete = false;
    let encoded = serde_json::to_vec(&whole.document.items).unwrap().len();
    for index in 0..=encoded {
        let output = cli(
            &fixture.env(),
            &args.iter().map(String::as_str).collect::<Vec<_>>(),
        );
        let page = typed_page(&output);
        assert!(
            !page.document.items.is_empty(),
            "page made no item progress"
        );
        if index == 0 {
            assert!(
                !page
                    .document
                    .items
                    .iter()
                    .any(|item| matches!(item.kind, ItemKind::Attachment { .. })),
                "header-only page revealed attachment metadata"
            );
        }
        for item in &page.document.items {
            if let ItemKind::Fragment {
                text,
                position,
                until,
                total,
            } = &item.kind
            {
                assert_eq!(*until - position.character, text.chars().count() as u64);
                assert!(*until <= *total);
            }
        }
        reconstructed.push(&page);
        let Some(next) = page.document.footer.next else {
            complete = true;
            break;
        };
        assert!(
            seen.insert(serde_json::to_string(&next.request).unwrap()),
            "repeated continuation"
        );
        original_next.get_or_insert_with(|| next.clone());
        args = continuation_arguments(&next);
    }
    assert!(complete, "finite attachment reading did not finish");
    let joined = reconstructed.into_values();
    assert_eq!(joined.len(), whole.document.items.len());
    assert!(
        joined
            .iter()
            .any(|item| matches!(&item.kind, ItemKind::Text { text } if text == CAPTION))
    );
    for (actual, expected) in joined.iter().zip(&whole.document.items) {
        if !matches!(expected.kind, ItemKind::Attachment { .. }) {
            assert_eq!(actual, expected, "paging changed a source item");
        }
    }
    let actual = one_attachment(&joined);
    assert_eq!(actual.source, expected.source);
    let (
        ItemKind::Attachment {
            attachment: actual,
            availability: actual_state,
            ..
        },
        ItemKind::Attachment {
            attachment: expected,
            availability: expected_state,
            ..
        },
    ) = (&actual.kind, &expected.kind)
    else {
        unreachable!()
    };
    assert_eq!(
        actual, expected,
        "fragment reconstruction changed the complete attachment descriptor"
    );
    assert_eq!(actual_state, expected_state);
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
    let next = original_next.expect("small page must continue");
    let args = continuation_arguments(&next);
    let refused = cli(
        &fixture.env(),
        &args.iter().map(String::as_str).collect::<Vec<_>>(),
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

mod empty;
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
