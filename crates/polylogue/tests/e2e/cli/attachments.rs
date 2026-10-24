//! Actual export custody and the observed operator-only exact byte path.
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
impl World {
    fn env(&self) -> [(&str, &Path); 3] {
        [
            ("POLYLOGUE_HOME", &self.home),
            ("POLYLOGUE_ARCHIVE", &self.archive),
            ("XDG_CONFIG_HOME", &self.config),
        ]
    }
    fn new(zip: bool) -> Self {
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
        let conversation = serde_json::to_vec(&json!([{"id":"asset-session","title":"Synthetic export attachment","mapping":{"node":{"id":"node","parent":null,"children":[],"message":{"id":"asset-message","author":{"role":"user"},"create_time":1700000000,"content":{"content_type":"multimodal_text","parts":[{"content_type":"image_asset_pointer","asset_pointer":"file-service://file-Native"}]},"metadata":{}}}},"current_node":"node"}])).unwrap();
        let members = [
            ("conversations.json", conversation.as_slice()),
            ("file-Native.dat", SECRET),
            (
                "conversation_asset_file_names.json",
                br#"{"file-Native.dat":"supplied \u03bb\ud83d\ude42.png"}"#.as_slice(),
            ),
            ("notes-file-Native.dat", b"basename decoy".as_slice()),
        ];
        let source = dir.path().join(if zip { "export.zip" } else { "export" });
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
fn public_directory_and_zip_imports_describe_retained_bytes_without_exposing_them() {
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
            shown["content"]
                .as_str()
                .unwrap()
                .contains("retained bytes"),
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
