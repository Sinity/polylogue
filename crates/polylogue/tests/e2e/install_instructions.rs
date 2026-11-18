//! The real local installer under synthetic captured harness environments.
use std::{fs, path::Path};

fn run(root: &Path, harness: &str, extra: &[&str]) -> std::process::Output {
    crate::command(root)
        .env("CLAUDE_CONFIG_DIR", root.join("custom claude"))
        .env("CODEX_HOME", root.join("custom codex"))
        .args(["install", harness, "--instructions"])
        .args(extra)
        .output()
        .unwrap()
}

#[test]
fn normal_cli_instructions_use_captured_harness_homes_and_remove_exactly() {
    for (harness, folder, name) in [
        ("claude-code", "custom claude", "CLAUDE.md"),
        ("codex", "custom codex", "AGENTS.md"),
        ("gemini-cli", "home/.gemini", "GEMINI.md"),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let path = root.join(folder).join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let original = b"# Synthetic global guidance without a final newline";
        fs::write(&path, original).unwrap();
        let preview = run(root, harness, &["--dry-run"]);
        assert!(preview.status.success(), "{preview:?}");
        assert_eq!(
            preview.stdout,
            format!(
                "<!-- polylogue -->\n{}<!-- /polylogue -->\n",
                include_str!("../../../../docs/agent-manual.md")
            )
            .as_bytes()
        );
        assert_eq!(fs::read(&path).unwrap(), original);
        let first = run(root, harness, &[]);
        assert!(first.status.success(), "{first:?}");
        let installed = fs::read(&path).unwrap();
        assert!(installed.starts_with(original));
        assert!(String::from_utf8_lossy(&installed).contains("polylogue_read"));
        let repeat = run(root, harness, &[]);
        assert!(repeat.status.success(), "{repeat:?}");
        assert_eq!(fs::read(&path).unwrap(), installed);
        let removed = run(root, harness, &["--remove"]);
        assert!(removed.status.success(), "{removed:?}");
        assert_eq!(fs::read(&path).unwrap(), original);
        assert!(
            !root.join("archive").exists(),
            "the local installer created an archive"
        );
    }
}

#[test]
fn normal_cli_hermes_instructions_refuse_with_the_config_hint() {
    let temp = tempfile::tempdir().unwrap();
    let output = run(temp.path(), "hermes", &[]);
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(text.contains("config.yaml"), "{text}");
    assert!(!temp.path().join("home/.hermes").exists());
}
