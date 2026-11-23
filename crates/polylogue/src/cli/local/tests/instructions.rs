use crate::config::Startup;
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::ExitCode,
};

fn setup(overrides: bool) -> (tempfile::TempDir, Startup) {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::test_daemon::engine_config(dir.path().join("home"));
    if overrides {
        config
            .tool_dirs
            .insert("CLAUDE_CONFIG_DIR".into(), dir.path().join("custom claude"));
        config
            .tool_dirs
            .insert("CODEX_HOME".into(), PathBuf::from("custom codex"));
    }
    let startup = Startup::for_test(config, &dir.path().join("archive"));
    (dir, startup)
}
fn paths(startup: &Startup, harness: &str) -> (PathBuf, PathBuf) {
    let home = &startup.config.paths.home;
    let root = match harness {
        "claude-code" => startup.engine.root(polylogue_sources::claude_code::ROOT),
        "codex" => startup.engine.root(polylogue_sources::codex::ROOT),
        "gemini-cli" => home.join(".gemini"),
        _ => unreachable!(),
    };
    let (settings, instructions) = match harness {
        "claude-code" => ("settings.json", "CLAUDE.md"),
        "codex" => ("hooks.json", "AGENTS.md"),
        "gemini-cli" => ("settings.json", "GEMINI.md"),
        _ => unreachable!(),
    };
    (root.join(settings), root.join(instructions))
}
fn write(path: &Path, bytes: &[u8]) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}
fn install(startup: &Startup, harness: &str, extra: &[&str]) -> (ExitCode, Vec<u8>) {
    let mut words = vec!["polylogue", "install", harness];
    words.extend(extra);
    let (_, parsed) = crate::cli::parse(&words).unwrap();
    let mut output = Vec::new();
    let code = super::super::install_with_output(
        &parsed,
        startup,
        true,
        &mut output,
        &crate::auth::Drive::default(),
    );
    (code, output)
}
fn expected_block() -> String {
    format!(
        "<!-- polylogue -->\n{}<!-- /polylogue -->\n",
        crate::manual::card()
    )
}

#[test]
fn harness_instructions_use_configured_roots_restore_exact_bytes_and_repeat() {
    for override_root in [false, true] {
        for harness in ["claude-code", "codex", "gemini-cli"] {
            if override_root && harness == "gemini-cli" {
                continue;
            }
            for original in [
                b"# Existing guidance\r\nkeep this\r\n".as_slice(),
                b"# Existing guidance without newline".as_slice(),
                b"".as_slice(),
            ] {
                let (_dir, startup) = setup(override_root);
                let (settings, path) = paths(&startup, harness);
                let original_settings = b"{ \"theme\": \"neutral\", \"hooks\": {} }\n";
                write(&settings, original_settings);
                write(&path, original);
                fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
                assert_eq!(
                    install(&startup, harness, &["--instructions"]).0,
                    ExitCode::SUCCESS
                );
                let installed = fs::read(&path).unwrap();
                let separator = if original.is_empty() || original.ends_with(b"\n") {
                    b"".as_slice()
                } else {
                    b"\n".as_slice()
                };
                let expected = [original, separator, expected_block().as_bytes()].concat();
                assert_eq!(installed, expected);
                assert_eq!(
                    fs::read(crate::installation::backup_path(&path)).unwrap(),
                    original
                );
                assert_eq!(
                    fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                    0o640
                );
                assert_eq!(
                    install(&startup, harness, &["--instructions"]).0,
                    ExitCode::SUCCESS
                );
                assert_eq!(fs::read(&path).unwrap(), installed);
                assert_eq!(
                    fs::read(crate::installation::backup_path(&path)).unwrap(),
                    original
                );
                assert_eq!(
                    install(
                        &startup,
                        harness,
                        &["--instructions", "--remove", "--dry-run"]
                    )
                    .0,
                    ExitCode::SUCCESS
                );
                assert_eq!(fs::read(&path).unwrap(), installed);
                assert_eq!(
                    install(&startup, harness, &["--instructions", "--remove"]).0,
                    ExitCode::SUCCESS
                );
                assert_eq!(fs::read(&path).unwrap(), original);
                assert_eq!(fs::read(&settings).unwrap(), original_settings);
                assert!(!crate::installation::backup_path(&path).exists());
                assert_eq!(
                    fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                    0o640
                );
                assert_eq!(
                    install(&startup, harness, &["--instructions", "--remove"]).0,
                    ExitCode::SUCCESS
                );
                assert_eq!(fs::read(path).unwrap(), original);
            }
        }
    }
}

#[test]
fn instruction_dry_run_prints_the_mcp_card_without_writing_either_file() {
    for harness in ["claude-code", "codex", "gemini-cli"] {
        let (_dir, startup) = setup(false);
        let (settings, path) = paths(&startup, harness);
        let (code, output) = install(&startup, harness, &["--instructions", "--dry-run"]);
        assert_eq!(code, ExitCode::SUCCESS);
        assert_eq!(output, expected_block().as_bytes());
        assert!(!settings.exists());
        assert!(!path.exists());
        assert!(!crate::installation::backup_path(&path).exists());
        let target = startup.config.paths.home.join("managed instructions");
        write(&target, b"managed source");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(&target, &path).unwrap();
        assert_eq!(
            install(&startup, harness, &["--instructions", "--dry-run"]).1,
            expected_block().as_bytes()
        );
        assert_eq!(fs::read(target).unwrap(), b"managed source");
    }
}

#[test]
fn instructions_remove_only_the_owned_block_and_keep_later_outside_edits() {
    let (_dir, startup) = setup(false);
    let (settings, path) = paths(&startup, "codex");
    let before = b"# Before\n";
    write(&path, before);
    write(&settings, b"{}\n");
    assert_eq!(
        install(&startup, "codex", &["--instructions"]).0,
        ExitCode::SUCCESS
    );
    let mut installed = fs::read(&path).unwrap();
    installed.extend_from_slice(b"# Later user guidance\n");
    fs::write(&path, installed).unwrap();
    assert_eq!(
        install(&startup, "codex", &["--instructions", "--remove"]).0,
        ExitCode::SUCCESS
    );
    assert_eq!(
        fs::read(&path).unwrap(),
        b"# Before\n# Later user guidance\n"
    );
    assert_eq!(
        fs::read(crate::installation::backup_path(&path)).unwrap(),
        before
    );
    assert_eq!(
        install(&startup, "codex", &["--instructions", "--remove"]).0,
        ExitCode::SUCCESS
    );
    assert_eq!(
        fs::read(path).unwrap(),
        b"# Before\n# Later user guidance\n"
    );
}

#[test]
fn install_updates_old_owned_guidance_without_rewriting_surrounding_text() {
    let (_dir, startup) = setup(false);
    let (settings, path) = paths(&startup, "gemini-cli");
    write(&settings, b"{}\n");
    let original =
        b"# Before\r\n<!-- polylogue -->\r\nold guidance\r\n<!-- /polylogue -->\r\n# After\r\n";
    write(&path, original);
    assert_eq!(
        install(&startup, "gemini-cli", &["--instructions"]).0,
        ExitCode::SUCCESS
    );
    assert_eq!(
        fs::read(&path).unwrap(),
        format!("# Before\r\n{}# After\r\n", expected_block()).as_bytes()
    );
    assert_eq!(
        fs::read(crate::installation::backup_path(&path)).unwrap(),
        original
    );
    assert_eq!(
        install(&startup, "gemini-cli", &["--instructions", "--remove"]).0,
        ExitCode::SUCCESS
    );
    assert_eq!(fs::read(&path).unwrap(), b"# Before\r\n# After\r\n");
    assert_eq!(
        fs::read(crate::installation::backup_path(&path)).unwrap(),
        original
    );
}

#[test]
fn instructions_flag_is_required_and_created_files_are_removed() {
    for harness in ["claude-code", "codex", "gemini-cli"] {
        let (_dir, startup) = setup(false);
        let (settings, path) = paths(&startup, harness);
        assert_eq!(install(&startup, harness, &[]).0, ExitCode::SUCCESS);
        assert!(!path.exists());
        assert_eq!(
            install(&startup, harness, &["--instructions"]).0,
            ExitCode::SUCCESS
        );
        assert!(path.exists());
        let before = fs::read(&path).unwrap();
        assert_eq!(
            install(&startup, harness, &["--remove"]).0,
            ExitCode::SUCCESS
        );
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(
            install(&startup, harness, &["--instructions", "--remove"]).0,
            ExitCode::SUCCESS
        );
        assert!(!path.exists());
        assert!(!settings.exists());
        write(&path, b"");
        write(&settings, b"{}");
        assert_eq!(
            install(&startup, harness, &["--instructions", "--remove"]).0,
            ExitCode::SUCCESS
        );
        assert!(path.exists());
        assert_eq!(fs::read(&settings).unwrap(), b"{}");
    }
}

#[test]
fn malformed_or_managed_instructions_fail_before_hook_settings_change() {
    for original in [
        b"<!-- polylogue -->\nunfinished".as_slice(),
        b"<!-- /polylogue -->\n<!-- polylogue -->\n".as_slice(),
        b"<!-- polylogue -->\na\n<!-- polylogue -->\nb\n<!-- /polylogue -->\n".as_slice(),
        &[0xff],
    ] {
        let (_dir, startup) = setup(false);
        let (settings, path) = paths(&startup, "claude-code");
        write(&settings, b"{\"theme\":\"keep\"}");
        write(&path, original);
        assert_eq!(
            install(&startup, "claude-code", &["--instructions"]).0,
            ExitCode::from(4)
        );
        assert_eq!(fs::read(settings).unwrap(), b"{\"theme\":\"keep\"}");
        assert_eq!(fs::read(&path).unwrap(), original);
        assert!(!crate::installation::backup_path(&path).exists());
    }
    for managed_backup in [false, true] {
        let (dir, startup) = setup(false);
        let (settings, path) = paths(&startup, "codex");
        write(&settings, b"{\"theme\":\"keep\"}");
        let target = dir.path().join("declared source");
        write(&target, b"original");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let link = if managed_backup {
            write(&path, b"# Existing\n");
            crate::installation::backup_path(&path)
        } else {
            path.clone()
        };
        std::os::unix::fs::symlink(&target, link).unwrap();
        assert_eq!(
            install(&startup, "codex", &["--instructions"]).0,
            ExitCode::from(4)
        );
        assert_eq!(fs::read(settings).unwrap(), b"{\"theme\":\"keep\"}");
        assert_eq!(fs::read(target).unwrap(), b"original");
    }
}

#[test]
fn invalid_hook_settings_do_not_change_instructions_and_settings_override_does_not_move_them() {
    let (dir, startup) = setup(true);
    let (_, path) = paths(&startup, "claude-code");
    write(&path, b"# Keep\n");
    let settings = dir.path().join("explicit settings.json");
    write(&settings, b"not JSON");
    assert_eq!(
        install(
            &startup,
            "claude-code",
            &["--instructions", "--settings", settings.to_str().unwrap()]
        )
        .0,
        ExitCode::from(4)
    );
    assert_eq!(fs::read(&path).unwrap(), b"# Keep\n");
    assert!(!crate::installation::backup_path(&path).exists());
    fs::write(&settings, b"{}\n").unwrap();
    assert_eq!(
        install(
            &startup,
            "claude-code",
            &["--instructions", "--settings", settings.to_str().unwrap()]
        )
        .0,
        ExitCode::SUCCESS
    );
    assert!(
        fs::read_to_string(path)
            .unwrap()
            .contains("<!-- polylogue -->")
    );
    assert!(!settings.with_file_name("CLAUDE.md").exists());
}

#[test]
fn hermes_and_non_harness_targets_refuse_instructions_without_writes() {
    let (_dir, startup) = setup(false);
    for harness in ["hermes", "chrome", "zsh"] {
        assert_eq!(
            install(&startup, harness, &["--instructions"]).0,
            ExitCode::from(2)
        );
    }
    let (_, parsed) =
        crate::cli::parse(&["polylogue", "install", "hermes", "--instructions"]).unwrap();
    let arguments: super::super::arguments::Install =
        serde_json::from_value(parsed.request.arguments).unwrap();
    let error = super::super::instructions::install(&startup, &arguments, true, &mut Vec::new())
        .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    assert_eq!(
        error.to_string(),
        "Hermes reads its hooks from config.yaml; --instructions has no supported global instructions file for Hermes"
    );
    assert!(!startup.config.paths.home.exists());
}

#[test]
fn settings_and_instruction_targets_cannot_overwrite_each_other_or_their_backups() {
    for destination in ["instructions", "instructions-backup"] {
        let (_dir, startup) = setup(false);
        let (_, path) = paths(&startup, "claude-code");
        let settings = match destination {
            "instructions" => path.clone(),
            "instructions-backup" => crate::installation::backup_path(&path),
            _ => unreachable!(),
        };
        assert_eq!(
            install(
                &startup,
                "claude-code",
                &["--instructions", "--settings", settings.to_str().unwrap()]
            )
            .0,
            ExitCode::from(2)
        );
        assert!(!path.exists());
        assert!(!settings.exists());
    }
}
