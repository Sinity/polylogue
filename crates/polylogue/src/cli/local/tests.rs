use super::*;
use serde_json::json;

#[test]
fn install_zsh_dry_run_prints_completion_script() {
    let directory = tempfile::tempdir().unwrap();
    let startup = crate::config::Startup::for_test(
        crate::test_daemon::engine_config(directory.path().join("home")),
        &directory.path().join("archive"),
    );
    let (_, parsed) = crate::cli::parse(&[
        "polylogue",
        "install",
        "zsh",
        "--dry-run",
        "--program",
        "archive-view",
    ])
    .unwrap();
    let mut output = Vec::new();
    assert_eq!(
        install_with_output(
            &parsed,
            &startup,
            false,
            &mut output,
            &crate::auth::Drive::default()
        ),
        ExitCode::SUCCESS
    );
    let script = String::from_utf8(output).unwrap();
    assert!(script.contains("#compdef archive-view"), "{script}");
    assert!(
        script.contains("_POLYLOGUE_COMPLETE_PREFIX=\"$PREFIX\" COMPLETE=\"zsh\""),
        "{script}"
    );
    assert!(script.contains("archive-view"), "{script}");
}

#[test]
fn install_harness_writes_hooks() {
    let directory = tempfile::tempdir().unwrap();
    let startup = crate::config::Startup::for_test(
        crate::test_daemon::engine_config(directory.path().join("home")),
        &directory.path().join("archive"),
    );
    let settings = directory.path().join("settings.json");
    std::fs::write(&settings, "{\"theme\":\"dark\"}\n").unwrap();
    let (_, parsed) = crate::cli::parse(&[
        "polylogue",
        "install",
        "claude-code",
        "--settings",
        settings.to_str().unwrap(),
    ])
    .unwrap();
    let mut output = Vec::new();
    assert_eq!(
        install_with_output(
            &parsed,
            &startup,
            true,
            &mut output,
            &crate::auth::Drive::default()
        ),
        ExitCode::SUCCESS
    );
    let installed: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&settings).unwrap()).unwrap();
    assert_eq!(installed["theme"], "dark");
    assert_eq!(
        installed["hooks"]["SessionStart"][0]["hooks"][0],
        json!({"type": "command", "command": "polylogue internal hook SessionStart", "timeout": 10})
    );
    for event in [
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "Notification",
        "Stop",
        "SubagentStop",
        "PreCompact",
        "SessionEnd",
    ] {
        assert_eq!(
            installed["hooks"][event][0]["hooks"][0]["command"],
            format!("polylogue internal hook {event}"),
            "{event}"
        );
    }
    let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(report, json!({"removed":false,"files":[settings]}));
}

fn chrome_path(startup: &crate::config::Startup) -> std::path::PathBuf {
    host_manifest_dirs(&startup.config.paths.config)[0]
        .join(format!("{}.json", crate::browser::HOST_NAME))
}

fn chrome_install(startup: &crate::config::Startup, options: &[&str]) -> (ExitCode, Vec<u8>) {
    let program = std::env::current_exe().unwrap();
    let mut arguments = vec![
        "polylogue",
        "install",
        "chrome",
        "--program",
        program.to_str().unwrap(),
    ];
    arguments.extend_from_slice(options);
    let (_, parsed) = crate::cli::parse(&arguments).unwrap();
    let mut output = Vec::new();
    let exit = install_with_output(
        &parsed,
        startup,
        true,
        &mut output,
        &crate::auth::Drive::default(),
    );
    (exit, output)
}

#[test]
fn install_chrome_dry_run_changes_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let startup = crate::config::Startup::for_test(
        crate::test_daemon::engine_config(directory.path().join("home")),
        &directory.path().join("archive"),
    );
    let (exit, output) = chrome_install(
        &startup,
        &[
            "--extension-id",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--dry-run",
        ],
    );
    assert_eq!(exit, ExitCode::SUCCESS);
    let manifest: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(manifest["name"], crate::browser::HOST_NAME);
    assert_eq!(
        manifest["allowed_origins"],
        json!(["chrome-extension://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/"])
    );
    assert!(!startup.config.paths.config.exists());
}

#[test]
fn install_chrome_remove_restores_backup_bytes_and_permissions() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let startup = crate::config::Startup::for_test(
        crate::test_daemon::engine_config(directory.path().join("home")),
        &directory.path().join("archive"),
    );
    let path = chrome_path(&startup);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let original = [0xff, 0x00, b'\n', b'a'];
    std::fs::write(&path, original).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
    for _ in 0..2 {
        let (exit, _) = chrome_install(
            &startup,
            &["--extension-id", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
        );
        assert_eq!(exit, ExitCode::SUCCESS);
        assert_eq!(
            std::fs::read(crate::installation::backup_path(&path)).unwrap(),
            original
        );
    }
    let installed = std::fs::read(&path).unwrap();
    let (exit, _) = chrome_install(&startup, &["--remove", "--dry-run"]);
    assert_eq!(exit, ExitCode::SUCCESS);
    assert_eq!(std::fs::read(&path).unwrap(), installed);
    for _ in 0..2 {
        let (exit, _) = chrome_install(&startup, &["--remove"]);
        assert_eq!(exit, ExitCode::SUCCESS);
        assert_eq!(std::fs::read(&path).unwrap(), original);
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o640
        );
    }
}

#[test]
fn install_chrome_remove_preserves_changed_manifest_and_backup() {
    let directory = tempfile::tempdir().unwrap();
    let startup = crate::config::Startup::for_test(
        crate::test_daemon::engine_config(directory.path().join("home")),
        &directory.path().join("archive"),
    );
    let path = chrome_path(&startup);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let original = b"original browser configuration";
    std::fs::write(&path, original).unwrap();
    assert_eq!(
        chrome_install(
            &startup,
            &["--extension-id", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]
        )
        .0,
        ExitCode::SUCCESS
    );
    let changed = b"manually changed browser configuration";
    std::fs::write(&path, changed).unwrap();
    assert_eq!(chrome_install(&startup, &["--remove"]).0, ExitCode::from(4));
    assert_eq!(std::fs::read(&path).unwrap(), changed);
    assert_eq!(
        std::fs::read(crate::installation::backup_path(&path)).unwrap(),
        original
    );
}

#[test]
fn install_chrome_refuses_symlinked_manifest() {
    let directory = tempfile::tempdir().unwrap();
    let startup = crate::config::Startup::for_test(
        crate::test_daemon::engine_config(directory.path().join("home")),
        &directory.path().join("archive"),
    );
    let path = chrome_path(&startup);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let source = directory.path().join("managed.json");
    std::fs::write(&source, b"managed configuration").unwrap();
    std::os::unix::fs::symlink(&source, &path).unwrap();
    assert_eq!(
        chrome_install(
            &startup,
            &["--extension-id", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]
        )
        .0,
        ExitCode::from(4)
    );
    assert_eq!(chrome_install(&startup, &["--remove"]).0, ExitCode::from(4));
    assert!(path.is_symlink());
    assert_eq!(std::fs::read(source).unwrap(), b"managed configuration");
    assert!(!crate::installation::backup_path(&path).exists());
}

#[test]
fn install_chrome_remove_takes_back_a_new_manifest() {
    let directory = tempfile::tempdir().unwrap();
    let startup = crate::config::Startup::for_test(
        crate::test_daemon::engine_config(directory.path().join("home")),
        &directory.path().join("archive"),
    );
    let path = chrome_path(&startup);
    assert_eq!(
        chrome_install(
            &startup,
            &["--extension-id", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]
        )
        .0,
        ExitCode::SUCCESS
    );
    assert!(path.is_file());
    for _ in 0..2 {
        assert_eq!(chrome_install(&startup, &["--remove"]).0, ExitCode::SUCCESS);
        assert!(!path.exists());
    }
}

mod daemon_end;
mod drive;

mod instructions;
