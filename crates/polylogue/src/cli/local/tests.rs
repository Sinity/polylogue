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
        install_with_output(&parsed, &startup, false, &mut output),
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
        install_with_output(&parsed, &startup, true, &mut output),
        ExitCode::SUCCESS
    );
    let installed: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&settings).unwrap()).unwrap();
    assert_eq!(installed["theme"], "dark");
    let start = installed["hooks"]["SessionStart"].as_array().unwrap();
    assert!(start.iter().any(
        |matcher| matcher["hooks"].as_array().unwrap().iter().any(|hook| {
            hook["type"] == "command" && hook["command"].as_str().unwrap().contains("SessionStart")
        })
    ));
    let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(report, json!({"removed":false,"files":[settings]}));
}
