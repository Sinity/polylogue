//! How the binary exits before any command runs, by the failure table of
//! docs/interfaces.md#exit-codes.

use std::process::Stdio;

#[test]
fn startup_without_working_directory_exits_4() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = crate::Fixture::new(
        &dir.path().join("archive"),
        &dir.path().join("home"),
        &dir.path().join("config"),
    );
    let gone = dir.path().join("gone");
    std::fs::create_dir(&gone).unwrap();
    // The process starts in a directory removed under it.
    let mut command = std::process::Command::new("sh");
    fixture.environment(&mut command);
    let out = command
        .args([
            "-c",
            "cd \"$1\" && rmdir \"$1\" && exec \"$2\" status",
            "sh",
        ])
        .arg(&gone)
        .arg(crate::BIN)
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(4), "{stderr}");
    assert!(
        stderr.contains("the current directory is unavailable"),
        "{stderr}"
    );
}

#[test]
fn daemon_startup_io_failure_exits_4() {
    let dir = tempfile::tempdir().unwrap();
    // The archive path names a file, so no archive can be opened there.
    std::fs::write(dir.path().join("archive"), b"not an archive").unwrap();
    let out = crate::command(dir.path())
        .args(["daemon", "--until-idle", "--no-sources"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(4), "{stderr}");
    assert_eq!(
        std::fs::read(dir.path().join("archive")).unwrap(),
        b"not an archive"
    );
}

#[test]
fn hook_with_unusable_configuration_exits_0_after_valued_globals() {
    let dir = tempfile::tempdir().unwrap();
    let out = crate::command(dir.path())
        .args(["--deadline", "100", "--auth-token-file"])
        .arg(dir.path().join("missing-token"))
        .args(["internal", "hook", "--harness", "claude-code", "Stop"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    assert!(stderr.starts_with("polylogue hook: "), "{stderr}");
    assert!(stderr.contains("missing-token"), "{stderr}");
}
