//! The Python SDK consumes the same public answers as the CLI.
use super::*;

#[test]
fn the_python_client_reads_and_continues_the_real_daemons_answers() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let project = home.join(".claude/projects/-python");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::write(
        project.join("python.jsonl"),
        polylogue_testkit::claude_user("u1", "python", "a synthetic Python client session"),
    )
    .unwrap();
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let _daemon = start_daemon(&env);
    wait_status(&env, |s| s["store"]["messages"].as_u64() >= Some(1));
    let python = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../clients/python");
    let mut command = Command::new("python3");
    command
        .arg(python.join("tests/integration.py"))
        .env("PYTHONPATH", &python)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("POLYLOGUE_BIN", crate::BIN);
    for (key, value) in env {
        command.env(key, value);
    }
    let result = command.output().unwrap();
    assert!(result.status.success(), "{result:?}");
}
