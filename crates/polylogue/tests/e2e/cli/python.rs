//! The Python client consumes the same public answers as the CLI.
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
    std::fs::write(
        project.join("python-pages.jsonl"),
        polylogue_testkit::claude_user(
            "u-pages",
            "python-pages",
            &"Python λ🙂 continuation. ".repeat(300),
        ),
    )
    .unwrap();
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
    let _daemon = start_daemon(&env);
    wait_status(&env, |s| s["store"]["messages"].as_u64() == Some(2));
    let python = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../clients/python");
    let mut command = Command::new("python3");
    env.environment(&mut command);
    command
        .arg(python.join("tests/integration.py"))
        .arg(&archive)
        .arg(crate::BIN)
        .env("PYTHONPATH", &python)
        .env("PYTHONDONTWRITEBYTECODE", "1");
    let result = command.output().unwrap();
    assert!(result.status.success(), "{result:?}");
}
