use super::*;

#[test]
fn terminal_identity_requires_the_native_adapter_process_and_window() {
    let environment = Environment {
        host: "synthetic-host".into(),
        boot: Some("synthetic-boot".into()),
        kitty_pid: Some(std::process::id()),
        kitty_window: Some("42".into()),
    };
    let captured = snapshot("Stop", &json!({"cwd":"/synthetic"}), &environment);
    assert_eq!(captured["terminal_identity"]["host"], "synthetic-host");
    assert_eq!(captured["terminal_identity"]["adapter"], "kitty");
    assert_eq!(
        captured["terminal_identity"]["process_id"],
        std::process::id()
    );
    assert_eq!(captured["terminal_identity"]["window_id"], "42");
    assert_eq!(captured["terminal_identity"]["boot_id"], "synthetic-boot");
    let missing = Environment {
        kitty_pid: None,
        ..environment
    };
    assert!(
        snapshot("Stop", &json!({}), &missing)
            .get("terminal_identity")
            .is_none()
    );
}

#[test]
fn git_snapshot_records_common_directory_roots_and_head_from_git() {
    let dir = tempfile::tempdir().unwrap();
    let git = |args: &[&str]| {
        let output = Command::new("git")
            .arg("-C")
            .arg(dir.path())
            .args(args)
            .env("GIT_AUTHOR_DATE", "2026-01-01T00:00:00+00:00")
            .env("GIT_COMMITTER_DATE", "2026-01-01T00:00:00+00:00")
            .output()
            .unwrap();
        assert!(output.status.success(), "native fixture command failed");
        String::from_utf8(output.stdout)
            .unwrap()
            .trim_end_matches('\n')
            .to_owned()
    };
    git(&["init", "--initial-branch=topic"]);
    git(&[
        "-c",
        "user.name=Synthetic",
        "-c",
        "user.email=synthetic@example.test",
        "commit",
        "--allow-empty",
        "-m",
        "root",
    ]);
    git(&[
        "remote",
        "add",
        "origin",
        "https://example.test/synthetic.git",
    ]);
    let head = git(&["rev-parse", "HEAD"]);
    let common = git(&["rev-parse", "--path-format=absolute", "--git-common-dir"]);
    let (identity, repository, unavailable) =
        git_identity(dir.path(), Duration::from_secs(10)).unwrap();
    assert!(unavailable.is_empty());
    assert_eq!(identity["common_dir"], common);
    assert_eq!(identity["root_commits"], json!([head]));
    assert_eq!(identity["head"], head);
    assert_eq!(identity["branch"], "topic");
    assert_eq!(
        repository.as_deref(),
        Some("https://example.test/synthetic.git")
    );
}

#[test]
fn unavailable_git_stays_unknown_and_an_elapsed_budget_starts_no_probe() {
    let dir = tempfile::tempdir().unwrap();
    assert!(git_identity(dir.path(), Duration::ZERO).is_err());
    let captured = snapshot(
        "SessionStart",
        &json!({"cwd":dir.path()}),
        &Environment {
            host: "synthetic".into(),
            boot: None,
            kitty_pid: None,
            kitty_window: None,
        },
    );
    assert!(captured.get("git_identity").is_none());
    assert!(
        captured["native_unavailable"]
            .as_array()
            .is_some_and(|reasons| !reasons.is_empty())
    );
}

#[test]
fn captured_native_facts_survive_spooling_and_parsing_at_their_record_anchor() {
    use polylogue_sources::hook_spool::{HookSpool, ParserState};
    let dir = tempfile::tempdir().unwrap();
    let payload = json!({"session_id":"synthetic", "objectives":[{"kind":"task","reference":"task-1","repository":"https://example.test/project.git"}]});
    let captured = snapshot(
        "Stop",
        &payload,
        &Environment {
            host: "synthetic-host".into(),
            boot: Some("synthetic-boot".into()),
            kitty_pid: Some(std::process::id()),
            kitty_window: Some("42".into()),
        },
    );
    super::super::append_to_spool(
        dir.path(),
        super::super::Harness::Codex,
        "Stop",
        &payload,
        &captured,
        &super::super::tests::startup(),
    )
    .unwrap();
    let path = dir
        .path()
        .join(format!("{}.ndjson", std::os::unix::process::parent_id()));
    let bytes = std::fs::read(path).unwrap();
    let parsed =
        polylogue_sources::parse::run(&HookSpool, &mut ParserState::default(), &bytes, 0).unwrap();
    let event = &parsed.output.events[0];
    assert_eq!(event.range, Some((0, bytes.len() as u64 - 1)));
    assert_eq!(
        event.data["terminal_identity"],
        captured["terminal_identity"]
    );
    assert_eq!(event.data["boot_id"], "synthetic-boot");
    assert_eq!(event.data["objectives"], payload["objectives"]);
    assert!(parsed.output.sessions.is_empty());
}
