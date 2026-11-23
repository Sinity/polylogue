use super::*;
use std::process::Command;
use std::time::Duration;

#[test]
fn terminal_identity_needs_adapter_process_and_window() {
    let dir = tempfile::tempdir().unwrap();
    let environment = Environment {
        host: "synthetic-host".into(),
        boot: Some("synthetic-boot".into()),
        kitty_pid: Some(std::process::id()),
        kitty_window: Some("42".into()),
    };
    let captured = snapshot(
        "Stop",
        &json!({"cwd":"/synthetic"}),
        &environment,
        &paths(dir.path()),
        Instant::now() + Duration::from_secs(10),
    );
    assert_eq!(captured["terminal_identity"]["host"], "synthetic-host");
    assert_eq!(captured["terminal_identity"]["adapter"], "kitty");
    assert_eq!(
        captured["terminal_identity"]["process_id"],
        std::process::id()
    );
    assert_eq!(captured["terminal_identity"]["window_id"], "42");
    assert_eq!(captured["terminal_identity"]["boot_id"], "synthetic-boot");
    let incomplete = |host: &str, process: Option<u32>, window: Option<&str>| Environment {
        host: host.into(),
        boot: Some("synthetic-boot".into()),
        kitty_pid: process,
        kitty_window: window.map(str::to_owned),
    };
    for missing in [
        incomplete("synthetic-host", None, Some("42")),
        incomplete("synthetic-host", Some(std::process::id()), None),
        incomplete("", Some(std::process::id()), Some("42")),
    ] {
        assert!(
            snapshot(
                "Stop",
                &json!({}),
                &missing,
                &paths(dir.path()),
                Instant::now() + Duration::from_secs(10)
            )
            .get("terminal_identity")
            .is_none()
        );
    }
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
        assert!(output.status.success(), "fixture command failed");
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
    let (identity, repository, unavailable) = git_identity(
        dir.path(),
        &paths(dir.path()),
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap();
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
fn the_session_start_snapshot_never_stores_remote_credentials() {
    let dir = tempfile::tempdir().unwrap();
    let git = |args: &[&str]| {
        let output = Command::new("git")
            .arg("-C")
            .arg(dir.path())
            .args(args)
            .output()
            .unwrap();
        assert!(output.status.success(), "fixture command failed");
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
        "https://user:secret@example.test/synthetic.git",
    ]);
    let captured = snapshot(
        "SessionStart",
        &json!({"cwd":dir.path()}),
        &Environment {
            host: "synthetic".into(),
            boot: None,
            kitty_pid: None,
            kitty_window: None,
        },
        &paths(dir.path()),
        Instant::now() + Duration::from_secs(10),
    );
    assert_eq!(captured["repository"], "https://example.test/synthetic.git");
    assert!(!captured.to_string().contains("secret"), "{captured}");
}

#[test]
fn an_elapsed_budget_starts_no_git_probe() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(
        git_identity(dir.path(), &paths(dir.path()), Instant::now()).unwrap_err(),
        "git identity: git did not answer before the deadline"
    );
}

#[test]
fn a_directory_outside_any_repository_leaves_git_identity_unknown() {
    let dir = tempfile::tempdir().unwrap();
    let captured = snapshot(
        "SessionStart",
        &json!({"cwd":dir.path()}),
        &Environment {
            host: "synthetic".into(),
            boot: None,
            kitty_pid: None,
            kitty_window: None,
        },
        &paths(dir.path()),
        Instant::now() + Duration::from_secs(10),
    );
    assert!(captured.get("git_identity").is_none());
    assert_eq!(
        captured["native_unavailable"],
        json!(["git identity: not in a repository with a commit"])
    );
}

#[test]
fn captured_stated_facts_survive_spooling_and_parsing() {
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
        &paths(dir.path()),
        Instant::now() + Duration::from_secs(10),
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

/// The same fixture Sinex's shell-session capture asserts: a repository whose
/// history merges two unrelated roots. Both products read it through
/// common's `git-identity`, so both record the sorted pair.
#[test]
fn git_identity_roots_equal_across_products() {
    let dir = tempfile::tempdir().unwrap();
    let git = |args: &[&str]| {
        let output = Command::new("git")
            .arg("-C")
            .arg(dir.path())
            .args(["-c", "user.name=t", "-c", "user.email=t@example.invalid"])
            .args(args)
            .output()
            .unwrap();
        assert!(output.status.success(), "git {args:?}");
        String::from_utf8(output.stdout).unwrap().trim().to_owned()
    };
    git(&["init", "-q", "-b", "main"]);
    git(&["commit", "-q", "--allow-empty", "-m", "first"]);
    let first = git(&["rev-parse", "HEAD"]);
    git(&["checkout", "-q", "--orphan", "other"]);
    git(&["commit", "-q", "--allow-empty", "-m", "second"]);
    let second = git(&["rev-parse", "HEAD"]);
    git(&["checkout", "-q", "main"]);
    git(&[
        "merge",
        "-q",
        "--allow-unrelated-histories",
        "--no-edit",
        "other",
    ]);
    let mut both = vec![first, second];
    both.sort();
    let (identity, _, _) = git_identity(
        dir.path(),
        &paths(dir.path()),
        Instant::now() + Duration::from_secs(20),
    )
    .unwrap();
    assert_eq!(identity["root_commits"], json!(both));
}

fn paths(root: &Path) -> Paths {
    Paths {
        home: root.into(),
        config: root.join("config"),
        data: root.join("data"),
        cache: root.join("cache"),
        runtime: root.join("runtime"),
    }
}

#[test]
fn known_roots_are_reused_from_the_working_tree_cache() {
    let dir = tempfile::tempdir().unwrap();
    let paths = paths(dir.path());
    let git = |args: &[&str]| {
        let output = Command::new("git")
            .arg("-C")
            .arg(dir.path())
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap().trim().to_owned()
    };
    git(&["init", "--initial-branch=main"]);
    let commit = || {
        git(&[
            "-c",
            "user.name=Synthetic",
            "-c",
            "user.email=synthetic@example.test",
            "commit",
            "--allow-empty",
            "-m",
            "synthetic",
        ])
    };
    commit();
    let observe = || {
        git_identity(dir.path(), &paths, Instant::now() + Duration::from_secs(10))
            .unwrap()
            .0
    };
    let first = observe();
    let file = paths.git_roots(dir.path());
    let mut known: ::git_identity::KnownRoots =
        serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    known.root_commits.push("known-root-sentinel".into());
    std::fs::write(&file, serde_json::to_vec(&known).unwrap()).unwrap();
    let reused = observe();
    assert!(
        reused["root_commits"]
            .as_array()
            .unwrap()
            .contains(&json!("known-root-sentinel"))
    );
    git(&["branch", "-m", "renamed"]);
    assert_eq!(observe()["branch"], "renamed");
    commit();
    let moved = observe();
    assert_ne!(moved["head"], first["head"]);
    assert_eq!(moved["root_commits"], reused["root_commits"]);
    git(&["checkout", "--orphan", "unrelated"]);
    git(&[
        "-c",
        "user.name=Synthetic",
        "-c",
        "user.email=synthetic@example.test",
        "commit",
        "--allow-empty",
        "-m",
        "unrelated root",
    ]);
    let unrelated = observe();
    assert_eq!(
        unrelated["root_commits"],
        json!([git(&["rev-parse", "HEAD"])])
    );
    std::fs::write(&file, b"partial cache").unwrap();
    assert_eq!(observe(), unrelated);
    let restored: ::git_identity::KnownRoots =
        serde_json::from_slice(&std::fs::read(file).unwrap()).unwrap();
    assert_eq!(json!(restored.root_commits), unrelated["root_commits"]);
}

#[test]
fn an_unwritable_derived_cache_keeps_the_observed_identity() {
    let dir = tempfile::tempdir().unwrap();
    let mut paths = paths(dir.path());
    paths.cache = dir.path().join("not-a-directory");
    std::fs::write(&paths.cache, b"synthetic").unwrap();
    let status = Command::new("git")
        .arg("-C")
        .arg(dir.path())
        .args(["init", "--initial-branch=main"])
        .status()
        .unwrap();
    assert!(status.success());
    let status = Command::new("git")
        .arg("-C")
        .arg(dir.path())
        .args([
            "-c",
            "user.name=Synthetic",
            "-c",
            "user.email=synthetic@example.test",
            "commit",
            "--allow-empty",
            "-m",
            "root",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let observed = git_identity(dir.path(), &paths, Instant::now() + Duration::from_secs(10));
    assert!(
        matches!(&observed, Ok((identity, _, _))
        if identity["branch"] == "main" && identity["root_commits"] == json!([identity["head"]])),
        "{observed:?}"
    );
}
