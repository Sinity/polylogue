use super::*;
use std::io::Write;

#[test]
fn the_repository_is_the_nearest_git_directory_above() {
    let t = tempfile::tempdir().unwrap();
    let repo = t.path().join("repo");
    fs::create_dir_all(repo.join(".git")).unwrap();
    fs::create_dir_all(repo.join("src/deep")).unwrap();
    assert_eq!(repository_of(&repo.join("src/deep")), Some(repo.clone()));
    // Beside the repository, its directory is not the answer (the temp
    // directory may itself sit inside another repository).
    assert!(repository_of(t.path()).is_none_or(|r| !r.starts_with(t.path())));
}

#[test]
fn an_unchanged_lookup_is_written_once() {
    let t = tempfile::tempdir().unwrap();
    let repo = t.path().join("repo");
    let root = repository(&repo);
    fs::create_dir_all(repo.join("src")).unwrap();
    let out = t.path().join("lookups");
    let mut cache = Cache::default();
    let cwds = || vec![repo.join("src"), repo.clone()];
    assert_eq!(refresh(cwds(), &out, &mut cache, "t1").unwrap(), 1);
    assert_eq!(refresh(cwds(), &out, &mut cache, "t2").unwrap(), 0);
    let files: Vec<_> = fs::read_dir(&out).unwrap().collect();
    assert_eq!(files.len(), 1);
    let doc: serde_json::Value =
        serde_json::from_slice(&fs::read(files[0].as_ref().unwrap().path()).unwrap()).unwrap();
    assert_eq!(doc["path"], repo.display().to_string());
    assert!(doc["remote"].is_null());
    assert_eq!(doc["root_commits"], serde_json::json!([root]));
    assert_eq!(doc["read_at"], "t1");
}

#[test]
fn a_changed_remote_rewrites_the_lookup_file() {
    let t = tempfile::tempdir().unwrap();
    let repo = t.path().join("repo");
    let root = repository(&repo);
    let out = t.path().join("lookups");
    let mut cache = Cache::default();
    let cwds = || vec![repo.clone()];
    assert_eq!(refresh(cwds(), &out, &mut cache, "t1").unwrap(), 1);
    git(
        &repo,
        &[
            "remote",
            "add",
            "origin",
            "https://example.invalid/team/project.git",
        ],
    );
    assert_eq!(refresh(cwds(), &out, &mut cache, "t2").unwrap(), 1);
    assert_eq!(refresh(cwds(), &out, &mut cache, "t3").unwrap(), 0);
    let files: Vec<_> = fs::read_dir(&out).unwrap().collect();
    assert_eq!(files.len(), 1);
    let doc: serde_json::Value =
        serde_json::from_slice(&fs::read(files[0].as_ref().unwrap().path()).unwrap()).unwrap();
    assert_eq!(doc["remote"], "https://example.invalid/team/project.git");
    assert_eq!(doc["root_commits"], serde_json::json!([root]));
    assert_eq!(doc["read_at"], "t2");
}

#[test]
fn a_temporary_repository_has_exact_roots_remote_and_path() {
    // Run Git and the lookup with the same isolated configuration, without
    // changing the environment of other tests in this process.
    if std::env::var_os("POLYLOGUE_REPO_TEST").is_none() {
        let home = tempfile::tempdir().unwrap();
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "repo_lookup::tests::a_temporary_repository_has_exact_roots_remote_and_path",
                "--nocapture",
            ])
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap())
            .env("HOME", home.path())
            .env("TMPDIR", home.path())
            .env("POLYLOGUE_REPO_TEST", "1")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_AUTHOR_NAME", "Example")
            .env("GIT_AUTHOR_EMAIL", "example@example.invalid")
            .env("GIT_COMMITTER_NAME", "Example")
            .env("GIT_COMMITTER_EMAIL", "example@example.invalid")
            .env("GIT_AUTHOR_DATE", "2026-01-01T00:00:00Z")
            .env("GIT_COMMITTER_DATE", "2026-01-01T00:00:00Z")
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(0),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .find(|line| line.starts_with("test repo_lookup::")),
            Some(
                "test repo_lookup::tests::a_temporary_repository_has_exact_roots_remote_and_path ... ok"
            )
        );
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let repo = temp.path().join("repo");
    fs::create_dir_all(repo.join("src/deep")).unwrap();
    git(
        &repo,
        &["init", "--object-format=sha1", "--initial-branch=main"],
    );
    git(&repo, &["commit", "--allow-empty", "-m", "root"]);
    // With the author, committer and dates above:
    // git init --object-format=sha1 --initial-branch=main
    // git -c commit.gpgsign=false commit --allow-empty -m root
    // git rev-parse HEAD
    let root = "640243abd56538d9a7fe8eeb4c9da595db4d7e4e";
    git(
        &repo,
        &[
            "remote",
            "add",
            "origin",
            "git@example.invalid:team/project.git",
        ],
    );
    let out = temp.path().join("lookups");
    let mut cache = Cache::default();
    for remote in [
        "git@example.invalid:team/project.git",
        "https://example.invalid/team/project.git",
    ] {
        git(&repo, &["remote", "set-url", "origin", remote]);
        assert_eq!(repository_of(&repo.join("src/deep")), Some(repo.clone()));
        assert_eq!(
            read(&repo, &mut cache),
            Some(Lookup {
                remote: Some(remote.into()),
                root_commits: vec![root.into()]
            })
        );
        assert_eq!(
            refresh([repo.join("src/deep")], &out, &mut cache, "t1").unwrap(),
            1
        );
        let files = fs::read_dir(&out)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect::<Vec<_>>();
        assert_eq!(files.len(), 1);
        let document: serde_json::Value =
            serde_json::from_slice(&fs::read(&files[0]).unwrap()).unwrap();
        assert_eq!(
            document,
            serde_json::json!({"path":repo, "remote":remote, "root_commits":[root], "read_at":"t1"})
        );
    }
}

fn git(repo: &Path, args: &[&str]) -> String {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .args([
            "-c",
            "user.name=Example",
            "-c",
            "user.email=example@example.invalid",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

fn repository(path: &Path) -> String {
    fs::create_dir_all(path).unwrap();
    git(path, &["init", "--initial-branch=main"]);
    git(path, &["commit", "--allow-empty", "-m", "root"]);
    git(path, &["rev-parse", "HEAD"])
}

#[test]
fn linked_worktrees_packed_refs_and_conditional_config_are_read() {
    let temp = tempfile::tempdir().unwrap();
    let repo = temp.path().join("repo");
    let root = repository(&repo);
    fs::write(
        repo.join(".git/remote-config"),
        "[remote \"origin\"]\nurl = https://user:secret@example.invalid/team/project.git\n",
    )
    .unwrap();
    let mut config = fs::OpenOptions::new()
        .append(true)
        .open(repo.join(".git/config"))
        .unwrap();
    writeln!(
        config,
        "[includeIf \"onbranch:main\"]\npath = remote-config"
    )
    .unwrap();
    let mut cache = Cache::default();
    assert_eq!(
        read(&repo, &mut cache).unwrap(),
        Lookup {
            remote: Some("https://example.invalid/team/project.git".into()),
            root_commits: vec![root.clone()]
        }
    );
    git(&repo, &["pack-refs", "--all"]);
    assert_eq!(
        read(&repo, &mut cache).unwrap().root_commits,
        vec![root.clone()]
    );
    let linked = temp.path().join("linked");
    git(
        &repo,
        &[
            "-c",
            "branch.autoSetupMerge=always",
            "worktree",
            "add",
            "-b",
            "linked",
            linked.to_str().unwrap(),
        ],
    );
    // Creating a tracked branch replaces config, so reopen its current file.
    let mut config = fs::OpenOptions::new()
        .append(true)
        .open(repo.join(".git/config"))
        .unwrap();
    let git_dir = git(&linked, &["rev-parse", "--absolute-git-dir"]);
    fs::write(
        repo.join(".git/worktree-config"),
        "[remote \"origin\"]\nurl = git@example.invalid:team/project.git\n",
    )
    .unwrap();
    writeln!(
        config,
        "[includeIf \"gitdir:{}\"]\npath = worktree-config",
        git_dir
    )
    .unwrap();
    assert_eq!(
        git(&linked, &["config", "--get", "remote.origin.url"]),
        "git@example.invalid:team/project.git"
    );
    assert_eq!(
        read(&linked, &mut cache).unwrap(),
        Lookup {
            remote: Some("git@example.invalid:team/project.git".into()),
            root_commits: vec![root]
        }
    );
}

#[test]
fn roots_are_every_root_reachable_from_head() {
    // The set common's git-identity defines, so Sinex attribution and the
    // session hook name the same repository: a merge of an unrelated history
    // adds its root instead of keeping only the first-parent one.
    let temp = tempfile::tempdir().unwrap();
    let repo = temp.path().join("repo");
    let root = repository(&repo);
    let mut cache = Cache::default();
    assert_eq!(
        read(&repo, &mut cache).unwrap().root_commits,
        vec![root.clone()]
    );
    git(&repo, &["checkout", "--orphan", "other"]);
    git(&repo, &["commit", "--allow-empty", "-m", "other root"]);
    let other = git(&repo, &["rev-parse", "HEAD"]);
    assert_eq!(
        read(&repo, &mut cache).unwrap().root_commits,
        vec![other.clone()]
    );
    git(&repo, &["checkout", "main"]);
    git(
        &repo,
        &["merge", "--allow-unrelated-histories", "--no-edit", "other"],
    );
    let mut both = vec![root, other];
    both.sort();
    assert_eq!(read(&repo, &mut cache).unwrap().root_commits, both);
}

#[test]
fn missing_shallow_history_stays_unknown() {
    let temp = tempfile::tempdir().unwrap();
    let repo = temp.path().join("repo");
    let root = repository(&repo);
    git(&repo, &["commit", "--allow-empty", "-m", "later"]);
    let clone = temp.path().join("shallow");
    git(
        &repo,
        &[
            "clone",
            "--depth=1",
            &format!("file://{}", repo.display()),
            clone.to_str().unwrap(),
        ],
    );
    let mut cache = Cache::default();
    assert!(read(&clone, &mut cache).unwrap().root_commits.is_empty());
    git(&clone, &["fetch", "--unshallow"]);
    assert_eq!(read(&clone, &mut cache).unwrap().root_commits, vec![root]);
}

#[test]
fn the_first_fetch_url_uses_the_longest_rewrite_and_masks_credentials() {
    let temp = tempfile::tempdir().unwrap();
    let repo = temp.path().join("repo");
    repository(&repo);
    let mut config = fs::OpenOptions::new()
        .append(true)
        .open(repo.join(".git/config"))
        .unwrap();
    writeln!(config, "[remote \"origin\"]\nurl = short:team/project.git\nurl = https://other.invalid/ignored.git\n[url \"https://example.invalid/\"]\ninsteadOf = short:\n[url \"https://user:secret@example.invalid/selected/\"]\ninsteadOf = short:team/").unwrap();
    assert_eq!(
        read(&repo, &mut Cache::default())
            .unwrap()
            .remote
            .as_deref(),
        Some("https://example.invalid/selected/project.git")
    );
}

#[test]
fn malformed_head_values_and_reference_paths_stay_unknown() {
    let temp = tempfile::tempdir().unwrap();
    let repo = temp.path().join("repo");
    let root = repository(&repo);
    let dir = repo.join(".git");
    fs::write(repo.join("outside"), &root).unwrap();
    for head in [
        "--all",
        "not-a-commit",
        "ref: ../outside",
        "ref: refs/../../outside",
    ] {
        fs::write(dir.join("HEAD"), head).unwrap();
        // No remote and no root: nothing to write over an earlier lookup.
        assert_eq!(read(&repo, &mut Cache::default()), None, "{head}");
    }
}

#[test]
fn ownership_refusal_reports_the_repository_as_unavailable() {
    let temp = tempfile::tempdir().unwrap();
    let repo = temp.path().join("repo");
    repository(&repo);
    git(
        &repo,
        &["remote", "add", "origin", "https://example.invalid/project"],
    );
    let mut cache = Cache::default();
    assert!(read(&repo, &mut cache).unwrap().remote.is_some());
    let lookup = read_with(
        &repo,
        &mut cache,
        |args| {
            git_output(
                Command::new("git")
                    .arg("-C")
                    .arg(&repo)
                    .args(["-c", "safe.directory="])
                    .args(args)
                    .env("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1"),
            )
        },
        ANSWER_WITHIN,
    );
    assert_eq!(lookup, None);
}

#[test]
fn unchanged_head_and_config_reuse_the_lookup_and_included_config_changes_refresh_it() {
    let temp = tempfile::tempdir().unwrap();
    let repo = temp.path().join("repo");
    repository(&repo);
    git(&repo, &["config", "include.path", "remote-config"]);
    let config = repo.join(".git/remote-config");
    fs::write(
        &config,
        "[remote \"origin\"]\nurl = https://example.invalid/first\n",
    )
    .unwrap();
    let mut cache = Cache::default();
    let first = read(&repo, &mut cache).unwrap();
    assert_eq!(
        first.remote.as_deref(),
        Some("https://example.invalid/first")
    );
    let mut queries = Vec::new();
    let reused = read_with(
        &repo,
        &mut cache,
        |args| {
            queries.push(args.join(" "));
            super::git(&repo, args)
        },
        ANSWER_WITHIN,
    );
    assert_eq!(reused, Some(first));
    assert!(
        !queries
            .iter()
            .any(|query| query.starts_with("remote") || query.starts_with("rev-list"))
    );
    fs::write(
        &config,
        "[remote \"origin\"]\nurl = https://example.invalid/second-repository\n",
    )
    .unwrap();
    assert_eq!(
        read(&repo, &mut cache).unwrap().remote.as_deref(),
        Some("https://example.invalid/second-repository")
    );
}

#[test]
fn repository_discovery_follows_new_and_removed_nested_repositories() {
    let temp = tempfile::tempdir().unwrap();
    let outer = temp.path().join("outer");
    let inner = outer.join("inner");
    let cwd = inner.join("src");
    fs::create_dir_all(&cwd).unwrap();
    let mut cache = Cache::default();
    let out = temp.path().join("lookups");
    assert_eq!(refresh([cwd.clone()], &out, &mut cache, "t1").unwrap(), 0);
    repository(&outer);
    assert_eq!(refresh([cwd.clone()], &out, &mut cache, "t2").unwrap(), 1);
    repository(&inner);
    assert_eq!(refresh([cwd.clone()], &out, &mut cache, "t3").unwrap(), 1);
    fs::rename(inner.join(".git"), inner.join("retired-git")).unwrap();
    assert_eq!(repository_of(&cwd), Some(outer));
    assert_eq!(refresh([cwd], &out, &mut cache, "t4").unwrap(), 0);
}

#[test]
fn remote_paths_keep_their_spaces_and_newlines() {
    let temp = tempfile::tempdir().unwrap();
    let repo = temp.path().join("repo");
    repository(&repo);
    let remote = " local repository \n";
    git(&repo, &["remote", "add", "origin", remote]);
    assert_eq!(
        read(&repo, &mut Cache::default())
            .unwrap()
            .remote
            .as_deref(),
        Some(remote)
    );
}

#[test]
fn working_directories_are_the_last_cwd_and_listed_cwds_that_exist_here() {
    let temp = tempfile::tempdir().unwrap();
    let (first, second, third) = (
        temp.path().join("first"),
        temp.path().join("second"),
        temp.path().join("third"),
    );
    for dir in [&first, &second, &third] {
        fs::create_dir_all(dir).unwrap();
    }
    let list = |dirs: &[PathBuf]| serde_json::to_string(dirs).unwrap();
    let gone = temp.path().join("gone");
    let rows = vec![
        serde_json::json!({"cwd": first, "cwd_json": list(&[second.clone(), gone])}),
        serde_json::json!({"cwd": null, "cwd_json": list(std::slice::from_ref(&third))}),
        serde_json::json!({"cwd": "relative/dir", "cwd_json": null}),
        serde_json::json!({"cwd": second, "cwd_json": "not json"}),
    ];
    assert_eq!(
        working_directories(&rows),
        BTreeSet::from([first, second, third])
    );
}

#[test]
fn a_failed_repository_lookup_keeps_the_previous_lookup() {
    let t = tempfile::tempdir().unwrap();
    let repo = t.path().join("repo");
    let root = repository(&repo);
    git(
        &repo,
        &[
            "remote",
            "add",
            "origin",
            "https://example.invalid/team/project.git",
        ],
    );
    let out = t.path().join("lookups");
    let mut cache = Cache::default();
    assert_eq!(
        refresh(vec![repo.clone()], &out, &mut cache, "t1").unwrap(),
        1
    );
    let file = fs::read_dir(&out).unwrap().next().unwrap().unwrap().path();
    let written = fs::read(&file).unwrap();
    // A read past its deadline and a read Git refuses give no lookup.
    git(&repo, &["commit", "--allow-empty", "-m", "later"]);
    assert_eq!(
        read_with(
            &repo,
            &mut cache,
            |args| super::git(&repo, args),
            Duration::ZERO
        ),
        None
    );
    assert_eq!(read_with(&repo, &mut cache, |_| None, ANSWER_WITHIN), None);
    // Writing what such a read gave leaves the last good lookup in place.
    assert!(!write_lookup(&repo, &out, None, "t2").unwrap());
    assert_eq!(fs::read(&file).unwrap(), written);
    let doc: serde_json::Value = serde_json::from_slice(&written).unwrap();
    assert_eq!(doc["remote"], "https://example.invalid/team/project.git");
    assert_eq!(doc["root_commits"], serde_json::json!([root]));
}

/// A pass never waits for repository lookups, which may stall on a slow
/// filesystem: the pass reads its sessions while a lookup is held, and the
/// lookup file written afterwards is collected by a later pass.
#[cfg(feature = "faults")]
#[test]
fn a_pass_does_not_wait_for_repository_lookups() {
    use polylogue_store::fault::{self, Action};
    fault::clear();
    let home = polylogue_testkit::ClaudeHome::new();
    let repo = home.dir.path().join("project");
    repository(&repo);
    let record = |uuid: &str, text: &str| {
        format!(
            "{}\n",
            serde_json::json!({
                "type": "user", "uuid": uuid, "sessionId": "inrepo",
                "timestamp": "2026-03-01T10:00:00Z", "cwd": repo.display().to_string(),
                "message": {"role": "user", "content": text}
            })
        )
    };
    home.append("inrepo.jsonl", &record("u1", "first"));
    let (mut engine, _lock) = crate::import::tests::engine(&home, 1 << 20);
    let first = engine.run_pass().unwrap();
    assert!(first.errors.is_empty(), "{first:?}");
    while engine.lookups.running() {
        std::thread::yield_now();
    }
    let spool = engine
        .config
        .spool_dir(polylogue_sources::repo_lookup::ORIGIN);
    let lookup_files = || fs::read_dir(&spool).map_or(0, |files| files.count());
    assert_eq!(lookup_files(), 0, "no session named the repository yet");
    engine.lookups_at = None;
    fault::arm("repo_lookup.refresh", "", Action::Hold, 0, 1);
    home.append("inrepo.jsonl", &record("u2", "second"));
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert!(
        fault::wait_held_within(Duration::from_secs(60)),
        "the pass started a lookup"
    );
    assert_eq!(home.texts("inrepo"), ["first", "second"]);
    assert_eq!(lookup_files(), 0, "the pass returned before the lookup");
    fault::release();
    while engine.lookups.running() {
        std::thread::yield_now();
    }
    fault::clear();
    assert_eq!(lookup_files(), 1);
    engine.lookups_at = None;
    engine.collectors_at = None;
    let collected = engine.run_pass().unwrap();
    assert!(collected.errors.is_empty(), "{collected:?}");
    assert_eq!(home.count("SELECT count(*) AS n FROM repo_lookup"), 1);
}
