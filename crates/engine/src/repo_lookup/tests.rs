use super::*;

#[test]
fn credentials_never_reach_a_lookup() {
    assert_eq!(
        without_credentials("https://user:tok@github.com/o/r.git"),
        "https://github.com/o/r.git"
    );
    assert_eq!(
        without_credentials("https://github.com/o/r.git"),
        "https://github.com/o/r.git"
    );
    assert_eq!(
        without_credentials("git@github.com:o/r.git"),
        "git@github.com:o/r.git"
    );
    // An @ in the path is not userinfo.
    assert_eq!(
        without_credentials("https://host/a@b/r.git"),
        "https://host/a@b/r.git"
    );
}

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
fn a_lookup_is_written_once_and_again_only_when_it_changes() {
    let t = tempfile::tempdir().unwrap();
    let repo = t.path().join("repo");
    // Not a readable repository: remote and root stay unknown.
    fs::create_dir_all(repo.join(".git")).unwrap();
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
    assert!(doc["remote"].is_null() && doc["root_commit"].is_null());
    assert_eq!(doc["read_at"], "t1");
}

#[test]
fn the_checkout_this_crate_is_built_from_reads_as_a_repository() {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let Some(repo) = repository_of(here) else {
        return; // a build outside a checkout (the Nix package)
    };
    let l = read(&repo, &mut Cache::default());
    let root = l.root_commit.expect("a full clone has a root commit");
    assert_eq!(root.len(), 40);
    if let Some(remote) = l.remote {
        assert!(!remote.contains('@') || !remote.contains("://"), "{remote}");
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
        read(&repo, &mut cache),
        Lookup {
            remote: Some("https://example.invalid/team/project.git".into()),
            root_commit: Some(root.clone())
        }
    );
    git(&repo, &["pack-refs", "--all"]);
    assert_eq!(read(&repo, &mut cache).root_commit, Some(root.clone()));
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
    let (git_dir, _) = git_directories(&linked).unwrap();
    fs::write(
        repo.join(".git/worktree-config"),
        "[remote \"origin\"]\nurl = git@example.invalid:team/project.git\n",
    )
    .unwrap();
    writeln!(
        config,
        "[includeIf \"gitdir:{}\"]\npath = worktree-config",
        git_dir.display()
    )
    .unwrap();
    assert_eq!(
        git(&linked, &["config", "--get", "remote.origin.url"]),
        "git@example.invalid:team/project.git"
    );
    assert_eq!(
        read(&linked, &mut cache),
        Lookup {
            remote: Some("git@example.invalid:team/project.git".into()),
            root_commit: Some(root)
        }
    );
}

#[test]
fn roots_follow_first_parents_and_change_with_head() {
    let temp = tempfile::tempdir().unwrap();
    let repo = temp.path().join("repo");
    let root = repository(&repo);
    let mut cache = Cache::default();
    assert_eq!(read(&repo, &mut cache).root_commit, Some(root.clone()));
    git(&repo, &["checkout", "--orphan", "other"]);
    git(&repo, &["commit", "--allow-empty", "-m", "other root"]);
    let other = git(&repo, &["rev-parse", "HEAD"]);
    assert_eq!(read(&repo, &mut cache).root_commit, Some(other));
    git(&repo, &["checkout", "main"]);
    git(
        &repo,
        &["merge", "--allow-unrelated-histories", "--no-edit", "other"],
    );
    assert_eq!(read(&repo, &mut cache).root_commit, Some(root));
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
    assert_eq!(read(&clone, &mut cache).root_commit, None);
    git(&clone, &["fetch", "--unshallow"]);
    assert_eq!(read(&clone, &mut cache).root_commit, Some(root));
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
        read(&repo, &mut Cache::default()).remote.as_deref(),
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
        assert_eq!(head_id(&dir, &dir), None, "{head}");
    }
}
