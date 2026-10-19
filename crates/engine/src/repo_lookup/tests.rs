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
