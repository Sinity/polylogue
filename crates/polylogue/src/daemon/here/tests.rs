use super::*;

#[test]
fn here_matches_main_and_linked_worktree_paths() {
    let temp = tempfile::tempdir().unwrap();
    let main = temp.path().join("main");
    let branch = temp.path().join("branch");
    let common = main.join(".git");
    let gitdir = common.join("worktrees/branch");
    std::fs::create_dir_all(&gitdir).unwrap();
    std::fs::create_dir_all(branch.join("src")).unwrap();
    std::fs::write(
        branch.join(".git"),
        format!("gitdir: {}\n", gitdir.display()),
    )
    .unwrap();
    std::fs::write(gitdir.join("commondir"), "../..\n").unwrap();
    std::fs::write(
        gitdir.join("gitdir"),
        format!("{}\n", branch.join(".git").display()),
    )
    .unwrap();
    let expected = BTreeSet::from([
        main.to_str().unwrap().to_string(),
        branch.to_str().unwrap().to_string(),
    ]);
    assert_eq!(
        paths(&main).unwrap().into_iter().collect::<BTreeSet<_>>(),
        expected
    );
    assert_eq!(
        paths(&branch.join("src"))
            .unwrap()
            .into_iter()
            .collect::<BTreeSet<_>>(),
        expected
    );
    let plain = temp.path().join("plain");
    std::fs::create_dir(&plain).unwrap();
    assert_eq!(paths(&plain).unwrap(), vec![plain.to_str().unwrap()]);
}

#[test]
fn here_uses_recorded_repository_roots_for_the_worktree_paths() {
    let temp = tempfile::tempdir().unwrap();
    let main = temp.path().join("main");
    std::fs::create_dir_all(main.join(".git")).unwrap();
    let mut archive = polylogue_store::fixture::writer::WriterArchive::new();
    archive.lookups.push(polylogue_model::RepoLookup {
        path: main.to_str().unwrap().into(),
        remote: None,
        root_commits: vec!["b".repeat(40), "a".repeat(40)],
    });
    archive.write("reader", vec![]);
    let pool = polylogue_store::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let scoped = scope(&main, &reader).unwrap();
    assert_eq!(scoped.paths, vec![main.to_str().unwrap()]);
    assert_eq!(scoped.repo_roots, vec!["a".repeat(40)]);
}
