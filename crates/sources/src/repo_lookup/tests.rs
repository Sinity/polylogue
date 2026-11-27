use super::*;

/// The rule as a file found at a location meets it: its first bytes alone.
fn recognize(path: &std::path::Path, head: &[u8]) -> bool {
    super::recognize(path, &crate::document::Head::new(head))
}

#[test]
fn a_lookup_document_becomes_one_row() {
    let doc = br#"{"path":"/work/repo","remote":"https://github.com/o/r.git","root_commits":["abc123"],"read_at":"2026-10-19T00:00:00Z"}"#;
    assert!(recognize(Path::new("/s/polylogue/repo-lookup/x.json"), doc));
    let step = crate::run_document(&Lookup, doc).unwrap();
    let out = step.output;
    assert_eq!(
        out.repo_lookups,
        [RepoLookup {
            path: "/work/repo".into(),
            remote: Some("https://github.com/o/r.git".into()),
            root_commits: vec!["abc123".into()],
        }]
    );
    assert_eq!(step.counts.records, 1);
}

#[test]
fn a_lookup_without_a_remote_keeps_its_root() {
    let doc = br#"{"path":"/work/local","remote":null,"root_commits":["def"],"read_at":"x"}"#;
    let out = crate::run_document(&Lookup, doc).unwrap().output;
    assert_eq!(out.repo_lookups[0].remote, None);
    assert_eq!(out.repo_lookups[0].root_commits, vec!["def".to_owned()]);
}

#[test]
fn root_commits_that_are_not_a_list_of_strings_are_unreadable_not_empty() {
    for doc in [
        r#"{"path":"/w","root_commits":"abc"}"#,
        r#"{"path":"/w","root_commits":[1,"abc"]}"#,
    ] {
        let step = crate::run_document(&Lookup, doc.as_bytes()).unwrap();
        assert!(step.output.repo_lookups.is_empty(), "{doc}");
        assert_eq!(
            step.counts.unreadable["repo_lookup: root_commits is not a list of strings"],
            1
        );
    }
}
