use super::*;

#[test]
fn a_lookup_document_becomes_one_row() {
    let doc = br#"{"path":"/work/repo","remote":"https://github.com/o/r.git","root_commit":"abc123","read_at":"2026-10-19T00:00:00Z"}"#;
    assert!(recognize(Path::new("/s/polylogue/repo-lookup/x.json"), doc));
    let mut out = Output::default();
    let mut acc = Accounting::default();
    Lookup.document(doc, &mut out, &mut acc);
    assert_eq!(
        out.repo_lookups,
        [RepoLookup {
            path: "/work/repo".into(),
            remote: Some("https://github.com/o/r.git".into()),
            root_commit: Some("abc123".into()),
        }]
    );
    assert_eq!(acc.framed, 1);
}

#[test]
fn a_lookup_without_a_remote_keeps_its_root() {
    let doc = br#"{"path":"/work/local","remote":null,"root_commit":"def","read_at":"x"}"#;
    let mut out = Output::default();
    Lookup.document(doc, &mut out, &mut Accounting::default());
    assert_eq!(out.repo_lookups[0].remote, None);
    assert_eq!(out.repo_lookups[0].root_commit.as_deref(), Some("def"));
}
