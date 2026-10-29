use super::*;
use std::collections::BTreeMap;

#[test]
fn published_vocabulary_completes_offline_and_keeps_observed_values_open() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let data = Completion { values: BTreeMap::from([("repo".into(), vec!["two words".into()])]), ..Default::default() };
    write_at(&path, "archive-a", "generation:3:2", &data, 10).unwrap();
    let snapshot = read_at(&path, "archive-a").unwrap().unwrap();
    assert!(crate::daemon::warm::completions(&snapshot, "repo:two").iter().any(|(value, _)| value == "repo:\"two words\""));
    assert!(polylogue_query::parse::with_catalog("repo:unobserved", &snapshot.catalog).is_ok());
    assert!(read_at(&path, "archive-b").unwrap().is_none());
    #[cfg(unix)] { use std::os::unix::fs::PermissionsExt; assert_eq!(fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600); }
}

#[test]
fn incomplete_corrupt_and_other_version_catalogs_do_not_replace_grammar() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    for bytes in [b"PLCAT002".to_vec(), b"PLCAT001".to_vec(), vec![0; HEADER]] {
        fs::write(&path, bytes).unwrap();
        assert!(read_at(&path, "archive-a").unwrap().is_none());
    }
    let grammar = Completion::default();
    assert!(crate::daemon::warm::completions(&grammar, "messages | gr").iter().any(|(value, _)| value.ends_with("group by")));
}
