use super::*;

#[test]
fn a_checkout_root_joins_the_scope() {
    let dir = tempfile::tempdir().unwrap();
    let repo = dir.path().join("proj/repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::create_dir_all(repo.join("crates/x")).unwrap();
    let dirs = scope_dirs(&repo.join("crates/x"));
    assert_eq!(dirs.len(), 2);
    assert_eq!(dirs[1], repo.to_string_lossy());
}

#[test]
fn start_sources_parse_from_harness_names() {
    assert_eq!(StartSource::parse("compact"), StartSource::Compact);
    assert_eq!(StartSource::parse("anything"), StartSource::Startup);
    assert_eq!(StartSource::parse("resume"), StartSource::Resume);
}
