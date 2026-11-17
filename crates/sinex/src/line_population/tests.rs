use super::*;
use serde_json::json;
use std::collections::BTreeMap;

fn bytes(
    root: &Path,
) -> BTreeMap<std::path::PathBuf, (Option<blake3::Hash>, std::time::SystemTime)> {
    fn visit(
        root: &Path,
        path: &Path,
        files: &mut BTreeMap<std::path::PathBuf, (Option<blake3::Hash>, std::time::SystemTime)>,
    ) {
        let metadata = std::fs::metadata(path).unwrap();
        let digest = if metadata.is_file() {
            Some(blake3::hash(&std::fs::read(path).unwrap()))
        } else {
            None
        };
        files.insert(
            path.strip_prefix(root).unwrap().to_owned(),
            (digest, metadata.modified().unwrap()),
        );
        if metadata.is_dir() {
            for entry in std::fs::read_dir(path).unwrap() {
                visit(root, &entry.unwrap().path(), files);
            }
        }
    }
    let mut files = BTreeMap::new();
    visit(root, root, &mut files);
    files
}

#[test]
fn preserved_spool_export_uses_checked_files_without_mutation() {
    let root = tempfile::tempdir().unwrap();
    let spool = crate::spool::Spool::open(root.path(), "preserved-producer").unwrap();
    let run = json!({"product":"query_run","run":{"id":"preserved-run","binding":{"kind":"original","execution":{"query_options":{"direct_reference":null}}}}});
    let frame=serde_json::from_value(json!({"kind":"events","frame":"query-run:preserved-run",
        "events":[{"kind":{"name":"polylogue.query_run","version":4},"payload":{"record":run},"class":"open"}],"failures":[]})).unwrap();
    spool.prepare(frame).unwrap();
    assert!(
        extract(root.path(), "preserved-producer", &mut Vec::new()).is_err(),
        "an active producer must be frozen first"
    );
    drop(spool);
    let original = bytes(root.path());
    let mut first = Vec::new();
    assert_eq!(
        extract(root.path(), "preserved-producer", &mut first).unwrap(),
        1
    );
    let writes: Vec<Value> = serde_json::from_slice(&first).unwrap();
    assert_eq!(writes[0]["product"], run);
    assert_eq!(bytes(root.path()), original);
    let mut repeat = Vec::new();
    assert_eq!(
        extract(root.path(), "preserved-producer", &mut repeat).unwrap(),
        1
    );
    assert_eq!(repeat, first);
    assert_eq!(bytes(root.path()), original);
}

#[test]
fn missing_preserved_spool_lock_creates_no_entries() {
    let root = tempfile::tempdir().unwrap();
    let original = bytes(root.path());
    assert!(extract(root.path(), "absent-producer", &mut Vec::new()).is_err());
    assert_eq!(bytes(root.path()), original);
}
