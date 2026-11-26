use super::*;
use polylogue_testkit::{ClaudeHome, claude_user};

#[cfg(feature = "faults")]
mod commits;
mod pool;
#[cfg(feature = "faults")]
mod watched;

fn assert_catalogued(engine: &Engine, hash: &hashed_blobs::Hash, present: bool) {
    match engine.blobs.extent(hash) {
        Ok(extent) => {
            assert!(present, "unexpected committed blob extent");
            assert_eq!(extent.hash, *hash);
        }
        Err(polylogue_store::StoreError::NotFound(_)) => {
            assert!(!present, "committed blob missing")
        }
        Err(error) => panic!("reading the blob catalog failed: {error}"),
    }
}

#[test]
/// A refused import commits no source rows; the blob its put recorded
/// stays, cited by nothing, and the next import of the same bytes finds it.
fn a_refused_import_commits_no_source_rows_and_keeps_its_put_blob() {
    for changed in [false, true] {
        let home = ClaudeHome::new();
        let path = home.dir.path().join("manual.jsonl");
        let original = claude_user("u1", "original-session", "synthetic original import");
        std::fs::write(&path, &original).unwrap();
        let hash = hashed_blobs::Hash::of(original.as_bytes());
        {
            let (mut engine, _lock) = engine(&home, 64);
            let conn = rusqlite::Connection::open(home.root.join("archive.db")).unwrap();
            conn.execute_batch("CREATE TRIGGER reject_first_import BEFORE INSERT ON input BEGIN SELECT RAISE(ABORT,'synthetic first write refusal'); END;").unwrap();
            let rejected = engine
                .import_with_options(
                    std::slice::from_ref(&path),
                    &ImportOptions {
                        origin: Some("claude-code".into()),
                        ..Default::default()
                    },
                )
                .unwrap();
            assert!(!rejected.pass.errors.is_empty());
            assert_catalogued(&engine, &hash, true);
            assert_eq!(home.count("SELECT count(*) AS n FROM a.blob"), 1);
            for table in ["input", "revision", "extent", "observation"] {
                assert_eq!(
                    home.count(&format!("SELECT count(*) AS n FROM a.{table}")),
                    0,
                    "{table}"
                );
            }
        }
        rusqlite::Connection::open(home.root.join("archive.db"))
            .unwrap()
            .execute_batch("DROP TRIGGER reject_first_import")
            .unwrap();
        let (mut engine, _lock) = engine(&home, 64);
        assert!(engine.run_pass().unwrap().errors.is_empty());
        assert_catalogued(&engine, &hash, true);
        if changed {
            std::fs::write(
                &path,
                claude_user("u2", "replacement-session", "synthetic replacement import"),
            )
            .unwrap();
        }
        let retry = engine
            .import_with_options(
                std::slice::from_ref(&path),
                &ImportOptions {
                    origin: Some("claude-code".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(retry.pass.errors.is_empty(), "{:?}", retry.pass.errors);
        assert_catalogued(&engine, &hash, true);
    }
}

#[test]
fn equal_small_bytes_share_one_segment_extent_across_inputs() {
    let home = ClaudeHome::new();
    let original = claude_user("small-user", "small-session", "synthetic shared content");
    home.append("small-session.jsonl", &original);
    let path = home.dir.path().join("manual-small.jsonl");
    std::fs::write(&path, &original).unwrap();
    let hash = hashed_blobs::Hash::of(original.as_bytes());
    let (mut engine, _lock) = engine(&home, 1024);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let extent = engine.blobs.extent(&hash).unwrap();
    let segment = home.root.join(&extent.segment);
    let before = std::fs::metadata(&segment).unwrap().len();
    let report = engine
        .import_with_options(
            &[path],
            &ImportOptions {
                origin: Some("claude-code".into()),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    assert_eq!(engine.blobs.extent(&hash).unwrap(), extent);
    assert_eq!(std::fs::metadata(segment).unwrap().len(), before);
    assert_eq!(home.count("SELECT count(*) AS n FROM a.blob"), 1);
    assert_eq!(home.count("SELECT count(*) AS n FROM a.extent"), 2);
    assert_eq!(home.texts("small-session"), ["synthetic shared content"]);
}
