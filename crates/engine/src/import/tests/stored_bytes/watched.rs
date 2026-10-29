use super::*;
use polylogue_store::fault::{self, Action};

fn prompt(text: &str) -> Vec<u8> {
    serde_json::to_vec(
        &serde_json::json!({"chunkedPrompt":{"chunks":[{"role":"user","text":text}]}}),
    )
    .unwrap()
}

#[test]
fn a_refused_watched_export_commits_no_blob_extent() {
    let home = ClaudeHome::new();
    let dir = home.config(64).spool_dir("ai-studio");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("stored.json");
    let original = prompt("synthetic watched import");
    let hash = hashed_blobs::Hash::of(&original);
    std::fs::write(&path, &original).unwrap();
    {
        let (mut engine, _lock) = engine(&home, 64);
        let conn = rusqlite::Connection::open(home.root.join("archive.db")).unwrap();
        conn.execute_batch("CREATE TRIGGER reject_capture BEFORE INSERT ON input BEGIN SELECT RAISE(ABORT,'synthetic watched capture refusal'); END;").unwrap();
        assert!(!engine.run_pass().unwrap().errors.is_empty());
        assert_catalogued(&engine, &hash, false);
        assert_eq!(home.count("SELECT count(*) AS n FROM a.extent"), 0);
        conn.execute_batch("DROP TRIGGER reject_capture").unwrap();
    }
    std::fs::write(&path, prompt("synthetic edited watched import")).unwrap();
    let (mut engine, _lock) = engine(&home, 64);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_catalogued(&engine, &hash, false);
    assert_eq!(home.count("SELECT count(*) AS n FROM a.revision"), 1);
}

#[test]
fn an_interrupted_watched_export_replays_only_committed_original_bytes() {
    for point in ["segment.after_sync", "segment.after_commit"] {
        fault::clear();
        let home = ClaudeHome::new();
        let dir = home.config(64).spool_dir("ai-studio");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stored.json");
        let original = prompt("synthetic watched stored version");
        let hash = hashed_blobs::Hash::of(&original);
        std::fs::write(&path, &original).unwrap();
        let committed = point == "segment.after_commit";
        {
            let (mut engine, _lock) = engine(&home, 64);
            fault::arm(point, "", Action::Fail, 0, 1);
            let pass = engine.run_pass().unwrap();
            assert!(!pass.errors.is_empty(), "{point}");
            assert_eq!(
                home.count("SELECT count(*) AS n FROM a.extent"),
                i64::from(committed)
            );
            assert_catalogued(&engine, &hash, committed);
            fault::clear();
        }
        std::fs::remove_file(&path).unwrap();
        let (mut engine, _lock) = engine(&home, 64);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{point}: {:?}", pass.errors);
        assert_eq!(
            home.count("SELECT count(*) AS n FROM message"),
            i64::from(committed)
        );
        assert_catalogued(&engine, &hash, committed);
    }
    fault::clear();
}
