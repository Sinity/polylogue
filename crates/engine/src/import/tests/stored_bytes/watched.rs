use super::*;
use polylogue_store::fault::{self, Action};

fn prompt(text: &str) -> Vec<u8> {
    serde_json::to_vec(
        &serde_json::json!({"chunkedPrompt":{"chunks":[{"role":"user","text":text}]}}),
    )
    .unwrap()
}

#[test]
fn a_refused_watched_export_installs_no_file() {
    let world = ClaudeHome::new();
    let dir = world.config(64).spool_dir("ai-studio");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("stored.json");
    let original = prompt("synthetic watched acquisition");
    let hash = hashed_blobs::Hash::of(&original);
    std::fs::write(&path, &original).unwrap();
    {
        let (mut engine, _lock) = engine(&world, 64);
        let conn = rusqlite::Connection::open(world.root.join("archive.db")).unwrap();
        conn.execute_batch("CREATE TRIGGER reject_capture BEFORE INSERT ON input BEGIN SELECT RAISE(ABORT,'synthetic watched capture refusal'); END;").unwrap();
        assert!(!engine.converge().unwrap().errors.is_empty());
        assert!(!engine.blobs.contains(&hash));
        assert_eq!(world.count("SELECT count(*) AS n FROM a.extent"), 0);
        conn.execute_batch("DROP TRIGGER reject_capture").unwrap();
    }
    std::fs::write(&path, prompt("synthetic edited watched acquisition")).unwrap();
    let (mut engine, _lock) = engine(&world, 64);
    let pass = engine.converge().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert!(!engine.blobs.contains(&hash));
    assert_eq!(world.count("SELECT count(*) AS n FROM a.revision"), 1);
}

#[test]
fn a_stopped_watched_export_finishes_only_its_stored_bytes() {
    for point in ["import.install", "import.installed"] {
        let world = ClaudeHome::new();
        let dir = world.config(64).spool_dir("ai-studio");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stored.json");
        let original = prompt("synthetic watched reserved version");
        let hash = hashed_blobs::Hash::of(&original);
        std::fs::write(&path, &original).unwrap();
        {
            let (mut engine, _lock) = engine(&world, 64);
            fault::arm(point, "", Action::Fail, 0, 1);
            let pass = engine.converge().unwrap();
            assert!(!pass.errors.is_empty(), "{point}");
            assert_eq!(world.count("SELECT count(*) AS n FROM a.extent"), 1);
            assert_eq!(engine.blobs.contains(&hash), point == "import.installed");
        }
        std::fs::remove_file(&path).unwrap();
        let (mut engine, _lock) = engine(&world, 64);
        let pass = engine.converge().unwrap();
        assert!(pass.errors.is_empty(), "{point}: {:?}", pass.errors);
        if point == "import.install" {
            assert_eq!(pass.unfinished_inputs.len(), 1);
            assert_eq!(world.count("SELECT count(*) AS n FROM message"), 0);
        } else {
            assert!(pass.unfinished_inputs.is_empty());
            assert_eq!(world.count("SELECT count(*) AS n FROM message"), 1);
        }
    }
}
