use super::*;
use polylogue_testkit::{ClaudeHome, claude_user};

#[cfg(feature = "faults")]
mod pending;
#[cfg(feature = "faults")]
mod watched;

#[test]
fn first_admission_refusal_does_not_publish_unowned_import_bytes() {
    for changed in [false, true] {
        let world = ClaudeHome::new();
        let path = world.dir.path().join("manual.jsonl");
        let original = claude_user("u1", "original-session", "synthetic original acquisition");
        std::fs::write(&path, &original).unwrap();
        let hash = hashed_blobs::Hash::of(original.as_bytes());
        {
            let (mut engine, _lock) = engine(&world, 64);
            let conn = rusqlite::Connection::open(world.root.join("archive.db")).unwrap();
            conn.execute_batch("CREATE TRIGGER reject_first_import BEFORE INSERT ON input BEGIN SELECT RAISE(ABORT,'synthetic first admission refusal'); END;").unwrap();
            let rejected = engine.import(std::slice::from_ref(&path)).unwrap();
            assert!(!rejected.pass.errors.is_empty());
            assert!(
                !engine.blobs.contains(&hash),
                "a rejected first acquisition must not publish a CAS object without canonical ownership"
            );
            for table in ["input", "revision", "extent", "blob", "observation"] {
                assert_eq!(
                    world.count(&format!("SELECT count(*) AS n FROM a.{table}")),
                    0,
                    "{table}"
                );
            }
        }
        let (mut engine, _lock) = engine(&world, 64);
        assert!(engine.converge().unwrap().errors.is_empty());
        assert!(!engine.blobs.contains(&hash));
        if changed {
            std::fs::write(
                &path,
                claude_user(
                    "u2",
                    "replacement-session",
                    "synthetic replacement acquisition",
                ),
            )
            .unwrap();
        }
        let conn = rusqlite::Connection::open(world.root.join("archive.db")).unwrap();
        conn.execute_batch("DROP TRIGGER reject_first_import")
            .unwrap();
        let retry = engine.import(std::slice::from_ref(&path)).unwrap();
        assert!(retry.pass.errors.is_empty(), "{:?}", retry.pass.errors);
        assert_eq!(engine.blobs.contains(&hash), !changed);
        let id = blake3::hash(&std::fs::read(&path).unwrap())
            .to_hex()
            .to_string();
        engine
            .excise(ExciseRequest {
                id: format!("remove-{changed}"),
                scope: ExciseScope::Input(id),
                reason: None,
                dry_run: false,
            })
            .unwrap();
        assert!(
            !engine.blobs.contains(&hash),
            "edited retries must not leave the refused original hidden in CAS"
        );
    }
}

#[test]
fn deduplicating_inline_material_does_not_publish_an_untracked_file_copy() {
    let world = ClaudeHome::new();
    let original = claude_user("inline-user", "inline-session", "synthetic inline content");
    world.append("inline-session.jsonl", &original);
    let path = world.dir.path().join("manual-inline.jsonl");
    std::fs::write(&path, &original).unwrap();
    let hash = hashed_blobs::Hash::of(original.as_bytes());
    let (mut engine, _lock) = engine(&world, 1024);
    assert!(engine.converge().unwrap().errors.is_empty());
    assert!(!engine.blobs.contains(&hash));
    let report = engine.import(&[path]).unwrap();
    assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    assert!(
        !engine.blobs.contains(&hash),
        "inline deduplication must discard its private file stage"
    );
    assert_eq!(
        world.count("SELECT count(*) AS n FROM a.blob WHERE place='inline'"),
        1
    );
    assert_eq!(world.count("SELECT count(*) AS n FROM a.extent"), 2);
    assert_eq!(world.texts("inline-session"), ["synthetic inline content"]);
}
