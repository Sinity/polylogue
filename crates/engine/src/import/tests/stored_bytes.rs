use super::*;
use polylogue_testkit::{ClaudeHome, claude_user};

#[cfg(feature = "faults")]
mod pending;
mod pool;
#[cfg(feature = "faults")]
mod watched;

#[test]
fn a_refused_import_installs_no_file() {
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
                        format: Some("claude-code".into()),
                        explain: false,
                    },
                )
                .unwrap();
            assert!(!rejected.pass.errors.is_empty());
            assert!(
                !engine.blobs.contains(&hash),
                "a rejected first import must not write a blob file without an archive row"
            );
            for table in ["input", "revision", "extent", "blob", "observation"] {
                assert_eq!(
                    home.count(&format!("SELECT count(*) AS n FROM a.{table}")),
                    0,
                    "{table}"
                );
            }
        }
        // A trigger is a table change the archive refuses at open.
        rusqlite::Connection::open(home.root.join("archive.db"))
            .unwrap()
            .execute_batch("DROP TRIGGER reject_first_import")
            .unwrap();
        let (mut engine, _lock) = engine(&home, 64);
        assert!(engine.run_pass().unwrap().errors.is_empty());
        assert!(!engine.blobs.contains(&hash));
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
                    format: Some("claude-code".into()),
                    explain: false,
                },
            )
            .unwrap();
        assert!(retry.pass.errors.is_empty(), "{:?}", retry.pass.errors);
        assert_eq!(engine.blobs.contains(&hash), !changed);
    }
}

#[test]
fn equal_inline_bytes_install_no_file_copy() {
    let home = ClaudeHome::new();
    let original = claude_user("inline-user", "inline-session", "synthetic inline content");
    home.append("inline-session.jsonl", &original);
    let path = home.dir.path().join("manual-inline.jsonl");
    std::fs::write(&path, &original).unwrap();
    let hash = hashed_blobs::Hash::of(original.as_bytes());
    let (mut engine, _lock) = engine(&home, 1024);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    assert!(!engine.blobs.contains(&hash));
    let report = engine
        .import_with_options(
            &[path],
            &ImportOptions {
                format: Some("claude-code".into()),
                explain: false,
            },
        )
        .unwrap();
    assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    assert!(
        !engine.blobs.contains(&hash),
        "inline deduplication must discard its private file stage"
    );
    assert_eq!(
        home.count("SELECT count(*) AS n FROM a.blob WHERE place='inline'"),
        1
    );
    assert_eq!(home.count("SELECT count(*) AS n FROM a.extent"), 2);
    assert_eq!(home.texts("inline-session"), ["synthetic inline content"]);
}
