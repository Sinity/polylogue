use super::*;
use polylogue_store::fault::{self, Action};

#[test]
fn an_interrupted_import_retains_exact_committed_bytes_across_restart() {
    for point in ["segment.after_sync", "segment.after_commit"] {
        fault::clear();
        let home = ClaudeHome::new();
        let path = home.dir.path().join("manual.jsonl");
        let original = claude_user(
            "stored-user",
            "stored-session",
            &format!("synthetic capture at {point}"),
        );
        std::fs::write(&path, &original).unwrap();
        let hash = hashed_blobs::Hash::of(original.as_bytes());
        let committed = point == "segment.after_commit";
        {
            let (mut engine, _lock) = engine(&home, 64);
            fault::arm(point, "", Action::Fail, 0, 1);
            let report = engine
                .import_with_options(
                    std::slice::from_ref(&path),
                    &ImportOptions {
                        format: Some("claude-code".into()),
                        explain: false,
                    },
                )
                .unwrap();
            assert!(!report.pass.errors.is_empty(), "{point}");
            assert_catalogued(&engine, &hash, committed);
            assert_eq!(
                home.count("SELECT count(*) AS n FROM a.revision"),
                i64::from(committed)
            );
            fault::clear();
        }
        std::fs::write(
            &path,
            claude_user(
                "replacement-user",
                "replacement-session",
                "synthetic replacement",
            ),
        )
        .unwrap();
        let (mut engine, _lock) = engine(&home, 64);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        if committed {
            assert_eq!(
                home.texts("stored-session"),
                [format!("synthetic capture at {point}")]
            );
        } else {
            assert_eq!(home.count("SELECT count(*) AS n FROM message"), 0);
        }
        let replacement = engine
            .import_with_options(
                std::slice::from_ref(&path),
                &ImportOptions {
                    format: Some("claude-code".into()),
                    explain: false,
                },
            )
            .unwrap();
        assert!(
            replacement.pass.errors.is_empty(),
            "{:?}",
            replacement.pass.errors
        );
        assert_eq!(home.texts("replacement-session"), ["synthetic replacement"]);
        assert_catalogued(&engine, &hash, committed);
    }
    fault::clear();
}
