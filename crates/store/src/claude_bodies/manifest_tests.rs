//! Exercise the native manifest query, including its actual SQLite work.
use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

fn measured(connection: &rusqlite::Connection, file: &Path, main: &str) -> (Vec<Companion>, usize) {
    let steps = Arc::new(AtomicUsize::new(0));
    let counter = steps.clone();
    connection
        .progress_handler(
            1,
            Some(move || {
                counter.fetch_add(1, Ordering::Relaxed);
                false
            }),
        )
        .unwrap();
    let result = manifest(connection, file, main);
    connection
        .progress_handler(0, None::<fn() -> bool>)
        .unwrap();
    (result.unwrap(), steps.load(Ordering::Relaxed))
}

#[test]
fn configured_claude_manifest_preserves_exact_ownership_without_population_scan() {
    let root = tempfile::tempdir().unwrap();
    let archive = root.path().join("archive");
    let main_path = root.path().join("session.jsonl");
    let directory = root.path().join("session/tool-results");
    std::fs::create_dir_all(&directory).unwrap();
    let provider = directory.join("result.txt");
    let mut writer = Writer::open(&archive).unwrap();
    let file = archive.join("archive.db");
    let stop = AtomicBool::new(false);
    // Repeated content still uses the latest local observation, not its timestamp.
    for (request, main, bytes, seen_at) in [
        ("a", "main", b"first".as_slice(), 3),
        ("b", "main", b"second".as_slice(), 2),
        ("c", "main", b"first".as_slice(), 1),
        ("foreign", "main-other", b"foreign".as_slice(), 4),
    ] {
        std::fs::write(&provider, bytes).unwrap();
        writer
            .acquire_claude_body(
                main,
                &main_path,
                &provider,
                &ImportObservation {
                    id: request,
                    machine: "test",
                    host: "test",
                    path: "copy:synthetic",
                    seen_at,
                },
                &stop,
                None,
            )
            .unwrap();
    }
    let names = ["main", "main-other", "absent", "1", ""];
    let before: Vec<_> = names
        .iter()
        .map(|main| measured(&writer.archive, &file, main))
        .collect();
    assert_eq!(before[0].0.len(), 1);
    assert_eq!(before[1].0.len(), 1);
    assert_ne!(before[0].0, before[1].0);
    assert_eq!(
        writer
            .read_handle()
            .unwrap()
            .claude_body(&before[0].0[0])
            .unwrap(),
        b"first"
    );
    assert!(before[2..].iter().all(|(rows, _)| rows.is_empty()));

    // Metadata-only synthetic population isolates lookup work from acquisition IO.
    // NULL observed contexts, unrelated formats and non-text JSON values must not
    // become matching companion ownership through the expression index.
    writer
        .archive
        .execute_batch(
            r#"BEGIN;
        WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<5000)
        INSERT INTO input(id,format) SELECT 'unrelated-'||x,'claude-code-jsonl' FROM n;
        INSERT INTO input(id,format) VALUES('null-companion','claude-code-tool-result');
        INSERT INTO input(id,format,original_hash,context_json) VALUES
          ('numeric-companion','claude-code-tool-result',zeroblob(32),'[1]'),
          ('unknown-format','future-provider',zeroblob(32),'["main"]'),
          ('null-owner','claude-code-tool-result',zeroblob(32),'[null]');
        INSERT INTO revision(input,number,produced,chain,length,local_seq)
          SELECT id,1,'first',1,0,1 FROM input WHERE id LIKE 'unrelated-%'
            OR id IN ('null-companion','numeric-companion','unknown-format','null-owner');
        COMMIT;"#,
        )
        .unwrap();
    for (main, (expected, old_steps)) in names.iter().zip(&before) {
        let (actual, steps) = measured(&writer.archive, &file, main);
        assert_eq!(&actual, expected, "exact ownership {main:?}");
        assert!(
            steps <= old_steps + 64,
            "{main:?}: {old_steps} -> {steps} VM steps"
        );
    }
    // Damaging control: the unchanged native query still returns the same answer
    // without the index, but its actual SQLite work exposes the population scan.
    writer
        .archive
        .execute_batch("DROP INDEX input_claude_tool_result_main")
        .unwrap();
    let (actual, steps) = measured(&writer.archive, &file, "absent");
    assert_eq!(actual, before[2].0);
    assert!(
        steps > 10_000,
        "missing-index control did not expose scan: {steps}"
    );
}
