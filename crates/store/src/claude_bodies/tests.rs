use super::*;
use std::os::unix::fs::PermissionsExt;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
fn observation(id: &str, seen_at: i64) -> ImportObservation<'_> {
    ImportObservation {
        id,
        machine: "test",
        host: "test",
        path: "copy:synthetic",
        seen_at,
    }
}
#[test]
fn body_open_refuses_a_symlink_in_any_component() {
    use std::os::unix::fs::symlink;
    for component in ["session", "tool-results", "result.txt"] {
        let root = tempfile::tempdir().unwrap();
        let main = root.path().join("session.jsonl");
        let directory = root.path().join("session/tool-results");
        std::fs::create_dir_all(&directory).unwrap();
        let body = directory.join("result.txt");
        std::fs::write(&body, b"outside synthetic body").unwrap();
        let target = match component {
            "session" => root.path().join("session"),
            "tool-results" => directory.clone(),
            _ => body.clone(),
        };
        let outside = root.path().join("outside");
        std::fs::rename(&target, &outside).unwrap();
        symlink(&outside, &target).unwrap();
        assert!(open_body(&main, &body).is_err(), "{component}");
    }
}
#[test]
fn body_open_follows_a_base_symlink_and_reports_missing_files() {
    use std::io::Read;
    use std::os::unix::fs::symlink;
    let root = tempfile::tempdir().unwrap();
    let base = root.path().join("base");
    std::fs::create_dir(&base).unwrap();
    let configured = root.path().join("configured");
    symlink(&base, &configured).unwrap();
    let main = configured.join("session.jsonl");
    let body = configured.join("session/tool-results/result.txt");
    assert!(open_body(&main, &body).unwrap().is_none());
    std::fs::create_dir(base.join("session")).unwrap();
    assert!(open_body(&main, &body).unwrap().is_none());
    std::fs::create_dir(base.join("session/tool-results")).unwrap();
    assert!(open_body(&main, &body).unwrap().is_none());
    std::fs::write(&body, b"inside configured root").unwrap();
    let mut bytes = Vec::new();
    open_body(&main, &body)
        .unwrap()
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    assert_eq!(bytes, b"inside configured root");
}
#[test]
fn latest_body_wins_and_survives_file_deletion() {
    let root = tempfile::tempdir().unwrap();
    let archive = root.path().join("archive");
    let main_path = root.path().join("session.jsonl");
    let directory = root.path().join("session/tool-results");
    std::fs::create_dir_all(&directory).unwrap();
    let provider = directory.join("result.txt");
    let mut writer = Writer::open(&archive).unwrap();
    let stop = AtomicBool::new(false);
    for (request, bytes) in [
        ("a", b"first".as_slice()),
        ("b", b"second"),
        ("c", b"first"),
    ] {
        std::fs::write(&provider, bytes).unwrap();
        writer
            .acquire_claude_body(
                "main",
                &main_path,
                &provider,
                &observation(request, 1),
                &stop,
                None,
            )
            .unwrap();
        let pinned = manifest(&writer.archive, &archive.join("archive.db"), "main").unwrap();
        assert_eq!(pinned.len(), 1);
        assert_eq!(
            writer
                .read_handle()
                .unwrap()
                .claude_body(&pinned[0])
                .unwrap(),
            bytes
        );
    }
    let pinned = manifest(&writer.archive, &archive.join("archive.db"), "main")
        .unwrap()
        .pop()
        .unwrap();
    std::fs::remove_file(&provider).unwrap();
    writer
        .acquire_claude_body(
            "main",
            &main_path,
            &provider,
            &observation("missing", 1),
            &stop,
            None,
        )
        .unwrap();
    assert_eq!(
        writer.read_handle().unwrap().claude_body(&pinned).unwrap(),
        b"first"
    );
    writer
        .archive
        .execute(
            "UPDATE blob SET state='deleting' WHERE hash=?1",
            [pinned.blob.bytes().as_slice()],
        )
        .unwrap();
    assert!(writer.read_handle().unwrap().claude_body(&pinned).is_err());
}
#[test]
fn unreadable_body_or_excision_is_an_error() {
    let root = tempfile::tempdir().unwrap();
    let main_path = root.path().join("session.jsonl");
    let directory = root.path().join("session/tool-results");
    std::fs::create_dir_all(&directory).unwrap();
    let provider = directory.join("result.txt");
    let mut writer = Writer::open(&root.path().join("archive")).unwrap();
    let observation = observation("a", 1);
    std::fs::write(&provider, b"private synthetic body").unwrap();
    std::fs::set_permissions(&provider, std::fs::Permissions::from_mode(0)).unwrap();
    assert!(
        writer
            .acquire_claude_body(
                "main",
                &main_path,
                &provider,
                &observation,
                &AtomicBool::new(false),
                None
            )
            .is_err()
    );
    std::fs::set_permissions(&provider, std::fs::Permissions::from_mode(0o600)).unwrap();
    writer.archive.execute("INSERT INTO excision(id,scope,created_at,local_seq) VALUES('x','session:synthetic',1,1)",[]).unwrap();
    assert!(matches!(
        writer.acquire_claude_body(
            "main",
            &main_path,
            &provider,
            &observation,
            &AtomicBool::new(false),
            None
        ),
        Err(Error::ExcisionCheckRequired)
    ));
}

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
fn manifest_matches_only_its_main_and_uses_the_index() {
    let root = tempfile::tempdir().unwrap();
    let archive = root.path().join("archive");
    let main_path = root.path().join("session.jsonl");
    let directory = root.path().join("session/tool-results");
    std::fs::create_dir_all(&directory).unwrap();
    let provider = directory.join("result.txt");
    let mut writer = Writer::open(&archive).unwrap();
    let file = archive.join("archive.db");
    let stop = AtomicBool::new(false);
    // The latest observation wins, whatever its seen_at.
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
                &observation(request, seen_at),
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

    // Add many unrelated inputs, plus NULL, numeric and foreign-format
    // contexts that must not match through the expression index.
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
    // Without the index the answer is the same but the query scans every input.
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
