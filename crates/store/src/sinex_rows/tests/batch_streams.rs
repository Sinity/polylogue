use super::*;
use crate::{
    Archive,
    archive::{cache, open_write},
};

#[test]
fn publishing_reads_one_sqlite_view_until_the_pass_ends() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    conn.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,commit_seq) VALUES ('one','one','note','before','operator','test',1,1)",[]).unwrap();
    let published = Published::open(dir.path()).unwrap();
    let before = units(&published.conn, 0).unwrap();
    assert_eq!(before.len(), 1);
    conn.execute_batch("BEGIN; UPDATE note SET value='after' WHERE id='one';
        INSERT INTO note(id,root,kind,value,author,machine,created_at,commit_seq) VALUES ('two','two','note','second','operator','test',1,2); COMMIT;").unwrap();
    assert_eq!(
        units(&published.conn, 0).unwrap(),
        before,
        "a pass must not splice a later source commit"
    );
    let batch = published.read(&before[0]).unwrap();
    let row: Value = serde_json::from_slice(
        batch
            .contents()
            .split(|b| *b == b'\n')
            .find(|line| !line.is_empty())
            .unwrap(),
    )
    .unwrap();
    assert_eq!(row["row"]["value"], "before");
    let next = Published::open(dir.path()).unwrap();
    assert_eq!(units(&next.conn, 0).unwrap().len(), 2);
}

#[test]
fn row_batch_bytes_keep_the_existing_jsonl_representation() {
    let source = archive();
    source.execute_batch("INSERT INTO note VALUES ('n1','lesson',x'00ff',7); INSERT INTO note_target VALUES ('n1','session:claude-code:s1');").unwrap();
    let batch = read(
        &source,
        &Unit {
            commit_seq: 7,
            kind: "note".into(),
            key: "n1".into(),
        },
    )
    .unwrap();
    let expected = concat!(
        r#"{"row":{"body":{"hex":"00ff"},"commit_seq":7,"id":"n1","kind":"lesson"},"table":"note"}"#,
        "\n",
        r#"{"row":{"note_id":"n1","ref":"session:claude-code:s1"},"table":"note_target"}"#,
        "\n"
    );
    assert_eq!(batch.contents(), expected.as_bytes());
}

#[test]
fn an_unknown_unit_kind_refuses_instead_of_becoming_an_empty_delivery() {
    let source = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&source).unwrap();
    source
        .execute_batch(include_str!("../../schema/migrations/005-append-facts.sql"))
        .unwrap();
    assert!(
        matches!(read(&source,&Unit{commit_seq:1,kind:"unknown".into(),key:"one".into()}),
        Err(crate::StoreError::Invalid(message)) if message.contains("unknown publication unit kind"))
    );
}

#[test]
fn a_stored_legacy_batch_keeps_its_original_bytes_and_restores_them() {
    let dir = tempfile::tempdir().unwrap();
    let (_archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let store = hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap();
    let legacy = concat!(
        r#"{"table":"note","row":{"id":"one","root":"one","kind":"note","value":"body","author":"operator","machine":"test","created_at":1,"commit_seq":1}}"#,
        "\n"
    );
    let hash = store.put(legacy.as_bytes(), 1).unwrap();
    let name = format!("blake3:{}", hash.hex());
    let published = Published::open(dir.path()).unwrap();
    assert_eq!(
        file_bytes(published.replacement(&name).unwrap()),
        legacy.as_bytes()
    );
    let selected = select(&[RestoreUnit {
        unit: Unit {
            commit_seq: 1,
            kind: "note".into(),
            key: "one".into(),
        },
        batch: Some(name.clone()),
        bytes: Some(legacy.len() as u64),
        rows: 1,
        blobs: vec![],
        excision: None,
    }])
    .unwrap();
    let mut target = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&target).unwrap();
    target
        .execute_batch(include_str!("../../schema/migrations/005-append-facts.sql"))
        .unwrap();
    assert_eq!(
        restore::install_fixture(&mut target, &selected, None, |_| Ok(Some(
            published.replacement(&name)?
        )))
        .unwrap(),
        1
    );
    assert_eq!(
        target
            .query_row("SELECT value FROM note WHERE id='one'", [], |r| r
                .get::<_, String>(0))
            .unwrap(),
        "body"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn a_many_row_batch_is_written_to_disk_without_collecting_its_body() {
    fn peak_kib() -> u64 {
        std::fs::read_to_string("/proc/self/status")
            .unwrap()
            .lines()
            .find_map(|line| {
                line.strip_prefix("VmHWM:")
                    .map(|v| v.split_whitespace().next().unwrap().parse().unwrap())
            })
            .unwrap()
    }
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let mut conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    conn.pragma_update(None, "cache_size", -1024).unwrap();
    let tx = conn.transaction().unwrap();
    tx.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,commit_seq) VALUES ('one','one','note','body','operator','test',1,1)",[]).unwrap();
    let suffix = "x".repeat(3072);
    {
        let mut insert = tx
            .prepare("INSERT INTO note_target(note_id,ref) VALUES ('one',?1)")
            .unwrap();
        for n in 0..12_000 {
            insert
                .execute([format!("session:neutral:{n:05}:{suffix}")])
                .unwrap();
        }
    }
    tx.commit().unwrap();
    drop(conn);
    let published = Published::open(dir.path()).unwrap();
    let before = peak_kib();
    let batch = published
        .read(&Unit {
            commit_seq: 1,
            kind: "note".into(),
            key: "one".into(),
        })
        .unwrap();
    assert_eq!(batch.rows, 12_001);
    assert!(batch.bytes > 32 * 1024 * 1024);
    assert_eq!(batch.file.metadata().unwrap().len(), batch.bytes);
    let growth = peak_kib().saturating_sub(before);
    eprintln!(
        "12,001 rows / {}bytes: producer high-water growth {growth}KiB",
        batch.bytes
    );
    assert!(growth < 32 * 1024, "whole-batch memory growth {growth}KiB");
}
