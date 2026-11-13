use super::*;
use crate::archive::{cache, open_read, open_write, restart_full_log};

#[test]
fn fresh_context_leaves_checkpoint_free_while_resumed_context_keeps_its_snapshot() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    let path = archive.index_db();
    let writer = open_write(&path, cache::SETUP).unwrap();
    writer.pragma_update(None, "wal_autocheckpoint", 0).unwrap();
    writer.execute_batch("INSERT INTO input_ref(pk,id,format) VALUES(1,'synthetic-continuation','claude-code');
        INSERT INTO parse_state(input_pk,chain,status,parsed_upto,reading_version) VALUES(1,3,'parsing',2,'synthetic=1');
        INSERT INTO parser_context_state VALUES(1,3,'synthetic=1',2);
        INSERT INTO meta VALUES('checkpoint-control','initial');").unwrap();
    let key = Key::Alias {
        namespace: String::new(),
        uuid: "event".into(),
    };
    let sql_key = serde_json::to_string(&key).unwrap();
    writer
        .execute(
            "INSERT INTO parser_context VALUES(1,3,'synthetic=1',2,?1,?2,NULL)",
            params![
                sql_key,
                crate::encoding::pack_json(&Reference::Alias { target: None }).unwrap()
            ],
        )
        .unwrap();
    let restart = || restart_full_log(&writer, &path, 0, std::time::Duration::ZERO).unwrap();
    assert!(restart());

    // An independent real reader proves that the checkpoint control detects
    // an admitted snapshot, rather than succeeding regardless of readers.
    let held = open_read(&path).unwrap();
    held.execute_batch("BEGIN").unwrap();
    held.query_row("SELECT count(*) FROM input_ref", [], |row| {
        row.get::<_, i64>(0)
    })
    .unwrap();
    writer
        .execute(
            "UPDATE meta SET value='held' WHERE key='checkpoint-control'",
            [],
        )
        .unwrap();
    assert!(!restart());
    drop(held);
    assert!(restart());

    // Fresh parsing ignores admitted context even when its chain is known.
    // Its private values and source binding survive a complete WAL restart.
    for chain in [None, Some(3)] {
        let mut fresh =
            Overlay::open(&archive, "synthetic-continuation", chain, "synthetic=1", 0).unwrap();
        assert!(fresh.get(&key).unwrap().is_none());
        fresh
            .check_source("synthetic-continuation", chain, "synthetic=1", 0)
            .unwrap();
        assert!(
            fresh
                .check_source("synthetic-continuation", Some(4), "synthetic=1", 0)
                .is_err()
        );
        fresh
            .put(key.clone(), Value::Alias { target: None })
            .unwrap();
        writer
            .execute(
                "UPDATE meta SET value=value||'x' WHERE key='checkpoint-control'",
                [],
            )
            .unwrap();
        assert!(restart());
        assert_eq!(
            std::fs::metadata(path.with_extension("db-wal"))
                .unwrap()
                .len(),
            0
        );
        assert!(matches!(
            fresh.get(&key).unwrap(),
            Some(Value::Alias { target: None })
        ));
        let delta = fresh.checkpoint_delta(1).unwrap();
        assert_eq!((delta.base, delta.through, delta.entries.len()), (0, 1, 1));
        assert_eq!(delta.entries[0].key, key);
        fresh
            .check_source("synthetic-continuation", chain, "synthetic=1", 1)
            .unwrap();
    }

    // A resumed parser must retain the exact admitted view it selected,
    // including when a later writer changes that same mapping.
    let mut resumed = Overlay::open(
        &archive,
        "synthetic-continuation",
        Some(3),
        "synthetic=1",
        2,
    )
    .unwrap();
    assert!(matches!(
        resumed.get(&key).unwrap(),
        Some(Value::Alias { target: None })
    ));
    let later: Reference =
        serde_json::from_value(serde_json::json!({"kind":"alias","target":"claude-code:later"}))
            .unwrap();
    writer
        .execute(
            "UPDATE parser_context SET value=?1 WHERE key=?2",
            params![crate::encoding::pack_json(&later).unwrap(), sql_key],
        )
        .unwrap();
    assert!(!restart());
    assert!(matches!(
        resumed.get(&key).unwrap(),
        Some(Value::Alias { target: None })
    ));
    let delta = resumed.checkpoint_delta(3).unwrap();
    assert_eq!((delta.base, delta.through, delta.entries.len()), (2, 3, 0));
    drop(resumed);
    assert!(restart());
    let mut reopened = Overlay::open(
        &archive,
        "synthetic-continuation",
        Some(3),
        "synthetic=1",
        2,
    )
    .unwrap();
    let Some(Value::Alias {
        target: Some(target),
    }) = reopened.get(&key).unwrap()
    else {
        panic!("later admitted alias is absent")
    };
    assert_eq!(target.to_string(), "claude-code:later");
}
