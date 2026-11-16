use super::*;
use crate::archive::{cache, open_read, open_write, restart_full_log};

#[test]
fn fresh_context_leaves_checkpoint_free_while_resumed_context_keeps_its_snapshot() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    // A resumed context's snapshot and the checkpoints it holds back are
    // those of the installed index's WAL (docs/architecture.md, Durability).
    crate::Writer::start(&archive).unwrap().derive().unwrap();
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
    writer
        .execute(
            "INSERT INTO parser_context_alias VALUES(1,3,'','event',NULL)",
            [],
        )
        .unwrap();
    let restart = || restart_full_log(&writer, &path, 0, std::time::Duration::ZERO).unwrap();
    assert!(restart());

    // An independent real reader shows that the checkpoint control detects
    // an committed snapshot, rather than succeeding regardless of readers.
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

    // Fresh parsing ignores committed context even when its chain is known.
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

    // A resumed parser must keep the committed view it selected,
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
    writer
        .execute(
            "UPDATE parser_context_alias SET target='claude-code:later' WHERE uuid='event'",
            [],
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
        panic!("later committed alias is absent")
    };
    assert_eq!(target.to_string(), "claude-code:later");
}

#[test]
fn cleared_parser_state_stays_cleared_after_checkpoint_and_restart() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    crate::Writer::start(&archive).unwrap().derive().unwrap();
    let mut writer = open_write(&archive.index_db(), cache::SETUP).unwrap();
    writer.execute_batch("INSERT INTO input_ref(pk,id,format) VALUES(1,'state-clear','codex');
        INSERT INTO parse_state(input_pk,chain,status,parsed_upto,reading_version) VALUES(1,1,'parsing',0,'synthetic=1');").unwrap();
    let key = |namespace: &str, key: &str| Key::State {
        namespace: namespace.into(),
        key: key.into(),
    };
    let put = |context: &mut Overlay, namespace: &str, key_name: &str, value: u64| {
        context
            .put(
                key(namespace, key_name),
                Value::State {
                    value: value.into(),
                },
            )
            .unwrap();
    };
    let commit = |writer: &mut Connection, delta: Delta| {
        let mut step = crate::ParseStep::new(
            Default::default(),
            Default::default(),
            delta.base..delta.through,
            false,
        );
        step.reading_version = "synthetic=1".into();
        step.checkpoint = Some(polylogue_model::batch::ParserCheckpoint::new(
            "synthetic=1".into(),
            delta.through,
            Some(serde_json::json!({"_polylogue_context":delta})),
        ));
        let tx = writer.transaction().unwrap();
        apply(&tx, 1, 1, &step, &Default::default()).unwrap();
        tx.execute(
            "UPDATE parse_state SET parsed_upto=?1 WHERE input_pk=1 AND chain=1",
            [step.parsed_upto],
        )
        .unwrap();
        tx.commit().unwrap();
    };
    let mut context = Overlay::open(&archive, "state-clear", Some(1), "synthetic=1", 0).unwrap();
    put(&mut context, "calls", "old", 1);
    put(&mut context, "names", "keep", 2);
    commit(&mut writer, context.checkpoint_delta(1).unwrap());
    drop(context);
    let mut context = Overlay::open(&archive, "state-clear", Some(1), "synthetic=1", 1).unwrap();
    context.clear_state("calls").unwrap();
    assert!(context.get(&key("calls", "old")).unwrap().is_none());
    put(&mut context, "calls", "new", 3);
    commit(&mut writer, context.checkpoint_delta(2).unwrap());
    // A private multi-range reading keeps the clear mask after sealing a slice.
    assert!(context.get(&key("calls", "old")).unwrap().is_none());
    assert!(
        context
            .checkpoint_delta(3)
            .unwrap()
            .cleared_states
            .is_empty()
    );
    drop(context);
    let mut context = Overlay::open(&archive, "state-clear", Some(1), "synthetic=1", 2).unwrap();
    assert!(context.get(&key("calls", "old")).unwrap().is_none());
    assert!(
        matches!(context.get(&key("calls","new")).unwrap(),Some(Value::State { value }) if value==3)
    );
    assert!(
        matches!(context.get(&key("names","keep")).unwrap(),Some(Value::State { value }) if value==2)
    );
    context
        .put(
            key("calls", "new"),
            Value::State {
                value: serde_json::Value::Null,
            },
        )
        .unwrap();
    assert!(context.get(&key("calls", "new")).unwrap().is_none());
    commit(&mut writer, context.checkpoint_delta(3).unwrap());
    drop(context);
    let mut context = Overlay::open(&archive, "state-clear", Some(1), "synthetic=1", 3).unwrap();
    assert!(context.get(&key("calls", "new")).unwrap().is_none());
    let count: i64 = writer
        .query_row("SELECT count(*) FROM parser_context_value", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(count, 1, "only the uncleared names namespace remains");
}
