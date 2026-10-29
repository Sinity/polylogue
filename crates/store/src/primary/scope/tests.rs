use super::*;
use crate::primary::{Confirmed, ConfirmedChange, Product, projection};

fn cache(archive: &Archive) -> Connection {
    let index = Connection::open(archive.index_db()).unwrap();
    index.pragma_update(None, "journal_mode", "wal").unwrap();
    index.pragma_update(None, "foreign_keys", true).unwrap();
    index.execute_batch(crate::schema::INDEX).unwrap();
    index.execute_batch(crate::variants::SCHEMA).unwrap();
    index
        .execute_batch(&crate::source::schema().unwrap().ddl)
        .unwrap();
    index.execute_batch(projection::SCHEMA).unwrap();
    crate::archive::generation::create(&index).unwrap();
    index
        .execute(
            "INSERT INTO meta VALUES ('primary_cache','persistent-cache')",
            [],
        )
        .unwrap();
    index
}

fn input(position: i64, id: &str) -> Confirmed {
    Confirmed {
        position,
        event: format!("event-{id}"),
        output_index: 0,
        change: ConfirmedChange::Live(Box::new(Product::Input {
            rows: crate::input_rows::InputRows {
                input: id.into(),
                format: "synthetic".into(),
                revision: None,
                extent: None,
                observation: None,
            },
            material: None,
        })),
    }
}
fn barrier(position: i64) -> Confirmed {
    Confirmed {
        position,
        event: String::new(),
        output_index: 0,
        change: ConfirmedChange::PositionComplete,
    }
}

#[test]
fn narrowing_and_widening_reset_cached_payloads_and_require_a_new_barrier() {
    let root = tempfile::tempdir().unwrap();
    let archive = Archive::on_sinex(root.path(), &root.path().join("sinex"));
    let mut index = cache(&archive);
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    std::fs::write(
        root.path().join("embeddings.db"),
        b"paid content hash vectors",
    )
    .unwrap();
    let private = root.path().join("sinex/producer-spool");
    std::fs::create_dir_all(&private).unwrap();
    std::fs::write(
        private.join("pending"),
        b"private authored and captured intent",
    )
    .unwrap();
    projection::apply(
        &mut index,
        &resolver,
        &[input(1, "private"), barrier(1), input(2, "uncommitted")],
    )
    .unwrap();
    index
        .execute(
            "INSERT INTO text_fts(rowid,body) VALUES (1,'private words')",
            [],
        )
        .unwrap();
    let before: String = index
        .query_row("SELECT value FROM meta WHERE key='generation'", [], |row| {
            row.get(0)
        })
        .unwrap();
    let token = archive.read_scope.token().unwrap();
    assert!(reset(&mut index, &archive, &resolver, "narrow").unwrap());
    assert!(matches!(token.check(), Err(StoreError::ScopeChanged)));
    assert_eq!(projection::checkpoint(&index).unwrap(), None);
    for table in [
        "primary_fact",
        "primary_pending",
        "source_input",
        "source_material",
    ] {
        assert_eq!(
            index
                .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }
    assert_eq!(
        index
            .query_row(
                "SELECT count(*) FROM text_fts WHERE text_fts MATCH 'private'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
    assert_eq!(
        super::super::cache_identity(&index).unwrap(),
        "persistent-cache"
    );
    assert_eq!(current(&index).unwrap().as_deref(), Some("narrow"));
    assert_ne!(
        index
            .query_row("SELECT value FROM meta WHERE key='generation'", [], |row| {
                row.get::<_, String>(0)
            })
            .unwrap(),
        before
    );
    assert!(!reset(&mut index, &archive, &resolver, "narrow").unwrap());
    projection::apply(&mut index, &resolver, &[barrier(0)]).unwrap();
    assert_eq!(projection::checkpoint(&index).unwrap(), Some(0));
    drop(index);
    let mut index = Connection::open(archive.index_db()).unwrap();
    assert_eq!(current(&index).unwrap().as_deref(), Some("narrow"));
    assert!(reset(&mut index, &archive, &resolver, "wide").unwrap());
    assert_eq!(projection::checkpoint(&index).unwrap(), None);
    projection::apply(&mut index, &resolver, &[input(1, "private"), barrier(1)]).unwrap();
    assert_eq!(
        index
            .query_row("SELECT id FROM source_input", [], |row| row
                .get::<_, String>(0))
            .unwrap(),
        "private"
    );
    assert_eq!(
        std::fs::read(root.path().join("embeddings.db")).unwrap(),
        b"paid content hash vectors"
    );
    assert_eq!(
        std::fs::read(private.join("pending")).unwrap(),
        b"private authored and captured intent"
    );
}

#[test]
fn a_failed_scope_transaction_keeps_readers_blocked_until_retry_commits() {
    let root = tempfile::tempdir().unwrap();
    let archive = Archive::on_sinex(root.path(), &root.path().join("sinex"));
    let mut index = cache(&archive);
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    projection::apply(&mut index, &resolver, &[input(1, "private"), barrier(1)]).unwrap();
    index.execute_batch("CREATE TRIGGER stop_reset BEFORE DELETE ON source_input BEGIN SELECT raise(ABORT,'stop'); END").unwrap();
    assert!(reset(&mut index, &archive, &resolver, "changed").is_err());
    assert!(matches!(
        archive.read_scope.token(),
        Err(StoreError::ScopeChanged)
    ));
    assert_eq!(current(&index).unwrap(), None);
    assert_eq!(projection::checkpoint(&index).unwrap(), Some(1));
    index.execute_batch("DROP TRIGGER stop_reset").unwrap();
    assert!(reset(&mut index, &archive, &resolver, "changed").unwrap());
    assert!(archive.read_scope.token().is_ok());
    assert_eq!(projection::checkpoint(&index).unwrap(), None);
}
