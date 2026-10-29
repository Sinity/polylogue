use super::*;

fn cache(path: &std::path::Path) -> Connection {
    let index = Connection::open(path).unwrap();
    index.execute_batch(crate::schema::INDEX).unwrap();
    index
        .execute_batch(&crate::source::schema().unwrap().ddl)
        .unwrap();
    index.execute_batch(SCHEMA).unwrap();
    index
}

fn input(position: i64, id: &str) -> Confirmed {
    Confirmed {
        position,
        output_index: 0,
        event: format!("event-{id}"),
        change: ConfirmedChange::Live(Product::Input {
            rows: crate::input_rows::InputRows {
                input: id.into(),
                format: "synthetic".into(),
                revision: None,
                extent: None,
                observation: None,
            },
            material: None,
        }),
    }
}
fn barrier(position: i64) -> Confirmed {
    Confirmed {
        position,
        output_index: 0,
        event: String::new(),
        change: ConfirmedChange::PositionComplete,
    }
}

#[test]
fn staged_pages_survive_restart_and_a_cumulative_barrier_commits_them_atomically() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.db");
    let mut index = cache(&path);
    let archive = crate::Archive::on_sinex(&dir.path().join("cache"), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    assert_eq!(
        apply(&mut index, &resolver, &[input(2, "first")]).unwrap(),
        0
    );
    assert_eq!(checkpoint(&index).unwrap(), None);
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM source_input", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    drop(index);
    let mut index = Connection::open(&path).unwrap();
    index.execute_batch("CREATE TRIGGER stop_projection BEFORE INSERT ON source_input WHEN new.id='second' BEGIN SELECT raise(ABORT,'stop'); END").unwrap();
    assert!(apply(&mut index, &resolver, &[input(4, "second"), barrier(5)]).is_err());
    assert_eq!(checkpoint(&index).unwrap(), None);
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM source_input", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM primary_fact", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    index.execute_batch("DROP TRIGGER stop_projection").unwrap();
    assert_eq!(apply(&mut index, &resolver, &[]).unwrap(), 5);
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM source_input", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        2
    );
    assert_eq!(
        apply(&mut index, &resolver, &[input(2, "first"), barrier(5)]).unwrap(),
        5
    );
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM primary_fact", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        2
    );
    assert!(!archive.root.join("archive.db").exists());
}

#[test]
fn an_empty_confirmed_database_is_known_and_zero_data_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    assert_eq!(checkpoint(&index).unwrap(), None);
    let archive = crate::Archive::on_sinex(&dir.path().join("cache"), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(&mut index, &resolver, &[barrier(0)]).unwrap();
    assert_eq!(checkpoint(&index).unwrap(), Some(0));
    assert!(apply(&mut index, &resolver, &[input(0, "invalid")]).is_err());
    assert_eq!(checkpoint(&index).unwrap(), Some(0));
}

#[test]
fn source_lifecycle_preserves_independent_frozen_authored_effects() {
    use polylogue_model::user::{UserRequest, UserWrite};
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    let archive = crate::Archive::on_sinex(&dir.path().join("cache"), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    let request = UserRequest {
        id: "setting-request".into(),
        machine: "machine".into(),
        at: polylogue_model::Timestamp::from_unix_ms(1).unwrap(),
        write: UserWrite::SetSetting {
            key: "saved-key".into(),
            value: "saved-value".into(),
            author: polylogue_model::Author::Operator,
            machine: None,
        },
    };
    let prepared =
        crate::primary::authored::prepare(&index, &request, |_| Ok(()), |_| Ok(())).unwrap();
    apply(
        &mut index,
        &resolver,
        &[
            input(1, "source"),
            Confirmed {
                position: 2,
                output_index: 0,
                event: "authored".into(),
                change: ConfirmedChange::Live(Product::Request(prepared)),
            },
            barrier(2),
        ],
    )
    .unwrap();
    for (position, change, present) in [
        (3, ConfirmedChange::Archived, 0),
        (4, ConfirmedChange::Restored, 1),
        (5, ConfirmedChange::Tombstoned, 0),
    ] {
        apply(
            &mut index,
            &resolver,
            &[
                Confirmed {
                    position,
                    output_index: 0,
                    event: "event-source".into(),
                    change,
                },
                barrier(position),
            ],
        )
        .unwrap();
        assert_eq!(
            index
                .query_row("SELECT count(*) FROM source_input", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            present
        );
        assert_eq!(
            index
                .query_row(
                    "SELECT value FROM source_setting WHERE key='saved-key'",
                    [],
                    |row| row.get::<_, String>(0)
                )
                .unwrap(),
            "saved-value"
        );
    }
    assert!(
        apply(
            &mut index,
            &resolver,
            &[
                Confirmed {
                    position: 6,
                    output_index: 0,
                    event: "event-source".into(),
                    change: ConfirmedChange::Restored
                },
                barrier(6)
            ]
        )
        .is_err()
    );
    assert_eq!(position(&index).unwrap(), 5);
    let body: Option<Vec<u8>> = index
        .query_row(
            "SELECT product FROM primary_fact WHERE event='event-source'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(body, None);
}
