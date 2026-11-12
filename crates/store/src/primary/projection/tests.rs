use super::*;

fn cache(path: &std::path::Path) -> Connection {
    let index = Connection::open(path).unwrap();
    crate::schema::ensure_primary_index(&index).unwrap();
    index
}

fn input(position: i64, id: &str) -> Confirmed {
    Confirmed {
        native: None,
        position,
        output_index: 0,
        event: format!("event-{id}"),
        change: ConfirmedChange::Live(Box::new(Product::Input {
            rows: Box::new(crate::input_rows::InputRows {
                acquisition: None,
                input: id.into(),
                format: "synthetic".into(),
                revision: None,
                extents: Vec::new(),
                observation: None,
            }),
            material: None,
        })),
    }
}
fn barrier(position: i64) -> Confirmed {
    Confirmed {
        native: None,
        position,
        output_index: 0,
        event: String::new(),
        change: ConfirmedChange::Control(Box::new(crate::primary::ReplayControl::Completion {
            position,
            delivery: None,
        })),
    }
}

#[test]
fn staged_pages_survive_restart_and_a_cumulative_barrier_commits_them_atomically() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.db");
    let mut index = cache(&path);
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
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
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
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
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    let request = UserRequest {
        authorship: polylogue_model::authorship::Authorship {
            author: polylogue_model::Author::Operator,
            via: polylogue_model::authorship::Channel::Cli,
        },
        id: "setting-request".into(),
        machine: "machine".into(),
        at: polylogue_model::Timestamp::from_unix_ms(1).unwrap(),
        write: UserWrite::SetSetting {
            key: "browser.actions".into(),
            value: "on".into(),
            machine: None,
        },
    };
    let prepared =
        crate::primary::authored::prepare(&index, &request, None, None, |_| Ok(()), |_| Ok(()))
            .unwrap();
    apply(
        &mut index,
        &resolver,
        &[
            input(1, "source"),
            Confirmed {
                native: None,
                position: 2,
                output_index: 0,
                event: "authored".into(),
                change: ConfirmedChange::Live(Box::new(Product::Request(prepared))),
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
                    native: None,
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
        let setting: String = index.query_row(
            "SELECT json(value) FROM source_assertion WHERE kind='setting' AND key='browser.actions'",
            [],|row|row.get(0),
        ).unwrap();
        let setting: polylogue_model::assertion::values::Setting =
            serde_json::from_str(&setting).unwrap();
        assert_eq!(
            setting.value,
            polylogue_model::settings::Value::Boolean(true)
        );
    }
    assert!(
        apply(
            &mut index,
            &resolver,
            &[
                Confirmed {
                    native: None,
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

fn setting(index: &Connection, id: &str, key: &str) -> Product {
    use polylogue_model::user::{UserRequest, UserWrite};
    let request = UserRequest {
        authorship: polylogue_model::authorship::Authorship {
            author: polylogue_model::Author::Operator,
            via: polylogue_model::authorship::Channel::Cli,
        },
        id: id.into(),
        machine: "machine".into(),
        at: polylogue_model::Timestamp::from_unix_ms(1).unwrap(),
        write: UserWrite::SetSetting {
            key: key.into(),
            value: "on".into(),
            machine: None,
        },
    };
    Product::Request(
        crate::primary::authored::prepare(index, &request, None, None, |_| Ok(()), |_| Ok(()))
            .unwrap(),
    )
}
fn settings(index: &Connection) -> Vec<String> {
    index
        .prepare("SELECT key FROM source_assertion WHERE kind='setting' ORDER BY key")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

#[test]
fn new_requests_apply_after_the_applied_ones_and_a_lifecycle_change_rebuilds_authored_state() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    let fact = |position, event: &str, product| Confirmed {
        native: None,
        position,
        output_index: 0,
        event: event.into(),
        change: ConfirmedChange::Live(Box::new(product)),
    };
    let first = setting(&index, "first-request", "browser.actions");
    apply(
        &mut index,
        &resolver,
        &[fact(2, "first", first), barrier(2)],
    )
    .unwrap();
    assert_eq!(settings(&index), ["browser.actions"]);
    let second = setting(&index, "second-request", "drive.ai-studio");
    apply(
        &mut index,
        &resolver,
        &[fact(3, "second", second), barrier(3)],
    )
    .unwrap();
    assert_eq!(settings(&index), ["browser.actions", "drive.ai-studio"]);
    apply(
        &mut index,
        &resolver,
        &[
            Confirmed {
                native: None,
                position: 4,
                output_index: 0,
                event: "first".into(),
                change: ConfirmedChange::Archived,
            },
            barrier(4),
        ],
    )
    .unwrap();
    assert_eq!(settings(&index), ["drive.ai-studio"]);
    apply(
        &mut index,
        &resolver,
        &[
            Confirmed {
                native: None,
                position: 5,
                output_index: 0,
                event: "first".into(),
                change: ConfirmedChange::Restored,
            },
            barrier(5),
        ],
    )
    .unwrap();
    assert_eq!(settings(&index), ["browser.actions", "drive.ai-studio"]);
}

#[test]
fn confirmed_slices_keep_full_material_descriptors_without_opening_source_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    let bytes_root = dir.path().join("unavailable-bytes");
    let archive = crate::Archive::on_sinex(dir.path(), &bytes_root);
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    let hash = polylogue_model::BlobHash::of(b"abcd");
    let extent = hashed_blobs::Extent {
        segment: "segments/confirmed.hb".into(),
        offset: 0,
        len: 100,
        decoded_len: 4,
        hash: hashed_blobs::Hash(hash.0),
        kind: hashed_blobs::Kind::Plain,
        index_hash: hashed_blobs::Hash([1; 32]),
        encoded_hash: hashed_blobs::Hash([2; 32]),
    };
    let material = Material {
        id: "shared-material".into(),
        hash,
        length: 4,
        extents: vec![extent.encode()],
    };
    let facts = |position, id: &str, blob_offset| Confirmed {
        native: None,
        position,
        output_index: 0,
        event: format!("event-{id}"),
        change: ConfirmedChange::Live(Box::new(Product::Input {
            rows: Box::new(crate::input_rows::InputRows {
                acquisition: None,
                input: id.into(),
                format: "synthetic".into(),
                observation: None,
                revision: Some(crate::input_rows::Revision {
                    number: 1,
                    produced: "first".into(),
                    chain: 1,
                    length: 2,
                }),
                extents: vec![crate::input_rows::Extent {
                    chain: 1,
                    offset: 0,
                    length: 2,
                    blob: hash,
                    blob_offset,
                }],
            }),
            material: Some(material.clone()),
        })),
    };
    apply(
        &mut index,
        &resolver,
        &[facts(7, "first", 1), facts(8, "second", 0), barrier(8)],
    )
    .unwrap();
    assert_eq!(
        material_at(&index, "first", 1, 1).unwrap(),
        Some(super::super::SourceLocation {
            material: "shared-material".into(),
            anchor: 2
        })
    );
    assert_eq!(
        material_at(&index, "second", 1, 1).unwrap(),
        Some(super::super::SourceLocation {
            material: "shared-material".into(),
            anchor: 1
        })
    );
    assert_eq!(
        crate::source::Source::admitted_through(&index, "first", 1, 2).unwrap(),
        Some(7)
    );
    let retained: (i64, Vec<u8>) = index
        .query_row("SELECT size,storage FROM source_blob", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    assert_eq!(retained, (4, extent.encode()));
    assert!(!bytes_root.exists());
}

mod chunks;
mod eligibility;
mod source_memberships;

mod replay;

mod completion;

#[test]
fn has_fact_follows_applied_positions_and_erasures_seen_first() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    let erased = Confirmed {
        native: None,
        position: 3,
        output_index: 0,
        event: "event-erased".into(),
        change: ConfirmedChange::Tombstoned,
    };
    // Staged but not yet complete: not in index.db.
    apply(&mut index, &resolver, &[input(2, "staged"), erased]).unwrap();
    assert!(!has_fact(&index, "event-staged").unwrap());
    apply(&mut index, &resolver, &[barrier(4)]).unwrap();
    assert!(has_fact(&index, "event-staged").unwrap());
    // An event erased before this index read it still leaves a fact.
    assert!(has_fact(&index, "event-erased").unwrap());
    assert!(!has_fact(&index, "event-never-delivered").unwrap());
}
