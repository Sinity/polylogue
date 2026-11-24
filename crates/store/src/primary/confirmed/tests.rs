use super::*;
use std::sync::Arc;

fn cache(path: &std::path::Path) -> Connection {
    let index = Connection::open(path).unwrap();
    crate::schema::ensure_primary_index(&index).unwrap();
    index
}

fn input(position: i64, id: &str) -> Confirmed {
    Confirmed {
        sinex_context: None,
        position,
        output_index: 0,
        event: format!("event-{id}"),
        change: ConfirmedChange::Live(Box::new(Product::Input {
            rows: Box::new(crate::input_rows::InputRows {
                capture_read: None,
                input: id.into(),
                origin: "synthetic".into(),
                revision: None,
                extents: Vec::new(),
                observation: None,
            }),
            material: None,
        })),
    }
}
fn complete_at(position: i64) -> Confirmed {
    Confirmed {
        sinex_context: None,
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
fn completion_marker_commits_staged_pages_atomically() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.db");
    let mut index = cache(&path);
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    assert_eq!(
        apply(
            &crate::text_rows::TextIds::new(
                &index,
                false,
                crate::archive::MemoryBudget::default().keys() / 3
            )
            .unwrap(),
            &mut index,
            &[input(2, "first")]
        )
        .unwrap(),
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
    index.execute_batch("CREATE TRIGGER stop_confirmed_apply BEFORE INSERT ON source_input WHEN new.id='second' BEGIN SELECT raise(ABORT,'stop'); END").unwrap();
    assert!(
        apply(
            &crate::text_rows::TextIds::new(
                &index,
                false,
                crate::archive::MemoryBudget::default().keys() / 3
            )
            .unwrap(),
            &mut index,
            &[input(4, "second"), complete_at(5)]
        )
        .is_err()
    );
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
            .query_row("SELECT count(*) FROM confirmed_event", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    index
        .execute_batch("DROP TRIGGER stop_confirmed_apply")
        .unwrap();
    assert_eq!(
        apply(
            &crate::text_rows::TextIds::new(
                &index,
                false,
                crate::archive::MemoryBudget::default().keys() / 3
            )
            .unwrap(),
            &mut index,
            &[]
        )
        .unwrap(),
        5
    );
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM source_input", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        2
    );
    assert_eq!(
        apply(
            &crate::text_rows::TextIds::new(
                &index,
                false,
                crate::archive::MemoryBudget::default().keys() / 3
            )
            .unwrap(),
            &mut index,
            &[input(2, "first"), complete_at(5)]
        )
        .unwrap(),
        5
    );
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM confirmed_event", [], |row| row
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
    let _archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    apply(
        &crate::text_rows::TextIds::new(
            &index,
            false,
            crate::archive::MemoryBudget::default().keys() / 3,
        )
        .unwrap(),
        &mut index,
        &[complete_at(0)],
    )
    .unwrap();
    assert_eq!(checkpoint(&index).unwrap(), Some(0));
    assert!(
        apply(
            &crate::text_rows::TextIds::new(
                &index,
                false,
                crate::archive::MemoryBudget::default().keys() / 3
            )
            .unwrap(),
            &mut index,
            &[input(0, "invalid")]
        )
        .is_err()
    );
    assert_eq!(checkpoint(&index).unwrap(), Some(0));
}

#[test]
fn source_lifecycle_keeps_recorded_authored_effects() {
    use polylogue_model::user::{UserRequest, UserWrite};
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    let _archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
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
        &crate::text_rows::TextIds::new(
            &index,
            false,
            crate::archive::MemoryBudget::default().keys() / 3,
        )
        .unwrap(),
        &mut index,
        &[
            input(1, "source"),
            Confirmed {
                sinex_context: None,
                position: 2,
                output_index: 0,
                event: "authored".into(),
                change: ConfirmedChange::Live(Box::new(Product::Request(prepared))),
            },
            complete_at(2),
        ],
    )
    .unwrap();
    for (position, change, present) in [
        (3, ConfirmedChange::Archived, 0),
        (4, ConfirmedChange::Restored, 1),
        (5, ConfirmedChange::Tombstoned, 0),
    ] {
        apply(
            &crate::text_rows::TextIds::new(
                &index,
                false,
                crate::archive::MemoryBudget::default().keys() / 3,
            )
            .unwrap(),
            &mut index,
            &[
                Confirmed {
                    sinex_context: None,
                    position,
                    output_index: 0,
                    event: "event-source".into(),
                    change,
                },
                complete_at(position),
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
            "SELECT json(value) FROM source_annotation WHERE kind='setting' AND key='browser.actions'",
            [],|row|row.get(0),
        ).unwrap();
        let setting: polylogue_model::annotation::values::Setting =
            serde_json::from_str(&setting).unwrap();
        assert_eq!(
            setting.value,
            polylogue_model::settings::Value::Boolean(true)
        );
    }
    assert!(
        apply(
            &crate::text_rows::TextIds::new(
                &index,
                false,
                crate::archive::MemoryBudget::default().keys() / 3
            )
            .unwrap(),
            &mut index,
            &[
                Confirmed {
                    sinex_context: None,
                    position: 6,
                    output_index: 0,
                    event: "event-source".into(),
                    change: ConfirmedChange::Restored
                },
                complete_at(6)
            ]
        )
        .is_err()
    );
    assert_eq!(position(&index).unwrap(), 5);
    let body: Option<Vec<u8>> = index
        .query_row(
            "SELECT product FROM confirmed_event WHERE event='event-source'",
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
        .prepare("SELECT key FROM source_annotation WHERE kind='setting' ORDER BY key")
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
    let _archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let live = |position, event: &str, product| Confirmed {
        sinex_context: None,
        position,
        output_index: 0,
        event: event.into(),
        change: ConfirmedChange::Live(Box::new(product)),
    };
    let first = setting(&index, "first-request", "browser.actions");
    apply(
        &crate::text_rows::TextIds::new(
            &index,
            false,
            crate::archive::MemoryBudget::default().keys() / 3,
        )
        .unwrap(),
        &mut index,
        &[live(2, "first", first), complete_at(2)],
    )
    .unwrap();
    assert_eq!(settings(&index), ["browser.actions"]);
    let second = setting(&index, "second-request", "drive.ai-studio");
    apply(
        &crate::text_rows::TextIds::new(
            &index,
            false,
            crate::archive::MemoryBudget::default().keys() / 3,
        )
        .unwrap(),
        &mut index,
        &[live(3, "second", second), complete_at(3)],
    )
    .unwrap();
    assert_eq!(settings(&index), ["browser.actions", "drive.ai-studio"]);
    apply(
        &crate::text_rows::TextIds::new(
            &index,
            false,
            crate::archive::MemoryBudget::default().keys() / 3,
        )
        .unwrap(),
        &mut index,
        &[
            Confirmed {
                sinex_context: None,
                position: 4,
                output_index: 0,
                event: "first".into(),
                change: ConfirmedChange::Archived,
            },
            complete_at(4),
        ],
    )
    .unwrap();
    assert_eq!(settings(&index), ["drive.ai-studio"]);
    apply(
        &crate::text_rows::TextIds::new(
            &index,
            false,
            crate::archive::MemoryBudget::default().keys() / 3,
        )
        .unwrap(),
        &mut index,
        &[
            Confirmed {
                sinex_context: None,
                position: 5,
                output_index: 0,
                event: "first".into(),
                change: ConfirmedChange::Restored,
            },
            complete_at(5),
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
    let _archive = crate::Archive::on_sinex(dir.path(), &bytes_root);
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
    let input_at = |position, id: &str, blob_offset| Confirmed {
        sinex_context: None,
        position,
        output_index: 0,
        event: format!("event-{id}"),
        change: ConfirmedChange::Live(Box::new(Product::Input {
            rows: Box::new(crate::input_rows::InputRows {
                capture_read: None,
                input: id.into(),
                origin: "synthetic".into(),
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
        &crate::text_rows::TextIds::new(
            &index,
            false,
            crate::archive::MemoryBudget::default().keys() / 3,
        )
        .unwrap(),
        &mut index,
        &[
            input_at(7, "first", 1),
            input_at(8, "second", 0),
            complete_at(8),
        ],
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
        crate::source::Source::committed_through(&index, "first", 1, 2).unwrap(),
        Some(7)
    );
    let kept: (i64, Vec<u8>) = index
        .query_row("SELECT size,storage FROM source_blob", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    assert_eq!(kept, (4, extent.encode()));
    assert!(!bytes_root.exists());
}

mod chunks;
mod sending;
mod source_memberships;

mod replay;

mod completion;

#[test]
fn has_confirmed_event_follows_positions_and_early_erasures() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    let _archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let erased = Confirmed {
        sinex_context: None,
        position: 3,
        output_index: 0,
        event: "event-erased".into(),
        change: ConfirmedChange::Tombstoned,
    };
    // Staged but not yet complete: not in index.db.
    apply(
        &crate::text_rows::TextIds::new(
            &index,
            false,
            crate::archive::MemoryBudget::default().keys() / 3,
        )
        .unwrap(),
        &mut index,
        &[input(2, "staged"), erased],
    )
    .unwrap();
    assert!(!has_confirmed_event(&index, "event-staged").unwrap());
    apply(
        &crate::text_rows::TextIds::new(
            &index,
            false,
            crate::archive::MemoryBudget::default().keys() / 3,
        )
        .unwrap(),
        &mut index,
        &[complete_at(4)],
    )
    .unwrap();
    assert!(has_confirmed_event(&index, "event-staged").unwrap());
    // An event erased before this index read it still leaves a row.
    assert!(has_confirmed_event(&index, "event-erased").unwrap());
    assert!(!has_confirmed_event(&index, "event-never-delivered").unwrap());
}

#[test]
fn prepared_position_cannot_skip_new_sinex_events() {
    let directory = tempfile::tempdir().unwrap();
    let mut index = cache(&directory.path().join("index.db"));
    let _archive = crate::Archive::on_sinex(directory.path(), &directory.path().join("bytes"));
    let Step::Prepare(prepared) =
        stage_next(&mut index, &[input(2, "first"), complete_at(5)]).unwrap()
    else {
        panic!("position should need application");
    };
    stage_next(&mut index, &[input(4, "second")]).unwrap();
    assert!(matches!(
        apply_prepared(
            &crate::text_rows::TextIds::new(
                &index,
                false,
                crate::archive::MemoryBudget::default().keys() / 3
            )
            .unwrap(),
            &mut index,
            *prepared
        ),
        Err(StoreError::ScopeChanged)
    ));
    assert_eq!(checkpoint(&index).unwrap(), None);
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM source_input", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        apply(
            &crate::text_rows::TextIds::new(
                &index,
                false,
                crate::archive::MemoryBudget::default().keys() / 3
            )
            .unwrap(),
            &mut index,
            &[]
        )
        .unwrap(),
        5
    );
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM source_input", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        2
    );
}

#[test]
fn prepared_application_respects_a_failed_scope_reset_fence() {
    let directory = tempfile::tempdir().unwrap();
    let mut index = cache(&directory.path().join("index.db"));
    let _archive = crate::Archive::on_sinex(directory.path(), &directory.path().join("bytes"));
    let Step::Prepare(prepared) =
        stage_next(&mut index, &[input(2, "first"), complete_at(2)]).unwrap()
    else {
        panic!("position should need application");
    };
    index
        .execute(
            "INSERT INTO meta(key,value) VALUES ('confirmed_scope_pending','replacement')",
            [],
        )
        .unwrap();
    assert!(matches!(
        apply_prepared(
            &crate::text_rows::TextIds::new(
                &index,
                false,
                crate::archive::MemoryBudget::default().keys() / 3
            )
            .unwrap(),
            &mut index,
            *prepared
        ),
        Err(StoreError::ScopeChanged)
    ));
    assert_eq!(checkpoint(&index).unwrap(), None);
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM source_input", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    index
        .execute("DELETE FROM meta WHERE key='confirmed_scope_pending'", [])
        .unwrap();
    assert_eq!(
        apply(
            &crate::text_rows::TextIds::new(
                &index,
                false,
                crate::archive::MemoryBudget::default().keys() / 3
            )
            .unwrap(),
            &mut index,
            &[]
        )
        .unwrap(),
        2
    );
}

#[test]
fn mixed_live_and_restored_authored_records_keep_first_execution_and_commit_order() {
    use polylogue_model::{
        BlobHash,
        authored::RunsHere,
        user::{NewChatAction, UserWrite},
    };
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    let _archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let context = crate::user_writes::PreparationContext::new(vec!["codex".into()]);
    let prepared = |id: &str| {
        let request = crate::fixture::authored::operator_request(
            id.into(),
            UserWrite::ChatEnqueue(NewChatAction {
                id: id.into(),
                request_key: format!("key-{id}"),
                request_hash: BlobHash::of(id.as_bytes()),
                request: serde_json::json!({"text":id}),
                stage_only: true,
                needs_approval: false,
                attachments: vec![],
            }),
        );
        crate::primary::authored::prepare(
            &index,
            &request,
            Some(&context),
            None,
            |_| Ok(()),
            |_| Ok(()),
        )
        .unwrap()
    };
    let live = prepared("live");
    let historical = prepared("historical");
    let historical_first = prepared("historical-first");
    let live_first = prepared("live-first");
    let data = [
        (2, "z-live", Product::Request(live)),
        (
            3,
            "y-historical",
            Product::RestoredRequest {
                prepared: historical,
            },
        ),
        (
            4,
            "x-historical-first",
            Product::RestoredRequest {
                prepared: historical_first.clone(),
            },
        ),
        (5, "c-later-live", Product::Request(historical_first)),
        (6, "b-live-first", Product::Request(live_first.clone())),
        (
            7,
            "a-later-historical",
            Product::RestoredRequest {
                prepared: live_first,
            },
        ),
    ];
    let mut events: Vec<_> = data
        .into_iter()
        .rev()
        .map(|(position, event, product)| Confirmed {
            sinex_context: None,
            position,
            output_index: 0,
            event: event.into(),
            change: ConfirmedChange::Live(Box::new(product)),
        })
        .collect();
    events.push(complete_at(7));
    let texts = crate::text_rows::TextIds::new(
        &index,
        false,
        crate::archive::MemoryBudget::default().keys() / 3,
    )
    .unwrap();
    for _ in 0..2 {
        assert_eq!(apply(&texts, &mut index, &events).unwrap(), 7);
        for (id, execution, committed) in [
            ("live", RunsHere::Live, 2),
            ("historical", RunsHere::Historical, 3),
            ("historical-first", RunsHere::Historical, 4),
            ("live-first", RunsHere::Live, 6),
        ] {
            let action = crate::chat_actions::read(&index, "source_", id)
                .unwrap()
                .unwrap();
            assert_eq!(action.execution, execution, "{id}");
            assert_eq!(action.events.len(), 1, "{id}");
            let position: i64 = index
                .query_row(
                    "SELECT commit_seq FROM source_client_request WHERE id=?1",
                    [id],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(position, committed, "{id}");
        }
        let count: i64 = index
            .query_row("SELECT count(*) FROM source_chat_action", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 4);
    }
}
