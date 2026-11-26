use super::*;
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn every_schema_applies_to_an_empty_database() {
    for (name, create) in [
        ("archive", ensure_archive as fn(&Connection) -> Result<()>),
        ("index", ensure_index),
        ("embeddings", ensure_embeddings),
    ] {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap_or_else(|e| panic!("{name}: {e}"));
        create(&conn).unwrap_or_else(|e| panic!("{name} again: {e}"));
    }
}

#[test]
fn every_database_starts_at_its_current_version() {
    assert_eq!(ARCHIVE_VERSION, 9);
    assert_eq!(INDEX_VERSION, 24);
    for (create, expected) in [
        (
            ensure_archive as fn(&Connection) -> Result<()>,
            ARCHIVE_VERSION,
        ),
        (ensure_index, INDEX_VERSION),
        (ensure_embeddings, 1),
    ] {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap();
        assert_eq!(version(&conn).unwrap(), expected);
    }
}

#[test]
fn an_index_at_another_version_is_refused_without_removing_rows() {
    let conn = Connection::open_in_memory().unwrap();
    ensure_index(&conn).unwrap();
    conn.execute("INSERT INTO input_ref(id) VALUES ('synthetic')", [])
        .unwrap();
    // An index written by an older build of another lineage.
    set_version(&conn, INDEX_VERSION + 1).unwrap();
    assert!(matches!(
        ensure_index(&conn),
        Err(StoreError::IndexVersion {
            found,
            wanted: INDEX_VERSION
        }) if found == INDEX_VERSION + 1
    ));
    assert_eq!(version(&conn).unwrap(), INDEX_VERSION + 1);
    assert_eq!(
        conn.query_row("SELECT id FROM input_ref", [], |row| row
            .get::<_, String>(0))
            .unwrap(),
        "synthetic"
    );
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE input_ref(pk INTEGER PRIMARY KEY, id TEXT) STRICT;
         INSERT INTO input_ref(id) VALUES ('synthetic');
         PRAGMA user_version = 1;",
    )
    .unwrap();
    // No baseline format identity: an index written by an earlier Polylogue.
    assert!(matches!(
        ensure_index(&conn),
        Err(StoreError::OlderFormat { database: "index" })
    ));
    assert_eq!(
        ensure_index(&conn).unwrap_err().to_string(),
        "index.db was written before this format's baseline and is left as it was; \
         import the original files it was built from with `polylogue import`"
    );
    assert_eq!(
        conn.query_row("SELECT id FROM input_ref", [], |row| row
            .get::<_, String>(0))
            .unwrap(),
        "synthetic"
    );
}

#[test]
fn a_newer_database_or_one_with_unknown_tables_is_refused() {
    let conn = Connection::open_in_memory().unwrap();
    ensure_archive(&conn).unwrap();
    set_version(&conn, ARCHIVE_VERSION + 1).unwrap();
    assert!(matches!(
        ensure_archive(&conn),
        Err(StoreError::NewerSchema { database: "archive", found, current })
            if found == ARCHIVE_VERSION + 1 && current == ARCHIVE_VERSION
    ));
    let conn = Connection::open_in_memory().unwrap();
    ensure_embeddings(&conn).unwrap();
    set_version(&conn, 2).unwrap();
    assert!(matches!(
        ensure_embeddings(&conn),
        Err(StoreError::NewerSchema {
            database: "embeddings",
            ..
        })
    ));
    // The current version with a table missing, as an archive written by a
    // build that declared the same version with other tables.
    for change in [
        "DROP TABLE revision",
        "ALTER TABLE annotation ADD COLUMN synthetic TEXT",
    ] {
        let conn = Connection::open_in_memory().unwrap();
        ensure_archive(&conn).unwrap();
        conn.execute_batch(change).unwrap();
        assert!(
            matches!(
                ensure_archive(&conn),
                Err(StoreError::OtherFormat {
                    database: "archive",
                    ..
                })
            ),
            "{change}"
        );
    }
}

#[test]
fn comments_and_spacing_in_the_declarations_do_not_change_the_schema() {
    let respaced: String = ARCHIVE
        .lines()
        .map(|line| {
            format!(
                "{}   -- synthetic remark, 'quoted'\n\n",
                line.split_whitespace().collect::<Vec<_>>().join("  ")
            )
        })
        .collect();
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(&respaced).unwrap();
    set_version(&conn, ARCHIVE_VERSION).unwrap();
    conn.pragma_update(None, "application_id", FORMAT_ID)
        .unwrap();
    ensure_archive(&conn).unwrap();
}

#[test]
fn open_indexes_are_added_in_place_and_survive_reopen() {
    let conn = Connection::open_in_memory().unwrap();
    base_index(&conn).unwrap();
    set_version(&conn, INDEX_VERSION).unwrap();
    conn.pragma_update(None, "application_id", FORMAT_ID)
        .unwrap();
    conn.execute("INSERT INTO input_ref(id) VALUES ('synthetic')", [])
        .unwrap();
    // Views are part of the schema; open indexes are recreated before writing.
    conn.execute_batch(OPEN_INDEXES).unwrap();
    assert!(same_index_schema_in(&conn).unwrap());
    ensure_index(&conn).unwrap();
    let base = Connection::open_in_memory().unwrap();
    base_index(&base).unwrap();
    let base_shape = schema_rows(&base).unwrap();
    base.execute_batch(OPEN_INDEXES).unwrap();
    let expected = schema_rows(&base).unwrap();
    let open_indexes: Vec<_> = schema_rows(&conn)
        .unwrap()
        .into_iter()
        .filter(|item| item.0 == "index" && !base_shape.contains(item))
        .map(|item| item.1)
        .collect();
    assert!(!open_indexes.is_empty());
    for name in open_indexes {
        conn.execute_batch(&format!("DROP INDEX {name}")).unwrap();
        assert!(same_index_schema_in(&conn).unwrap(), "missing {name}");
        ensure_index(&conn).unwrap();
        assert_eq!(schema_rows(&conn).unwrap(), expected);
    }
    assert_eq!(schema_rows(&conn).unwrap(), expected);
    assert_eq!(version(&conn).unwrap(), INDEX_VERSION);
    assert_eq!(
        conn.query_row("SELECT id FROM input_ref", [], |row| row
            .get::<_, String>(0))
            .unwrap(),
        "synthetic"
    );
    conn.execute_batch("DROP INDEX event_time; CREATE INDEX event_time ON event(kind)")
        .unwrap();
    assert!(!same_index_schema_in(&conn).unwrap());
}

#[test]
fn an_archive_written_before_the_baseline_is_an_older_format_and_stays_as_it_is() {
    // An archive with tables and a version but no format identity, as every
    // earlier Polylogue wrote it (master's archives are at version 4).
    for found in [1, 4] {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            "CREATE TABLE input(id TEXT PRIMARY KEY) STRICT;
             INSERT INTO input VALUES ('synthetic');
             PRAGMA user_version = {found};"
        ))
        .unwrap();
        let error = ensure_archive(&conn).unwrap_err();
        assert!(
            matches!(
                error,
                StoreError::OlderFormat {
                    database: "archive"
                }
            ),
            "{error}"
        );
        assert_eq!(
            error.to_string(),
            "archive.db was written before this format's baseline and is left as it was; \
             import the original files it was built from with `polylogue import`"
        );
        assert_eq!(version(&conn).unwrap(), found);
        assert_eq!(
            conn.query_row("SELECT id FROM input", [], |row| row.get::<_, String>(0))
                .unwrap(),
            "synthetic"
        );
    }
    let conn = Connection::open_in_memory().unwrap();
    ensure_archive(&conn).unwrap();
    assert_eq!(format_id(&conn).unwrap(), FORMAT_ID);
}

/// An unversioned file that already holds tables is not ours to create into:
/// it is refused before it takes the format identity or any table.
#[test]
fn an_unversioned_file_with_tables_is_refused_unchanged() {
    for (name, create) in [
        ("archive", ensure_archive as fn(&Connection) -> Result<()>),
        ("index", ensure_index),
    ] {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE unrelated(x INTEGER) STRICT; INSERT INTO unrelated VALUES (1);",
        )
        .unwrap();
        let error = create(&conn).unwrap_err();
        assert!(
            matches!(error, StoreError::OtherFormat { found: 0, .. }),
            "{name}: {error}"
        );
        assert_eq!(format_id(&conn).unwrap(), 0, "{name}");
        assert_eq!(version(&conn).unwrap(), 0, "{name}");
        assert_eq!(
            schema_rows(&conn).unwrap(),
            vec![(
                "table".to_owned(),
                "unrelated".to_owned(),
                statement("CREATE TABLE unrelated(x INTEGER) STRICT")
            )],
            "{name}"
        );
    }
}

#[test]
fn a_parse_state_row_cannot_hold_a_half_stated_failure() {
    let conn = Connection::open_in_memory().unwrap();
    ensure_index(&conn).unwrap();
    conn.execute_batch("INSERT INTO input_ref(pk,id,format) VALUES(1,'parsed','synthetic'),(2,'failed','synthetic');
        INSERT INTO parse_state(input_pk,chain,state,parser_version,parsed_to,waiting,resume) VALUES
          (1,1,'parsed','v1',40,60,x'00');
        INSERT INTO parse_state(input_pk,chain,state,parser_version,parsed_to,waiting,failure_kind,failure_detail) VALUES
          (2,1,'failed','v1',90,0,'malformed','line 3 is not JSON');").unwrap();
    assert_eq!(version(&conn).unwrap(), INDEX_VERSION);
    let refused = |sql: &str| conn.execute(sql, []).is_err();
    // A failure names its kind and its sentence, and only a failure does.
    assert!(refused(
        "UPDATE parse_state SET failure_detail=NULL WHERE input_pk=2"
    ));
    assert!(refused(
        "UPDATE parse_state SET failure_kind=NULL WHERE input_pk=2"
    ));
    assert!(refused(
        "UPDATE parse_state SET state='parsed' WHERE input_pk=2"
    ));
    assert!(refused(
        "UPDATE parse_state SET failure_kind='malformed',failure_detail='x' WHERE input_pk=1"
    ));
    // A failed chain carries no resume state; a kind outside the model's is refused.
    assert!(refused(
        "UPDATE parse_state SET resume=x'00' WHERE input_pk=2"
    ));
    assert!(refused(
        "UPDATE parse_state SET failure_kind='lost' WHERE input_pk=2"
    ));
    assert!(refused(
        "UPDATE parse_state SET waiting=-1 WHERE input_pk=1"
    ));
    for kind in polylogue_model::ChainFailureKind::NAMES {
        conn.execute(
            "UPDATE parse_state SET failure_kind=?1 WHERE input_pk=2",
            [kind],
        )
        .unwrap();
    }
}

mod embeddings;

/// The `CHECK(column IN ('a','b',...))` lists of every table, by table and
/// column, read back from `sqlite_master`.
fn check_lists(conn: &Connection) -> BTreeMap<(String, String), BTreeSet<String>> {
    let mut lists = BTreeMap::new();
    for (kind, table, sql) in schema_rows(conn).unwrap() {
        if kind != "table" {
            continue;
        }
        let mut rest = sql.as_str();
        while let Some(start) = rest.find("CHECK") {
            rest = &rest[start + "CHECK".len()..];
            let Some(inner) = rest.trim_start().strip_prefix('(') else {
                continue;
            };
            let Some((column, list)) = inner.trim_start().split_once(" IN") else {
                continue;
            };
            let column = column.trim();
            let Some(list) = list.trim_start().strip_prefix("('") else {
                continue;
            };
            if !column
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                continue;
            }
            let Some((list, after)) = list.split_once("')") else {
                continue;
            };
            if !after.trim_start().starts_with(')') {
                continue;
            }
            let names = list
                .split(',')
                .map(|name| name.trim().trim_matches('\'').to_owned());
            lists.insert((table.clone(), column.to_owned()), names.collect());
        }
    }
    lists
}

/// Every closed set the archive and the index constrain
/// with a `CHECK` list is exactly its enum's `NAMES`.
#[test]
fn schema_check_lists_come_from_names() {
    use polylogue_model::{
        Completion, DeliveryOutcome, FieldGroup, IdKind, Route, authored::RunsHere,
        query_objects::QueryEdgeKind,
    };
    let archive = Connection::open_in_memory().unwrap();
    ensure_archive(&archive).unwrap();
    let index = Connection::open_in_memory().unwrap();
    ensure_index(&index).unwrap();
    let mut found = check_lists(&archive);
    found.extend(check_lists(&index));
    // The content group's route is the message's main route, so
    // `message_route` records every other group.
    let routed: Vec<&str> = FieldGroup::ALL
        .iter()
        .filter(|group| **group != FieldGroup::Content)
        .map(|group| group.as_str())
        .collect();
    let record: &[(&str, &str, &[&str])] = &[
        ("delivery_event", "outcome", DeliveryOutcome::NAMES),
        ("chat_action", "execution", RunsHere::NAMES),
        ("import_job", "execution", RunsHere::NAMES),
        ("query_edge", "kind", QueryEdgeKind::NAMES),
        ("revision", "produced", crate::input_rows::Produced::NAMES),
        (
            "observation",
            "class",
            crate::input_rows::ObservationClass::NAMES,
        ),
    ];
    let precisions: Vec<&str> = polylogue_model::Precision::ALL
        .iter()
        .map(|precision| precision.as_str())
        .collect();
    let index_sets: [(&str, &str, &[&str]); 29] = [
        ("session", "kind", polylogue_model::SessionKind::NAMES),
        ("session", "started_precision", &precisions[..]),
        ("session", "last_active_precision", &precisions[..]),
        ("message", "role", polylogue_model::Role::NAMES),
        ("message", "written_by", polylogue_model::WrittenBy::NAMES),
        ("message", "prompt_kind", polylogue_model::PromptKind::NAMES),
        ("message", "at_precision", &precisions[..]),
        ("event", "at_precision", &precisions[..]),
        ("document", "at_precision", &precisions[..]),
        ("block", "kind", polylogue_model::BlockKind::NAMES),
        ("tool_call", "action", polylogue_model::ToolAction::NAMES),
        ("tool_call", "outcome", polylogue_model::CallOutcome::NAMES),
        ("link_statement", "kind", polylogue_model::LinkKind::NAMES),
        (
            "link_statement",
            "how",
            polylogue_model::ReferenceReason::NAMES,
        ),
        ("session_link", "kind", polylogue_model::LinkKind::NAMES),
        (
            "session_link",
            "how",
            polylogue_model::ReferenceReason::NAMES,
        ),
        ("link_parse", "kind", polylogue_model::LinkKind::NAMES),
        ("link_parse", "how", polylogue_model::ReferenceReason::NAMES),
        ("input_ref", "route", Route::NAMES),
        ("message_parse", "id_kind", IdKind::NAMES),
        ("message_parse", "route", Route::NAMES),
        ("document_parse", "route", Route::NAMES),
        ("event_parse", "route", Route::NAMES),
        ("link_parse", "route", Route::NAMES),
        ("message_parse", "completion", Completion::NAMES),
        ("message_route", "field_group", &routed[..]),
        ("parse_state", "state", crate::parse_state::StateWord::NAMES),
        (
            "parse_state",
            "failure_kind",
            polylogue_model::ChainFailureKind::NAMES,
        ),
        ("session_time", "party", crate::measure::time::Party::NAMES),
    ];
    let expected: BTreeMap<(String, String), BTreeSet<String>> = index_sets
        .into_iter()
        .map(|(table, column, names)| (table.to_owned(), column, names))
        .chain(
            record
                .iter()
                .map(|(table, column, names)| ((*table).to_owned(), *column, *names)),
        )
        .map(|(table, column, names)| {
            (
                (table, column.to_owned()),
                names.iter().map(|name| (*name).to_owned()).collect(),
            )
        })
        .collect();
    assert_eq!(found, expected);
}
