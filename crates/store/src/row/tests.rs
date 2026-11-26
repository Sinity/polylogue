use super::*;
use crate::fixture::Fixture;
use polylogue_model::{Role, SessionId};
use proptest::prelude::*;

row! {
    /// An input reference, inserted and read back through declared statements.
    struct InputRefRow {
        pk: InputPk,
        id: String,
        format: String,
        route: polylogue_model::Route,
    }
    statements {
        const INSERT = "INSERT INTO input_ref(pk, id, format, route) VALUES (?1, ?2, ?3, ?4)";
        const ALL = "SELECT pk, id, format, route FROM input_ref ORDER BY pk";
    }
}

row! {
    /// A row naming its columns in a different order than its fields.
    struct SwappedRow {
        id: String,
        pk: InputPk,
    }
    statements {
        const READ = "SELECT pk, id FROM input_ref";
        const INSERT = "INSERT INTO input_ref(pk, id) VALUES (?1, ?2)";
    }
}

#[test]
fn every_declared_statement_prepares_and_names_its_struct_fields() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut check = Check {
        conn: reader.conn().unwrap(),
        checked: 0,
        failures: Vec::new(),
    };
    check.row::<InputRefRow>();
    crate::read::declared_rows(&mut check);
    crate::lineage::declared_rows(&mut check);
    crate::documents::declared_rows(&mut check);
    crate::commit_positions::declared_rows(&mut check);
    crate::text_rows::declared_rows(&mut check);
    crate::archive::writer::agreement::declared_rows(&mut check);
    crate::summary::lines::declared_rows(&mut check);
    assert!(check.failures.is_empty(), "{:#?}", check.failures);
    assert!(check.checked > 2);
}

#[test]
fn a_statement_whose_columns_differ_from_its_fields_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut check = Check {
        conn: reader.conn().unwrap(),
        checked: 0,
        failures: Vec::new(),
    };
    check.row::<SwappedRow>();
    assert_eq!(
        check.failures,
        [
            "SwappedRow::READ reads columns [\"pk\", \"id\"], but its row has fields [\"id\", \"pk\"]",
            "SwappedRow::INSERT inserts columns [\"pk\", \"id\"], but its row has fields [\"id\", \"pk\"]",
        ]
    );
}

#[test]
fn a_row_inserted_through_its_statement_reads_back_equal() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE input_ref (pk INTEGER PRIMARY KEY, id TEXT NOT NULL, format TEXT NOT NULL,
         route TEXT NOT NULL) STRICT",
    )
    .unwrap();
    let rows = [
        InputRefRow {
            pk: InputPk(7),
            id: "input:a".into(),
            format: "codex".into(),
            route: polylogue_model::Route::Live,
        },
        InputRefRow {
            pk: InputPk(9),
            id: "input:b".into(),
            format: String::new(),
            route: polylogue_model::Route::Export,
        },
    ];
    for row in &rows {
        conn.insert(&InputRefRow::INSERT, row).unwrap();
    }
    assert_eq!(conn.all(&InputRefRow::ALL, []).unwrap(), rows);
    assert_eq!(
        conn.maybe(&InputRefRow::ALL, []).unwrap().as_ref(),
        Some(&rows[0])
    );
    assert_eq!(
        conn.value::<InputPk>("SELECT max(pk) FROM input_ref", [])
            .unwrap(),
        Some(InputPk(9))
    );
}

#[test]
fn a_stored_value_the_field_cannot_hold_names_the_field() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE input_ref (pk INTEGER, id TEXT, format TEXT, route TEXT);
         INSERT INTO input_ref VALUES (1, 'input:a', 'codex', 'sideways');",
    )
    .unwrap();
    let error = conn.all(&InputRefRow::ALL, []).unwrap_err();
    assert_eq!(
        error.to_string(),
        StoreError::stored(3, "field route: unknown stored name \"sideways\"").to_string()
    );
}

row! {
    /// One field of every kind a column can be declared as.
    struct EveryKind {
        signed: i64,
        small: i32,
        count: u64,
        narrow: u32,
        flag: bool,
        real: f64,
        text: String,
        bytes: Vec<u8>,
        hash: [u8; 32],
        maybe_text: Option<String>,
        maybe_number: Option<i64>,
        session_pk: SessionPk,
        role: Role,
        session: SessionId,
        at: Timestamp,
        json: Json<Vec<String>>,
    }
}

fn every_kind() -> impl Strategy<Value = EveryKind> {
    (
        (
            any::<i64>(),
            any::<i32>(),
            0..=i64::MAX as u64,
            any::<u32>(),
            any::<bool>(),
        ),
        (
            -1e300f64..1e300,
            ".*",
            prop::collection::vec(any::<u8>(), 0..64),
        ),
        (
            any::<[u8; 32]>(),
            any::<Option<String>>(),
            any::<Option<i64>>(),
            any::<i64>(),
        ),
        (
            prop::sample::select(Role::ALL.to_vec()),
            "[a-z]{1,8}",
            -62_000_000_000_000i64..250_000_000_000_000,
            prop::collection::vec(".*", 0..4),
        ),
    )
        .prop_map(
            |(
                (signed, small, count, narrow, flag),
                (real, text, bytes),
                (hash, maybe_text, maybe_number, key),
                (role, native, ms, json),
            )| EveryKind {
                signed,
                small,
                count,
                narrow,
                flag,
                real,
                text,
                bytes,
                hash,
                maybe_text,
                maybe_number,
                session_pk: SessionPk(key),
                role,
                session: SessionId::parse(&format!("codex:{native}")).unwrap(),
                at: Timestamp::from_unix_ms(ms).unwrap(),
                json: Json(json),
            },
        )
}

proptest! {
    #[test]
    fn rows_read_back_from_their_byte_form_and_from_sqlite(rows in prop::collection::vec(every_kind(), 1..8)) {
        let mut bytes = Vec::new();
        for row in &rows {
            row.to_bytes(&mut bytes);
        }
        let mut input = bytes.as_slice();
        let mut read = Vec::new();
        while !input.is_empty() {
            read.push(EveryKind::from_bytes(&mut input).unwrap());
        }
        prop_assert_eq!(&read, &rows);

        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            "CREATE TABLE every_kind ({})",
            EveryKind::FIELDS.join(",")
        ))
        .unwrap();
        let insert = Statement::<EveryKind>::built(format!(
            "INSERT INTO every_kind({}) VALUES ({})",
            EveryKind::FIELDS.join(","),
            vec!["?"; EveryKind::FIELDS.len()].join(",")
        ));
        for row in &rows {
            conn.insert(&insert, row).unwrap();
        }
        let select = Statement::<EveryKind>::built(format!(
            "SELECT {} FROM every_kind ORDER BY rowid",
            EveryKind::FIELDS.join(",")
        ));
        prop_assert_eq!(conn.all(&select, []).unwrap(), rows);
    }
}

#[test]
fn a_truncated_byte_form_is_refused() {
    let row = InputRefRow {
        pk: InputPk(300),
        id: "input:a".into(),
        format: "codex".into(),
        route: polylogue_model::Route::Live,
    };
    let mut bytes = Vec::new();
    row.to_bytes(&mut bytes);
    for end in 0..bytes.len() {
        assert!(
            InputRefRow::from_bytes(&mut &bytes[..end]).is_err(),
            "{end}"
        );
    }
    assert_eq!(InputRefRow::from_bytes(&mut bytes.as_slice()).unwrap(), row);
}
