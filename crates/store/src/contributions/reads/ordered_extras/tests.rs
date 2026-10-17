use super::*;
#[test]
fn ordered_extras_connection_allows_only_private_temp_writes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("main.db");
    Connection::open(&path)
        .unwrap()
        .execute_batch(
            "CREATE TABLE retained(value TEXT); INSERT INTO retained VALUES('original');",
        )
        .unwrap();
    let connection = crate::open(&path, true, false).unwrap();
    configure(&connection, &path).unwrap();
    assert_eq!(
        connection
            .query_row("PRAGMA temp_store", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        connection
            .query_row("PRAGMA temp.cache_size", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        -64
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT sqlite_compileoption_used('TEMP_STORE=3')",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
    connection.execute_batch("BEGIN; CREATE TEMP TABLE polylogue_extra_order_1(sort_key BLOB PRIMARY KEY,semantic_key BLOB NOT NULL) STRICT,WITHOUT ROWID; INSERT INTO temp.polylogue_extra_order_1 VALUES(x'61',x'62'); DROP TABLE temp.polylogue_extra_order_1;").unwrap();
    for sql in [
        "UPDATE main.retained SET value='changed'",
        "CREATE TABLE main.bad(value)",
        "ATTACH ':memory:' AS other",
        "PRAGMA query_only=OFF",
        "CREATE TEMP TABLE unrelated(value)",
    ] {
        assert!(connection.execute_batch(sql).is_err(), "{sql}");
    }
    let denied = dir.path().join("must-not-exist.db");
    assert!(
        connection
            .execute("ATTACH ?1 AS durable", [denied.to_str().unwrap()])
            .is_err()
    );
    assert!(!denied.exists());
    assert_eq!(
        connection
            .query_row("SELECT value FROM retained", [], |r| r.get::<_, String>(0))
            .unwrap(),
        "original"
    );
}

#[test]
fn ordered_extras_corrupt_oversized_scratch_cells_are_sql_null_before_materialization() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("main.db");
    Connection::open(&path).unwrap();
    let connection = crate::open(&path, true, false).unwrap();
    configure(&connection, &path).unwrap();
    connection.execute_batch("CREATE TEMP TABLE polylogue_extra_order_1(sort_key BLOB PRIMARY KEY,semantic_key BLOB NOT NULL) STRICT,WITHOUT ROWID; INSERT INTO temp.polylogue_extra_order_1 VALUES(x'61',x'62');").unwrap();
    assert_eq!(
        read_next_cells(&connection, "polylogue_extra_order_1", None, 64)
            .unwrap()
            .unwrap()
            .admit(64)
            .unwrap(),
        (vec![b'a'], vec![b'b'])
    );
    assert_eq!(
        read_next_cells(&connection, "polylogue_extra_order_1", Some(b""), 64)
            .unwrap()
            .unwrap()
            .admit(64)
            .unwrap(),
        (vec![b'a'], vec![b'b'])
    );
    for column in ["sort_key", "semantic_key"] {
        connection.execute_batch("DELETE FROM temp.polylogue_extra_order_1; INSERT INTO temp.polylogue_extra_order_1 VALUES(x'61',x'62');").unwrap();
        connection
            .execute_batch(&format!(
                "UPDATE temp.polylogue_extra_order_1 SET {column}=zeroblob(1048576)"
            ))
            .unwrap();
        let cells = read_next_cells(&connection, "polylogue_extra_order_1", None, 64)
            .unwrap()
            .unwrap();
        if column == "sort_key" {
            assert!(cells.key.is_none());
            assert_eq!(cells.key_length, 1048576);
        } else {
            assert!(cells.semantic.is_none());
            assert_eq!(cells.semantic_length, 1048576);
        }
        assert!(matches!(
            cells.admit(64),
            Err(Error::ParserAdmission {
                resource: "ordered extra key",
                required: 1048576
            })
        ));
    }
}
